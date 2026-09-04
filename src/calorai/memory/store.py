"""Durable memory: profile facts and learned aliases.

Two stores, deliberately separate, and neither one is conversation history.

**profile_facts** -- small, durable attributes ('vegetarian', 'protein target
140g'). The whole store is loaded on every turn. That sounds naive until you
notice it is the point: keeping the corpus to a couple of dozen one-line facts
means retrieval never needs to be solved. No embeddings, no vector store, no
similarity threshold to tune, and nothing on the latency path.

**aliases** -- learned shorthand ('my usual' -> two parathas and a chai).
Resolved by exact phrase match *before* the model is called, so the model sees
concrete food, not a pronoun it has to guess at.

Note what is NOT here: 'same as yesterday'. That is a query against the meal
log, not a memory lookup, and it is handled by the find_meals tool. Conflating
the two is the mistake that makes people reach for a vector store they do not
need.
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import date, timedelta
from typing import Any

from ..db import utcnow

# An alias unused for this long stops being shorthand and starts being noise.
ALIAS_DECAY_DAYS = 60

# How many times a meal must repeat before it is offered as "your usual".
ALIAS_INFERENCE_THRESHOLD = 3

# Phrases that mean "you know the one". Matched literally -- cheap, and it
# cannot hallucinate a meal the way a model asked to guess would.
_ALIAS_TRIGGERS = [
    "my usual", "the usual", "usual please", "as usual", "same usual",
]


# ---------------------------------------------------------------------------
# profile facts
# ---------------------------------------------------------------------------
def get_facts(conn: sqlite3.Connection, user_id: str) -> list[dict[str, Any]]:
    """Live facts only. Superseded rows stay in the table for audit."""
    rows = conn.execute(
        "SELECT key, value, confidence FROM profile_facts"
        " WHERE user_id = ? AND superseded_by IS NULL ORDER BY id",
        (user_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def put_fact(
    conn: sqlite3.Connection,
    user_id: str,
    key: str,
    value: str,
    confidence: float = 0.8,
    source_message: str | None = None,
) -> dict[str, Any]:
    """Write a fact, superseding any previous value for the same key.

    Forgetting is explicit: the old row is marked, never deleted, so 'when did
    it learn this' stays answerable. Newest wins on contradiction.
    """
    existing = conn.execute(
        "SELECT id, value FROM profile_facts"
        " WHERE user_id = ? AND key = ? AND superseded_by IS NULL",
        (user_id, key),
    ).fetchone()

    if existing and existing["value"].strip().lower() == value.strip().lower():
        return {"ok": True, "changed": False, "key": key, "value": value}

    cur = conn.execute(
        "INSERT INTO profile_facts (user_id, key, value, confidence, source_message, created_at)"
        " VALUES (?,?,?,?,?,?)",
        (user_id, key, value, confidence, source_message, utcnow()),
    )
    new_id = cur.lastrowid
    if existing:
        conn.execute(
            "UPDATE profile_facts SET superseded_by = ? WHERE id = ?",
            (new_id, existing["id"]),
        )
    conn.commit()
    return {
        "ok": True,
        "changed": True,
        "key": key,
        "value": value,
        "replaced": existing["value"] if existing else None,
    }


def forget_fact(conn: sqlite3.Connection, user_id: str, key: str) -> bool:
    cur = conn.execute(
        "UPDATE profile_facts SET superseded_by = -1"
        " WHERE user_id = ? AND key = ? AND superseded_by IS NULL",
        (user_id, key),
    )
    conn.commit()
    return cur.rowcount > 0


# ---------------------------------------------------------------------------
# aliases
# ---------------------------------------------------------------------------
def put_alias(
    conn: sqlite3.Connection,
    user_id: str,
    phrase: str,
    items: list[dict[str, Any]],
    slot: str | None = None,
    source: str = "explicit",
) -> None:
    conn.execute(
        "INSERT INTO aliases (user_id, phrase, items_json, slot, source, created_at, last_used_at)"
        " VALUES (?,?,?,?,?,?,?)"
        " ON CONFLICT(user_id, phrase) DO UPDATE SET"
        " items_json=excluded.items_json, slot=excluded.slot, source=excluded.source",
        (user_id, phrase.lower().strip(), json.dumps(items), slot, source, utcnow(), utcnow()),
    )
    conn.commit()


def get_aliases(conn: sqlite3.Connection, user_id: str) -> list[dict[str, Any]]:
    cutoff = (date.today() - timedelta(days=ALIAS_DECAY_DAYS)).isoformat()
    rows = conn.execute(
        "SELECT phrase, items_json, slot, hits, source FROM aliases"
        " WHERE user_id = ? AND (last_used_at IS NULL OR last_used_at >= ?)"
        " ORDER BY hits DESC",
        (user_id, cutoff),
    ).fetchall()
    out = []
    for row in rows:
        entry = dict(row)
        entry["items"] = json.loads(entry.pop("items_json"))
        out.append(entry)
    return out


def resolve_alias(
    conn: sqlite3.Connection, user_id: str, text: str
) -> dict[str, Any] | None:
    """Deterministic lookup, run before the model sees the message.

    Returns the stored meal for 'my usual' and friends. Cheap, exact, and it
    keeps a pronoun from ever reaching the model as something to guess at.
    """
    low = text.lower().strip()
    stored = get_aliases(conn, user_id)

    for entry in stored:
        if entry["phrase"] and entry["phrase"] in low:
            _touch(conn, user_id, entry["phrase"])
            return entry

    # Generic trigger with nothing learned yet -- try to infer one from history
    # rather than replying "I don't know what your usual is".
    if any(trigger in low for trigger in _ALIAS_TRIGGERS):
        inferred = infer_usual(conn, user_id)
        if inferred:
            return inferred
    return None


def _touch(conn: sqlite3.Connection, user_id: str, phrase: str) -> None:
    conn.execute(
        "UPDATE aliases SET hits = hits + 1, last_used_at = ?"
        " WHERE user_id = ? AND phrase = ?",
        (utcnow(), user_id, phrase),
    )
    conn.commit()


def infer_usual(
    conn: sqlite3.Connection, user_id: str, slot: str | None = None
) -> dict[str, Any] | None:
    """Derive 'my usual' from repetition instead of asking.

    Groups past meals by their item signature and returns the most repeated one,
    provided it has happened at least ALIAS_INFERENCE_THRESHOLD times. Below
    that it is a coincidence, not a habit.
    """
    sql = (
        "SELECT m.id, m.slot, mi.name, mi.qty, mi.unit FROM meals m"
        " JOIN meal_items mi ON mi.meal_id = m.id"
        " WHERE m.user_id = ? AND mi.deleted_at IS NULL AND m.deleted_at IS NULL"
    )
    params: list[Any] = [user_id]
    if slot:
        sql += " AND m.slot = ?"
        params.append(slot)
    sql += " ORDER BY m.id"

    meals: dict[int, dict[str, Any]] = {}
    for row in conn.execute(sql, params).fetchall():
        meal = meals.setdefault(row["id"], {"slot": row["slot"], "items": []})
        meal["items"].append({"name": row["name"], "qty": row["qty"], "unit": row["unit"]})

    counts: dict[tuple, dict[str, Any]] = {}
    for meal in meals.values():
        signature = tuple(sorted((i["name"], i["qty"]) for i in meal["items"]))
        if not signature:
            continue
        bucket = counts.setdefault(
            signature, {"count": 0, "items": meal["items"], "slot": meal["slot"]}
        )
        bucket["count"] += 1

    if not counts:
        return None
    best = max(counts.values(), key=lambda b: b["count"])
    if best["count"] < ALIAS_INFERENCE_THRESHOLD:
        return None
    return {
        "phrase": "my usual",
        "items": best["items"],
        "slot": best["slot"],
        "hits": best["count"],
        "source": "inferred",
    }


def learn_usual_if_repeated(conn: sqlite3.Connection, user_id: str) -> bool:
    """Promote an inferred habit into a stored alias. Runs in the background
    pass, never on the reply path."""
    inferred = infer_usual(conn, user_id)
    if not inferred:
        return False
    existing = conn.execute(
        "SELECT source FROM aliases WHERE user_id = ? AND phrase = 'my usual'",
        (user_id,),
    ).fetchone()
    if existing and existing["source"] == "explicit":
        return False  # never overwrite something the user stated outright
    put_alias(
        conn, user_id, "my usual", inferred["items"], inferred["slot"], source="inferred"
    )
    return True


# ---------------------------------------------------------------------------
# explicit "my usual is X" statements
# ---------------------------------------------------------------------------
_DEFINE_RE = re.compile(
    r"\b(?:my |the )?usual(?:\s+\w+)?\s+is\b(?P<body>.+)", re.I
)


def detect_alias_definition(text: str) -> str | None:
    """'my usual is 2 parathas and chai' -> '2 parathas and chai'."""
    match = _DEFINE_RE.search(text)
    if not match:
        return None
    body = match.group("body").strip(" .,:")
    return body or None
