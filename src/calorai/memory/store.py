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

# How far back "my usual" looks, and the row ceiling on that scan. A habit is
# recent behaviour; and this runs on the background pass of every turn, so it
# must not grow with the lifetime of the account.
HABIT_WINDOW_DAYS = 45
HABIT_SCAN_LIMIT = 400

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
    """Store shorthand, scoped to a meal slot when one is known.

    "my usual" means porridge at 8am and something else entirely at 8pm, so the
    same phrase can hold one entry per slot plus one unscoped fallback.

    Replace-then-insert rather than an upsert, because the natural key is
    (user, phrase, slot) and SQLite will not treat two NULL slots as a conflict
    — an upsert would quietly accumulate duplicate unscoped rows.
    """
    phrase = phrase.lower().strip()
    conn.execute(
        "DELETE FROM aliases WHERE user_id = ? AND phrase = ?"
        " AND COALESCE(slot,'') = COALESCE(?,'')",
        (user_id, phrase, slot),
    )
    conn.execute(
        "INSERT INTO aliases (user_id, phrase, items_json, slot, source, created_at, last_used_at)"
        " VALUES (?,?,?,?,?,?,?)",
        (user_id, phrase, json.dumps(items), slot, source, utcnow(), utcnow()),
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
    conn: sqlite3.Connection,
    user_id: str,
    text: str,
    slot: str | None = None,
) -> dict[str, Any] | None:
    """Deterministic lookup, run before the model sees the message.

    Returns the stored meal for 'my usual' and friends. Cheap, exact, and it
    keeps a pronoun from ever reaching the model as something to guess at.

    Scoping: "my usual" said at breakfast should mean the breakfast one. The
    order is slot match, then an unscoped entry, then any entry at all —
    because a wrong-slot usual is still far more useful than "I don't know what
    your usual is", and the user can correct it in three words.

    `slot` defaults to the current time of day. That is right for someone with
    three meals a day and only a guess for a grazer, which is why an unscoped
    alias is kept as a fallback rather than everything being forced into a slot.
    """
    from ..repository import infer_slot

    low = text.lower().strip()
    slot = slot or infer_slot()
    matches = [e for e in get_aliases(conn, user_id) if e["phrase"] and e["phrase"] in low]

    if matches:
        best = (
            next((e for e in matches if e.get("slot") == slot), None)
            or next((e for e in matches if not e.get("slot")), None)
            or matches[0]
        )
        _touch(conn, user_id, best["phrase"], best.get("slot"))
        return best

    # Generic trigger with nothing learned yet -- try to infer one from history
    # rather than replying "I don't know what your usual is". Prefer this slot's
    # habit, then fall back to any.
    if any(trigger in low for trigger in _ALIAS_TRIGGERS):
        return infer_usual(conn, user_id, slot=slot) or infer_usual(conn, user_id)
    return None


def _touch(
    conn: sqlite3.Connection, user_id: str, phrase: str, slot: str | None = None
) -> None:
    conn.execute(
        "UPDATE aliases SET hits = hits + 1, last_used_at = ?"
        " WHERE user_id = ? AND phrase = ? AND COALESCE(slot,'') = COALESCE(?,'')",
        (utcnow(), user_id, phrase, slot),
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
    # Bounded on both axes. Without a floor this scanned every meal the user
    # had ever logged, on a background pass that runs every turn -- fine on day
    # one, quadratic-feeling by month six. The window is also the *correct*
    # semantics: a habit is recent behaviour, and what someone ate every day
    # last spring is not "their usual" today.
    floor = (date.today() - timedelta(days=HABIT_WINDOW_DAYS)).isoformat()
    sql = (
        "SELECT m.id, m.slot, mi.name, mi.qty, mi.unit FROM meals m"
        " JOIN meal_items mi ON mi.meal_id = m.id"
        " WHERE m.user_id = ? AND mi.deleted_at IS NULL AND m.deleted_at IS NULL"
        " AND m.local_date >= ?"
    )
    params: list[Any] = [user_id, floor]
    if slot:
        sql += " AND m.slot = ?"
        params.append(slot)
    sql += " ORDER BY m.id DESC LIMIT ?"
    params.append(HABIT_SCAN_LIMIT)

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

# "remember this as my usual", "that's my usual", "make this my usual".
#
# This is the phrasing people actually reach for, and it is a different shape
# from "my usual is X": it names no food at all. The meal is the one that was
# just logged, which means the alias has to be built from the log rather than
# parsed out of the sentence. Missing this was why "remember this dinner thats
# my usual" did nothing.
_REMEMBER_RECENT_RE = re.compile(
    r"("
    r"(?:remember|save|store|keep)\b[^.]{0,40}?\b(?:my|the)?\s*usual"
    r"|(?:that'?s|this is|thats)\s+(?:my|the)\s+usual"
    r"|make\s+(?:this|that)\s+(?:my|the)\s+usual"
    r"|(?:my|the)\s+usual\s*[.!]?$"
    r")",
    re.I,
)


def detect_alias_definition(text: str) -> str | None:
    """'my usual is 2 parathas and chai' -> '2 parathas and chai'."""
    match = _DEFINE_RE.search(text)
    if not match:
        return None
    body = match.group("body").strip(" .,:")
    return body or None


def means_remember_recent(text: str) -> bool:
    """True for "remember this as my usual" and friends, which point at the
    last meal rather than naming one."""
    if detect_alias_definition(text):
        return False  # "my usual is X" names its own food; handled above
    low = (text or "").lower().strip()
    # "my usual" on its own is someone *using* the alias, not defining it.
    if low in {"my usual", "the usual", "my usual please"}:
        return False
    return bool(_REMEMBER_RECENT_RE.search(low))


_SLOT_WORDS = ("breakfast", "lunch", "dinner", "snack")


def learn_alias_from_recent_meal(
    conn: sqlite3.Connection,
    user_id: str,
    phrase: str = "my usual",
    text: str = "",
) -> dict[str, Any] | None:
    """Store what was just eaten under `phrase`.

    "this dinner" is not one row. Someone logs the naan and curry, then adds
    the rice they forgot -- two meals in the table, one dinner to a person. So
    the whole slot's worth of today's food is collected, not the last INSERT.
    Taking only the last meal captured "rice" and nothing else, which is a
    usual nobody would recognise.

    Explicit, so it overwrites an inferred alias: being told outright beats a
    habit guessed from repetition.
    """
    last = conn.execute(
        "SELECT mi.meal_id, m.slot, mi.local_date FROM meal_items mi"
        " JOIN meals m ON m.id = mi.meal_id"
        " WHERE mi.user_id = ? AND mi.deleted_at IS NULL ORDER BY mi.id DESC LIMIT 1",
        (user_id,),
    ).fetchone()
    if last is None:
        return None

    # A slot named in the message wins over the one inferred from the last row.
    named = next((w for w in _SLOT_WORDS if w in (text or "").lower()), None)
    slot = named or last["slot"]

    if slot:
        rows = conn.execute(
            "SELECT mi.name, mi.qty, mi.unit FROM meal_items mi"
            " JOIN meals m ON m.id = mi.meal_id"
            " WHERE mi.user_id = ? AND mi.deleted_at IS NULL"
            " AND mi.local_date = ? AND m.slot = ? ORDER BY mi.id",
            (user_id, last["local_date"], slot),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT name, qty, unit FROM meal_items"
            " WHERE meal_id = ? AND deleted_at IS NULL",
            (last["meal_id"],),
        ).fetchall()

    items = [{"name": r["name"], "qty": r["qty"], "unit": r["unit"]} for r in rows]
    if not items:
        return None
    put_alias(conn, user_id, phrase, items, slot, source="explicit")
    return {"phrase": phrase, "items": items, "slot": slot}
