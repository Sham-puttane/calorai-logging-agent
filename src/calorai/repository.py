"""All meal reads and writes. Tools are thin wrappers over this module.

The rule that keeps totals correct: **nothing here ever stores a total.**
`daily_totals` sums `meal_items` on demand. A correction is an UPDATE to an
existing row and a delete is a timestamp, so neither can desynchronise a
counter -- there is no counter to desynchronise.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable

from .db import log_edit, utcnow
from .nutrition import Resolved, normalize, resolve
from .schemas import FoodItem, Nutrition

# How far back a bare correction ("actually that was 3") is allowed to reach.
# Beyond this the agent should be asking which meal, not guessing. Two days
# covers "I forgot to fix yesterday's dinner" without letting a stale row get
# silently rewritten a week later.
CORRECTION_WINDOW_DAYS = 2


def local_today() -> str:
    return date.today().isoformat()


def parse_day(value: str | None) -> str:
    """Accepts 'today', 'yesterday', an ISO date, or None (-> today)."""
    if not value or value.lower() in {"today", "now"}:
        return local_today()
    if value.lower() == "yesterday":
        return (date.today() - timedelta(days=1)).isoformat()
    try:
        return datetime.fromisoformat(value).date().isoformat()
    except ValueError:
        return local_today()


def infer_slot(when: datetime | None = None) -> str:
    hour = (when or datetime.now()).hour
    if hour < 11:
        return "breakfast"
    if hour < 16:
        return "lunch"
    if hour < 21:
        return "dinner"
    return "snack"


# --------------------------------------------------------------------------
# writes
# --------------------------------------------------------------------------
def log_meal(
    conn: sqlite3.Connection,
    user_id: str,
    items: list[FoodItem],
    slot: str | None = None,
    day: str | None = None,
    source: str = "text",
    note: str | None = None,
    is_estimate: bool = False,
    confidence: float = 1.0,
    estimator: Callable[[str, str], Nutrition | None] | None = None,
) -> dict[str, Any]:
    """Insert one meal and its items. Returns a summary the agent can phrase."""
    if not items:
        return {"ok": False, "error": "no items to log"}

    local_date = parse_day(day)
    slot = slot or infer_slot()
    now = utcnow()

    cur = conn.execute(
        "INSERT INTO meals (user_id, slot, occurred_at, local_date, source, note, created_at)"
        " VALUES (?,?,?,?,?,?,?)",
        (user_id, slot, now, local_date, source, note, now),
    )
    meal_id = cur.lastrowid

    logged: list[dict[str, Any]] = []
    for item in items:
        r: Resolved = resolve(conn, item, estimator=estimator)
        item_conf = min(confidence, r.confidence) if r.source != "seed" else confidence
        cur = conn.execute(
            "INSERT INTO meal_items (meal_id, user_id, local_date, name, qty, unit,"
            " kcal, protein_g, carbs_g, fat_g, confidence, is_estimate, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                meal_id, user_id, local_date, r.name, r.qty, r.unit,
                r.nutrition.kcal, r.nutrition.protein_g, r.nutrition.carbs_g,
                r.nutrition.fat_g, item_conf, int(is_estimate or r.source == "unknown"), now,
            ),
        )
        logged.append(
            {
                "item_id": cur.lastrowid, "name": r.name, "qty": r.qty, "unit": r.unit,
                "kcal": round(r.nutrition.kcal), "source": r.source,
            }
        )
        log_edit(conn, user_id, cur.lastrowid, "log", None, logged[-1], note)

    conn.commit()
    unknown = [i["name"] for i in logged if i["source"] == "unknown"]
    return {
        "ok": True, "meal_id": meal_id, "slot": slot, "date": local_date,
        "items": logged,
        "meal_kcal": round(sum(i["kcal"] for i in logged)),
        "unknown_foods": unknown,
        "totals_after": daily_totals(conn, user_id, local_date),
    }


def _find_recent_item(
    conn: sqlite3.Connection, user_id: str, hint: str
) -> sqlite3.Row | None:
    """Most recent live item matching `hint`. Empty hint -> the last thing logged."""
    cutoff = (date.today() - timedelta(days=CORRECTION_WINDOW_DAYS)).isoformat()
    rows = conn.execute(
        "SELECT * FROM meal_items WHERE user_id = ? AND deleted_at IS NULL"
        " AND local_date >= ? ORDER BY id DESC",
        (user_id, cutoff),
    ).fetchall()
    if not rows:
        return None
    key = normalize(hint or "")
    if not key:
        return rows[0]
    for row in rows:
        name = normalize(row["name"])
        if key == name or key in name or name in key:
            return row
    # fall back to token overlap: "roti count" should still find "roti"
    tokens = set(key.split())
    for row in rows:
        if tokens & set(normalize(row["name"]).split()):
            return row
    return None


def _substitution_target(
    conn: sqlite3.Connection, user_id: str, new_unit: str
) -> sqlite3.Row | None:
    """Which row did the user actually misname?

    "2 parathas and chai" then "actually that was 3 rotis" should rewrite the
    paratha, not the chai -- even though the chai is the more recent row. Unit
    is the cheap signal that gets this right: roti and paratha are both
    counted in pieces, chai is a cup. Falls back to the most recent item when
    nothing matches, because a wrong guess is still better than adding a
    duplicate meal.
    """
    cutoff = (date.today() - timedelta(days=CORRECTION_WINDOW_DAYS)).isoformat()
    rows = conn.execute(
        "SELECT * FROM meal_items WHERE user_id = ? AND deleted_at IS NULL"
        " AND local_date >= ? ORDER BY id DESC",
        (user_id, cutoff),
    ).fetchall()
    if not rows:
        return None
    for row in rows:
        if row["unit"] == new_unit:
            return row
    return rows[0]


def correct_meal(
    conn: sqlite3.Connection,
    user_id: str,
    target_hint: str = "",
    new_qty: float | None = None,
    new_name: str | None = None,
    new_unit: str | None = None,
    estimator: Callable[[str, str], Nutrition | None] | None = None,
) -> dict[str, Any]:
    """UPDATE an existing item in place. Never inserts.

    This is a separate tool from log_meal on purpose: it makes double-counting
    structurally impossible rather than something the prompt has to remember.
    """
    row = _find_recent_item(conn, user_id, target_hint)

    # A correction can name a food that is not in the log yet, because the food
    # itself is what is being corrected: "2 parathas and chai", then "actually
    # that was 3 rotis not 2". There is no roti to find -- the paratha is the
    # thing that was wrong.
    #
    # Before this, that returned ok:False, the agent fell back to log_meal, and
    # the rotis were ADDED on top of the parathas. Double counting, arriving
    # through the one path the separate-tools design was meant to close.
    #
    # Substitution is only allowed when the named food is one we recognise. An
    # unrecognisable hint is more likely a mistake than a correction, and
    # rewriting the most recent row on a guess is worse than refusing.
    substituting = False
    if row is None and target_hint.strip():
        known = resolve(conn, FoodItem(name=target_hint, qty=1), estimator=None)
        if known.source != "unknown":
            row = _substitution_target(conn, user_id, known.unit)
            substituting = row is not None

    if row is None:
        return {
            "ok": False,
            "error": f"nothing matching '{target_hint}' in the last {CORRECTION_WINDOW_DAYS} days",
        }

    before = dict(row)
    name = new_name or (target_hint if substituting else row["name"])
    qty = new_qty if new_qty is not None else row["qty"]
    # A substitution must not inherit the old food's unit -- correcting a chai
    # into rotis should not produce "3 cup roti". Leaving it empty lets the
    # nutrition table supply the right one.
    unit = new_unit or ("" if substituting else row["unit"])

    # Re-resolve from scratch so a name change recomputes nutrition correctly
    # instead of rescaling numbers that belonged to the old food.
    r = resolve(conn, FoodItem(name=name, qty=qty, unit=unit), estimator=estimator)

    conn.execute(
        "UPDATE meal_items SET name=?, qty=?, unit=?, kcal=?, protein_g=?, carbs_g=?,"
        " fat_g=? WHERE id = ?",
        (
            r.name, r.qty, r.unit, r.nutrition.kcal, r.nutrition.protein_g,
            r.nutrition.carbs_g, r.nutrition.fat_g, row["id"],
        ),
    )
    after = dict(conn.execute("SELECT * FROM meal_items WHERE id = ?", (row["id"],)).fetchone())
    log_edit(conn, user_id, row["id"], "correct", before, after, target_hint)
    conn.commit()

    return {
        "ok": True,
        "changed": {
            "item_id": row["id"],
            "from": f"{before['qty']:g} {before['unit']} {before['name']}",
            "to": f"{after['qty']:g} {after['unit']} {after['name']}",
            "kcal_delta": round(after["kcal"] - before["kcal"]),
        },
        "totals_after": daily_totals(conn, user_id, after["local_date"]),
    }


def _meal_items(conn: sqlite3.Connection, meal_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM meal_items WHERE meal_id = ? AND deleted_at IS NULL", (meal_id,)
    ).fetchall()


def delete_meal(
    conn: sqlite3.Connection, user_id: str, target_hint: str = ""
) -> dict[str, Any]:
    """Soft delete: rows stay for audit, the sum stops counting them.

    Scope follows the hint, which is what people actually mean:

    * no hint -- "scratch that" -- removes the whole most recent MEAL. A photo
      logs seven items in one meal, and removing one of them because the user
      said "scratch that" would silently leave the other six on their day. This
      was a real bug: delete_meal("") on a five-item meal removed one item and
      left 705 of 755 calories logged.
    * a named food -- "I didn't have the chai" -- removes just that item.
    """
    row = _find_recent_item(conn, user_id, target_hint)
    if row is None:
        return {"ok": False, "error": f"nothing matching '{target_hint}' to remove"}

    rows = [row] if target_hint.strip() else _meal_items(conn, row["meal_id"])
    now = utcnow()
    for item in rows:
        conn.execute("UPDATE meal_items SET deleted_at = ? WHERE id = ?", (now, item["id"]))
        log_edit(conn, user_id, item["id"], "delete", dict(item), None, target_hint)
    # If nothing live is left, mark the parent meal too. Otherwise
    # meals.deleted_at is a column that is never written -- dead schema that
    # makes a reader wonder what they are missing.
    if not _meal_items(conn, row["meal_id"]):
        conn.execute("UPDATE meals SET deleted_at = ? WHERE id = ?", (now, row["meal_id"]))
    conn.commit()

    removed = ", ".join(f"{i['qty']:g} {i['unit']} {i['name']}" for i in rows)
    return {
        "ok": True,
        "removed": removed,
        "items_removed": len(rows),
        "kcal_removed": round(sum(i["kcal"] for i in rows)),
        "totals_after": daily_totals(conn, user_id, row["local_date"]),
    }


def scale_meal(
    conn: sqlite3.Connection, user_id: str, factor: float, target_hint: str = ""
) -> dict[str, Any]:
    """Multiply every item in the most recent matching meal by `factor`.

    This is what "half of this was my brother's" means when it arrives *after*
    a meal is already logged. Without it the agent reached for correct_meal,
    which only touches one item -- observed live as "i've updated the naan to 1
    piece" on a seven-item plate, leaving the other six at full size.

    Proportional, so it composes: halving twice is a quarter, which is what
    someone correcting themselves would expect.
    """
    if factor <= 0 or factor > 1:
        return {"ok": False, "error": "factor must be between 0 and 1"}

    row = _find_recent_item(conn, user_id, target_hint)
    if row is None:
        return {"ok": False, "error": "nothing recent to resize"}

    rows = _meal_items(conn, row["meal_id"])
    if not rows:
        return {"ok": False, "error": "that meal has no items left"}

    before_kcal = sum(i["kcal"] for i in rows)
    for item in rows:
        conn.execute(
            "UPDATE meal_items SET qty=?, kcal=?, protein_g=?, carbs_g=?, fat_g=?"
            " WHERE id = ?",
            (
                round(item["qty"] * factor, 3), item["kcal"] * factor,
                item["protein_g"] * factor, item["carbs_g"] * factor,
                item["fat_g"] * factor, item["id"],
            ),
        )
        after = dict(conn.execute("SELECT * FROM meal_items WHERE id=?", (item["id"],)).fetchone())
        log_edit(conn, user_id, item["id"], "scale", dict(item), after, f"x{factor:g}")
    conn.commit()

    return {
        "ok": True,
        "scaled_by": factor,
        "items_scaled": len(rows),
        "kcal_removed": round(before_kcal * (1 - factor)),
        "items": [f"{i['qty'] * factor:g} {i['unit']} {i['name']}" for i in rows],
        "totals_after": daily_totals(conn, user_id, row["local_date"]),
    }


# --------------------------------------------------------------------------
# reads
# --------------------------------------------------------------------------
def daily_totals(
    conn: sqlite3.Connection, user_id: str, day: str | None = None
) -> dict[str, Any]:
    """The single source of truth for 'how am I doing today'.

    Derived, never stored. This is why corrections and deletes cannot break it.
    """
    local_date = parse_day(day)
    row = conn.execute(
        "SELECT COALESCE(SUM(kcal),0) kcal, COALESCE(SUM(protein_g),0) protein_g,"
        " COALESCE(SUM(carbs_g),0) carbs_g, COALESCE(SUM(fat_g),0) fat_g,"
        " COUNT(*) n_items, COALESCE(SUM(is_estimate),0) n_estimated"
        " FROM meal_items WHERE user_id = ? AND local_date = ? AND deleted_at IS NULL",
        (user_id, local_date),
    ).fetchone()
    return {
        "date": local_date,
        "kcal": round(row["kcal"]),
        "protein_g": round(row["protein_g"], 1),
        "carbs_g": round(row["carbs_g"], 1),
        "fat_g": round(row["fat_g"], 1),
        "items_logged": row["n_items"],
        # surfaced so the agent can hedge a number built partly on guesses
        "items_estimated": row["n_estimated"],
    }


def find_meals(
    conn: sqlite3.Connection,
    user_id: str,
    query: str = "",
    day: str | None = None,
    slot: str | None = None,
    limit: int = 25,
) -> dict[str, Any]:
    """Retrieval over the log. This is what answers 'same as yesterday' --
    a database question, not a memory question."""
    sql = (
        "SELECT mi.id, mi.name, mi.qty, mi.unit, mi.kcal, mi.local_date, m.slot"
        " FROM meal_items mi JOIN meals m ON m.id = mi.meal_id"
        " WHERE mi.user_id = ? AND mi.deleted_at IS NULL"
    )
    params: list[Any] = [user_id]
    if day:
        sql += " AND mi.local_date = ?"
        params.append(parse_day(day))
    if slot:
        sql += " AND m.slot = ?"
        params.append(slot)
    if query:
        sql += " AND LOWER(mi.name) LIKE ?"
        params.append(f"%{query.lower()}%")
    sql += " ORDER BY mi.id DESC LIMIT ?"
    params.append(limit)

    rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
    for r in rows:
        r["kcal"] = round(r["kcal"])
    return {"ok": True, "count": len(rows), "meals": rows}


def transcript_append(conn: sqlite3.Connection, user_id: str, role: str, content: str) -> None:
    conn.execute(
        "INSERT INTO transcript (user_id, role, content, created_at) VALUES (?,?,?,?)",
        (user_id, role, content, utcnow()),
    )
    conn.commit()


# --------------------------------------------------------------------------
# pending photo confirmations
# --------------------------------------------------------------------------
# How long a photo waits for a yes before it is treated as abandoned. Long
# enough to answer a question, short enough that "yeah" an hour later about
# something else does not log last lunch.
PENDING_TTL_MINUTES = 15


def put_pending(
    conn: sqlite3.Connection, user_id: str, items: list[dict[str, Any]], summary: str
) -> None:
    conn.execute(
        "INSERT INTO pending_meals (user_id, items_json, summary, created_at)"
        " VALUES (?,?,?,?) ON CONFLICT(user_id) DO UPDATE SET"
        " items_json=excluded.items_json, summary=excluded.summary,"
        " created_at=excluded.created_at",
        (user_id, json.dumps(items), summary, utcnow()),
    )
    conn.commit()


def get_pending(conn: sqlite3.Connection, user_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT items_json, summary, created_at FROM pending_meals WHERE user_id = ?",
        (user_id,),
    ).fetchone()
    if row is None:
        return None
    age = datetime.now(timezone.utc) - datetime.fromisoformat(row["created_at"])
    if age > timedelta(minutes=PENDING_TTL_MINUTES):
        clear_pending(conn, user_id)
        return None
    return {"items": json.loads(row["items_json"]), "summary": row["summary"]}


def clear_pending(conn: sqlite3.Connection, user_id: str) -> None:
    conn.execute("DELETE FROM pending_meals WHERE user_id = ?", (user_id,))
    conn.commit()
