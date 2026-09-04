"""All meal reads and writes. Tools are thin wrappers over this module.

The rule that keeps totals correct: **nothing here ever stores a total.**
`daily_totals` sums `meal_items` on demand. A correction is an UPDATE to an
existing row and a delete is a timestamp, so neither can desynchronise a
counter -- there is no counter to desynchronise.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta
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
    if row is None:
        return {
            "ok": False,
            "error": f"nothing matching '{target_hint}' in the last {CORRECTION_WINDOW_DAYS} days",
        }

    before = dict(row)
    name = new_name or row["name"]
    qty = new_qty if new_qty is not None else row["qty"]
    unit = new_unit or row["unit"]

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


def delete_meal(
    conn: sqlite3.Connection, user_id: str, target_hint: str = ""
) -> dict[str, Any]:
    """Soft delete: the row stays for audit, the sum stops counting it."""
    row = _find_recent_item(conn, user_id, target_hint)
    if row is None:
        return {"ok": False, "error": f"nothing matching '{target_hint}' to remove"}
    conn.execute("UPDATE meal_items SET deleted_at = ? WHERE id = ?", (utcnow(), row["id"]))
    log_edit(conn, user_id, row["id"], "delete", dict(row), None, target_hint)
    conn.commit()
    return {
        "ok": True,
        "removed": f"{row['qty']:g} {row['unit']} {row['name']} ({round(row['kcal'])} kcal)",
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


def transcript_tail(conn: sqlite3.Connection, user_id: str, limit: int = 6) -> list[dict]:
    """Recent turns, oldest first. NOT memory -- just continuity across restarts,
    and hard-capped so it cannot grow into a prompt-bloat problem."""
    rows = conn.execute(
        "SELECT role, content FROM transcript WHERE user_id = ? ORDER BY id DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    return [dict(r) for r in reversed(rows)]
