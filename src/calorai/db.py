"""SQLite persistence.

Two schema decisions carry the whole "totals stay correct" requirement:

1. There is no stored running total anywhere. Daily totals are a SUM over
   `meal_items` at query time. A correction or a delete therefore cannot
   desynchronise a counter, because no counter exists.

2. `meal_items` denormalises `user_id` and `local_date` off its parent meal.
   That is redundant on paper, but it means the hot query -- today's totals --
   is a single indexed scan with no join, which matters on the latency path.

Soft deletes throughout (`deleted_at`): edits stay auditable and reversible,
and `edit_log` keeps a before/after trail.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

DEFAULT_DB = os.environ.get("CALORAI_DB_PATH", "calorai.db")

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS meals (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      TEXT NOT NULL,
    slot         TEXT,
    occurred_at  TEXT NOT NULL,
    local_date   TEXT NOT NULL,
    source       TEXT NOT NULL DEFAULT 'text',
    note         TEXT,
    created_at   TEXT NOT NULL,
    deleted_at   TEXT
);

CREATE TABLE IF NOT EXISTS meal_items (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    meal_id      INTEGER NOT NULL REFERENCES meals(id),
    user_id      TEXT NOT NULL,
    local_date   TEXT NOT NULL,
    name         TEXT NOT NULL,
    qty          REAL NOT NULL,
    unit         TEXT NOT NULL,
    kcal         REAL NOT NULL DEFAULT 0,
    protein_g    REAL NOT NULL DEFAULT 0,
    carbs_g      REAL NOT NULL DEFAULT 0,
    fat_g        REAL NOT NULL DEFAULT 0,
    confidence   REAL NOT NULL DEFAULT 1.0,
    is_estimate  INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL,
    deleted_at   TEXT
);

-- the index the daily-totals query rides on
CREATE INDEX IF NOT EXISTS idx_items_user_date
    ON meal_items(user_id, local_date) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_items_meal ON meal_items(meal_id);
CREATE INDEX IF NOT EXISTS idx_meals_user_date ON meals(user_id, local_date);

CREATE TABLE IF NOT EXISTS edit_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      TEXT NOT NULL,
    meal_item_id INTEGER,
    action       TEXT NOT NULL,
    before_json  TEXT,
    after_json   TEXT,
    reason       TEXT,
    created_at   TEXT NOT NULL
);

-- Durable user attributes. Small on purpose: the whole store is loaded every
-- turn, which is why it needs no retrieval logic.
CREATE TABLE IF NOT EXISTS profile_facts (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id        TEXT NOT NULL,
    key            TEXT NOT NULL,
    value          TEXT NOT NULL,
    confidence     REAL NOT NULL DEFAULT 0.8,
    source_message TEXT,
    created_at     TEXT NOT NULL,
    superseded_by  INTEGER
);
CREATE INDEX IF NOT EXISTS idx_facts_user
    ON profile_facts(user_id) WHERE superseded_by IS NULL;

-- Learned shorthand: "my usual" -> a concrete item list.
CREATE TABLE IF NOT EXISTS aliases (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      TEXT NOT NULL,
    phrase       TEXT NOT NULL,
    items_json   TEXT NOT NULL,
    slot         TEXT,
    hits         INTEGER NOT NULL DEFAULT 1,
    source       TEXT NOT NULL DEFAULT 'explicit',
    created_at   TEXT NOT NULL,
    last_used_at TEXT
);
-- Not unique on (user, phrase): the same phrase holds one entry per meal slot
-- plus an unscoped fallback, so "my usual" can mean porridge at 8am and
-- something else at 8pm.
CREATE INDEX IF NOT EXISTS idx_alias_user_phrase ON aliases(user_id, phrase);

-- A photo the agent has read but NOT yet logged, waiting on the user to say
-- yes. Photos are the one input where the user delegates the entire
-- description to a model, and vision models get portions and counts wrong in
-- ways the user can see instantly and the agent cannot see at all.
CREATE TABLE IF NOT EXISTS pending_meals (
    user_id    TEXT PRIMARY KEY,
    items_json TEXT NOT NULL,
    summary    TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS nutrition_cache (
    key        TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    unit       TEXT NOT NULL,
    kcal       REAL NOT NULL,
    protein_g  REAL NOT NULL,
    carbs_g    REAL NOT NULL,
    fat_g      REAL NOT NULL,
    source     TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- Rolling transcript. This is NOT memory -- it is replay context for
-- continuity across restarts, and it is capped. See memory/ for real memory.
CREATE TABLE IF NOT EXISTS transcript (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    TEXT NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_transcript_user ON transcript(user_id, id DESC);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


_CONNS: dict[str, sqlite3.Connection] = {}


def connect(db_path: str | None = None) -> sqlite3.Connection:
    """One long-lived connection per path -- reconnecting per turn costs ms we
    do not have to spend."""
    path = str(db_path or DEFAULT_DB)
    conn = _CONNS.get(path)
    if conn is None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA)
        conn.commit()
        _CONNS[path] = conn
    return conn


def reset_connections() -> None:
    for conn in _CONNS.values():
        conn.close()
    _CONNS.clear()


@contextmanager
def tx(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(r) for r in rows]


def log_edit(
    conn: sqlite3.Connection,
    user_id: str,
    meal_item_id: int | None,
    action: str,
    before: dict | None,
    after: dict | None,
    reason: str | None = None,
) -> None:
    conn.execute(
        "INSERT INTO edit_log (user_id, meal_item_id, action, before_json, after_json,"
        " reason, created_at) VALUES (?,?,?,?,?,?,?)",
        (
            user_id,
            meal_item_id,
            action,
            json.dumps(before) if before else None,
            json.dumps(after) if after else None,
            reason,
            utcnow(),
        ),
    )
