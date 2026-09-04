"""Nutrition resolution: cache -> seed table -> fuzzy -> model estimate.

Ordering is a latency decision. The first three tiers are pure local work
(microseconds) and cover the overwhelming majority of real logging, because
people eat the same fifty things. Only a genuinely novel food reaches the model,
and when it does the answer is written back to `nutrition_cache` so it is a
one-time cost per food per install.

Values are per ONE household unit. See data/seed_foods.json for the trade-off.
"""

from __future__ import annotations

import difflib
import json
import re
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

from .db import utcnow
from .schemas import FoodItem, Nutrition

_SEED_PATH = Path(__file__).resolve().parents[2] / "data" / "seed_foods.json"

# Fuzzy matching is deliberately tight. At 0.86 "parathas" -> "paratha" and
# "chapatti" -> "chapati" both land, while "chicken" does not silently become
# "chickpea". A loose cutoff here would log the wrong food confidently, which
# is worse than admitting we do not know.
_FUZZY_CUTOFF = 0.86

# Noise words people attach to foods that carry no nutritional signal for a
# table this coarse. Stripping them raises the hit rate on the seed table.
_NOISE = {
    "some", "a", "an", "the", "my", "one", "two", "half", "plate", "of",
    "leftover", "leftovers", "little", "bit", "small", "large", "big", "hot",
    "cold", "fresh", "homemade", "home", "made",
}


# Units that carry no information. A model asked for a unit will often say
# "serving" rather than commit, and "2 serving paratha" reads worse than
# "2 piece paratha" for the same number of calories -- so when the caller is
# vague the table's own unit wins.
_GENERIC_UNITS = {"", "serving", "servings", "portion", "portions", "unit", "units"}


def _best_unit(given: str | None, canonical: str) -> str:
    return canonical if (given or "").strip().lower() in _GENERIC_UNITS else given


@dataclass(frozen=True)
class Resolved:
    """A FoodItem with nutrition attached. `nutrition` is the TOTAL for `qty`,
    not per unit -- callers write this straight to meal_items."""

    name: str
    qty: float
    unit: str
    nutrition: Nutrition
    source: str  # seed | cache | fuzzy | model | unknown
    confidence: float


def normalize(name: str) -> str:
    """Lowercase, drop punctuation and noise words, collapse whitespace."""
    text = re.sub(r"[^a-z0-9\s]", " ", name.lower())
    words = [w for w in text.split() if w and w not in _NOISE]
    return " ".join(words).strip()


def _depluralize(word: str) -> list[str]:
    """Candidate singulars, most likely first. 'rotis'->'roti', 'momos'->'momo',
    'batches'->'batche'/'batch'. We generate candidates rather than guessing
    once, and let the index decide which one actually exists."""
    out = [word]
    if word.endswith("ies") and len(word) > 4:
        out.append(word[:-3] + "y")
    if word.endswith("es") and len(word) > 3:
        out.append(word[:-2])
    if word.endswith("s") and not word.endswith("ss"):
        out.append(word[:-1])
    return out


@lru_cache(maxsize=1)
def _index() -> dict[str, dict]:
    """name/alias -> entry. Built once per process."""
    raw = json.loads(_SEED_PATH.read_text(encoding="utf-8"))
    index: dict[str, dict] = {}
    for name, entry in raw["foods"].items():
        record = {
            "name": name,
            "unit": entry["unit"],
            "nutrition": Nutrition(
                kcal=entry["kcal"],
                protein_g=entry["protein_g"],
                carbs_g=entry["carbs_g"],
                fat_g=entry["fat_g"],
            ),
        }
        index[normalize(name)] = record
        for alias in entry.get("aliases", []):
            index.setdefault(normalize(alias), record)
    return index


def _seed_lookup(name: str) -> dict | None:
    idx = _index()
    key = normalize(name)
    if key in idx:
        return idx[key]

    # try singularising each word: "2 parathas" -> "paratha"
    words = key.split()
    if words:
        for candidate in _depluralize(words[-1]):
            probe = " ".join(words[:-1] + [candidate]).strip()
            if probe in idx:
                return idx[probe]
        # last word alone: "veg biryani box" -> "biryani"
        for word in reversed(words):
            for candidate in _depluralize(word):
                if candidate in idx:
                    return idx[candidate]
    return None


def _fuzzy_lookup(name: str) -> tuple[dict, float] | None:
    idx = _index()
    key = normalize(name)
    if not key:
        return None
    matches = difflib.get_close_matches(key, idx.keys(), n=1, cutoff=_FUZZY_CUTOFF)
    if not matches:
        return None
    score = difflib.SequenceMatcher(None, key, matches[0]).ratio()
    return idx[matches[0]], score


def _cache_get(conn: sqlite3.Connection, key: str) -> dict | None:
    row = conn.execute(
        "SELECT name, unit, kcal, protein_g, carbs_g, fat_g FROM nutrition_cache WHERE key = ?",
        (key,),
    ).fetchone()
    if row is None:
        return None
    return {
        "name": row["name"],
        "unit": row["unit"],
        "nutrition": Nutrition(
            kcal=row["kcal"],
            protein_g=row["protein_g"],
            carbs_g=row["carbs_g"],
            fat_g=row["fat_g"],
        ),
    }


def cache_put(
    conn: sqlite3.Connection, name: str, unit: str, nutrition: Nutrition, source: str
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO nutrition_cache"
        " (key, name, unit, kcal, protein_g, carbs_g, fat_g, source, created_at)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        (
            normalize(name),
            name,
            unit,
            nutrition.kcal,
            nutrition.protein_g,
            nutrition.carbs_g,
            nutrition.fat_g,
            source,
            utcnow(),
        ),
    )
    conn.commit()


def resolve(
    conn: sqlite3.Connection,
    item: FoodItem,
    estimator: Callable[[str, str], Nutrition | None] | None = None,
) -> Resolved:
    """Resolve one item to totals. `estimator` is the model fallback; it is
    optional so that tools, tests and evals can run entirely offline."""
    key = normalize(item.name)

    hit = _cache_get(conn, key)
    if hit:
        return Resolved(
            name=hit["name"], qty=item.qty, unit=_best_unit(item.unit, hit["unit"]),
            nutrition=hit["nutrition"].scaled(item.qty), source="cache", confidence=0.9,
        )

    seed = _seed_lookup(item.name)
    if seed:
        return Resolved(
            name=seed["name"], qty=item.qty, unit=_best_unit(item.unit, seed["unit"]),
            nutrition=seed["nutrition"].scaled(item.qty), source="seed", confidence=1.0,
        )

    fuzzy = _fuzzy_lookup(item.name)
    if fuzzy:
        record, score = fuzzy
        return Resolved(
            name=record["name"], qty=item.qty, unit=_best_unit(item.unit, record["unit"]),
            nutrition=record["nutrition"].scaled(item.qty), source="fuzzy", confidence=score,
        )

    if estimator is not None:
        estimated = estimator(item.name, item.unit or "serving")
        if estimated is not None:
            cache_put(conn, item.name, item.unit or "serving", estimated, "model")
            return Resolved(
                name=item.name, qty=item.qty, unit=item.unit or "serving",
                nutrition=estimated.scaled(item.qty), source="model", confidence=0.6,
            )

    # Unknown foods log as zero rather than as a fabricated number. A visible
    # zero is honest; an invented 250 kcal silently corrupts the day's total.
    return Resolved(
        name=item.name, qty=item.qty, unit=item.unit or "serving",
        nutrition=Nutrition(), source="unknown", confidence=0.0,
    )
