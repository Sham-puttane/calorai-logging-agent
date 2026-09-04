"""The agent's tool surface.

Six tools, split by what they *do to the data* rather than by topic:

  writes   log_meal, correct_meal, delete_meal
  reads    get_daily_totals, find_meals
  lookup   lookup_nutrition

The boundary that matters most is `correct_meal` being separate from
`log_meal`. A single "record what they said" tool would let the model handle
"actually that was 3 rotis" by logging three more rotis, and the day's total
would be wrong in a way no prompt wording reliably prevents. Splitting them
makes double-counting *structurally* impossible: correct_meal has no INSERT
path, and log_meal has no UPDATE path.

Note what log_meal does NOT do: make the agent look nutrition up first.
Chaining lookup_nutrition -> log_meal would cost an extra round trip on the
most common action in the product, to make a decision the model has no input
into. So logging resolves nutrition internally, and lookup_nutrition exists for
the case where the user asks *without* logging ("how many calories in a
paratha?").

Tools are built per session by `make_tools`, closing over the connection and
user id, so no tool can read or write another user's rows.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from . import repository as repo
from .nutrition import resolve
from .schemas import FoodItem, Nutrition


class ItemArg(BaseModel):
    name: str = Field(description="Food name, e.g. 'paratha'")
    qty: float = Field(default=1.0, description="How many units, e.g. 2")
    unit: str = Field(
        default="serving",
        description="piece | katori | cup | glass | slice | serving | plate",
    )


def _dump(payload: dict[str, Any]) -> str:
    return json.dumps(payload, default=str)


def make_estimator(enabled: bool = True):
    """Model fallback for foods missing from the table. Returns None when
    unavailable so the caller degrades to logging zero rather than inventing."""
    if not enabled:
        return None

    def estimate(name: str, unit: str) -> Nutrition | None:
        from .llm import get_text_model
        from .schemas import NutritionEstimate

        model = get_text_model()
        if getattr(model, "_llm_type", "") == "calorai-mock":
            return None
        try:
            structured = model.with_structured_output(NutritionEstimate)
            result = structured.invoke(
                f"Give typical nutrition for one {unit} of '{name}' as eaten at home. "
                "Be approximate; a sensible everyday serving is fine."
            )
            data = result if isinstance(result, NutritionEstimate) else NutritionEstimate(**dict(result))
            return Nutrition(
                kcal=data.kcal, protein_g=data.protein_g,
                carbs_g=data.carbs_g, fat_g=data.fat_g,
            )
        except Exception:
            return None

    return estimate


def make_tools(
    conn: sqlite3.Connection, user_id: str, estimator=None
) -> list[BaseTool]:
    """Build the tool surface bound to one user's session."""

    # -- writes ---------------------------------------------------------------
    def log_meal(
        items: list[ItemArg],
        slot: str = "",
        day: str = "",
        note: str = "",
        is_estimate: bool = False,
    ) -> str:
        """Record a NEW meal the user just told you about. Use for anything
        newly eaten. Do NOT use this to fix a meal already logged -- that is
        correct_meal. Returns the logged items and the day's updated totals."""
        parsed = [FoodItem(name=i.name, qty=i.qty, unit=i.unit) for i in items]
        return _dump(
            repo.log_meal(
                conn, user_id, parsed,
                slot=slot or None, day=day or None, note=note or None,
                is_estimate=is_estimate, estimator=estimator,
            )
        )

    def correct_meal(
        target_hint: str = "",
        new_qty: float | None = None,
        new_name: str = "",
        new_unit: str = "",
    ) -> str:
        """Fix something ALREADY logged -- 'actually that was 3 rotis not 2',
        'that was dal not rice'. Updates the existing entry in place so totals
        change by the difference and nothing is double counted. target_hint is
        the food to fix; leave it empty to mean the most recent thing logged."""
        return _dump(
            repo.correct_meal(
                conn, user_id, target_hint=target_hint,
                new_qty=new_qty, new_name=new_name or None,
                new_unit=new_unit or None, estimator=estimator,
            )
        )

    def delete_meal(target_hint: str = "") -> str:
        """Remove something logged by mistake -- 'scratch that', 'I didn't
        actually have the chai'. Leave target_hint empty for the most recent."""
        return _dump(repo.delete_meal(conn, user_id, target_hint=target_hint))

    # -- reads ----------------------------------------------------------------
    def get_daily_totals(day: str = "today") -> str:
        """Calories and macros for a day. Use for 'how am I doing', 'how much
        protein have I had'. Accepts 'today', 'yesterday' or YYYY-MM-DD."""
        return _dump(repo.daily_totals(conn, user_id, day or "today"))

    def find_meals(query: str = "", day: str = "", slot: str = "") -> str:
        """Look up meals already logged. Use this for 'same as yesterday' --
        fetch that meal, then log_meal the items it returns. Also answers
        'what did I have for lunch'."""
        return _dump(
            repo.find_meals(
                conn, user_id, query=query, day=day or None, slot=slot or None
            )
        )

    def lookup_nutrition(food: str, qty: float = 1.0, unit: str = "serving") -> str:
        """Nutrition for a food WITHOUT logging it -- 'how many calories in a
        samosa?'. Never call this before log_meal; logging handles its own
        lookup."""
        result = resolve(conn, FoodItem(name=food, qty=qty, unit=unit), estimator=estimator)
        return _dump(
            {
                "food": result.name, "qty": result.qty, "unit": result.unit,
                "kcal": round(result.nutrition.kcal),
                "protein_g": round(result.nutrition.protein_g, 1),
                "carbs_g": round(result.nutrition.carbs_g, 1),
                "fat_g": round(result.nutrition.fat_g, 1),
                "source": result.source,
                "known": result.source != "unknown",
            }
        )

    return [
        StructuredTool.from_function(log_meal),
        StructuredTool.from_function(correct_meal),
        StructuredTool.from_function(delete_meal),
        StructuredTool.from_function(get_daily_totals),
        StructuredTool.from_function(find_meals),
        StructuredTool.from_function(lookup_nutrition),
    ]
