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
    # Field descriptions are sent on every single call, so they are kept to the
    # minimum that still steers the model. This schema alone was 189 tokens
    # before trimming -- the most expensive of the six tools.
    name: str = Field(description="food, e.g. paratha")
    qty: float = Field(
        default=1.0,
        # Spelled out because the model was observed saying "2 parathas" in its
        # reply while sending qty=1 in the call -- the summary was right and the
        # data was wrong, which is the worst way to be wrong.
        description="how many, copied from the user: '2 parathas' -> 2, 'two thirds' -> 0.67",
    )
    unit: str = Field(default="serving", description="piece|katori|cup|glass|serving")


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
    conn: sqlite3.Connection,
    user_id: str,
    estimator=None,
    note_ref: dict[str, str] | None = None,
) -> list[BaseTool]:
    """Build the tool surface bound to one user's session.

    `note_ref` is a mutable box the graph fills with the raw user message each
    turn. Provenance is worth storing, but it is not a decision the model
    should be spending tokens on -- exposing a `note` parameter cost schema
    tokens on every call to have the model retype what we already have.
    """
    note_ref = {} if note_ref is None else note_ref

    # -- writes ---------------------------------------------------------------
    def log_meal(
        items: list[ItemArg],
        slot: str = "",
        day: str = "",
        is_estimate: bool = False,
    ) -> str:
        """Log a NEW meal. Not for fixing something already logged."""
        parsed = [FoodItem(name=i.name, qty=i.qty, unit=i.unit) for i in items]
        return _dump(
            repo.log_meal(
                conn, user_id, parsed,
                slot=slot or None, day=day or None, note=note_ref.get("text"),
                is_estimate=is_estimate, estimator=estimator,
            )
        )

    def correct_meal(
        target_hint: str = "",
        new_qty: float | None = None,
        new_name: str = "",
        new_unit: str = "",
        scale_whole_meal: float | None = None,
    ) -> str:
        """Fix something already logged that was WRONG. Updates in place.

        NOT for additions. 'plus rice', 'also had a chai', 'forgot the dal'
        mean they ate more -- use log_meal. This tool REPLACES, so using it for
        an addition deletes food they really did eat.

        One food: 'actually that was 3 rotis not 2' -> target_hint='roti',
        new_qty=3. Empty target_hint means the most recent item.

        The WHOLE last meal at once: 'half of that was my brother's',
        'only ate a third' -> scale_whole_meal=0.5 or 0.33. Use this whenever a
        share applies to everything on the plate, not one dish."""
        if scale_whole_meal is not None:
            return _dump(
                repo.scale_meal(conn, user_id, scale_whole_meal, target_hint=target_hint)
            )
        return _dump(
            repo.correct_meal(
                conn, user_id, target_hint=target_hint,
                new_qty=new_qty, new_name=new_name or None,
                new_unit=new_unit or None, estimator=estimator,
                # the raw turn, so the repository can veto a substitution that
                # is really an addition
                message=note_ref.get("text"),
            )
        )

    def delete_meal(target_hint: str = "") -> str:
        """Remove something logged by mistake. Empty hint = most recent."""
        return _dump(repo.delete_meal(conn, user_id, target_hint=target_hint))

    # -- reads ----------------------------------------------------------------
    def get_daily_totals(day: str = "today") -> str:
        """Calories and macros for a day ('today', 'yesterday', YYYY-MM-DD)."""
        return _dump(repo.daily_totals(conn, user_id, day or "today"))

    def find_meals(query: str = "", day: str = "", slot: str = "") -> str:
        """Look up past meals. For 'same as yesterday': call this, then
        log_meal what it returns."""
        return _dump(
            repo.find_meals(
                conn, user_id, query=query, day=day or None, slot=slot or None
            )
        )

    def lookup_nutrition(food: str, qty: float = 1.0, unit: str = "serving") -> str:
        """Nutrition for a food WITHOUT logging it. Never call before
        log_meal."""
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
