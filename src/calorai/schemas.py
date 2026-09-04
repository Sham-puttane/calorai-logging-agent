"""Typed contracts between the models, the tools and the database.

The shape that matters most here is `PlateAnalysis`. Its confidence is split in
two -- see docs/RESEARCH.md -- because a VLM's failure modes are independent:
it is usually sure *what* the food is and usually guessing *how much*. One
blended score cannot tell the agent which question to ask.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Slot = Literal["breakfast", "lunch", "dinner", "snack"]

# --- clarification thresholds -------------------------------------------------
# Deliberately asymmetric. Identification below 0.6 means the agent genuinely
# does not know what it is looking at, and logging would produce garbage.
#
# Portion confidence is held to a *lower* bar on purpose. The literature is
# clear that scale is close to unknowable from one photo, so a symmetric
# threshold would make the agent ask about portions on nearly every image --
# and over-asking is the failure mode the brief warns about. Below this bar the
# agent still logs, but says out loud what it assumed.
ID_CONFIDENCE_THRESHOLD = 0.60
PORTION_CONFIDENCE_THRESHOLD = 0.35


class Nutrition(BaseModel):
    """Per one unit of the food, never per meal."""

    kcal: float = 0.0
    protein_g: float = 0.0
    carbs_g: float = 0.0
    fat_g: float = 0.0

    def scaled(self, qty: float) -> "Nutrition":
        return Nutrition(
            kcal=self.kcal * qty,
            protein_g=self.protein_g * qty,
            carbs_g=self.carbs_g * qty,
            fat_g=self.fat_g * qty,
        )


class FoodItem(BaseModel):
    """One food in a meal, in household units -- '2 roti', '1 katori dal'.

    Household units rather than grams: it is how people speak, it is how the
    nutrition table is keyed, and asking a VLM for grams forces a conversion it
    is measurably bad at.
    """

    name: str = Field(description="Food name, singular and lowercase, e.g. 'paratha'")
    qty: float = Field(default=1.0, description="How many units")
    unit: str = Field(default="serving", description="piece | katori | cup | glass | serving | plate")


class DetectedItem(FoodItem):
    """A FoodItem the vision model proposed, with its two uncertainties."""

    id_confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    portion_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    alternatives: list[str] = Field(
        default_factory=list,
        description="Other plausible identifications, e.g. ['tofu'] for paneer",
    )


class PlateAnalysis(BaseModel):
    """The vision model's entire output. This is the handoff object: it enters
    agent state as a structured observation, never as free text."""

    items: list[DetectedItem] = Field(default_factory=list)
    scale_reference: str | None = Field(
        default=None,
        description="Which object was used as the ruler, e.g. 'dinner plate ~27cm'",
    )
    notes: str = ""
    failed: bool = False
    failure_reason: str | None = None

    # -- uncertainty routing ---------------------------------------------------
    def unidentified(self) -> list[DetectedItem]:
        return [i for i in self.items if i.id_confidence < ID_CONFIDENCE_THRESHOLD]

    def unsized(self) -> list[DetectedItem]:
        return [i for i in self.items if i.portion_confidence < PORTION_CONFIDENCE_THRESHOLD]

    def needs_user_input(self) -> bool:
        """True only when the agent genuinely cannot proceed. A shaky portion is
        not a blocker -- it is logged with a stated assumption."""
        return self.failed or not self.items or bool(self.unidentified())

    def clarifying_question(self) -> str | None:
        """One consolidated question, never a checklist."""
        if self.failed:
            # Blaming the photo for a rate limit is a lie that sends the user
            # off to check their picture when the picture was fine. Observed
            # live: a perfectly readable plate of paneer came back as "I
            # couldn't read that photo clearly" because both vision providers
            # were throttled at that moment.
            reason = (self.failure_reason or "").lower()
            if any(k in reason for k in ("429", "rate", "quota", "resource_exhausted")):
                return (
                    "i'm rate limited on photos right now, so i couldn't look at "
                    "that one -- try again in a moment, or just tell me what was "
                    "on the plate?"
                )
            if "no image at" in reason or "limit is" in reason:
                return "i couldn't open that file -- what was on the plate?"
            return "i couldn't make that photo out -- what was on the plate?"
        if not self.items:
            return "I couldn't make out any food in that one. What was it?"
        unknown = self.unidentified()
        if not unknown:
            return None
        parts = []
        for item in unknown[:3]:
            if item.alternatives:
                options = " or ".join([item.name] + item.alternatives[:2])
                parts.append(options)
            else:
                parts.append(f"something I couldn't place ({item.name}?)")
        joined = ", and ".join(parts)
        return f"I can see most of it -- is that {joined}?"

    def assumptions(self) -> list[str]:
        """Portion guesses worth admitting to in the confirmation message."""
        notes = []
        for item in self.unsized():
            notes.append(f"guessed {item.qty:g} {item.unit} of {item.name}")
        return notes


class NutritionEstimate(BaseModel):
    """Structured output when the text model has to invent nutrition data."""

    kcal: float
    protein_g: float
    carbs_g: float
    fat_g: float
    basis: str = Field(default="", description="One line on what serving this assumes")


class ExtractedFact(BaseModel):
    """One durable thing worth remembering. See memory/extractor.py."""

    key: str = Field(description="snake_case, e.g. diet, protein_target_g, allergy")
    value: str
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class FactExtraction(BaseModel):
    """Extractor envelope. Empty list is the common and correct answer."""

    facts: list[ExtractedFact] = Field(default_factory=list)
