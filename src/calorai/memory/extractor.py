"""Deciding what is worth remembering.

Runs *after* the reply has been sent, so nothing here appears in p50 or p95.

The design goal is selectivity. "Store every message and hope" is explicitly a
red flag, and it is also expensive: an extraction call on every turn doubles
model spend to learn almost nothing, because almost nothing anyone says about
food is durable. "had 2 rotis" is an event, not a fact about the person.

So there are three tiers, cheapest first:

1. **A regex gate.** Most messages contain no durable-fact signal at all and
   exit here having cost nothing. This is where the selectivity actually comes
   from -- the model is never asked about a message that obviously has no fact.
2. **Rules** for the handful of patterns that carry most of the real value
   (diet, allergies, macro targets). Deterministic and free.
3. **The model**, only for signal-bearing messages the rules did not resolve.
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any

from ..schemas import ExtractedFact, FactExtraction
from . import store

# --- tier 1: the gate --------------------------------------------------------
# If none of these fire, the message cannot plausibly contain a durable fact
# about the person, and we stop without calling anything.
# Stems are matched as prefixes on purpose: a trailing \b would make "allerg"
# fail on "allergic" and "target" fail on "targeting". Only the leading
# boundary is anchored.
_SIGNAL_RE = re.compile(
    r"\b("
    r"i'?m\s|i\s+am\s|my\s|me\b"
    r"|vegetarian|vegan|non-?veg|eggetarian|pescatarian|jain|halal|kosher"
    r"|allerg|intoleran|lactose|gluten|diabetic"
    r"|target|goal|aiming|trying to|cutting|bulking|deficit"
    r"|don'?t eat|do not eat|avoid|can'?t eat|stopped eating|gave up"
    r"|usual|always|never|every day|prefer"
    r")",
    re.I,
)

# --- tier 2: rules -----------------------------------------------------------
_DIET_RE = re.compile(
    r"\b(?:i'?m|i am|im)\s+(?:a\s+)?(vegetarian|vegan|pescatarian|eggetarian|non-?veg)\b",
    re.I,
)
_DIET_BARE_RE = re.compile(r"\b(vegetarian|vegan|pescatarian|eggetarian)\b(?:\s+btw)?", re.I)
_PROTEIN_RE = re.compile(
    r"(?:target|goal|aiming for|hitting|trying to (?:hit|get))\D{0,20}?(\d{2,4})\s*g?\b"
    r"[^.]{0,20}protein|protein[^.]{0,20}?(\d{2,4})\s*g\b",
    re.I,
)
_CALORIE_RE = re.compile(
    r"(?:target|goal|aiming for|budget|cap)\D{0,20}?(\d{3,5})\s*(?:kcal|cal|calories)\b",
    re.I,
)
_ALLERGY_RE = re.compile(r"\b(?:allergic to|allergy to)\s+([a-z\s]{2,25})", re.I)
_AVOID_RE = re.compile(
    r"\b(?:i\s+)?(?:don'?t|do not|can'?t|cannot)\s+eat\s+([a-z\s]{2,25})", re.I
)


def _rule_extract(text: str) -> list[ExtractedFact]:
    facts: list[ExtractedFact] = []

    diet = _DIET_RE.search(text) or _DIET_BARE_RE.search(text)
    if diet:
        value = diet.group(1).lower().replace("nonveg", "non-vegetarian")
        facts.append(ExtractedFact(key="diet", value=value, confidence=0.95))

    protein = _PROTEIN_RE.search(text)
    if protein:
        grams = protein.group(1) or protein.group(2)
        if grams:
            facts.append(
                ExtractedFact(key="protein_target_g", value=str(int(grams)), confidence=0.9)
            )

    calories = _CALORIE_RE.search(text)
    if calories:
        facts.append(
            ExtractedFact(
                key="calorie_target", value=str(int(calories.group(1))), confidence=0.9
            )
        )

    allergy = _ALLERGY_RE.search(text)
    if allergy:
        facts.append(
            ExtractedFact(key="allergy", value=allergy.group(1).strip(), confidence=0.9)
        )

    avoid = _AVOID_RE.search(text)
    if avoid:
        facts.append(ExtractedFact(key="avoids", value=avoid.group(1).strip(), confidence=0.8))

    return facts


# --- tier 3: the model -------------------------------------------------------
_EXTRACTOR_PROMPT = """\
You extract durable facts about a person from one message they sent a food-logging assistant.

A durable fact is true next month: a diet, an allergy, a macro target, a standing preference.
What someone ate is an EVENT, not a fact -- never extract it.

Return an empty list unless the message genuinely states something lasting.
Empty is the correct answer most of the time.

Use snake_case keys. Prefer these when they fit: diet, allergy, avoids,
protein_target_g, calorie_target, cuisine, name.

Message: {message}"""


def _model_extract(text: str) -> list[ExtractedFact]:
    from ..llm import get_text_model

    model = get_text_model()
    if model._llm_type == "calorai-mock":
        return []  # the rules already are the mock's behaviour
    try:
        structured = model.with_structured_output(FactExtraction)
        result = structured.invoke(_EXTRACTOR_PROMPT.format(message=text))
        if isinstance(result, FactExtraction):
            return result.facts
        if isinstance(result, dict):
            return FactExtraction(**result).facts
    except Exception:
        # Memory is best-effort. A failed extraction must never surface as a
        # broken turn -- the reply has already gone out.
        return []
    return []


def extract_and_store(
    conn: sqlite3.Connection, user_id: str, text: str, use_model: bool = True
) -> list[dict[str, Any]]:
    """The background pass. Returns what it wrote, for logging and the demo."""
    if not text or not _SIGNAL_RE.search(text):
        return []

    facts = _rule_extract(text)
    if not facts and use_model:
        facts = _model_extract(text)

    written = []
    for fact in facts:
        result = store.put_fact(
            conn, user_id, fact.key, fact.value, fact.confidence, source_message=text[:200]
        )
        if result.get("changed"):
            written.append(result)
    return written


def maybe_learn_alias(conn: sqlite3.Connection, user_id: str, text: str) -> bool:
    """Two ways shorthand gets learned, both off the reply path.

    Explicit ('my usual is 2 parathas and chai') beats inferred, and an
    explicit definition is never overwritten by a later inference.
    """
    body = store.detect_alias_definition(text)
    if body:
        from ..llm.mock import parse_foods

        items = parse_foods(body)
        if items:
            store.put_alias(conn, user_id, "my usual", items, source="explicit")
            return True
    return store.learn_usual_if_repeated(conn, user_id)
