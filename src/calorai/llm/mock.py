"""A deterministic, offline stand-in for the text model.

Why this exists: the eval suite needs to assert *agent behaviour* -- did the
right tool fire, did the database end up in the right state -- and a real model
makes that non-deterministic and key-dependent. With this backend the whole
repo, tests and evals included, runs on a clean clone with no API keys and no
network.

It is a rule-based approximation of the agent, not a language model. It is good
enough to exercise every control path and honest about being a stub: it never
appears in a latency number, and README says so.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from ..nutrition import _index, normalize

# --- surface parsing ---------------------------------------------------------
_WORD_QTY: dict[str, float] = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "couple": 2,
    "half": 0.5, "quarter": 0.25,
}
_FRACTIONS: list[tuple[str, float]] = [
    ("two thirds", 0.67), ("two-thirds", 0.67), ("three quarters", 0.75),
    ("three-quarters", 0.75), ("one third", 0.33), ("a third", 0.33),
    ("half", 0.5), ("quarter", 0.25),
]

_SLOTS = ("breakfast", "lunch", "dinner", "snack")

_CORRECTION_RE = re.compile(
    r"\b(actually|sorry|i meant|make (?:that|it)|no,|not \d)\b", re.I
)
_QTY_FIX_RE = re.compile(r"\b(?:that was|make it|it was)?\s*(\d+(?:\.\d+)?)\b", re.I)
_NOT_QTY_RE = re.compile(r"\bnot\s+(\d+(?:\.\d+)?)\b", re.I)
_TOTALS_RE = re.compile(
    r"\b(how (?:am i|much|many)|what.{0,12}(?:total|calorie|protein|carb|fat)"
    r"|calories? (?:so far|today)|doing (?:on|today)|left today)\b",
    re.I,
)
_YESTERDAY_RE = re.compile(r"\b(same as (?:yesterday|last night)|like yesterday)\b", re.I)
_SKIPPED_RE = re.compile(r"\b(grazed|grazing|snacked|picked at|nibbled)\b", re.I)


def _detect_slot(text: str) -> str | None:
    low = text.lower()
    for slot in _SLOTS:
        # "skipped lunch" names a slot it did NOT eat -- do not attribute to it
        if re.search(rf"\bskipped\s+{slot}\b", low):
            continue
        if re.search(rf"\b{slot}\b", low):
            return slot
    return None


def _scan_text(text: str) -> str:
    """Lowercase and depunctuate, but keep every word.

    Deliberately *not* `normalize()`: that strips quantity words like 'half'
    and 'two', which is right for looking a food up in the table and wrong for
    reading how much of it someone ate.
    """
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text.lower())).strip()


def _quantity_for(text: str, start: int, end: int, floor: int) -> float:
    """Quantity for the food occupying [start, end).

    Looks backwards first, bounded by `floor` -- the end of the previous food --
    so '2 parathas and chai' does not give the chai a count of two. Falls back
    to a short forward look, because people trail the amount after the food:
    'biryani, maybe two thirds of the box'.
    """
    before = text[max(floor, start - 30) : start]
    for phrase, value in _FRACTIONS:
        if phrase in before:
            return value
    numbers = re.findall(r"(\d+(?:\.\d+)?)", before)
    if numbers:
        return float(numbers[-1])
    for word in reversed(re.findall(r"[a-z]+", before)[-3:]):
        if word in _WORD_QTY:
            return _WORD_QTY[word]

    # nothing before it -- look just ahead for a trailing fraction only.
    # Counts are not read forwards: in "chai and 2 parathas" the 2 belongs to
    # the parathas, and reading ahead would hand it to the chai.
    after = text[end : end + 35]
    for phrase, value in _FRACTIONS:
        if phrase in after:
            return value
    return 1.0


def parse_foods(text: str) -> list[dict[str, Any]]:
    """Find known foods in free text.

    Longest key first so 'aloo paratha' wins over 'paratha', then re-sorted
    into sentence order so each food's quantity lookback can be bounded by the
    previous one.
    """
    index = _index()
    scan = _scan_text(text)
    matches: list[tuple[int, int, dict]] = []
    claimed: list[tuple[int, int]] = []

    for key in sorted(index.keys(), key=len, reverse=True):
        if not key:
            continue
        for match in re.finditer(rf"\b{re.escape(key)}s?\b", scan):
            span = match.span()
            if any(span[0] < c_end and c_start < span[1] for c_start, c_end in claimed):
                continue
            claimed.append(span)
            matches.append((span[0], span[1], index[key]))
            break

    found: list[dict[str, Any]] = []
    floor = 0
    for start, end, record in sorted(matches):
        found.append(
            {
                "name": record["name"],
                "qty": _quantity_for(scan, start, end, floor),
                "unit": record["unit"],
            }
        )
        floor = end
    return found


class MockChatModel(BaseChatModel):
    """Rule-based agent stand-in. Emits real tool calls so the graph, the tool
    node and the database all run exactly as they do in production."""

    @property
    def _llm_type(self) -> str:
        return "calorai-mock"

    def bind_tools(self, tools: Sequence, **kwargs: Any):  # noqa: ARG002
        # Tool schemas are irrelevant to a rule engine, but the graph binds
        # them unconditionally, so accept and ignore.
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = self._decide(messages)
        return ChatResult(generations=[ChatGeneration(message=message)])

    # -- the rule engine -------------------------------------------------------
    def _decide(self, messages: list[BaseMessage]) -> AIMessage:
        # Second pass: tools have run, so phrase an answer and stop.
        tool_results = [m for m in messages if isinstance(m, ToolMessage)]
        if tool_results and isinstance(messages[-1], ToolMessage):
            return AIMessage(content=self._summarise(tool_results[-1]))

        human = next(
            (m for m in reversed(messages) if isinstance(m, HumanMessage)), None
        )
        text = str(human.content) if human else ""
        low = text.lower()

        if _TOTALS_RE.search(low):
            return self._call("get_daily_totals", {"day": "today"})

        if _CORRECTION_RE.search(low):
            foods = parse_foods(text)
            hint = foods[0]["name"] if foods else ""
            explicit = _NOT_QTY_RE.search(low)
            numbers = [
                float(n) for n in re.findall(r"\b(\d+(?:\.\d+)?)\b", low)
            ]
            # "3 rotis not 2" -> the corrected value is the one that is NOT
            # after the word "not"
            if explicit and len(numbers) >= 2:
                wrong = float(explicit.group(1))
                new_qty = next((n for n in numbers if n != wrong), numbers[0])
            elif numbers:
                new_qty = numbers[0]
            else:
                new_qty = foods[0]["qty"] if foods else 1.0
            return self._call(
                "correct_meal", {"target_hint": hint, "new_qty": new_qty}
            )

        if _YESTERDAY_RE.search(low):
            return self._call(
                "find_meals", {"day": "yesterday", "slot": _detect_slot(low) or ""}
            )

        foods = parse_foods(text)
        if foods:
            return self._call(
                "log_meal",
                {
                    "items": foods,
                    "slot": _detect_slot(text) or "",
                    "note": text[:120],
                },
            )

        if _SKIPPED_RE.search(low):
            return self._call(
                "log_meal",
                {
                    "items": [{"name": "assorted snacks", "qty": 1, "unit": "serving"}],
                    "slot": "snack",
                    "is_estimate": True,
                    "note": text[:120],
                },
            )

        # Nothing actionable: a statement of fact ("i'm vegetarian") lands here
        # and is picked up by the memory extractor, not by a tool.
        return AIMessage(content="Got it.")

    def _call(self, name: str, args: dict[str, Any]) -> AIMessage:
        return AIMessage(
            content="",
            tool_calls=[{"name": name, "args": args, "id": f"call_{name}"}],
        )

    def _summarise(self, result: ToolMessage) -> str:
        text = str(result.content)
        if '"kcal"' in text and '"items_logged"' in text:
            return f"Here's where you're at: {text}"
        return f"Done. {text[:240]}"
