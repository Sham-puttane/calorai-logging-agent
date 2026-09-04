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
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
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
# Matches the lines _describe_plate renders: "- 1 katori rice (portion uncertain...)"
_PLATE_ITEM_RE = re.compile(
    r"^-\s*(\d+(?:\.\d+)?)\s+(\S+)\s+([a-z][a-z ]*?)\s*(?:\(portion uncertain[^)]*\))?\s*$",
    re.M,
)

_SKIPPED_RE = re.compile(r"\b(grazed|grazing|snacked|picked at|nibbled)\b", re.I)

# Deletes are tested BEFORE foods are parsed: "scratch the samosas" names a
# food, and a food-first rule would cheerfully log it a second time.
_DELETE_RE = re.compile(
    r"\b(scratch that|scratch the|remove|delete|undo|forget that"
    r"|didn'?t (?:actually )?(?:have|eat)|never mind)\b",
    re.I,
)

# Last resort for a food the table has never heard of. A real model logs
# "zorblax casserole" without blinking; without this the rule engine could only
# ever log foods it already knows, and the unknown-food path would go untested.
_LOG_INTENT_RE = re.compile(
    r"\b(?:had|ate|having|just had|finished|grabbed)\s+"
    r"(?:some |a |an |the |my )?([a-z][a-z\s]{2,40})",
    re.I,
)

# The alias expansion ingest writes:
#   ("my usual" for this person means: 2 piece paratha, 1 cup chai)
_ALIAS_BLOCK_RE = re.compile(r"for this person means:\s*(.+?)\)\s*$", re.S)
_QTY_UNIT_NAME_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s+(\S+)\s+(.+?)\s*$")


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

        # A photo that made it past the confidence gate: the vision handoff is
        # already in context, so log exactly those items in ONE call. This is
        # the branch that keeps a photo plus its caption from becoming two
        # meals -- there is no separate log for the caption text.
        plate = self._plate_items(messages)
        if plate:
            return self._call(
                "log_meal", {"items": plate, "slot": "", "note": "from photo"}
            )

        # "my usual" was already expanded to concrete food by ingest, before
        # this model was ever called. Log what it resolved to.
        alias_items = self._alias_items(messages)
        if alias_items:
            return self._call(
                "log_meal",
                {"items": alias_items, "slot": _detect_slot(text) or "", "note": text[:120]},
            )

        if _TOTALS_RE.search(low):
            return self._call("get_daily_totals", {"day": "today"})

        if _DELETE_RE.search(low):
            foods = parse_foods(text)
            return self._call(
                "delete_meal", {"target_hint": foods[0]["name"] if foods else ""}
            )

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

        # a food the table does not know, stated as something eaten
        intent = _LOG_INTENT_RE.search(text)
        if intent and not _SKIPPED_RE.search(low):
            phrase = intent.group(1).strip(" .,!?")
            if phrase:
                return self._call(
                    "log_meal",
                    {"items": [{"name": phrase, "qty": 1, "unit": "serving"}],
                     "slot": _detect_slot(text) or "", "note": text[:120]},
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

    def _plate_items(self, messages: list[BaseMessage]) -> list[dict[str, Any]]:
        """Read the vision handoff back out of context.

        A real model reads these lines from the prompt and decides to log them;
        the mock parses the same lines. Keeping it to the rendered text rather
        than a side channel means the mock exercises the same handoff the real
        model does.
        """
        for message in reversed(messages):
            content = str(getattr(message, "content", ""))
            if isinstance(message, SystemMessage) and content.startswith("[photo analysed"):
                return [
                    {"name": name.strip(), "qty": float(qty), "unit": unit}
                    for qty, unit, name in _PLATE_ITEM_RE.findall(content)
                ]
        return []

    def _alias_items(self, messages: list[BaseMessage]) -> list[dict[str, Any]]:
        for message in reversed(messages):
            content = str(getattr(message, "content", ""))
            if isinstance(message, SystemMessage) and "for this person means:" in content:
                block = _ALIAS_BLOCK_RE.search(content)
                if not block:
                    return []
                items = []
                for chunk in block.group(1).split(","):
                    parsed = _QTY_UNIT_NAME_RE.match(chunk)
                    if parsed:
                        qty, unit, name = parsed.groups()
                        items.append({"name": name, "qty": float(qty), "unit": unit})
                return items
        return []

    def _call(self, name: str, args: dict[str, Any]) -> AIMessage:
        return AIMessage(
            content="",
            tool_calls=[{"name": name, "args": args, "id": f"call_{name}"}],
        )

    def _summarise(self, result: ToolMessage) -> str:
        """Phrase the tool result the way the real model is asked to.

        A real model writes this sentence; the mock templates it. Keeping the
        shape close means eval assertions about phrasing stay meaningful and
        the offline demo does not read like a debug dump.
        """
        import json

        try:
            data = json.loads(str(result.content))
        except (json.JSONDecodeError, TypeError):
            return "done."

        if not data.get("ok", True):
            return f"hmm, {data.get('error', 'I could not do that')} -- want to tell me again?"

        if "changed" in data:  # correct_meal
            change = data["changed"]
            totals = data.get("totals_after", {})
            return (
                f"fixed -- {change['from']} is now {change['to']}. "
                f"that puts you at {totals.get('kcal', 0)} cal today."
            )

        if "removed" in data:
            return f"removed {data['removed']}. now at {data['totals_after']['kcal']} cal today."

        if "items" in data and "meal_kcal" in data:  # log_meal
            items = ", ".join(f"{i['qty']:g} {i['unit']} {i['name']}" for i in data["items"])
            reply = f"logged {items} -- about {data['meal_kcal']} cal."
            if data.get("unknown_foods"):
                reply += f" (i don't know {data['unknown_foods'][0]}, so it's counted as 0)"
            reply += f" you're at {data['totals_after']['kcal']} cal today."
            return reply

        if "meals" in data:  # find_meals
            if not data["meals"]:
                return "i couldn't find that one -- what did you have?"
            items = ", ".join(f"{m['qty']:g} {m['unit']} {m['name']}" for m in data["meals"])
            return f"that was {items}."

        if "kcal" in data and "items_logged" in data:  # get_daily_totals
            if not data["items_logged"]:
                return "nothing logged yet today -- what have you had?"
            return (
                f"{data['kcal']} cal so far today -- {data['protein_g']:g}g protein, "
                f"{data['carbs_g']:g}g carbs, {data['fat_g']:g}g fat."
            )

        if "food" in data:  # lookup_nutrition
            return f"{data['qty']:g} {data['unit']} of {data['food']} is about {data['kcal']} cal."

        return "done."
