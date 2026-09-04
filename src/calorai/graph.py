"""The agent: a LangGraph tool-calling loop.

    message -> ingest -> [vision] -> agent <-> tools -> reply
                  |                     ^________|
                  |                     (bounded)
                  '-> fast path -----------------> reply   (optional, opt-out)

`ingest` does the deterministic work the model should not be paying for:
resolving "my usual" to concrete food, loading memory, spotting an image.

`agent` is a genuine tool-calling node -- the model sees all six tool schemas
and decides. It is not a classifier dispatching to handlers. That distinction
matters: a router would be faster but it would not generalise past the phrasings
someone thought to write a rule for.

The fast path is a narrow optimisation, not the architecture. It short-circuits
only unambiguous read-only questions ("how am I doing today"), and
CALORAI_FAST_PATH=0 disables it. The eval suite runs with it off, so the agent
has to earn every case on its own.
"""

from __future__ import annotations

import os
import re
import sqlite3
import time
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from . import repository as repo
from .memory import render, store
from .schemas import PlateAnalysis
from .tools import make_estimator, make_tools

# The loop is bounded. Three passes is enough for find_meals -> log_meal (the
# "same as yesterday" shape); beyond that the model is thrashing, and on a
# messaging surface a quick honest "say that again?" beats a long silence.
MAX_TOOL_ROUNDS = 3

SYSTEM_PROMPT = """\
You are CalorAI, a food logging assistant people text like a friend. Replies go \
to WhatsApp: short, warm, lowercase-friendly, no bullet points, no tables, no \
emoji spam. One or two sentences.

WHEN TO LOG WITHOUT ASKING
Log it. Assume a normal home portion and say what you assumed, so they can \
correct you in three words. "logged 2 parathas and a chai, ~430 cal" is a good \
reply. Guessing and saying so beats interrogating.

WHEN TO ASK
Ask only when logging would produce garbage: you genuinely cannot tell what the \
food is, or the amount could swing the calories by more than about 40%. Ask ONE \
question, never a list. Never ask about exact grams, cooking oil, or brand.

CORRECTIONS
"actually that was 3 not 2" means fix what is already there. Use correct_meal, \
never log_meal -- logging again would double count. Then confirm the new total.

REFERENCES TO PAST MEALS
"same as yesterday" means look it up with find_meals, then log those items with \
log_meal. Tell them what you logged so they can catch a wrong guess.

WHAT YOU KNOW ABOUT THEM
Anything under [what I know about you] is durable and already confirmed. Use it \
without re-asking. If they are vegetarian, do not offer them chicken.

NUMBERS
Never do arithmetic yourself. Totals come from get_daily_totals. Report what the \
tools return."""


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    image_path: str | None
    caption: str | None
    plate_analysis: dict[str, Any] | None
    memory_block: str
    alias_expansion: str | None
    spans: dict[str, float]
    short_circuit: str | None
    rounds: int


# --------------------------------------------------------------------------
# fast path
# --------------------------------------------------------------------------
# Deliberately narrow. These are read-only, unambiguous, and among the most
# frequent things a calorie tracker is asked. Anything not matching falls
# through to the agent, so a miss costs nothing but a normal turn.
_FAST_TOTALS_RE = re.compile(
    r"^\s*(?:hey\s+|so\s+|ok\s+)?(?:"
    r"how(?:'s|s| is| am i| are we| much| many)?\s*(?:am\s+i\s+)?"
    r"(?:doing|going|calories|protein|carbs?|fat)"
    r"|what'?s my (?:total|count|protein|calories|macros)"
    r"|(?:my\s+)?(?:totals?|macros)"
    r"|calories?\s+(?:today|so far)"
    r")\b",
    re.I,
)


def fast_path_answer(conn: sqlite3.Connection, user_id: str, text: str) -> str | None:
    """Short-circuit unambiguous read-only questions.

    Two guards keep this from mis-routing. The message must open with a totals
    question, AND it must not name a food -- otherwise "how many calories in a
    samosa" (a lookup) and "how am I doing, also had 2 rotis" (a log) would both
    be answered with today's totals and their real intent silently dropped.
    """
    if os.environ.get("CALORAI_FAST_PATH", "1") not in {"1", "true", "True"}:
        return None
    if not text or not _FAST_TOTALS_RE.match(text):
        return None

    from .llm.mock import parse_foods

    if parse_foods(text):
        return None

    totals = repo.daily_totals(conn, user_id)
    if totals["items_logged"] == 0:
        return "nothing logged yet today -- what have you had?"

    facts = {f["key"]: f["value"] for f in store.get_facts(conn, user_id)}
    low = text.lower()
    # Answer the macro they actually asked about, not a generic dump. A target
    # from memory turns the number into progress, but the lead is the same
    # either way.
    if "protein" in low:
        reply = f"{totals['protein_g']:g}g protein today"
        if target := facts.get("protein_target_g"):
            remaining = float(target) - totals["protein_g"]
            reply += (
                f", {remaining:g}g to go on your {target}g target."
                if remaining > 0
                else f" -- past your {target}g target."
            )
        else:
            reply += f", across {totals['kcal']} cal."
    else:
        reply = (
            f"{totals['kcal']} cal so far today -- {totals['protein_g']:g}g protein, "
            f"{totals['carbs_g']:g}g carbs, {totals['fat_g']:g}g fat."
        )
        if target := facts.get("calorie_target"):
            reply += f" that's {round(float(target) - totals['kcal'])} left of your {target}."
    if totals["items_estimated"]:
        reply += " (some of that is estimated)"
    return reply


# --------------------------------------------------------------------------
# graph
# --------------------------------------------------------------------------
def build_graph(conn: sqlite3.Connection, user_id: str, streaming: bool = False):
    from .llm import get_fallback_text_model, get_text_model

    estimator = make_estimator()
    tools = make_tools(conn, user_id, estimator=estimator)

    primary = get_text_model(streaming).bind_tools(tools)
    fallback = get_fallback_text_model()
    if fallback is not None:
        # Free tiers are rate limited; the benchmark alone will trip Groq's
        # 30 rpm. Falling over keeps a run alive instead of dying at request 31.
        primary = primary.with_fallbacks([fallback.bind_tools(tools)])

    tool_node = ToolNode(tools)

    def ingest(state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        uid = state["user_id"]
        last = state["messages"][-1] if state["messages"] else None
        text = str(last.content) if isinstance(last, HumanMessage) else ""

        # 1. shorthand -> concrete food, before the model is involved
        expansion = None
        alias = store.resolve_alias(conn, uid, text) if text else None
        if alias and alias.get("items"):
            items = ", ".join(
                f"{i.get('qty', 1):g} {i.get('unit', '')} {i['name']}".strip()
                for i in alias["items"]
            )
            expansion = f'("{alias["phrase"]}" for this person means: {items})'

        # 2. everything memory knows, rendered small
        memory_block = render.render_memory_block(conn, uid)

        # 3. the narrow read-only short circuit
        short = fast_path_answer(conn, uid, text) if text and not state.get("image_path") else None

        return {
            "memory_block": memory_block,
            "alias_expansion": expansion,
            "short_circuit": short,
            "rounds": 0,
            "spans": {**state.get("spans", {}), "ingest": time.perf_counter() - started},
        }

    def vision(state: AgentState) -> dict[str, Any]:
        from .vision import analyse_plate

        started = time.perf_counter()
        uid = state["user_id"]
        priors = render.render_vision_priors(conn, uid)
        analysis = analyse_plate(state["image_path"], state.get("caption"), priors)
        return {
            "plate_analysis": analysis.model_dump(),
            "spans": {**state.get("spans", {}), "vision": time.perf_counter() - started},
        }

    def agent(state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        preamble: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]

        if state.get("memory_block"):
            preamble.append(SystemMessage(content=state["memory_block"]))
        if state.get("alias_expansion"):
            preamble.append(SystemMessage(content=state["alias_expansion"]))

        analysis = state.get("plate_analysis")
        if analysis:
            preamble.append(SystemMessage(content=_describe_plate(analysis)))

        response = primary.invoke(preamble + list(state["messages"]))
        spans = dict(state.get("spans", {}))
        spans["agent"] = spans.get("agent", 0.0) + (time.perf_counter() - started)
        return {
            "messages": [response],
            "rounds": state.get("rounds", 0) + 1,
            "spans": spans,
        }

    def short_circuit_reply(state: AgentState) -> dict[str, Any]:
        return {"messages": [AIMessage(content=state["short_circuit"])]}

    def vision_question(state: AgentState) -> dict[str, Any]:
        analysis = PlateAnalysis(**state["plate_analysis"])
        return {"messages": [AIMessage(content=analysis.clarifying_question() or "what was that?")]}

    # -- routing --------------------------------------------------------------
    def after_ingest(state: AgentState) -> str:
        if state.get("short_circuit"):
            return "short_circuit_reply"
        return "vision" if state.get("image_path") else "agent"

    def after_vision(state: AgentState) -> str:
        analysis = PlateAnalysis(**state["plate_analysis"])
        # Surface uncertainty instead of guessing: an unidentifiable food is
        # the one case where asking beats logging.
        return "vision_question" if analysis.needs_user_input() else "agent"

    def after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None) and state.get("rounds", 0) < MAX_TOOL_ROUNDS:
            return "tools"
        return END

    builder = StateGraph(AgentState)
    builder.add_node("ingest", ingest)
    builder.add_node("vision", vision)
    builder.add_node("vision_question", vision_question)
    builder.add_node("agent", agent)
    builder.add_node("tools", tool_node)
    builder.add_node("short_circuit_reply", short_circuit_reply)

    builder.add_edge(START, "ingest")
    builder.add_conditional_edges(
        "ingest", after_ingest,
        {"vision": "vision", "agent": "agent", "short_circuit_reply": "short_circuit_reply"},
    )
    builder.add_conditional_edges(
        "vision", after_vision, {"agent": "agent", "vision_question": "vision_question"}
    )
    builder.add_conditional_edges("agent", after_agent, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")
    builder.add_edge("short_circuit_reply", END)
    builder.add_edge("vision_question", END)

    return builder.compile()


def _describe_plate(analysis: dict[str, Any]) -> str:
    """Render the vision handoff for the text model.

    The two models meet here, and they meet as a typed object rendered to text
    -- not by piping one model's prose into the other's prompt. Portion
    uncertainty is passed through explicitly so the reply can admit to it.
    """
    plate = PlateAnalysis(**analysis)
    if plate.failed:
        return f"[photo] could not be read: {plate.failure_reason}. Ask what they ate."

    lines = [
        f"- {i.qty:g} {i.unit} {i.name}"
        + (f" (portion uncertain, {i.portion_confidence:.0%} sure)" if i.portion_confidence < 0.5 else "")
        for i in plate.items
    ]
    body = "\n".join(lines)
    note = f"\nMeasured against: {plate.scale_reference}." if plate.scale_reference else ""
    hedge = (
        "\nPortions are estimates from a photo -- log them and say so plainly, "
        "e.g. 'rough guess from the picture'."
        if plate.unsized()
        else ""
    )
    return (
        "[photo analysed by the vision model -- this is ONE meal, log it in a "
        f"single log_meal call]\n{body}{note}{hedge}"
    )


def run_turn(
    conn: sqlite3.Connection,
    user_id: str,
    text: str,
    image_path: str | None = None,
    graph=None,
) -> dict[str, Any]:
    """One synchronous turn. Returns the reply plus per-stage timings."""
    graph = graph or build_graph(conn, user_id)
    started = time.perf_counter()

    # A photo with a caption is ONE message, not two. Keeping them in a single
    # HumanMessage is what stops the models producing two meals.
    content = text or ("[sent a photo]" if image_path else "")
    state: AgentState = {
        "messages": [HumanMessage(content=content)],
        "user_id": user_id,
        "image_path": image_path,
        "caption": text or None,
        "spans": {},
    }
    result = graph.invoke(state)
    elapsed = time.perf_counter() - started

    reply = ""
    for message in reversed(result["messages"]):
        if isinstance(message, AIMessage) and message.content:
            reply = str(message.content)
            break

    return {
        "reply": reply,
        "elapsed": elapsed,
        "spans": result.get("spans", {}),
        "used_fast_path": bool(result.get("short_circuit")),
        "plate_analysis": result.get("plate_analysis"),
        "tool_calls": [
            call["name"]
            for message in result["messages"]
            if isinstance(message, AIMessage)
            for call in (getattr(message, "tool_calls", None) or [])
        ],
        "messages": result["messages"],
    }
