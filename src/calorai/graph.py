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

from langchain_core.messages import (
    AIMessageChunk,
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
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

# Every token here rides on every call, and the free tier's binding limit is
# tokens per minute -- so this prompt is kept terse on purpose. It was roughly
# twice this length and behaved no better; the rules that survived are the ones
# that changed a measured outcome.
SYSTEM_PROMPT = """\
You are CalorAI. People text you what they ate, like texting a friend. Reply in \
one or two short lowercase sentences. No bullets, tables or emoji.

Write like a person, not a receipt. "logged 3 idlis and a katori of sambar, \
~284 cal -- you're at 1125 for the day" is right. "kcal 284. total today 1125" \
is wrong: no field:value pairs, no bare unit names, and say "cal" not "kcal". \
Round to whole numbers; nobody wants 38.4g.

Default to logging, not asking. Assume a normal home portion and say what you \
assumed. Ask only if you cannot tell what the food is, and then ask ONE \
question. Never ask about grams, oil or brand.

Report ONLY what the tool actually returned -- the foods in its result, and its
kcal number. Do not carry food or numbers over from earlier in the conversation.

Vague amounts are not a reason to ask: "grazed all afternoon" -> log 1 serving \
of "assorted snacks" and call it a rough guess.

Fractions are amounts of one item, not counts: "two thirds of the box" is \
qty 0.67, "half" 0.5, "a couple" 2.

"actually that was 3 not 2" -> correct_meal, never log_meal; logging again \
double counts. Then say what changed and the new day total, nothing else.

COPY THE NUMBER THE USER SAID into qty. "2 parathas" is qty 2, not qty 1. \
Counts belong to their own food: "2 parathas and chai" is qty 2 paratha AND \
qty 1 chai -- do not spread one number across every item, and do not drop it.

NOT EVERY MESSAGE IS A MEAL. Facts about the person -- "i'm vegetarian", \
"i'm aiming for 140g protein", "allergic to peanuts" -- are things to remember, \
NOT things to log. Call no tool, and reply like a friend would: "got it, i'll \
remember that". Never narrate your own plumbing -- "no log", "not logged", \
"no tool called" are things the user should never see. Logging "1 vegetarian \
meal" because someone told you their diet is badly wrong.

"same as yesterday" / "my usual" needs TWO calls: find_meals, then log_meal \
with exactly the items find_meals returned. Finding is not logging, and \
"exactly" means from the tool result -- never from what you remember and never \
from the usual, which is a different meal.

Never say "logged" unless log_meal returned ok.

[what I know about you] is already confirmed -- use it, never re-ask it.

Never do arithmetic. Totals come from get_daily_totals."""


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
    note_ref: dict[str, str] = {}
    tools = make_tools(conn, user_id, estimator=estimator, note_ref=note_ref)

    base = get_text_model(streaming)
    fallback = get_fallback_text_model()

    def _with_fallback(model, backup):
        # Groq's free tier limits tokens per minute, not just requests, and
        # this agent reaches it in about two turns. The fallback is a different
        # provider on purpose, so a 429 costs a model swap rather than a wait.
        # The backup must be bound the same way as the primary -- a fallback
        # without tool schemas cannot answer a call that needs a tool.
        return model.with_fallbacks([backup]) if backup is not None else model

    # Two bindings of the same model. `deciding` carries the tool schemas;
    # `phrasing` does not, and is used once a turn's work is already done.
    # The schemas are 616 of the ~870 fixed tokens per call, so dropping them
    # from the reply call cuts roughly a third of the tokens in a normal
    # logging turn -- which matters because tokens per minute, not latency,
    # is the binding constraint on the free tier.
    deciding = _with_fallback(
        base.bind_tools(tools), fallback.bind_tools(tools) if fallback else None
    )
    phrasing = _with_fallback(base, fallback)

    def _tool_failed(exc: Exception) -> str:
        """Turn a tool exception into something the agent can talk about.

        LangGraph's ToolNode re-raises by default, which on a messaging surface
        means a database hiccup reaches the user as a stack trace and the turn
        is lost. Returning a ToolMessage instead lets the agent apologise and
        keep the conversation alive, which is the correct failure mode when
        someone is mid-sentence about their lunch.
        """
        return (
            f"That tool failed: {type(exc).__name__}: {exc}. "
            "Tell the user briefly that it didn't save and ask them to try again."
        )

    tool_node = ToolNode(tools, handle_tool_errors=_tool_failed)

    def ingest(state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        uid = state["user_id"]
        last = state["messages"][-1] if state["messages"] else None
        text = str(last.content) if isinstance(last, HumanMessage) else ""

        # provenance for whatever gets logged this turn, without spending
        # schema tokens asking the model to repeat the message back to us
        note_ref["text"] = text[:200]

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

        # After a terminal write the turn's work is finished and all that is
        # left is wording, so the tool schemas are dead weight. find_meals and
        # get_daily_totals are NOT terminal -- "same as yesterday" still has a
        # log_meal to make -- so they keep the tools bound.
        done = _work_is_done(state["messages"])
        model = phrasing if done else deciding
        if done:
            # Say so explicitly. A model that has had tools all turn will try to
            # call one anyway, and Groq rejects a call to a tool that was not in
            # the request with a 400 -- observed as a hallucinated
            # 'ask_question' tool. Cheaper than re-sending 597 tokens of schema.
            preamble.append(
                SystemMessage(
                    content="The work is done and saved. Reply in plain words only. "
                    "Do not call any tool."
                )
            )

        try:
            response = model.invoke(preamble + list(state["messages"]))
        except Exception as exc:  # noqa: BLE001
            # Every provider is down or throttled. The user gets a sentence
            # rather than a stack trace -- but *which* sentence depends on
            # whether the write already went through, because this failure can
            # land on the reply-phrasing call after the food is safely logged.
            response = AIMessage(
                content=_degraded_reply(exc, saved=_work_is_done(state["messages"]))
            )
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


def _degraded_reply(exc: Exception, saved: bool = False) -> str:
    """What to say when no model is reachable.

    `saved` matters more than the error does. This failure can land on the
    *reply-phrasing* call, after the food is already committed -- observed live
    as "nothing was saved" while the day's total had just moved by 1620 kcal.
    Telling someone their meal was lost when it wasn't is worse than the rate
    limit itself: they will log it again, and now the total really is wrong.

    Beyond that, throttling and an outage get different advice, because a rate
    limit clears itself in under a minute and a missing key does not.
    """
    text = f"{type(exc).__name__} {exc}".lower()
    throttled = any(k in text for k in ("429", "rate", "quota", "resource_exhausted"))

    if saved:
        return (
            "saved that. i'm rate limited so i can't give you the breakdown right now"
            " -- ask me for your total in a moment."
            if throttled
            else "saved that, but something went wrong writing the reply. ask me for your total."
        )
    if throttled:
        return (
            "i'm being rate limited right now -- nothing was saved. "
            "give it a few seconds and send that again?"
        )
    if "401" in text or "api key" in text or "auth" in text:
        return "my connection to the model isn't set up right, so i couldn't log that."
    return "something went wrong on my side and i didn't save that -- try again?"


#: Tools after which nothing further can usefully happen this turn.
_TERMINAL_TOOLS = {"log_meal", "correct_meal", "delete_meal"}


def _work_is_done(messages: list[BaseMessage]) -> bool:
    """True when the last thing that happened was a successful terminal write.

    Used to decide whether the next model call still needs the tool schemas.
    A failed write is not terminal -- the agent may want to try something else.
    """
    if not messages or not isinstance(messages[-1], ToolMessage):
        return False
    last = messages[-1]
    if getattr(last, "name", None) not in _TERMINAL_TOOLS:
        return False
    return '"ok": true' in str(last.content).lower()


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


def _initial_state(user_id: str, text: str, image_path: str | None) -> AgentState:
    # A photo with a caption is ONE message, not two. Keeping them in a single
    # HumanMessage is what stops the two models producing two meals.
    return {
        "messages": [HumanMessage(content=text or ("[sent a photo]" if image_path else ""))],
        "user_id": user_id,
        "image_path": image_path,
        "caption": text or None,
        "spans": {},
    }


def stream_turn(
    conn: sqlite3.Connection,
    user_id: str,
    text: str,
    image_path: str | None = None,
    graph=None,
):
    """Same turn as `run_turn`, yielding the reply as it is generated.

    Yields ("token", str) as words arrive, then ("done", result) with the same
    shape run_turn returns.

    Worth streaming because time-to-first-token, not total time, is what a
    person waiting on a message actually feels: a 900ms reply that starts
    appearing at 250ms reads as fast, and the identical reply delivered in one
    lump at 900ms reads as a pause. The tool-deciding call produces no visible
    text -- only the final phrasing call does -- so what streams is exactly the
    sentence the user reads.
    """
    graph = graph or build_graph(conn, user_id, streaming=True)
    started = time.perf_counter()
    ttft: float | None = None
    final: dict[str, Any] | None = None

    for mode, payload in graph.stream(
        _initial_state(user_id, text, image_path), stream_mode=["messages", "values"]
    ):
        if mode == "values":
            final = payload
            continue
        chunk, meta = payload if isinstance(payload, tuple) else (payload, {})

        # Stream ONLY the agent node's own words. "messages" mode is
        # indiscriminate: it also carries ToolMessages, whose content is the raw
        # JSON a tool returned, and the vision model's structured output, which
        # is a wall of PlateAnalysis JSON. Both were observed dumped on screen
        # ahead of the sentence the user is meant to read. Filtering by node is
        # what makes this precise -- filtering by message type alone let the
        # vision JSON through, because it arrives as assistant content too.
        if (meta or {}).get("langgraph_node") != "agent":
            continue
        if isinstance(chunk, ToolMessage) or not isinstance(chunk, (AIMessage, AIMessageChunk)):
            continue

        piece = getattr(chunk, "content", "")
        if isinstance(piece, list):  # some providers emit content blocks
            piece = "".join(b.get("text", "") for b in piece if isinstance(b, dict))
        if not piece:
            continue
        if ttft is None:
            ttft = time.perf_counter() - started
        yield "token", piece

    result = _collect(final or {}, time.perf_counter() - started)
    result["ttft"] = ttft
    yield "done", result


def _collect(state: dict[str, Any], elapsed: float) -> dict[str, Any]:
    messages = state.get("messages", [])
    reply = ""
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.content:
            content = message.content
            if isinstance(content, list):
                content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
            reply = str(content)
            break
    return {
        "reply": reply,
        "elapsed": elapsed,
        "spans": state.get("spans", {}),
        "used_fast_path": bool(state.get("short_circuit")),
        "plate_analysis": state.get("plate_analysis"),
        "tool_calls": [
            call["name"]
            for message in messages
            if isinstance(message, AIMessage)
            for call in (getattr(message, "tool_calls", None) or [])
        ],
        "messages": messages,
    }


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
    result = graph.invoke(_initial_state(user_id, text, image_path))
    return _collect(result, time.perf_counter() - started)
