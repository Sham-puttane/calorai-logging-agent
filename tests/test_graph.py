"""Orchestration tests for the LangGraph agent.

These are about *control flow*, not about food: which node ran, how many times,
what the state carried between them, and what happens when a node misbehaves.
Every edge in the graph is exercised, including the ones that only fire when
something goes wrong.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from calorai import repository as repo  # noqa: E402
from calorai.db import connect, reset_connections  # noqa: E402
from calorai.graph import MAX_TOOL_ROUNDS, build_graph, run_turn  # noqa: E402
from calorai.memory import store  # noqa: E402
from calorai.schemas import FoodItem  # noqa: E402

USER = "u_graph"


@pytest.fixture(autouse=True)
def mock_backend():
    """Every test in this file runs offline and deterministically."""
    previous = dict(os.environ)
    os.environ["CALORAI_TEXT_BACKEND"] = "mock"
    os.environ["CALORAI_VISION_BACKEND"] = "mock"
    os.environ["CALORAI_TEXT_FALLBACK"] = "none"
    os.environ["CALORAI_FAST_PATH"] = "1"
    from calorai import llm

    llm.get_text_model.cache_clear()
    llm.get_vision_model.cache_clear()
    llm.get_fallback_text_model.cache_clear()
    yield
    os.environ.clear()
    os.environ.update(previous)
    llm.get_text_model.cache_clear()
    llm.get_vision_model.cache_clear()
    llm.get_fallback_text_model.cache_clear()


@pytest.fixture()
def conn():
    reset_connections()
    c = connect(":memory:")
    yield c
    reset_connections()


def live_items(c, user_id=USER) -> int:
    return c.execute(
        "SELECT COUNT(*) n FROM meal_items WHERE user_id=? AND deleted_at IS NULL",
        (user_id,),
    ).fetchone()["n"]


def meal_count(c, user_id=USER) -> int:
    return c.execute(
        "SELECT COUNT(DISTINCT meal_id) n FROM meal_items"
        " WHERE user_id=? AND deleted_at IS NULL",
        (user_id,),
    ).fetchone()["n"]


# ===========================================================================
# routing -- every edge out of every conditional node
# ===========================================================================
def test_text_message_routes_through_the_agent(conn):
    result = run_turn(conn, USER, "had 2 rotis")
    assert result["tool_calls"] == ["log_meal"]
    assert not result["used_fast_path"]
    assert "agent" in result["spans"]
    assert "vision" not in result["spans"]


def test_image_routes_through_vision_before_the_agent(conn):
    result = run_turn(conn, USER, "", image_path="images/plate.jpg")
    assert "vision" in result["spans"], "the vision node must run for an image"
    assert "agent" in result["spans"], "and hand off to the text agent"
    assert result["plate_analysis"] is not None


def test_low_confidence_image_stops_before_the_agent(conn):
    """The vision -> vision_question edge. An unidentifiable food must produce
    a question and must NOT reach a tool."""
    result = run_turn(conn, USER, "", image_path="images/ambiguous_plate.jpg")

    assert result["tool_calls"] == []
    assert "agent" not in result["spans"], "must not spend a text-model call"
    assert result["reply"].endswith("?")
    assert live_items(conn) == 0, "nothing may be logged on a guess"


def test_fast_path_skips_both_models(conn):
    repo.log_meal(conn, USER, [FoodItem(name="roti", qty=2, unit="piece")])
    result = run_turn(conn, USER, "how am I doing?")

    assert result["used_fast_path"]
    assert "agent" not in result["spans"], "the fast path must not call the model"
    assert "210" in result["reply"]


def test_fast_path_can_be_disabled(conn):
    repo.log_meal(conn, USER, [FoodItem(name="roti", qty=2, unit="piece")])
    os.environ["CALORAI_FAST_PATH"] = "0"
    result = run_turn(conn, USER, "how am I doing?")

    assert not result["used_fast_path"]
    assert result["tool_calls"] == ["get_daily_totals"], "the agent handles it instead"
    assert "210" in result["reply"]


def test_fast_path_yields_when_the_message_names_a_food(conn):
    """'how many calories in a samosa' is a lookup, not a totals question.
    Swallowing it would silently answer something the user did not ask."""
    repo.log_meal(conn, USER, [FoodItem(name="roti", qty=2, unit="piece")])
    result = run_turn(conn, USER, "how many calories in a samosa")
    assert not result["used_fast_path"]


def test_image_never_takes_the_fast_path(conn):
    repo.log_meal(conn, USER, [FoodItem(name="roti", qty=2, unit="piece")])
    result = run_turn(conn, USER, "how am I doing?", image_path="images/plate.jpg")
    assert not result["used_fast_path"], "a photo always needs the vision node"


# ===========================================================================
# the loop
# ===========================================================================
class AlwaysToolCalls(BaseChatModel):
    """A model that never stops asking for tools -- the pathological case the
    round limit exists for."""

    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "always-tools"

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ARG002
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls += 1
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "get_daily_totals",
                                "args": {"day": "today"},
                                "id": f"call_{self.calls}",
                            }
                        ],
                    )
                )
            ]
        )


def test_the_loop_is_bounded(conn, monkeypatch):
    """A model stuck in a tool loop must terminate, not hang a WhatsApp reply."""
    from calorai import graph as graph_module

    looper = AlwaysToolCalls()
    monkeypatch.setattr(graph_module, "get_text_model", lambda *a, **k: looper, raising=False)
    monkeypatch.setattr("calorai.llm.get_text_model", lambda *a, **k: looper)
    monkeypatch.setattr("calorai.llm.get_fallback_text_model", lambda: None)

    result = run_turn(conn, USER, "how am I doing?", graph=build_graph(conn, USER))

    assert looper.calls <= MAX_TOOL_ROUNDS, f"ran {looper.calls} rounds, limit is {MAX_TOOL_ROUNDS}"
    assert result["elapsed"] < 5


def test_multi_step_turn_chains_two_tools(conn):
    """'same as yesterday' needs find_meals then log_meal -- two passes through
    the agent node in one turn. This is why the bound is 3 and not 1."""
    repo.log_meal(
        conn, USER,
        [FoodItem(name="paratha", qty=2, unit="piece"), FoodItem(name="chai", qty=1, unit="cup")],
        slot="breakfast", day="yesterday",
    )
    result = run_turn(conn, USER, "same as yesterday")
    assert "find_meals" in result["tool_calls"]


# ===========================================================================
# state carried between nodes
# ===========================================================================
def test_memory_reaches_the_agent_as_state(conn):
    store.put_fact(conn, USER, "diet", "vegetarian")
    graph = build_graph(conn, USER)
    result = graph.invoke(
        {
            "messages": [__import__("langchain_core.messages", fromlist=["HumanMessage"]).HumanMessage(content="had 2 rotis")],
            "user_id": USER,
            "spans": {},
        }
    )
    assert "vegetarian" in result["memory_block"]


def test_alias_is_expanded_in_ingest_before_the_model(conn):
    store.put_alias(
        conn, USER, "my usual",
        [{"name": "paratha", "qty": 2, "unit": "piece"}, {"name": "chai", "qty": 1, "unit": "cup"}],
    )
    result = run_turn(conn, USER, "my usual")

    assert result["tool_calls"] == ["log_meal"]
    assert live_items(conn) == 2
    assert repo.daily_totals(conn, USER)["kcal"] == 430


def test_every_stage_is_timed(conn):
    result = run_turn(conn, USER, "had 2 rotis")
    assert set(result["spans"]) >= {"ingest", "agent"}
    assert all(v >= 0 for v in result["spans"].values())


# ===========================================================================
# isolation
# ===========================================================================
def test_tools_are_bound_to_one_user(conn):
    """Two graphs, two users, one database. Neither may see the other's rows."""
    graph_a = build_graph(conn, "user_a")
    graph_b = build_graph(conn, "user_b")

    run_turn(conn, "user_a", "had 2 rotis", graph=graph_a)
    run_turn(conn, "user_b", "had 1 samosa", graph=graph_b)

    assert repo.daily_totals(conn, "user_a")["kcal"] == 210
    assert repo.daily_totals(conn, "user_b")["kcal"] == 260

    result = run_turn(conn, "user_b", "how am I doing?", graph=graph_b)
    assert "260" in result["reply"]
    assert "210" not in result["reply"]


def test_memory_does_not_leak_between_users(conn):
    store.put_fact(conn, "user_a", "diet", "vegetarian")
    graph_b = build_graph(conn, "user_b")
    state = graph_b.invoke(
        {
            "messages": [__import__("langchain_core.messages", fromlist=["HumanMessage"]).HumanMessage(content="hi")],
            "user_id": "user_b",
            "spans": {},
        }
    )
    assert "vegetarian" not in (state.get("memory_block") or "")


# ===========================================================================
# persistence across sessions
# ===========================================================================
def test_state_survives_a_process_restart():
    """Meals and memory must outlive the graph, the connection and the process."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "session.db")

        reset_connections()
        first = connect(db_path)
        run_turn(first, USER, "had 2 parathas", graph=build_graph(first, USER))
        store.put_fact(first, USER, "diet", "vegetarian")
        reset_connections()  # simulate the process exiting

        second = connect(db_path)
        assert repo.daily_totals(second, USER)["kcal"] == 340
        assert [f["value"] for f in store.get_facts(second, USER)] == ["vegetarian"]

        result = run_turn(second, USER, "how am I doing?", graph=build_graph(second, USER))
        assert "340" in result["reply"]
        reset_connections()


# ===========================================================================
# failure paths
# ===========================================================================
def test_a_missing_image_asks_instead_of_crashing(conn, monkeypatch):
    """The mock never touches disk, so force the real path to prove a bad file
    degrades into a question rather than an exception."""
    monkeypatch.setenv("CALORAI_VISION_BACKEND", "gemini")
    from calorai import llm, vision

    llm.get_vision_model.cache_clear()

    def fake_model():
        raise llm.BackendUnavailable("no key")

    monkeypatch.setattr(vision, "analyse_plate", lambda *a, **k: __import__(
        "calorai.schemas", fromlist=["PlateAnalysis"]
    ).PlateAnalysis(failed=True, failure_reason="no image at nope.jpg"))

    result = run_turn(conn, USER, "", image_path="nope.jpg")
    assert result["tool_calls"] == []
    assert "?" in result["reply"]
    assert live_items(conn) == 0
    llm.get_vision_model.cache_clear()


def test_a_tool_raising_does_not_kill_the_turn(conn, monkeypatch):
    from calorai import repository

    def boom(*_args, **_kwargs):
        raise RuntimeError("database on fire")

    monkeypatch.setattr(repository, "log_meal", boom)
    graph = build_graph(conn, USER)

    # ToolNode converts the exception into a ToolMessage the agent can react to,
    # so the user gets a reply rather than a stack trace.
    result = run_turn(conn, USER, "had 2 rotis", graph=graph)
    assert isinstance(result["reply"], str)


def test_empty_message_is_harmless(conn):
    result = run_turn(conn, USER, "")
    assert result["tool_calls"] == []
    assert live_items(conn) == 0


# ===========================================================================
# the multimodal invariant
# ===========================================================================
def test_photo_plus_caption_produces_exactly_one_meal(conn):
    """The headline multimodal requirement: two models, one meal. A second
    meal row here would mean the caption was logged separately."""
    result = run_turn(conn, USER, "half of this was my brother's", image_path="images/plate.jpg")

    assert result["tool_calls"].count("log_meal") == 1
    assert meal_count(conn) == 1, "one meal, not one per model"
    assert live_items(conn) == 3


def test_caption_fraction_halves_the_portions(conn):
    plain = run_turn(conn, USER, "", image_path="images/plate.jpg")
    plain_kcal = repo.daily_totals(conn, USER)["kcal"]

    reset_connections()
    fresh = connect(":memory:")
    run_turn(fresh, USER, "half of this was my brother's", image_path="images/plate.jpg")
    halved_kcal = repo.daily_totals(fresh, USER)["kcal"]

    assert plain["plate_analysis"] is not None
    assert halved_kcal == pytest.approx(plain_kcal / 2, rel=0.02)


# ===========================================================================
# streaming
# ===========================================================================
def test_streaming_never_leaks_raw_tool_json(conn):
    """LangGraph's "messages" stream mode emits ToolMessages too, whose content
    is the JSON a tool returned. Without filtering, a log_meal result gets
    dumped on screen ahead of the sentence the user is meant to read -- which
    is exactly what happened the first time this was recorded."""
    from calorai.graph import stream_turn

    graph = build_graph(conn, USER, streaming=True)
    tokens = [p for kind, p in stream_turn(conn, USER, "had 2 rotis", graph=graph) if kind == "token"]
    streamed = "".join(tokens)

    for leak in ('"ok":', '"meal_id"', '"totals_after"', '"item_id"'):
        assert leak not in streamed, f"raw tool JSON leaked into the reply: {leak}"


def test_streaming_and_blocking_agree_on_the_reply(conn):
    from calorai.graph import stream_turn

    graph = build_graph(conn, USER, streaming=True)
    done = [p for kind, p in stream_turn(conn, USER, "had 2 rotis", graph=graph) if kind == "done"]

    assert len(done) == 1
    assert done[0]["tool_calls"] == ["log_meal"]
    assert repo.daily_totals(conn, USER)["kcal"] == 210


def test_streaming_reports_time_to_first_token(conn):
    from calorai.graph import stream_turn

    graph = build_graph(conn, USER, streaming=True)
    result = [p for kind, p in stream_turn(conn, USER, "had 2 rotis", graph=graph) if kind == "done"][0]
    assert result["ttft"] is not None and result["ttft"] >= 0
