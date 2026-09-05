"""Memory tests: selectivity, supersession, and alias learning."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from calorai import repository as repo  # noqa: E402
from calorai.db import connect, reset_connections  # noqa: E402
from calorai.memory import extractor, render, store  # noqa: E402
from calorai.schemas import FoodItem  # noqa: E402

USER = "u_mem"


@pytest.fixture()
def conn():
    reset_connections()
    c = connect(":memory:")
    yield c
    reset_connections()


# ---------------------------------------------------------------------------
# selectivity -- the thing the brief warns about
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "message",
    [
        "had 2 parathas and chai for breakfast",
        "leftover biryani, maybe two thirds of the box",
        "actually that was 3 rotis not 2",
        "how am I doing on calories?",
        "ok thanks",
    ],
)
def test_ordinary_food_talk_stores_nothing(conn, message):
    """What someone ate is an event, not a fact about them. If these wrote
    rows, memory would just be conversation history with extra steps."""
    written = extractor.extract_and_store(conn, USER, message, use_model=False)
    assert written == []
    assert store.get_facts(conn, USER) == []


@pytest.mark.parametrize(
    "message,key,value",
    [
        ("i'm vegetarian btw", "diet", "vegetarian"),
        ("im vegan now", "diet", "vegan"),
        ("i'm targeting 140g of protein", "protein_target_g", "140"),
        ("allergic to peanuts", "allergy", "peanuts"),
        ("i don't eat beef", "avoids", "beef"),
    ],
)
def test_durable_statements_are_stored(conn, message, key, value):
    extractor.extract_and_store(conn, USER, message, use_model=False)
    facts = {f["key"]: f["value"] for f in store.get_facts(conn, USER)}
    assert facts.get(key) == value


def test_the_gate_runs_before_any_model_call(conn):
    """A message with no fact signal must not reach the model at all -- that
    is where the cost saving comes from, not from the model saying 'none'."""
    called = False

    def explode(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("model must not be consulted for plain food talk")

    original = extractor._model_extract
    extractor._model_extract = explode
    try:
        assert extractor.extract_and_store(conn, USER, "had 2 rotis") == []
    finally:
        extractor._model_extract = original
    assert not called


# ---------------------------------------------------------------------------
# contradiction and forgetting
# ---------------------------------------------------------------------------
def test_newer_fact_supersedes_older_and_keeps_the_trail(conn):
    store.put_fact(conn, USER, "diet", "vegetarian")
    store.put_fact(conn, USER, "diet", "vegan")

    live = store.get_facts(conn, USER)
    assert len(live) == 1, "only one live value per key"
    assert live[0]["value"] == "vegan"

    total = conn.execute(
        "SELECT COUNT(*) n FROM profile_facts WHERE user_id=?", (USER,)
    ).fetchone()["n"]
    assert total == 2, "the superseded row is retained for audit, not deleted"


def test_restating_the_same_fact_is_a_no_op(conn):
    store.put_fact(conn, USER, "diet", "vegetarian")
    result = store.put_fact(conn, USER, "diet", "Vegetarian")
    assert result["changed"] is False
    assert (
        conn.execute(
            "SELECT COUNT(*) n FROM profile_facts WHERE user_id=?", (USER,)
        ).fetchone()["n"]
        == 1
    )


def test_facts_are_isolated_per_user(conn):
    store.put_fact(conn, USER, "diet", "vegetarian")
    store.put_fact(conn, "someone_else", "diet", "carnivore")
    assert [f["value"] for f in store.get_facts(conn, USER)] == ["vegetarian"]


# ---------------------------------------------------------------------------
# aliases -- "my usual"
# ---------------------------------------------------------------------------
def test_usual_is_inferred_only_after_it_becomes_a_habit(conn):
    breakfast = [FoodItem(name="paratha", qty=2, unit="piece"), FoodItem(name="chai", qty=1, unit="cup")]

    repo.log_meal(conn, USER, breakfast, slot="breakfast")
    repo.log_meal(conn, USER, breakfast, slot="breakfast")
    assert store.infer_usual(conn, USER) is None, "twice is a coincidence"

    repo.log_meal(conn, USER, breakfast, slot="breakfast")
    usual = store.infer_usual(conn, USER)
    assert usual is not None, "three times is a habit"
    assert {i["name"] for i in usual["items"]} == {"paratha", "chai"}


def test_explicit_definition_beats_inference(conn):
    store.put_alias(conn, USER, "my usual", [{"name": "oats", "qty": 1, "unit": "katori"}], source="explicit")
    breakfast = [FoodItem(name="paratha", qty=2, unit="piece")]
    for _ in range(4):
        repo.log_meal(conn, USER, breakfast, slot="breakfast")

    extractor.maybe_learn_alias(conn, USER, "")
    resolved = store.resolve_alias(conn, USER, "my usual")
    assert resolved["items"][0]["name"] == "oats", "a stated usual must not be overwritten"


def test_alias_resolves_before_the_model_sees_it(conn):
    store.put_alias(
        conn, USER, "my usual",
        [{"name": "paratha", "qty": 2, "unit": "piece"}, {"name": "chai", "qty": 1, "unit": "cup"}],
    )
    resolved = store.resolve_alias(conn, USER, "my usual please")
    assert resolved is not None
    assert {i["name"] for i in resolved["items"]} == {"paratha", "chai"}


def test_alias_definition_is_detected_from_prose(conn):
    assert store.detect_alias_definition("my usual is 2 parathas and chai") == "2 parathas and chai"
    assert store.detect_alias_definition("had my usual") is None


# ---------------------------------------------------------------------------
# rendering into context
# ---------------------------------------------------------------------------
def test_memory_block_is_compact_and_readable(conn):
    store.put_fact(conn, USER, "diet", "vegetarian")
    store.put_fact(conn, USER, "protein_target_g", "140")
    store.put_alias(conn, USER, "my usual", [{"name": "paratha", "qty": 2, "unit": "piece"}])

    block = render.render_memory_block(conn, USER)
    assert "vegetarian" in block
    assert "protein target 140g" in block
    assert '"my usual"' in block
    assert len(block) <= render.MAX_CHARS + 40


def test_empty_memory_renders_nothing(conn):
    assert render.render_memory_block(conn, USER) == ""


def test_vision_priors_carry_only_plate_relevant_facts(conn):
    """The vision prompt gets diet, not macro targets -- a protein goal cannot
    change what is in the photo."""
    store.put_fact(conn, USER, "diet", "vegetarian")
    store.put_fact(conn, USER, "protein_target_g", "140")

    priors = render.render_vision_priors(conn, USER)
    assert "vegetarian" in priors
    assert "140" not in priors


def test_forgetting_everything_keeps_the_meals(conn):
    """The two stores are separate, and clearing one must not touch the other.

    This is the assertion behind the sidebar's two buttons. Meals are what
    happened; facts and aliases are what the agent remembers. Wiping memory
    while the day's log survives is the shortest proof that memory here is not
    the conversation replayed back.
    """
    store.put_fact(conn, USER, "diet", "vegetarian")
    store.put_fact(conn, USER, "protein_target_g", "140")
    store.put_alias(conn, USER, "my usual", [{"name": "oats", "qty": 1, "unit": "katori"}])
    repo.log_meal(conn, USER, [FoodItem(name="roti", qty=2, unit="piece")], slot="dinner")

    before = repo.daily_totals(conn, USER)["kcal"]
    assert before > 0
    assert render.render_memory_block(conn, USER)

    dropped = store.forget_everything(conn, USER)

    assert dropped == {"facts": 2, "aliases": 1}
    assert render.render_memory_block(conn, USER) == ""
    assert store.resolve_alias(conn, USER, "my usual") is None
    assert repo.daily_totals(conn, USER)["kcal"] == before


def test_forgetting_is_scoped_to_one_user(conn):
    store.put_fact(conn, USER, "diet", "vegetarian")
    store.put_fact(conn, "someone_else", "diet", "vegan")

    store.forget_everything(conn, USER)

    assert render.render_memory_block(conn, USER) == ""
    assert "vegan" in render.render_memory_block(conn, "someone_else")


def test_forgetting_nothing_is_not_an_error(conn):
    assert store.forget_everything(conn, USER) == {"facts": 0, "aliases": 0}


def test_naming_a_slot_that_holds_nothing_still_remembers_the_meal(conn):
    """"remember this as my usual dinner" said over food filed as a snack.

    The named slot wins over the inferred one, which is right -- the person is
    telling you what to call it. But when that slot holds no rows, searching it
    and giving up stored nothing at all, silently, and the failure depended on
    what time of day the message was sent. They are plainly pointing at the
    meal they just logged.
    """
    repo.log_meal(conn, USER, [FoodItem(name="roti", qty=2, unit="piece")], slot="snack")

    learned = store.learn_alias_from_recent_meal(
        conn, USER, "my usual", text="remember this as my usual dinner"
    )

    assert learned is not None, "named a slot with no rows and lost the meal"
    assert [i["name"] for i in learned["items"]] == ["roti"]
    assert learned["slot"] == "dinner", "should file it under the name they gave it"
    assert store.resolve_alias(conn, USER, "my usual", slot="dinner") is not None
