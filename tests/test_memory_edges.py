"""Edge cases for the memory layer.

The happy paths are covered in test_memory.py. This file is the awkward stuff:
people who do not eat three tidy meals a day, phrasings that point at a meal
instead of naming one, and the boundaries where "a habit" starts and stops.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from calorai import repository as repo  # noqa: E402
from calorai.db import connect, reset_connections  # noqa: E402
from calorai.memory import extractor, store  # noqa: E402
from calorai.schemas import FoodItem  # noqa: E402

USER = "u_edge"


@pytest.fixture()
def conn():
    reset_connections()
    c = connect(":memory:")
    yield c
    reset_connections()


def items(name_qty: list[tuple[str, float, str]]) -> list[FoodItem]:
    return [FoodItem(name=n, qty=q, unit=u) for n, q, u in name_qty]


# ===========================================================================
# "remember this as my usual" -- points at a meal instead of naming one
# ===========================================================================
@pytest.mark.parametrize(
    "phrasing",
    [
        "remember this dinner thats my usual",
        "remember this as my usual",
        "that's my usual",
        "thats my usual",
        "make this my usual",
        "save this as my usual",
    ],
)
def test_remembering_the_last_meal_by_reference(conn, phrasing):
    """None of these name a food. The meal is the one just logged, so the alias
    has to be built from the log rather than parsed from the sentence."""
    repo.log_meal(conn, USER, items([("naan", 1, "piece"), ("dal", 1, "katori")]), slot="dinner")

    assert extractor.maybe_learn_alias(conn, USER, phrasing) is True

    resolved = store.resolve_alias(conn, USER, "my usual", slot="dinner")
    assert {i["name"] for i in resolved["items"]} == {"naan", "dal"}


def test_using_the_alias_does_not_redefine_it(conn):
    """"my usual" on its own is someone *using* the shorthand. If that also
    rewrote it, ordering the usual after any other meal would silently replace
    it with that meal."""
    store.put_alias(conn, USER, "my usual", [{"name": "oats", "qty": 1, "unit": "katori"}])
    repo.log_meal(conn, USER, items([("biryani", 2, "katori")]))

    assert store.means_remember_recent("my usual") is False
    extractor.maybe_learn_alias(conn, USER, "my usual")

    resolved = store.resolve_alias(conn, USER, "my usual")
    assert {i["name"] for i in resolved["items"]} == {"oats"}, "still oats, not biryani"


def test_remembering_with_nothing_logged_is_harmless(conn):
    assert extractor.maybe_learn_alias(conn, USER, "remember this as my usual") is False
    assert store.get_aliases(conn, USER) == []


def test_remembering_a_deleted_meal_does_not_resurrect_it(conn):
    repo.log_meal(conn, USER, items([("samosa", 2, "piece")]))
    repo.delete_meal(conn, USER)

    assert extractor.maybe_learn_alias(conn, USER, "remember this as my usual") is False


def test_a_named_definition_still_wins_over_reference(conn):
    """"my usual is X" names its own food and must not be hijacked by the
    reference path."""
    repo.log_meal(conn, USER, items([("biryani", 2, "katori")]))
    extractor.maybe_learn_alias(conn, USER, "my usual is 2 parathas and chai")

    resolved = store.resolve_alias(conn, USER, "my usual")
    assert {i["name"] for i in resolved["items"]} == {"paratha", "chai"}


# ===========================================================================
# meal slots -- the three-meals-a-day assumption, and people who break it
# ===========================================================================
def test_the_same_phrase_holds_a_different_meal_per_slot(conn):
    store.put_alias(conn, USER, "my usual", [{"name": "oats", "qty": 1, "unit": "katori"}], slot="breakfast")
    store.put_alias(conn, USER, "my usual", [{"name": "dal", "qty": 1, "unit": "katori"}], slot="dinner")

    morning = store.resolve_alias(conn, USER, "my usual", slot="breakfast")
    evening = store.resolve_alias(conn, USER, "my usual", slot="dinner")

    assert {i["name"] for i in morning["items"]} == {"oats"}
    assert {i["name"] for i in evening["items"]} == {"dal"}


def test_an_unscoped_usual_covers_slots_with_no_entry(conn):
    """A grazer will not have a usual for every slot. An unscoped entry is the
    fallback, so lunch still resolves rather than coming back empty."""
    store.put_alias(conn, USER, "my usual", [{"name": "oats", "qty": 1, "unit": "katori"}], slot="breakfast")
    store.put_alias(conn, USER, "my usual", [{"name": "banana", "qty": 1, "unit": "piece"}])

    assert {i["name"] for i in store.resolve_alias(conn, USER, "my usual", slot="lunch")["items"]} == {"banana"}
    assert {i["name"] for i in store.resolve_alias(conn, USER, "my usual", slot="breakfast")["items"]} == {"oats"}


def test_a_slot_with_no_entry_and_no_fallback_still_answers(conn):
    """Better a wrong-slot usual than "I don't know" -- the user corrects it in
    three words, and an empty answer helps nobody."""
    store.put_alias(conn, USER, "my usual", [{"name": "oats", "qty": 1, "unit": "katori"}], slot="breakfast")
    assert store.resolve_alias(conn, USER, "my usual", slot="dinner") is not None


def test_redefining_one_slot_leaves_the_others_alone(conn):
    store.put_alias(conn, USER, "my usual", [{"name": "oats", "qty": 1, "unit": "katori"}], slot="breakfast")
    store.put_alias(conn, USER, "my usual", [{"name": "dal", "qty": 1, "unit": "katori"}], slot="dinner")

    store.put_alias(conn, USER, "my usual", [{"name": "poha", "qty": 1, "unit": "katori"}], slot="breakfast")

    assert conn.execute("SELECT COUNT(*) n FROM aliases WHERE user_id=?", (USER,)).fetchone()["n"] == 2
    assert {i["name"] for i in store.resolve_alias(conn, USER, "my usual", slot="breakfast")["items"]} == {"poha"}
    assert {i["name"] for i in store.resolve_alias(conn, USER, "my usual", slot="dinner")["items"]} == {"dal"}


def test_remembering_scopes_to_the_slot_of_the_meal(conn):
    repo.log_meal(conn, USER, items([("oats", 1, "katori")]), slot="breakfast")
    extractor.maybe_learn_alias(conn, USER, "remember this as my usual")

    stored = store.get_aliases(conn, USER)
    assert stored[0]["slot"] == "breakfast", "a breakfast habit is not a dinner one"


# ===========================================================================
# habits -- what counts as one, and what does not
# ===========================================================================
def test_a_grazer_with_no_repeated_meal_has_no_usual(conn):
    """Someone who snacks on something different every time has no habit to
    infer, and the agent must not invent one from the most recent thing."""
    for food in ("samosa", "banana", "chips", "biscuit", "almonds", "pakora"):
        repo.log_meal(conn, USER, items([(food, 1, "piece")]), slot="snack")

    assert store.infer_usual(conn, USER) is None


def test_a_grazer_who_repeats_one_snack_does_get_a_usual(conn):
    for _ in range(3):
        repo.log_meal(conn, USER, items([("almonds", 1, "serving")]), slot="snack")
    for food in ("samosa", "banana"):
        repo.log_meal(conn, USER, items([(food, 1, "piece")]), slot="snack")

    usual = store.infer_usual(conn, USER, slot="snack")
    assert usual is not None
    assert {i["name"] for i in usual["items"]} == {"almonds"}


def test_habits_are_counted_per_slot(conn):
    """Three identical breakfasts is a breakfast habit, not a dinner one."""
    for _ in range(3):
        repo.log_meal(conn, USER, items([("oats", 1, "katori")]), slot="breakfast")

    assert store.infer_usual(conn, USER, slot="breakfast") is not None
    assert store.infer_usual(conn, USER, slot="dinner") is None


def test_two_repeats_is_a_coincidence_three_is_a_habit(conn):
    meal = items([("paratha", 2, "piece"), ("chai", 1, "cup")])
    repo.log_meal(conn, USER, meal, slot="breakfast")
    repo.log_meal(conn, USER, meal, slot="breakfast")
    assert store.infer_usual(conn, USER) is None

    repo.log_meal(conn, USER, meal, slot="breakfast")
    assert store.infer_usual(conn, USER) is not None


def test_a_different_quantity_is_a_different_meal(conn):
    """2 parathas three times is a habit. 2, then 3, then 4 is not."""
    for qty in (2, 3, 4):
        repo.log_meal(conn, USER, items([("paratha", qty, "piece")]), slot="breakfast")

    assert store.infer_usual(conn, USER) is None


def test_habits_older_than_the_window_stop_counting(conn):
    """A habit is recent behaviour. What someone ate every day last season is
    not their usual today -- and the scan must not grow with the account."""
    old = (date.today() - timedelta(days=store.HABIT_WINDOW_DAYS + 5)).isoformat()
    for _ in range(4):
        repo.log_meal(conn, USER, items([("oats", 1, "katori")]), slot="breakfast", day=old)

    assert store.infer_usual(conn, USER) is None


def test_an_explicit_usual_is_never_overwritten_by_a_habit(conn):
    store.put_alias(conn, USER, "my usual", [{"name": "oats", "qty": 1, "unit": "katori"}], source="explicit")
    for _ in range(5):
        repo.log_meal(conn, USER, items([("paratha", 2, "piece")]), slot="breakfast")

    extractor.maybe_learn_alias(conn, USER, "")

    assert {i["name"] for i in store.resolve_alias(conn, USER, "my usual")["items"]} == {"oats"}


# ===========================================================================
# isolation and decay
# ===========================================================================
def test_aliases_do_not_leak_between_users(conn):
    store.put_alias(conn, USER, "my usual", [{"name": "oats", "qty": 1, "unit": "katori"}])
    assert store.resolve_alias(conn, "someone_else", "my usual") is None


def test_one_users_meals_do_not_become_anothers_habit(conn):
    for _ in range(4):
        repo.log_meal(conn, "heavy_user", items([("biryani", 2, "katori")]), slot="dinner")
    assert store.infer_usual(conn, USER) is None


def test_an_unused_alias_decays(conn):
    store.put_alias(conn, USER, "my usual", [{"name": "oats", "qty": 1, "unit": "katori"}])
    stale = (date.today() - timedelta(days=store.ALIAS_DECAY_DAYS + 1)).isoformat()
    conn.execute("UPDATE aliases SET last_used_at = ? WHERE user_id = ?", (stale, USER))
    conn.commit()

    assert store.get_aliases(conn, USER) == []


def test_using_an_alias_keeps_it_alive(conn):
    store.put_alias(conn, USER, "my usual", [{"name": "oats", "qty": 1, "unit": "katori"}])
    before = conn.execute("SELECT hits FROM aliases WHERE user_id=?", (USER,)).fetchone()["hits"]

    store.resolve_alias(conn, USER, "my usual")

    after = conn.execute("SELECT hits FROM aliases WHERE user_id=?", (USER,)).fetchone()["hits"]
    assert after == before + 1


# ===========================================================================
# facts alongside aliases
# ===========================================================================
def test_a_message_can_carry_a_fact_and_an_alias_at_once(conn):
    repo.log_meal(conn, USER, items([("dal", 1, "katori")]), slot="dinner")

    extractor.extract_and_store(conn, USER, "i'm vegetarian, and remember this as my usual", use_model=False)
    extractor.maybe_learn_alias(conn, USER, "i'm vegetarian, and remember this as my usual")

    facts = {f["key"]: f["value"] for f in store.get_facts(conn, USER)}
    assert facts.get("diet") == "vegetarian"
    assert store.resolve_alias(conn, USER, "my usual", slot="dinner") is not None


def test_alias_matching_is_case_insensitive(conn):
    store.put_alias(conn, USER, "My Usual", [{"name": "oats", "qty": 1, "unit": "katori"}])
    assert store.resolve_alias(conn, USER, "MY USUAL please") is not None


def test_ordinary_food_talk_never_touches_the_alias(conn):
    store.put_alias(conn, USER, "my usual", [{"name": "oats", "qty": 1, "unit": "katori"}])
    for message in ("had 2 rotis", "how am I doing", "actually that was 3"):
        extractor.maybe_learn_alias(conn, USER, message)

    assert {i["name"] for i in store.resolve_alias(conn, USER, "my usual")["items"]} == {"oats"}
