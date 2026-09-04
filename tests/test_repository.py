"""Correctness tests for the meal store.

These assert the *effect* -- row counts and summed totals -- rather than that a
call returned without raising. A correction that silently inserts a second row
still "succeeds"; only the sum catches it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from calorai import repository as repo  # noqa: E402
from calorai.db import connect, reset_connections  # noqa: E402
from calorai.schemas import FoodItem  # noqa: E402

USER = "u_test"
OTHER = "u_other"


@pytest.fixture()
def conn():
    reset_connections()
    c = connect(":memory:")
    yield c
    reset_connections()


def live_item_count(c, user_id=USER) -> int:
    return c.execute(
        "SELECT COUNT(*) n FROM meal_items WHERE user_id=? AND deleted_at IS NULL",
        (user_id,),
    ).fetchone()["n"]


# ---------------------------------------------------------------------------
# The one that matters: "actually that was 3 rotis not 2"
# ---------------------------------------------------------------------------
def test_correction_updates_in_place_and_does_not_double_count(conn):
    repo.log_meal(conn, USER, [FoodItem(name="roti", qty=2, unit="piece")])

    before = repo.daily_totals(conn, USER)
    assert before["kcal"] == 210, "2 rotis at 105 kcal"
    assert live_item_count(conn) == 1

    result = repo.correct_meal(conn, USER, target_hint="roti", new_qty=3)
    assert result["ok"]

    after = repo.daily_totals(conn, USER)

    # The whole point: ONE row, and the day moved by exactly one roti.
    assert live_item_count(conn) == 1, "correction must UPDATE, never INSERT a second row"
    assert after["kcal"] == 315, "3 rotis, not 2+3=5 rotis"
    assert after["kcal"] - before["kcal"] == 105, "delta is one roti"
    assert after["items_logged"] == 1


def test_repeated_corrections_stay_idempotent(conn):
    """Users correct the same thing twice. Totals must track the latest value,
    not accumulate every intermediate one."""
    repo.log_meal(conn, USER, [FoodItem(name="roti", qty=2, unit="piece")])
    repo.correct_meal(conn, USER, target_hint="roti", new_qty=3)
    repo.correct_meal(conn, USER, target_hint="roti", new_qty=4)
    repo.correct_meal(conn, USER, target_hint="roti", new_qty=1)

    assert live_item_count(conn) == 1
    assert repo.daily_totals(conn, USER)["kcal"] == 105


def test_correction_can_change_the_food_and_recomputes_nutrition(conn):
    """'that was dal not rice' must re-derive macros, not rescale rice's."""
    repo.log_meal(conn, USER, [FoodItem(name="rice", qty=1, unit="katori")])
    assert repo.daily_totals(conn, USER)["protein_g"] == 4.0

    repo.correct_meal(conn, USER, target_hint="rice", new_name="dal")

    totals = repo.daily_totals(conn, USER)
    assert live_item_count(conn) == 1
    assert totals["kcal"] == 150
    assert totals["protein_g"] == 9.0, "dal's protein, not rice's"


def test_correction_targets_the_right_item_in_a_mixed_meal(conn):
    repo.log_meal(
        conn, USER,
        [FoodItem(name="paratha", qty=2, unit="piece"), FoodItem(name="chai", qty=1, unit="cup")],
    )
    assert repo.daily_totals(conn, USER)["kcal"] == 430  # 340 + 90

    repo.correct_meal(conn, USER, target_hint="paratha", new_qty=1)

    totals = repo.daily_totals(conn, USER)
    assert live_item_count(conn) == 2, "the chai must survive untouched"
    assert totals["kcal"] == 260  # 170 + 90


def test_correction_with_no_match_fails_loudly(conn):
    repo.log_meal(conn, USER, [FoodItem(name="roti", qty=2, unit="piece")])
    result = repo.correct_meal(conn, USER, target_hint="sushi", new_qty=3)
    assert not result["ok"], "must not silently rewrite an unrelated row"
    assert repo.daily_totals(conn, USER)["kcal"] == 210


# ---------------------------------------------------------------------------
# deletes
# ---------------------------------------------------------------------------
def test_delete_removes_from_totals_but_keeps_the_row(conn):
    repo.log_meal(conn, USER, [FoodItem(name="samosa", qty=2, unit="piece")])
    assert repo.daily_totals(conn, USER)["kcal"] == 520

    repo.delete_meal(conn, USER, target_hint="samosa")

    assert repo.daily_totals(conn, USER)["kcal"] == 0
    assert live_item_count(conn) == 0
    total_rows = conn.execute(
        "SELECT COUNT(*) n FROM meal_items WHERE user_id=?", (USER,)
    ).fetchone()["n"]
    assert total_rows == 1, "soft delete: row retained for audit"


# ---------------------------------------------------------------------------
# scoping
# ---------------------------------------------------------------------------
def test_totals_are_isolated_per_user(conn):
    repo.log_meal(conn, USER, [FoodItem(name="roti", qty=2, unit="piece")])
    repo.log_meal(conn, OTHER, [FoodItem(name="biryani", qty=2, unit="katori")])

    assert repo.daily_totals(conn, USER)["kcal"] == 210
    assert repo.daily_totals(conn, OTHER)["kcal"] == 480


def test_one_user_cannot_correct_anothers_meal(conn):
    repo.log_meal(conn, USER, [FoodItem(name="roti", qty=2, unit="piece")])
    result = repo.correct_meal(conn, OTHER, target_hint="roti", new_qty=99)

    assert not result["ok"]
    assert repo.daily_totals(conn, USER)["kcal"] == 210


def test_totals_are_scoped_to_the_day(conn):
    repo.log_meal(conn, USER, [FoodItem(name="roti", qty=2, unit="piece")], day="yesterday")
    repo.log_meal(conn, USER, [FoodItem(name="chai", qty=1, unit="cup")], day="today")

    assert repo.daily_totals(conn, USER, "today")["kcal"] == 90
    assert repo.daily_totals(conn, USER, "yesterday")["kcal"] == 210


# ---------------------------------------------------------------------------
# retrieval -- powers "same as yesterday"
# ---------------------------------------------------------------------------
def test_find_meals_retrieves_yesterday_for_replay(conn):
    repo.log_meal(
        conn, USER,
        [FoodItem(name="paratha", qty=2, unit="piece"), FoodItem(name="chai", qty=1, unit="cup")],
        slot="breakfast", day="yesterday",
    )
    found = repo.find_meals(conn, USER, day="yesterday", slot="breakfast")

    assert found["count"] == 2
    assert {m["name"] for m in found["meals"]} == {"paratha", "chai"}


def test_unknown_food_logs_zero_and_is_flagged_not_invented(conn):
    result = repo.log_meal(conn, USER, [FoodItem(name="zorblax casserole", qty=1)])

    assert result["unknown_foods"] == ["zorblax casserole"]
    totals = repo.daily_totals(conn, USER)
    assert totals["kcal"] == 0, "an unknown food must not fabricate calories"
    assert totals["items_estimated"] == 1, "and must be flagged so the agent can hedge"


def test_edit_log_records_the_before_and_after(conn):
    repo.log_meal(conn, USER, [FoodItem(name="roti", qty=2, unit="piece")])
    repo.correct_meal(conn, USER, target_hint="roti", new_qty=3)

    rows = conn.execute(
        "SELECT action FROM edit_log WHERE user_id=? ORDER BY id", (USER,)
    ).fetchall()
    assert [r["action"] for r in rows] == ["log", "correct"]
