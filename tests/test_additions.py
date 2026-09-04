"""Adding food is not correcting food.

"plus rice, I forgot to add it" and "actually that was rice not dal" produce
the *same* tool call shape, so telling them apart is a language judgement only
the model can make. The prompt teaches it.

But substitution REPLACES a row, so getting it wrong deletes food the person
really ate — silently, and in a way that makes their day's total too low. That
is exactly the class of mistake the separate-tools design exists to prevent, so
it does not get to rest on prompt wording alone: the repository refuses a
substitution when the message that triggered it reads like an addition.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from calorai import repository as repo  # noqa: E402
from calorai.db import connect, reset_connections  # noqa: E402
from calorai.schemas import FoodItem  # noqa: E402

USER = "u_add"


@pytest.fixture()
def conn():
    reset_connections()
    c = connect(":memory:")
    yield c
    reset_connections()


def live(c) -> int:
    return c.execute(
        "SELECT COUNT(*) n FROM meal_items WHERE user_id=? AND deleted_at IS NULL", (USER,)
    ).fetchone()["n"]


@pytest.mark.parametrize(
    "message",
    [
        "plus rice in the dinner that i forgot to add",
        "also had a chai",
        "forgot the dal",
        "and some curd as well",
        "add a banana too",
    ],
)
def test_an_addition_never_silently_replaces_food(conn, message):
    """The bug: "plus rice ... I forgot to add" replaced the dal with rice.
    The user ate both, and their total came out too low."""
    repo.log_meal(conn, USER, [FoodItem(name="dal", qty=1, unit="katori")])

    result = repo.correct_meal(conn, USER, target_hint="rice", new_qty=1, message=message)

    assert not result["ok"], "must refuse rather than replace"
    assert "log_meal" in result["error"], "and must say what to do instead"
    names = {r["name"] for r in conn.execute(
        "SELECT name FROM meal_items WHERE user_id=? AND deleted_at IS NULL", (USER,))}
    assert names == {"dal"}, "the food they actually ate survives"


@pytest.mark.parametrize(
    "message",
    [
        "actually that was 3 rotis not 2",
        "that was dal not rice",
        "make it 1 paratha instead of 2",
        "i meant rotis",
    ],
)
def test_a_real_correction_still_substitutes(conn, message):
    """The guard must not block the case it was built around."""
    repo.log_meal(conn, USER, [FoodItem(name="paratha", qty=2, unit="piece")])

    result = repo.correct_meal(conn, USER, target_hint="roti", new_qty=3, message=message)

    assert result["ok"], f"{message!r} is a correction, not an addition"
    assert live(conn) == 1, "rewritten, not joined by a second row"
    assert repo.daily_totals(conn, USER)["kcal"] == 315


def test_the_guard_only_blocks_substitution_not_ordinary_corrections(conn):
    """A correction to a food that IS in the log is untouched by the guard --
    it never reaches the substitution branch."""
    repo.log_meal(conn, USER, [FoodItem(name="roti", qty=2, unit="piece")])

    result = repo.correct_meal(
        conn, USER, target_hint="roti", new_qty=3, message="also make the rotis 3"
    )

    assert result["ok"]
    assert repo.daily_totals(conn, USER)["kcal"] == 315


def test_no_message_falls_back_to_permitting_substitution(conn):
    """Callers that do not pass the raw turn (tests, scripts) keep the old
    behaviour rather than being silently blocked."""
    repo.log_meal(conn, USER, [FoodItem(name="paratha", qty=2, unit="piece")])
    assert repo.correct_meal(conn, USER, target_hint="roti", new_qty=3)["ok"]
