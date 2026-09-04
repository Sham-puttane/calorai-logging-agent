"""Full dress rehearsal against real models.

Walks the brief's entire test conversation set, including both photo cases, and
prints the database state after every turn so a wrong reply and a wrong write
can be told apart.

    python bench/_real_e2e.py            # paced, so the free tier keeps up
    python bench/_real_e2e.py --delay 0  # unpaced, expect throttling
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from calorai import repository as repo  # noqa: E402
from calorai.db import connect, reset_connections  # noqa: E402
from calorai.graph import build_graph, run_turn  # noqa: E402
from calorai.llm import active_backends  # noqa: E402
from calorai.memory import extractor, store  # noqa: E402
from calorai.schemas import FoodItem  # noqa: E402

USER = "real_test"

# (message, image) -- every line from the brief's test conversation set.
SCRIPT: list[tuple[str, str | None]] = [
    ("had 2 parathas and chai for breakfast", None),
    ("leftover biryani, maybe two thirds of the box", None),
    ("skipped lunch but grazed all afternoon", None),
    ("i'm vegetarian btw", None),
    ("2 rotis with dal", None),
    ("actually that was 3 rotis not 2", None),
    ("how much protein have I had today?", None),
    ("how am I doing on calories?", None),
    ("same as yesterday", None),
    ("my usual", None),
    ("", "images/plate.jpg"),
    ("half of this was my brother's", "images/plate.jpg"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay", type=float, default=7.0)
    args = parser.parse_args()

    print(active_backends(), "\n")
    reset_connections()
    conn = connect(":memory:")

    # seed yesterday so "same as yesterday" has something to find, and an alias
    # so "my usual" resolves without needing three prior breakfasts
    repo.log_meal(
        conn, USER,
        [FoodItem(name="idli", qty=3, unit="piece"), FoodItem(name="sambar", qty=1, unit="katori")],
        slot="breakfast", day="yesterday",
    )
    store.put_alias(
        conn, USER, "my usual",
        [{"name": "oats", "qty": 1, "unit": "katori"}, {"name": "banana", "qty": 1, "unit": "piece"}],
    )

    graph = build_graph(conn, USER)
    timings: list[float] = []
    throttled = 0

    for i, (message, image) in enumerate(SCRIPT):
        if args.delay and i:
            time.sleep(args.delay)
        started = time.perf_counter()
        try:
            result = run_turn(
                conn, USER, message, image_path=str(ROOT / image) if image else None, graph=graph
            )
        except Exception as exc:
            print(f"> {message or '[photo]'}\n  !! {type(exc).__name__}: {str(exc)[:130]}\n")
            throttled += 1
            continue
        elapsed = (time.perf_counter() - started) * 1000
        extractor.extract_and_store(conn, USER, message, use_model=False)

        if "rate limited" in result["reply"].lower():
            throttled += 1
        else:
            timings.append(elapsed)

        totals = repo.daily_totals(conn, USER)
        items = conn.execute(
            "SELECT COUNT(*) n FROM meal_items WHERE user_id=? AND local_date=date('now','localtime')"
            " AND deleted_at IS NULL",
            (USER,),
        ).fetchone()["n"]
        meals = conn.execute(
            "SELECT COUNT(DISTINCT meal_id) n FROM meal_items WHERE user_id=?"
            " AND local_date=date('now','localtime') AND deleted_at IS NULL",
            (USER,),
        ).fetchone()["n"]

        label = (message or "[photo]") + (" + [photo]" if image and message else "")
        tag = " [fast]" if result["used_fast_path"] else ""
        print(f"> {label}")
        print(f"  {elapsed:6.0f}ms {result['tool_calls']}{tag}")
        print(f"  {result['reply'][:150]}")
        print(f"  db: {totals['kcal']} cal · {items} items · {meals} meals\n")

    print("remembered:", {f["key"]: f["value"] for f in store.get_facts(conn, USER)})
    if timings:
        ordered = sorted(timings)
        print(
            f"turns: n={len(timings)} p50={ordered[len(ordered) // 2]:.0f}ms "
            f"max={ordered[-1]:.0f}ms throttled={throttled}"
        )


if __name__ == "__main__":
    main()
