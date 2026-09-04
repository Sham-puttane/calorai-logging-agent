"""Real-backend end-to-end walk through the brief's test conversation set."""

from __future__ import annotations

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

SCRIPT = [
    "had 2 parathas and chai for breakfast",
    "leftover biryani, maybe two thirds of the box",
    "skipped lunch but grazed all afternoon",
    "i'm vegetarian btw",
    "2 rotis with dal",
    "actually that was 3 rotis not 2",
    "how much protein have I had today?",
    "how am I doing on calories?",
    "same as yesterday",
    "my usual",
]


def main() -> None:
    print(active_backends(), "\n")
    reset_connections()
    conn = connect(":memory:")

    # seed yesterday so "same as yesterday" has something to find
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
    timings = []

    for message in SCRIPT:
        started = time.perf_counter()
        try:
            result = run_turn(conn, USER, message, graph=graph)
        except Exception as exc:
            print(f"> {message}\n  !! {type(exc).__name__}: {str(exc)[:160]}\n")
            continue
        elapsed = (time.perf_counter() - started) * 1000
        timings.append(elapsed)
        extractor.extract_and_store(conn, USER, message, use_model=False)

        totals = repo.daily_totals(conn, USER)
        items = conn.execute(
            "SELECT COUNT(*) n FROM meal_items WHERE user_id=? AND deleted_at IS NULL", (USER,)
        ).fetchone()["n"]
        tag = " [fast]" if result["used_fast_path"] else ""
        print(f"> {message}")
        print(f"  {elapsed:6.0f}ms {result['tool_calls']}{tag}")
        print(f"  {result['reply'][:150]}")
        print(f"  db: {totals['kcal']} kcal / {items} items\n")

    facts = {f["key"]: f["value"] for f in store.get_facts(conn, USER)}
    print("remembered:", facts)
    if timings:
        ordered = sorted(timings)
        print(f"turns: n={len(timings)} p50={ordered[len(ordered)//2]:.0f}ms max={ordered[-1]:.0f}ms")


if __name__ == "__main__":
    main()
