"""Self-playing demo, for recording a walkthrough.

    python -m calorai.demo                 # normal pace, good for recording
    python -m calorai.demo --rehearse      # fast, no typing animation
    python -m calorai.demo --manual        # wait for Enter between beats

Types each message out at reading speed and streams the real reply, so a
recording looks like a live session without the risk of a typo or a pause on
camera. Everything it shows is a real turn against real models -- nothing here
is scripted output.

Record it silently (Snipping Tool has screen recording built in on Windows 11)
and lay the voiceover over the top afterwards.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.rule import Rule  # noqa: E402

from calorai import repository as repo  # noqa: E402
from calorai.db import connect, reset_connections  # noqa: E402
from calorai.graph import build_graph, stream_turn  # noqa: E402
from calorai.llm import active_backends  # noqa: E402
from calorai.memory import extractor, render, store  # noqa: E402
from calorai.schemas import FoodItem  # noqa: E402

console = Console()
USER = "demo"

# (caption, message, image). A caption of None keeps the previous section
# heading -- used where two turns belong together, like the correction pair.
BEATS: list[tuple[str | None, str, str | None]] = [
    ("Plain logging — it assumes a portion and says so, rather than asking",
     "had 2 parathas and chai for breakfast", None),
    (None, "leftover biryani, maybe two thirds of the box", None),

    ("Vague amounts are not a reason to interrogate someone",
     "skipped lunch but grazed all afternoon", None),

    ("A fact about the person — remembered, NOT logged as food",
     "i'm vegetarian btw", None),
    (None, "i'm aiming for 140g of protein", None),

    ("THE CORRECTION CASE — watch the total move by one roti, not three",
     "2 rotis with dal", None),
    (None, "actually that was 3 rotis not 2", None),

    ("Read-only questions skip both models entirely",
     "how much protein have I had today?", None),
    (None, "how am I doing on calories?", None),

    ("Memory, not parsing — 'my usual' resolves before the model is called",
     "my usual", None),

    ("THE IMAGE CASE — one photo, one meal, seven dishes",
     "", "images/plate.jpg"),
    ("Two models, still ONE meal — the caption halves the whole plate",
     "half of this was my brother's", "images/plate.jpg"),
]


def type_out(text: str, delay: float) -> None:
    console.print("[bold cyan]you ›[/bold cyan] ", end="")
    if delay <= 0:
        console.print(text, markup=False)
        return
    for ch in text:
        console.print(ch, end="", markup=False, highlight=False)
        time.sleep(delay)
    console.print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-playing CalorAI demo")
    parser.add_argument("--rehearse", action="store_true", help="fast, for checking it still works")
    parser.add_argument("--manual", action="store_true", help="wait for Enter between beats")
    parser.add_argument("--pace", type=float, default=7.0,
                        help="seconds between beats; keeps the free tier from throttling on camera")
    parser.add_argument("--db", default="demo.db")
    args = parser.parse_args()

    typing_delay = 0.0 if args.rehearse else 0.035
    # Long enough that Groq's tokens-per-minute cap does not throttle mid-demo.
    # On camera it reads as the presenter talking; unpaced it reads as a bug.
    beat_pause = 0.0 if args.rehearse else args.pace

    # Fresh database every run, so the recording is reproducible and the totals
    # on screen always start from zero.
    db_path = Path(args.db)
    if db_path.exists():
        db_path.unlink()
    reset_connections()
    conn = connect(str(db_path))

    backends = active_backends()
    console.print(
        Panel(
            f"[bold]CalorAI[/bold] — conversational meal logging\n"
            f"text   [green]{backends['text']}[/green]\n"
            f"vision [green]{backends['vision']}[/green]",
            expand=False,
        )
    )
    if backends["text"].startswith("mock"):
        console.print("[yellow]offline mock backend — set CALORAI_TEXT_BACKEND=groq in .env[/yellow]")
    console.print()

    # Seed what a returning user would already have: yesterday's breakfast, and
    # a learned shorthand. Shown on screen so nothing looks like sleight of hand.
    repo.log_meal(
        conn, USER,
        [FoodItem(name="idli", qty=3, unit="piece"), FoodItem(name="sambar", qty=1, unit="katori")],
        slot="breakfast", day="yesterday",
    )
    store.put_alias(
        conn, USER, "my usual",
        [{"name": "oats", "qty": 1, "unit": "katori"}, {"name": "banana", "qty": 1, "unit": "piece"}],
    )
    console.print(
        "[dim]seeded, so this looks like a returning user: yesterday's breakfast "
        "(3 idli + sambar), and a learned alias \"my usual\" = oats + banana[/dim]\n"
    )

    graph = build_graph(conn, USER, streaming=True)
    last: dict = {}

    for caption, message, image in BEATS:
        if caption:
            console.print(Rule(f"[bold yellow]{caption}[/bold yellow]", align="left"))
        if args.manual:
            console.input("[dim](enter)[/dim]")

        shown = message or "[sends a photo]"
        if image and message:
            shown = f"{message}  [+ photo]"
        type_out(shown, typing_delay)

        started = time.perf_counter()
        console.print("[bold green]calorai ›[/bold green] ", end="")
        printed = False
        try:
            for kind, payload in stream_turn(
                conn, USER, message,
                image_path=str(Path(image)) if image else None, graph=graph,
            ):
                if kind == "status":
                    console.print(f"[dim]{payload}[/dim] ", end="")
                elif kind == "token":
                    console.print(payload, end="", markup=False, highlight=False)
                    printed = True
                else:
                    last = payload
            if not printed:
                console.print(last.get("reply", ""), end="", markup=False)
            console.print()
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]{type(exc).__name__}: {str(exc)[:110]}[/red]")
            continue

        elapsed = (time.perf_counter() - started) * 1000
        totals = repo.daily_totals(conn, USER)
        items = totals["items_logged"]
        detail = f"{elapsed:.0f} ms"
        if last.get("ttft"):
            detail = f"{last['ttft'] * 1000:.0f} ms to first word · {detail}"
        if last.get("used_fast_path"):
            detail += " · fast path, no model call"
        if last.get("tool_calls"):
            detail += f" · {', '.join(last['tool_calls'])}"
        console.print(f"[dim]{detail}[/dim]")
        console.print(
            f"[dim]  db: {totals['kcal']} cal · {items} items · "
            f"{totals['protein_g']:g}g protein[/dim]\n"
        )

        extractor.extract_and_store(conn, USER, message, use_model=False)
        time.sleep(beat_pause)

    console.print(Rule("[bold yellow]What it remembered, across the session[/bold yellow]", align="left"))
    console.print(render.render_memory_block(conn, USER) or "[dim]nothing[/dim]")
    console.print()

    console.print(Rule("[bold yellow]Where the time went, on the last turn[/bold yellow]", align="left"))
    for stage, seconds in sorted(last.get("spans", {}).items()):
        console.print(f"  {stage:8} {seconds * 1000:7.0f} ms")
    console.print()

    console.print(Rule("[bold yellow]The day's log[/bold yellow]", align="left"))
    for meal in repo.find_meals(conn, USER, day="today", limit=30)["meals"]:
        console.print(
            f"  {meal['slot'] or '':10} {meal['qty']:>5g} {meal['unit']:<8} "
            f"{meal['name']:<20} {meal['kcal']:>5} cal"
        )
    totals = repo.daily_totals(conn, USER)
    console.print(f"\n  [bold]{totals['kcal']} cal · {totals['protein_g']:g}g protein[/bold]")


if __name__ == "__main__":
    main()
