"""Chat REPL.

    python -m calorai.cli
    python -m calorai.cli --user priya
    > had 2 parathas and chai
    > img:images/plate.jpg half of this was my brother's
    > /totals            show today
    > /memory            show what it remembers about you
    > /debug             per-stage timings for the last turn

The interface is deliberately thin -- the brief says the agent code is what
gets read. The one thing it does carefully is fire the memory extractor on a
background thread *after* the reply is printed, so writing memory never shows
up in a latency number.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Windows terminals default to cp1252, and a model that emits an emoji or a
# non-Latin character then kills the process with UnicodeEncodeError mid-reply.
# Observed live: a model answered with a seedling emoji and the CLI crashed.
# The prompt asks for no emoji, but a prompt is not a guarantee, and losing a
# session over a decorative character is absurd.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):  # already wrapped, or not a real tty
        pass


from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402

from calorai import repository as repo  # noqa: E402
from calorai.db import connect  # noqa: E402
from calorai.graph import build_graph, run_turn, stream_turn  # noqa: E402
from calorai.llm import active_backends, tracing_status  # noqa: E402
from calorai.memory import extractor, render, store  # noqa: E402

console = Console()

#: In-flight background memory writes, joined on exit by _flush_memory_writes.
_PENDING_WRITES: list[threading.Thread] = []

HELP = """\
[bold]commands[/bold]
  img:PATH [caption]   send a photo, optionally with a caption
  /totals              today's calories and macros
  /memory              what the agent remembers about you
  /history             recent meals
  /forget              wipe remembered facts and aliases (meals kept)
  /debug               per-stage timings for the last turn
  /help  /quit"""


def _parse_image(line: str) -> tuple[str, str | None]:
    """'img:foo.jpg half was my brother's' -> (caption, path)."""
    for prefix in ("img:", "image:", "/img "):
        if line.lower().startswith(prefix):
            rest = line[len(prefix):].strip()
            parts = rest.split(" ", 1)
            return (parts[1].strip() if len(parts) > 1 else ""), parts[0]
    return line, None


def _remember_later(conn, user_id: str, text: str) -> None:
    """Background memory write.

    This is the whole reason memory costs nothing at p50: by the time it runs,
    the user already has their reply. Failures are swallowed deliberately -- a
    memory miss must never turn a successful turn into an error.
    """

    def work() -> None:
        try:
            extractor.extract_and_store(conn, user_id, text)
            extractor.maybe_learn_alias(conn, user_id, text)
        except Exception:
            pass

    thread = threading.Thread(target=work, daemon=True)
    _PENDING_WRITES.append(thread)
    thread.start()


def _flush_memory_writes(timeout: float = 3.0) -> None:
    """Let in-flight memory writes finish before the process exits.

    These threads are daemons so a hung model call can never wedge the REPL --
    but a daemon dies with the process, so teaching the agent something and
    quitting immediately lost the write. Observed exactly that: "remember this
    as my usual dinner" followed straight by /quit stored the facts and dropped
    the alias, because the slower of the two writes was still in flight.

    "It forgot the last thing I told it" is a memory bug to anyone watching,
    whatever the cause. Bounded so a wedged write delays exit by 3s at most.
    """
    deadline = time.monotonic() + timeout
    for thread in _PENDING_WRITES:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        thread.join(remaining)


def main() -> None:
    parser = argparse.ArgumentParser(description="CalorAI logging agent")
    parser.add_argument("--user", default="default", help="user id (sessions are isolated)")
    parser.add_argument("--db", default=os.environ.get("CALORAI_DB_PATH", "calorai.db"))
    parser.add_argument("--no-fast-path", action="store_true", help="force every turn through the agent")
    parser.add_argument("--no-stream", action="store_true", help="wait for the whole reply instead of streaming it")
    args = parser.parse_args()

    if args.no_fast_path:
        os.environ["CALORAI_FAST_PATH"] = "0"

    conn = connect(args.db)
    backends = active_backends()

    console.print(
        Panel(
            f"[bold]CalorAI[/bold]  ·  user [cyan]{args.user}[/cyan]  ·  db [dim]{args.db}[/dim]\n"
            f"text   [green]{backends['text']}[/green]\n"
            f"vision [green]{backends['vision']}[/green]\n"
            f"[dim]fast path {'off' if args.no_fast_path else 'on'} · "
            f"tracing {tracing_status()} · /help for commands[/dim]",
            expand=False,
        )
    )
    if backends["text"].startswith("mock"):
        console.print(
            "[yellow]running on the offline mock backend -- set CALORAI_TEXT_BACKEND=groq "
            "in .env for real replies and meaningful timings[/yellow]\n"
        )

    graph = build_graph(conn, args.user, streaming=not args.no_stream)
    last: dict = {}

    while True:
        try:
            line = console.input("[bold cyan]you ›[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            _flush_memory_writes()
            console.print("\n[dim]bye[/dim]")
            return
        if not line:
            continue

        low = line.lower()
        if low in {"/quit", "/exit", "quit", "exit"}:
            _flush_memory_writes()
            console.print("[dim]bye[/dim]")
            return
        if low == "/help":
            console.print(HELP)
            continue
        if low == "/totals":
            console.print(repo.daily_totals(conn, args.user))
            continue
        if low == "/memory":
            block = render.render_memory_block(conn, args.user)
            console.print(block or "[dim]nothing remembered yet[/dim]")
            continue
        if low == "/forget":
            dropped = store.forget_everything(conn, args.user)
            console.print(
                f"[dim]forgot {dropped['facts']} fact(s) and "
                f"{dropped['aliases']} alias(es) -- meals kept, /history still works[/dim]"
            )
            continue
        if low == "/history":
            for meal in repo.find_meals(conn, args.user, limit=10)["meals"]:
                console.print(
                    f"  {meal['local_date']} {meal['slot'] or '':9} "
                    f"{meal['qty']:g} {meal['unit']} {meal['name']:18} {meal['kcal']:>5} cal"
                )
            continue
        if low == "/debug":
            if not last:
                console.print("[dim]no turn yet[/dim]")
                continue
            console.print(f"  total   {last['elapsed'] * 1000:7.0f} ms")
            for stage, seconds in last.get("spans", {}).items():
                console.print(f"  {stage:8}{seconds * 1000:7.0f} ms")
            console.print(f"  tools   {last.get('tool_calls')}")
            console.print(f"  fastpath {last.get('used_fast_path')}")
            continue

        caption, image_path = _parse_image(line)
        if image_path and not Path(image_path).exists():
            console.print(f"[red]no file at {image_path}[/red]")
            continue

        started = time.perf_counter()
        try:
            if args.no_stream:
                last = run_turn(conn, args.user, caption, image_path=image_path, graph=graph)
                console.print(f"[bold green]calorai ›[/bold green] {last['reply']}")
            else:
                # Print tokens as they arrive. What the user feels is the moment
                # the first word appears, not the moment the last one does.
                console.print("[bold green]calorai ›[/bold green] ", end="")
                printed = False
                for kind, payload in stream_turn(
                    conn, args.user, caption, image_path=image_path, graph=graph
                ):
                    if kind == "status":
                        console.print(f"[dim]{payload}[/dim] ", end="")
                    elif kind == "token":
                        console.print(payload, end="", highlight=False, markup=False)
                        printed = True
                    else:
                        last = payload
                if not printed:
                    console.print(last.get("reply", ""), end="", markup=False)
                console.print()
        except Exception as exc:  # noqa: BLE001
            console.print(f"\n[red]turn failed:[/red] {exc}")
            continue
        elapsed_ms = (time.perf_counter() - started) * 1000

        timing = f"{elapsed_ms:.0f} ms"
        if last.get("ttft"):
            timing = f"{last['ttft'] * 1000:.0f} ms to first word · {timing} total"
        console.print(f"[dim]{timing}{' · fast path' if last['used_fast_path'] else ''}[/dim]\n")

        repo.transcript_append(conn, args.user, "user", caption or "[photo]")
        repo.transcript_append(conn, args.user, "assistant", last["reply"])
        _remember_later(conn, args.user, caption)


if __name__ == "__main__":
    main()
