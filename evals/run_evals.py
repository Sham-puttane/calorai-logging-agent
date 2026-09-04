"""Eval runner.

    python evals/run_evals.py                 # default backend (mock unless .env says otherwise)
    python evals/run_evals.py --no-fast-path  # force every case through the agent loop
    python evals/run_evals.py --backend groq  # score a real model
    python evals/run_evals.py --case correction_updates_not_appends

Each case runs against a fresh in-memory database, so cases cannot contaminate
each other and the run is order-independent.

Scoring is per-assertion rather than per-case: a case that logs the right food
with the wrong tool should not score zero, because the distinction between
"slightly wrong" and "corrupted the data" is the thing worth measuring.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calorai import repository as repo  # noqa: E402
from calorai.db import connect, reset_connections  # noqa: E402
from calorai.graph import build_graph, run_turn  # noqa: E402
from calorai.llm import active_backends  # noqa: E402
from calorai.memory import extractor, store  # noqa: E402
from calorai.schemas import FoodItem  # noqa: E402

USER = "eval_user"

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


class Case:
    def __init__(self, raw: dict):
        self.id = raw["id"]
        self.note = raw.get("note", "")
        self.message = raw.get("message", "")
        self.image = raw.get("image")
        self.setup_meals = raw.get("setup_meals", [])
        self.setup_messages = raw.get("setup_messages", [])
        self.setup_alias = raw.get("setup_alias")
        self.expect = raw.get("expect", {})


def _apply_setup(conn, case: Case, graph) -> None:
    """Meals go in through the repository (deterministic), messages through the
    agent (so multi-turn state is real)."""
    grouped: dict[int, list[dict]] = {}
    for spec in case.setup_meals:
        grouped.setdefault(spec.get("meal", 0), []).append(spec)

    for specs in grouped.values():
        first = specs[0]
        repo.log_meal(
            conn,
            first.get("user", USER),
            [FoodItem(name=s["name"], qty=s.get("qty", 1), unit=s.get("unit", "serving")) for s in specs],
            slot=first.get("slot"),
            day=first.get("day"),
        )

    if case.setup_alias:
        store.put_alias(
            conn, USER, case.setup_alias["phrase"], case.setup_alias["items"], source="explicit"
        )

    for message in case.setup_messages:
        run_turn(conn, USER, message, graph=graph)
        extractor.extract_and_store(conn, USER, message, use_model=False)


def _check(case: Case, result: dict, conn) -> list[tuple[str, bool, str]]:
    expect = case.expect
    checks: list[tuple[str, bool, str]] = []
    called = result.get("tool_calls", [])
    reply = (result.get("reply") or "").lower()

    for tool in expect.get("tools", []):
        checks.append((f"calls {tool}", tool in called, f"called {called or 'nothing'}"))

    for tool in expect.get("tools_absent", []):
        checks.append((f"never calls {tool}", tool not in called, f"called {called}"))

    if "writes" in expect and expect["writes"] is False:
        writers = {"log_meal", "correct_meal", "delete_meal"}
        offending = writers & set(called)
        checks.append(("makes no writes", not offending, f"wrote via {offending or '-'}"))

    db = expect.get("db", {})
    if db:
        totals = repo.daily_totals(conn, USER)
        live = conn.execute(
            "SELECT COUNT(*) n FROM meal_items WHERE user_id=? AND deleted_at IS NULL", (USER,)
        ).fetchone()["n"]
        meals = conn.execute(
            "SELECT COUNT(DISTINCT meal_id) n FROM meal_items"
            " WHERE user_id=? AND deleted_at IS NULL",
            (USER,),
        ).fetchone()["n"]

        if "kcal" in db:
            actual = totals["kcal"]
            ok = abs(actual - db["kcal"]) <= max(2, db["kcal"] * 0.02)
            checks.append((f"kcal == {db['kcal']}", ok, f"got {actual}"))
        if "items" in db:
            checks.append((f"live items == {db['items']}", live == db["items"], f"got {live}"))
        if "meals" in db:
            checks.append((f"meals == {db['meals']}", meals == db["meals"], f"got {meals}"))
        if "estimated" in db:
            got = totals["items_estimated"]
            checks.append(
                (f"estimated == {db['estimated']}", got == db["estimated"], f"got {got}")
            )

    if "asks" in expect:
        asked = reply.strip().endswith("?")
        checks.append((f"asks == {expect['asks']}", asked == expect["asks"], f"reply: {reply[:60]}"))

    for needle in expect.get("reply_contains", []):
        checks.append(
            (f"reply mentions '{needle}'", needle.lower() in reply, f"reply: {reply[:60]}")
        )

    for key, value in expect.get("facts", {}).items():
        facts = {f["key"]: f["value"] for f in store.get_facts(conn, USER)}
        checks.append((f"remembers {key}={value}", facts.get(key) == value, f"got {facts}"))

    return checks


def run(cases: list[Case], verbose: bool = False) -> int:
    passed_checks = total_checks = 0
    failed_cases: list[str] = []

    for case in cases:
        reset_connections()
        conn = connect(":memory:")
        graph = build_graph(conn, USER)

        _apply_setup(conn, case, graph)

        image = str(ROOT / case.image) if case.image else None
        result = run_turn(conn, USER, case.message, image_path=image, graph=graph)
        # memory writes happen after the reply in production too
        extractor.extract_and_store(conn, USER, case.message, use_model=False)
        extractor.maybe_learn_alias(conn, USER, case.message)

        checks = _check(case, result, conn)
        ok = all(passed for _, passed, _ in checks)
        passed_checks += sum(1 for _, passed, _ in checks if passed)
        total_checks += len(checks)

        mark = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
        # Cases drawn straight from the brief's test conversation set are
        # flagged, so a run shows coverage of it rather than just a score.
        tag = f"  {DIM}<- {case.note}{RESET}" if case.note.startswith("brief:") else ""
        print(f"{mark}  {case.id}{tag}")
        if not ok:
            failed_cases.append(case.id)
        for label, passed, detail in checks:
            if not passed or verbose:
                colour = GREEN if passed else RED
                print(f"        {colour}{'ok' if passed else 'no'}{RESET} {label} {DIM}({detail}){RESET}")

    print()
    backends = active_backends()
    print(f"{DIM}text {backends['text']} · vision {backends['vision']} · "
          f"fast path {os.environ.get('CALORAI_FAST_PATH', '1')}{RESET}")
    score = 100.0 * passed_checks / total_checks if total_checks else 0.0
    colour = GREEN if not failed_cases else YELLOW
    print(
        f"{colour}{len(cases) - len(failed_cases)}/{len(cases)} cases · "
        f"{passed_checks}/{total_checks} assertions ({score:.0f}%){RESET}"
    )
    from_brief = sum(1 for c in cases if c.note.startswith("brief:"))
    if from_brief:
        print(f"{DIM}{from_brief} of these come straight from the brief's test "
              f"conversation set{RESET}")
    if failed_cases:
        print(f"{RED}failed: {', '.join(failed_cases)}{RESET}")
    return 0 if not failed_cases else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", help="run one case by id")
    parser.add_argument("--backend", help="override CALORAI_TEXT_BACKEND")
    parser.add_argument("--no-fast-path", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    # Default to the offline mock so a clean clone scores deterministically with
    # no keys. .env may point at a real provider for interactive use; evals opt
    # in explicitly with --backend.
    os.environ["CALORAI_TEXT_BACKEND"] = args.backend or "mock"
    os.environ["CALORAI_VISION_BACKEND"] = args.backend or "mock"
    if args.no_fast_path:
        os.environ["CALORAI_FAST_PATH"] = "0"

    raw = yaml.safe_load((Path(__file__).parent / "cases.yaml").read_text(encoding="utf-8"))
    cases = [Case(entry) for entry in raw]
    if args.case:
        cases = [c for c in cases if c.id == args.case]
        if not cases:
            print(f"no case named {args.case}")
            return 2
    return run(cases, verbose=args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
