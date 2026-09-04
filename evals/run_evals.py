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
import json
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
        # A second message sent after the first, used for the photo flow: the
        # image turn asks for confirmation, and assertions run on what the
        # follow-up produced.
        self.follow_up = raw.get("follow_up")
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


BASELINE = Path(__file__).parent / "baseline.json"


def load_baseline() -> dict[str, list[int]]:
    """Per-case scores from a previous run, if one was committed."""
    if not BASELINE.exists():
        return {}
    return json.loads(BASELINE.read_text(encoding="utf-8")).get("cases", {})


def report_drift(scores: dict[str, list[int]], baseline: dict[str, list[int]]) -> None:
    """Say whether this run is better or worse than the recorded one.

    A pass/fail total answers "does it work". It does not answer "did that
    change help", which is the question you actually have when you have just
    rewritten a prompt -- and prompts regress sideways: you fix the phrasing of
    corrections and quietly break the one about fractions. Comparing
    per-assertion scores against a committed baseline catches that, and catches
    the case where the total is unchanged because one thing broke as another
    was fixed.
    """
    if not baseline:
        print(f"{DIM}no baseline recorded -- run with --save-baseline to set one{RESET}")
        return

    better, worse, added = [], [], []
    for case_id, (passed, total) in scores.items():
        if case_id not in baseline:
            added.append(case_id)
            continue
        was, _ = baseline[case_id]
        if passed > was:
            better.append(f"{case_id} {was}->{passed}")
        elif passed < was:
            worse.append(f"{case_id} {was}->{passed}")
    dropped = [c for c in baseline if c not in scores]

    if not (better or worse or added or dropped):
        print(f"{DIM}unchanged against baseline{RESET}")
        return
    if better:
        print(f"{GREEN}better than baseline: {', '.join(better)}{RESET}")
    if worse:
        print(f"{RED}WORSE than baseline: {', '.join(worse)}{RESET}")
    if added:
        print(f"{DIM}new cases: {', '.join(added)}{RESET}")
    if dropped:
        print(f"{DIM}cases no longer run: {', '.join(dropped)}{RESET}")


def run(cases: list[Case], verbose: bool = False, save_baseline: bool = False) -> int:
    passed_checks = total_checks = 0
    failed_cases: list[str] = []
    scores: dict[str, list[int]] = {}

    for case in cases:
        reset_connections()
        conn = connect(":memory:")
        graph = build_graph(conn, USER)

        _apply_setup(conn, case, graph)

        image = str(ROOT / case.image) if case.image else None
        result = run_turn(conn, USER, case.message, image_path=image, graph=graph)
        if case.follow_up:
            result = run_turn(conn, USER, case.follow_up, graph=graph)
        # memory writes happen after the reply in production too
        extractor.extract_and_store(conn, USER, case.message, use_model=False)
        extractor.maybe_learn_alias(conn, USER, case.message)

        checks = _check(case, result, conn)
        ok = all(passed for _, passed, _ in checks)
        case_passed = sum(1 for _, passed, _ in checks if passed)
        scores[case.id] = [case_passed, len(checks)]
        passed_checks += case_passed
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

    if save_baseline:
        BASELINE.write_text(
            json.dumps({"total": [passed_checks, total_checks], "cases": scores}, indent=2),
            encoding="utf-8",
        )
        print(f"{DIM}baseline written to {BASELINE.name}{RESET}")
    else:
        report_drift(scores, load_baseline())

    return 0 if not failed_cases else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", help="run one case by id")
    parser.add_argument("--backend", help="override CALORAI_TEXT_BACKEND")
    parser.add_argument("--no-fast-path", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--save-baseline", action="store_true",
                        help="record this run as the bar for future runs")
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
    return run(cases, verbose=args.verbose, save_baseline=args.save_baseline)


if __name__ == "__main__":
    raise SystemExit(main())
