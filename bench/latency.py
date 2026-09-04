"""Latency benchmark.

    python bench/latency.py --n 30
    python bench/latency.py --n 30 --no-fast-path   # cost of the agent loop alone
    python bench/latency.py --n 10 --backend ollama # the local comparison

Reports p50/p95 for the text path and the image path separately, plus a
per-stage breakdown so a slow number can be attributed rather than guessed at.

Two things this deliberately does NOT do:

* It does not report a number for the mock backend without shouting about it.
  The mock runs in microseconds; publishing that as a latency result would be
  a lie, so the summary is stamped with the backend that produced it.
* It does not warm up silently and hide the first-call cost. The first request
  pays TLS setup and client construction, and a benchmark that quietly discards
  that is measuring something the user never experiences. Warmup is a separate
  reported line.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calorai.db import connect, reset_connections  # noqa: E402
from calorai.graph import build_graph, run_turn  # noqa: E402
from calorai.llm import active_backends  # noqa: E402

USER = "bench_user"

# Representative of real traffic rather than of the happy path: a mix of
# logging, a correction, and a read-only question.
TEXT_MESSAGES = [
    "had 2 parathas and chai for breakfast",
    "leftover biryani, maybe two thirds of the box",
    "3 rotis and dal for lunch",
    "actually that was 4 rotis not 3",
    "how am I doing on calories?",
    "how much protein have I had today?",
    "a banana and some almonds",
    "skipped lunch but grazed all afternoon",
]

IMAGE_CAPTIONS = [
    "",
    "half of this was my brother's",
    "this was lunch",
]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * pct
    lower, upper = int(position), min(int(position) + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarise(name: str, samples: list[dict]) -> dict:
    times = [s["elapsed"] * 1000 for s in samples]
    if not times:
        return {"path": name, "n": 0}
    spans: dict[str, list[float]] = {}
    for sample in samples:
        for stage, seconds in sample.get("spans", {}).items():
            spans.setdefault(stage, []).append(seconds * 1000)
    return {
        "path": name,
        "n": len(times),
        "p50_ms": round(percentile(times, 0.50), 1),
        "p95_ms": round(percentile(times, 0.95), 1),
        "mean_ms": round(statistics.mean(times), 1),
        "min_ms": round(min(times), 1),
        "max_ms": round(max(times), 1),
        "stages_p50_ms": {k: round(percentile(v, 0.50), 1) for k, v in sorted(spans.items())},
        "fast_path_share": round(
            sum(1 for s in samples if s.get("used_fast_path")) / len(samples), 2
        ),
    }


def run_path(conn, graph, n: int, image: str | None, messages: list[str]) -> list[dict]:
    samples: list[dict] = []
    for i in range(n):
        message = messages[i % len(messages)]
        started = time.perf_counter()
        try:
            result = run_turn(conn, USER, message, image_path=image, graph=graph)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i + 1}/{n}] failed: {type(exc).__name__}: {str(exc)[:90]}")
            continue
        result["elapsed"] = time.perf_counter() - started
        samples.append(result)
        marker = "F" if result.get("used_fast_path") else "."
        print(marker, end="", flush=True)
    print()
    return samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30, help="samples per path")
    parser.add_argument("--image", default="images/plate.jpg")
    parser.add_argument("--backend", help="override CALORAI_TEXT_BACKEND")
    parser.add_argument("--vision-backend", help="override CALORAI_VISION_BACKEND")
    parser.add_argument("--no-fast-path", action="store_true")
    parser.add_argument("--out", default="bench/results/latest.json")
    parser.add_argument("--skip-image", action="store_true")
    args = parser.parse_args()

    if args.backend:
        os.environ["CALORAI_TEXT_BACKEND"] = args.backend
    if args.vision_backend:
        os.environ["CALORAI_VISION_BACKEND"] = args.vision_backend
    if args.no_fast_path:
        os.environ["CALORAI_FAST_PATH"] = "0"

    backends = active_backends()
    is_mock = backends["text"].startswith("mock")

    print(f"text   {backends['text']}")
    print(f"vision {backends['vision']}")
    print(f"fast path {os.environ.get('CALORAI_FAST_PATH', '1')} · n={args.n}\n")
    if is_mock:
        print(
            "!! MOCK BACKEND -- these numbers measure Python and SQLite, not a model.\n"
            "   They are the floor the agent adds on top of an LLM, nothing more.\n"
        )

    reset_connections()
    conn = connect(":memory:")
    graph = build_graph(conn, USER)

    # First call pays client construction and TLS. Reported, not hidden.
    warm_started = time.perf_counter()
    run_turn(conn, USER, "had a banana", graph=graph)
    warmup_ms = (time.perf_counter() - warm_started) * 1000
    print(f"warmup (first call, includes connection setup): {warmup_ms:.0f} ms\n")

    print("text path")
    text_samples = run_path(conn, graph, args.n, None, TEXT_MESSAGES)

    image_samples: list[dict] = []
    image_path = ROOT / args.image
    if args.skip_image:
        print("\nimage path skipped")
    elif not image_path.exists() and not is_mock:
        print(f"\nimage path skipped -- no file at {image_path}")
    else:
        print("\nimage path")
        n_image = max(1, args.n // 3)
        image_samples = run_path(
            conn, graph, n_image, str(image_path), IMAGE_CAPTIONS
        )

    report = {
        "backends": backends,
        "fast_path": os.environ.get("CALORAI_FAST_PATH", "1"),
        "warmup_ms": round(warmup_ms, 1),
        "mock": is_mock,
        "paths": [summarise("text", text_samples)],
    }
    if image_samples:
        report["paths"].append(summarise("image", image_samples))

    print()
    header = f"{'path':7}{'n':>4}{'p50 ms':>10}{'p95 ms':>10}{'mean':>9}{'max':>9}"
    print(header)
    print("-" * len(header))
    for path in report["paths"]:
        if not path.get("n"):
            continue
        print(
            f"{path['path']:7}{path['n']:>4}{path['p50_ms']:>10.0f}{path['p95_ms']:>10.0f}"
            f"{path['mean_ms']:>9.0f}{path['max_ms']:>9.0f}"
        )
    print()
    for path in report["paths"]:
        if path.get("stages_p50_ms"):
            stages = " · ".join(f"{k} {v:.0f}ms" for k, v in path["stages_p50_ms"].items())
            print(f"{path['path']:7} stage p50: {stages}")
    if text_samples:
        print(f"\nfast path handled {report['paths'][0]['fast_path_share']:.0%} of text turns")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten to {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
