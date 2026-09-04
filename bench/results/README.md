# Benchmark evidence

Committed so the numbers in the root README can be traced to the run that produced them. Each file
carries a `backends` block naming the exact models it used, the pacing (`delay_s`), and a
`throttled` count. Regenerate any of them with `bench/latency.py`.

Free-tier quotas are the reason this directory is curated rather than exhaustive: all three text
providers hit their daily limits during the final measurement session, so some runs cannot be
retaken until the caps reset.

| file | what it is | why it is here |
|---|---|---|
| `image_pixtral.json` | Mistral `pixtral-12b-2409` at 512 px, n=8, 0 throttled | The image-path numbers in the README: p50 5864 ms, p95 6837 ms. Taken with `CALORAI_VISION_FALLBACK=none`, which is what took p95 down from 13709 ms — a fallback to an out-of-quota provider was being paid for before Pixtral ever answered. |
| `text_groq_throttled.json` | Groq `gpt-oss-20b`, n=20, **16 throttled** | Not a latency result — evidence. This is the same command that produced the 766 ms figure, re-run after the day's 200k-token budget was gone. Its 14 ms p50 is meaningless (only 4 fast-path turns survived), and that is the point: it is the artifact behind the "the binding constraint is tokens per day" claim. |
| `text_openrouter_llama_rejected.json` | OpenRouter `llama-3.3-70b-instruct`, n=16 | The rejected failover model, kept because it is the counter-example. p50 2504 ms — *faster* than the model that replaced it — but walking the full conversation showed it confirming meals it had never written. Latency was never the deciding axis. |
| `local_ollama.json` | `qwen2.5:3b` on CPU, no GPU | The local-inference measurement behind the decision not to ship local as the default path. |

## What is missing and why

The Groq run that produced **766 ms / 1257 ms** was overwritten by `text_groq_throttled.json` — the
same command, re-run once the budget was spent. It has not been retaken because it cannot be until
the daily cap resets.

The failover p50 of **2212 ms** comes from `bench/_real_e2e.py` rather than the percentile sweep,
for the same reason. That script walks the brief's whole conversation and prints the database after
every turn, so it measures per-turn latency *and* asserts what was actually written — which is how
the rejected model above was caught. It has no p95 to report.
