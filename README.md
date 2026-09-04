# CalorAI Logging Agent

A conversational meal logger. You text what you ate the way you'd text a friend, and it gets
logged — no forms, no dropdowns, no searching a database for "paratha (medium)".

```
you › had 2 parathas and chai for breakfast
calorai › logged 2 parathas and a chai, ~430 cal.                          812 ms

you › actually that was 3 rotis not 2
calorai › fixed — 2 rotis is now 3. that puts you at 895 cal today.        714 ms

you › how much protein have I had today?
calorai › 34.4g protein today, 105.6g to go on your 140g target.            19 ms

you › img:images/plate.jpg half of this was my brother's
calorai › logged rice, dal and paneer at half portions, ~241 cal —
          rough guess on the amounts from the photo.
```

---

## Setup

Python 3.12+.

```bash
git clone <this repo> && cd calorai-agent
python -m venv .venv && .venv/Scripts/activate     # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

**The tests and evals run with no API keys at all** — they default to a deterministic offline
backend. To verify the clone before configuring anything:

```bash
pytest tests/ -q                 # 53 tests
python evals/run_evals.py        # 19 cases, 69 assertions
```

For real conversation, put free keys in `.env` (no credit card for either):

| Key | Where | Used for |
|---|---|---|
| `GROQ_API_KEY` | [console.groq.com/keys](https://console.groq.com/keys) | the agent loop |
| `GOOGLE_API_KEY` | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | vision, and text failover |

### Try it

```bash
python -m calorai.cli                       # chat
python -m calorai.cli --user priya          # a different user; sessions are isolated
python -m calorai.cli --no-fast-path        # force every turn through the agent loop
```

In the chat: `img:images/plate.jpg optional caption` sends a photo. `/totals`, `/memory`,
`/history`, `/debug` (per-stage timings for the last turn), `/help`, `/quit`.

---

## Architecture

```
message ──▶ ingest ──▶ [vision] ──▶ agent ◀────────┐   Groq, sees all six tool schemas
              │           │           │            │
       alias lookup,   Gemini,    tool_calls? ──▶ tools (parallel)
       memory load,   PlateAnalysis   │ no
       image detect       │           ▼
              │           │        reply
              │           ▼
              │      ask, if it can't identify the food
              │
              └──▶ fast path ──▶ reply     (totals questions only; opt-out)
                                    │
                                    ▼
                            memory extractor   ← after the reply, off the clock
```

**`agent` is a real tool-calling node**, not a classifier dispatching to handlers. The model
receives all six tool schemas and decides. That distinction is the point: a router is faster, but
it only ever handles the phrasings someone thought to write a rule for.

**`ingest` does the deterministic work the model shouldn't pay for** — resolving "my usual" to
concrete food, loading memory, spotting an image.

**The fast path is an optimisation, not the architecture.** It short-circuits only unambiguous
read-only questions, and `CALORAI_FAST_PATH=0` disables it. The eval suite scores 19/19 with it
**off**, so the agent earns every case on its own.

Worth reading in order: [`graph.py`](src/calorai/graph.py) (the agent),
[`tools.py`](src/calorai/tools.py) (the tool surface), [`memory/`](src/calorai/memory),
[`vision.py`](src/calorai/vision.py), [`db.py`](src/calorai/db.py) (why totals can't drift).

---

## Model choices, and why

Every model id here was **measured, not taken from a docs page** — and measuring changed two of the
three picks. Full write-up in [`docs/RESEARCH.md`](docs/RESEARCH.md).

| Path | Model | Warm | Why |
|---|---|---|---|
| **Text / agent loop** | Groq `openai/gpt-oss-20b` | **230 ms** | The loop is tool calling, so function-calling reliability and throughput are the axes that matter. ~1000 tok/s. |
| **Vision** | Gemini `gemini-2.5-flash-lite` | **429 ms** | A genuinely different model *and* provider. Mature multimodal, generous free tier. |
| **Failover** | Gemini | — | A different provider, so a Groq rate limit costs a model swap rather than a wait. |

Three things I got wrong first and fixed by measuring:

- **I picked `gemini-3.5-flash-lite` from the docs — it's a thinking model.** 8.1s warm, 20s cold,
  and it *rejects* `thinking_budget=0`. Nineteen times slower for a job that is structured
  extraction, not reasoning. The older `2.5-flash-lite` does it in 429ms. **The newest model was
  the wrong model.**
- **`gpt-oss-20b` is also a reasoning model.** At default effort, two-tool turns took 12–20s.
  `reasoning_effort="low"` brought them to ~1s. Choosing between six tools does not need
  deliberation.
- **Cerebras was my planned failover; its free tier is gone** (402 Payment Required on every model
  its key can see). Moved to Gemini.

**Groq is deliberately not used for vision.** Its vision model was Llama 4 Scout, preview-only and
**deprecated in June 2026**. An image path with a retirement date is how a submission breaks
between being written and being read.

### Considered and rejected

| Option | Why not |
|---|---|
| **Self-hosted `transformers`** | Slower than the Ollama path below — raw PyTorch on CPU is well behind llama.cpp's quantised kernels. And the deciding argument: p50 is already **dominated by network round-trip**, so self-hosting swaps a 766 ms network call for a multi-second local one. Wrong direction. |
| **Ollama (local GGUF)** | Ships as a real backend for the zero-key path, but measured infeasible here — see below. |
| **One model for both paths** | Would satisfy nothing the brief asks for, and the right model genuinely differs: the text path is graded on tool-calling reliability, the vision path on image understanding. |
| **`gemini-3.5-flash-lite`** | My first pick. Measured 8.1 s. See above. |
| **Cerebras** | 402 Payment Required — free tier gone. |
| **Groq for vision** | Deprecated vision model. |

### Optimisation techniques considered

I read around agent-latency work before optimising rather than guessing at it. What I adopted, and
what I consciously didn't:

| Technique | Verdict |
|---|---|
| **Inference avoidance** (semantic caching) | **Adopted**, in its cheapest form — the deterministic fast path answers totals questions with zero model calls, and the nutrition cache means a repeat food never costs an estimation call. |
| **Reducing prompt tokens** | **Adopted.** Dropping tool schemas from the reply call, trimming field descriptions, moving `note` out of the schema. 36% off a two-round turn. |
| **Payload reduction** | **Adopted.** Downscaling photos to 768 px cut uploads by 81–94%. |
| **Implicit prefix caching** | **Adopted passively.** Gemini 2.5+ caches repeated prefixes automatically on the free tier, so the static system prompt is deliberately placed *first* in the preamble and the per-user memory block after it, which keeps the longest possible prefix stable across turns. |
| **Streaming** | **Adopted.** Doesn't reduce total time, but TTFT is what a person waiting on a message feels. |
| **Speculative tool calling** — a draft model predicting the next tool so execution overlaps generation | **Deferred.** The literature reports 2–5× on long tool chains, but this agent's turns are 1–2 tool calls, so there's very little sequential bottleneck to hide. It would add a second model call per turn to a system whose binding constraint is tokens per minute. Wrong optimisation for this shape of workload. |
| **Explicit context caching** | **Deferred.** Guarantees the discount but adds a storage meter that needs a paid account, and the literature is clear that cache creation only pays off if the prefix is reused enough. Implicit caching gets most of it for free. |

### Why the vision prompt looks the way it does

I spent ~25 minutes on the food-image estimation literature before writing it, and it changed four
decisions ([`docs/RESEARCH.md`](docs/RESEARCH.md) has citations):

1. **Portion is the error, not identification.** Models name *biryani* reliably; they cannot tell
   200g from 500g, because a photo carries no absolute scale. So `PlateAnalysis` carries **two
   confidences** — `id_confidence` and `portion_confidence` — because they fail independently and
   deserve different questions: *"is that paneer or tofu?"* vs *"small katori or big bowl?"* One
   blended score gives a vague question for both.
2. **Scale comes from dining vessels.** Fiducial markers work but wreck the UX; the marker-less
   alternative is standard vessel sizes. The prompt supplies them (plate ~27cm, katori ~150ml) and
   makes the model **report which ruler it used** — an invisible guess becomes a correctable one.
3. **Locale priors measurably shift portion estimates**, so the vision prompt isn't static: the
   user's diet facts are injected into it. If memory knows they're vegetarian, the white cubes are
   paneer, not chicken. **Memory improves multimodal accuracy, not just conversation.**
4. **Staged reasoning beats one-shot**, but inside a single call — the image path shouldn't pay
   double latency. And **household units, never grams**: it's how people talk, it's how the
   nutrition table is keyed, and grams force a conversion models are measurably bad at.

Thresholds are **asymmetric on purpose**: ID below 0.60 asks, portion below 0.35 logs anyway and
states the assumption. Symmetric thresholds would ask about portions on nearly every photo — the
over-asking failure the brief warns about.

---

## How memory works

Three stores, and **none of them is conversation history**.

| Store | Holds | Written | Retrieved |
|---|---|---|---|
| `profile_facts` | `diet=vegetarian`, `protein_target_g=140` | Background pass, **after the reply ships** | *All of them, every turn* |
| `aliases` | `"my usual"` → a concrete item list | On an explicit statement, or inferred after 3 repeats | Exact phrase match in `ingest`, **before** the model |
| `meals` | The log itself | `log_meal` | The `find_meals` **tool** |

Five positions I'd defend:

**Profile facts need no retrieval logic, because the store is kept small.** Facts are *keyed*, and
a contradiction **supersedes** rather than appends — so the corpus stays bounded at a couple of
dozen one-liners no matter how long someone uses the app. The honest answer to "how do you retrieve
without bloating the prompt" is to make retrieval unnecessary. Measured cost: ~50 tokens a turn.

**No RAG, and that's a decision rather than a gap.** RAG solves *corpus larger than context*. This
memory cannot have that problem by construction. The one genuinely unbounded corpus — meal history
— is already retrieval-on-demand behind a **tool**, which is strictly better here than embeddings:
"same as yesterday" is a date predicate with an exact answer, and a similarity search would be
slower (an embedding call in the hot path) and less correct. When I measured where the tokens
actually go, memory was ~6% and tool schemas were 71%.

**Writes are off the critical path.** The extractor runs on a background thread after the reply is
printed, so memory never appears in p50 or p95.

**Extraction is selective in three tiers, cheapest first.** A regex gate most messages exit having
cost *nothing*; then rules for the patterns carrying the value (diet, allergies, macro targets);
then the model, only for signal-bearing messages the rules didn't resolve. "had 2 rotis" is an
**event**, not a fact about a person, and never reaches the model.
`test_the_gate_runs_before_any_model_call` asserts exactly that — the saving comes from not asking,
not from the model answering "none".

**"same as yesterday" is not memory — it's a database query.** "my usual" is memory. The brief
groups them; separating them is why there's no vector store here.

Forgetting is explicit: superseded facts are marked rather than deleted, so *when did it learn
this* stays answerable; aliases decay after 60 days unused; an explicitly stated "usual" is never
overwritten by an inferred one.

Rendered into the prompt as bounded prose rather than JSON:

```
[what I know about you]
vegetarian · protein target 140g
"my usual" = 2 piece paratha, 1 cup chai
```

---

## Tool design

Six tools, split by **what they do to the data**, not by topic:

| | Tool | Boundary rationale |
|---|---|---|
| write | `log_meal` | INSERT only |
| write | `correct_meal` | UPDATE only — the anti-double-count boundary |
| write | `delete_meal` | Soft delete, reversible |
| read | `get_daily_totals` | Derived sum + progress against a remembered target |
| read | `find_meals` | Powers "same as yesterday" |
| lookup | `lookup_nutrition` | Cache → seed table → model estimate |

**The split that carries the data-correctness requirement is `correct_meal` being separate from
`log_meal`.** A single "record what they said" tool lets the model handle *"actually that was 3
rotis"* by logging three more rotis, and no prompt wording reliably prevents that. Splitting them
makes double-counting **structurally impossible**: `correct_meal` has no INSERT path, `log_meal`
has no UPDATE path.

**`log_meal` resolves nutrition internally** rather than making the agent chain
`lookup_nutrition → log_meal`. That chain costs a full extra round trip on the commonest action in
the product, to make a decision the model has no input into. `lookup_nutrition` stays exposed for
*"how many calories in a samosa?"* — asked without logging.

Two deliberate omissions: **no `remember()` tool** (memory writes are the background extractor; as
a tool it would add a round trip and get over-called) and **no history tool** (history is graph
state, not a retrieval target).

### Where the correctness actually comes from

**There is no stored total anywhere.** `get_daily_totals` is a `SUM` over `meal_items` at query
time. A correction or delete therefore *cannot* desynchronise a counter, because no counter exists.
Deletes are soft, so edits stay auditable, and `edit_log` keeps before/after.

### When it asks, and when it doesn't

Written down rather than left to the model's mood:

- **Log without asking** when the food resolves and a typical portion is inferable.
- **Ask one batched question** only when the food is unidentifiable.
- **Never ask** about grams, cooking oil, or brand — assume and say so.
- **Vague amounts are not a reason to ask.** *"skipped lunch but grazed all afternoon"* logs one
  serving of "assorted snacks" flagged as an estimate. Asking *"what did you graze on?"* is exactly
  the form-filling this product exists to avoid — and it was a real bug I found by reading real
  transcripts.

---

## Latency

Measured on the real stack: Groq `openai/gpt-oss-20b` for text, Gemini `gemini-2.5-flash-lite` for
vision. Reproduce with `python bench/latency.py --n 20 --delay 8`; the raw report lands in
`bench/results/latest.json`.

| path | n | **p50** | **p95** | mean | max | throttled |
|---|---|---|---|---|---|---|
| **text** | 20 | **766 ms** | **1257 ms** | 661 ms | 1289 ms | 0 |
| **image** | — | *not yet measured* | | | | |

Cold start (first call, includes client construction and TLS): **924 ms**.
Stage p50: `agent` 790 ms · `ingest` 0 ms. The fast path served **20%** of turns.

Two things to read off that table. The agent's own overhead — alias resolution, memory load,
routing — is **sub-millisecond**; essentially all of p50 is the model round trip, which is where it
should be. And p95 is only 1.6× p50, so there's no long tail hiding in the loop.

**The image path is not yet measured.** It needs real plate photos in `images/`, which this repo
doesn't ship. The pipeline is exercised end-to-end by the eval suite and by
`tests/test_graph.py::test_photo_plus_caption_produces_exactly_one_meal`, and the vision model
itself measured 429 ms warm, so the expected shape is one vision call plus one text call. I'd
rather leave this blank than publish a number I didn't take.

### What I did to get there

Ordered by how much they actually bought:

1. **`reasoning_effort="low"` on Groq.** The single biggest win. `gpt-oss-20b` is a reasoning
   model, and at default effort two-tool turns measured **12–20s**. Deciding between six tools
   doesn't need deliberation. → ~1s.
2. **Choosing the older Gemini.** `3.5-flash-lite` thinks, and measured 8.1s warm. `2.5-flash-lite`
   does the same extraction in 429 ms.
3. **Dropping tool schemas from the reply call.** Once a terminal write succeeds the turn's work is
   done and only wording remains — but the schemas are 597 of the ~854 fixed tokens. Dropping them
   cut a two-round turn from ~1746 to ~1111 tokens (**36%**). `find_meals` and `get_daily_totals`
   are *not* terminal, so those keep the tools bound.
4. **`max_retries=0` on Groq.** Its default backoff sat on a 429 for **10–18 seconds** before
   giving up. Failing instantly into the Gemini fallback turns a throttled turn from ~18s into ~1s.
   On a messaging surface, a fast answer from the second-choice model beats a slow one from the
   first.
5. **The fast path.** Totals questions answer from SQLite in **2–23 ms** with zero model calls.
6. **Cheap deterministic work before the model.** Alias resolution, memory loading and nutrition
   lookup are local; only a genuinely novel food reaches a model, and that answer is cached.
7. **Memory writes on a background thread**, so they never enter the measurement.
8. **A cached client.** Constructing a chat model per turn discards the connection pool and pays a
   TLS handshake you can see in the numbers.

### What I couldn't fix, honestly

**The binding constraint is not latency, it's tokens per minute.** Groq's free tier caps TPM (~8k)
and this agent spends ~1.1k tokens a turn, so sustained use throttles after a handful of turns.
Gemini's free tier has a daily cap I exhausted while benchmarking. That's why the benchmark has a
`--delay` flag and reports a throttled count: firing 30 turns back-to-back measures the rate
limiter, not the agent. **The p50 above is the agent's real speed at 8s spacing; the throttled count
is the free tier's real ceiling.** Unpaced, expect roughly one throttled turn in three. On a paid
tier this disappears; on a free one it's the dominant fact, and pretending otherwise would make the
numbers a lie.

Token reduction is therefore a *correctness* lever here, not only a cost one — which is why #3 was
worth doing.

### Local inference: measured, and rejected

Ollama ships as a real backend (`CALORAI_TEXT_BACKEND=ollama`) so the repo has a zero-key path. But
on this hardware — i7-1165G7, Intel Iris Xe integrated, no CUDA, 4 cores — a 3B Q4 model generates
~8–15 tok/s with slow prompt eval. An agent turn carries ~1.1k prompt tokens across two hops, which
puts it at tens of seconds per turn, and small-model tool-calling reliability would damage the two
things this project cares most about. Reported rather than quietly dropped.

---

## Testing and evals

```bash
pytest tests/ -q                             # 53 tests
python evals/run_evals.py                    # 19 cases, 69 assertions
python evals/run_evals.py --no-fast-path     # same, with the short-circuit off
python evals/run_evals.py --backend groq     # score a real model
```

Both suites default to the offline mock, so **a clean clone passes with no keys and no network**.

**What "correct" means here.** Each case scores four independent axes, because a reply can read
perfectly while having corrupted the database:

| axis | asserts |
|---|---|
| `tools` | the right tool fired — and the wrong one didn't |
| `db` | calories, **live row count**, and distinct meal count |
| `asks` | asked vs logged, against the stated policy |
| `reply` | the wording mentions what it did |

**The row count is the load-bearing assertion.** A correction handled with `log_meal` instead of
`correct_meal` still produces plausible-looking calories; only the row count catches it. So
`2 rotis → 3 rotis` asserts *exactly one row* and *+105 kcal*, not 315.

All 11 messages from the brief's test conversation set are covered, plus correction, memory,
multi-user and failure traps. [`tests/test_graph.py`](tests/test_graph.py) covers orchestration
specifically: every edge in the graph, the loop bound under a model that never stops calling tools,
cross-user isolation, persistence across a simulated restart, and the failure paths.

### The mock, and what it does and doesn't prove

`MockChatModel` is a rule engine that emits **real tool calls through the real graph into the real
database**. It proves the plumbing and the data correctness. It proves **nothing** about model
behaviour, and it never appears in a latency number.

That distinction was not academic. Running the brief's conversation set against real models found
five bugs the mock could not have surfaced: the two thinking-model latency traps, a fraction parsed
as a count, the agent claiming it had logged a meal it had only looked up, and — my favourite — the
model **parroting the example in my own system prompt verbatim**, reporting *"3 rotis and a chai"*
after correcting rotis. Examples with memorable numbers get copied. The example is gone.

---

## Assumptions and trade-offs

- **Nutrition is a 68-food seed table** in household units, with a cache and a model-estimate
  fallback. The brief says accuracy isn't evaluated, so this trades precision for zero lookup
  latency and offline determinism. South-Asian-weighted because the test set is. An unknown food
  logs as **0 kcal and is flagged**, never a fabricated number — a visible zero is honest, an
  invented 250 silently corrupts the day.
- **A correction reaches back 2 days.** Beyond that the agent should ask which meal rather than
  guess, and a stale row shouldn't get silently rewritten a week later.
- **The tool loop is bounded at 3 rounds** — enough for `find_meals → log_meal`. Past that the
  model is thrashing, and a quick "say that again?" beats a long silence.
- **The fast path trades a small mis-routing risk for a large latency win.** Two guards keep it
  narrow (must open with a totals question *and* must not name a food), and the evals run with it
  off to prove nothing depends on it.
- **SQLite with explicit SQL, not an ORM.** The brief says the agent code is what gets read, and
  explicit SQL reads better than ORM indirection for the two queries that matter.
- **The transcript table exists but is not injected into prompts.** Cross-turn reference ("the
  first one") would need it; I spent the tokens elsewhere. Named as a gap rather than hidden.
- **Free tiers train on inputs.** Fine for a demo, unfit for real user data.

### One place I'd push back on the brief

It groups *"same as yesterday"* with *"my usual"* as memory problems. I think that's wrong, and
built them differently: one is a query against the meal log, the other is learned shorthand.
Treating the first as a memory problem is what leads people to reach for a vector store they don't
need.

---

## Time breakdown

Roughly 8 hours.

| | |
|---|---|
| Research (vision models, portion-estimation literature, free tiers) | 0:45 |
| Scaffold, schema, typed contracts | 0:45 |
| Nutrition resolution + repository + correctness tests | 1:00 |
| Memory: stores, extractor, rendering | 0:50 |
| Agent graph, tools, vision path | 1:15 |
| Mock backend + eval suite | 1:00 |
| **Real-model integration and debugging** | **1:30** |
| Benchmark, orchestration tests, README | 1:00 |

The largest single line is real-model debugging, and that's the honest shape of this work: the code
was written in a few hours and then *earned* over another ninety minutes of reading transcripts and
finding out which assumptions were wrong.

---

## What I'd do next

1. **Measure the image path** with real photos, and add image cases to the latency table.
2. **Streaming.** The architecture supports it (`build_graph(streaming=True)`); the CLI doesn't use
   it yet. Time-to-first-token is what a WhatsApp user actually feels — a 700 ms reply that starts
   at 200 ms feels twice as fast.
3. **Confirm-before-write on low-confidence photos.** A confident-enough plate logs straight away
   today; a one-tap "that right?" would catch vision errors before they enter the totals.
4. **Prompt caching.** The system prompt and tool schemas are identical every call. With a provider
   that supports caching that's most of the fixed cost gone — which on a TPM-limited tier converts
   directly into more usable turns.
5. **Better correction targeting.** `_find_recent_item` matches on normalised name overlap. Right
   for the cases tested, but it would mis-target "the second one", or two similar foods in one
   meal. It needs the transcript.
6. **Alias inference per meal slot** — "my usual" should mean different things at 8am and 8pm. The
   data model supports it; the inference doesn't use it yet.
7. **A larger, adversarial eval set**, scored against real models on a schedule rather than only
   against the mock.

---

## AI tool usage

Built with **Claude Code (Opus)**, and it's worth being specific about where it helped and where it
didn't.

**Genuinely fast:** scaffolding the schema and repository layer, the eval harness, and the test
suites — particularly the orchestration tests, where enumerating every edge of the graph is exactly
the kind of thorough-but-mechanical work that's easy to skimp on by hand.

**Where judgment had to come from outside the model:** the research pass on portion estimation and
the resulting two-confidence design, splitting `correct_meal` from `log_meal`, and the call not to
use RAG. The first draft of this project had a *worse* architecture — it short-circuited the LLM
for most intents, which would have been faster but wouldn't have been an agent. That got caught in
review and rebuilt.

**Where it was actively wrong, and measurement caught it:** every model id picked from
documentation. Two of three were wrong — the newest Gemini was 19× too slow, and Cerebras' free
tier no longer exists. No amount of reasoning would have found that; running it did.

The habit that mattered most was **verifying effects rather than exit codes** — asserting row counts
and summed totals after every correction, not that a call returned without raising. Two silent
failures in this build (a `str.replace` that matched nothing while reporting success, and a regex
gate whose trailing `\b` blocked every stem match) would otherwise have shipped quietly.
