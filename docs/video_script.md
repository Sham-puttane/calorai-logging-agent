# Walkthrough script (5–10 min)

Cue cards, not a read-aloud. The brief asks for: a working demo including **one image case** and
**one correction case**, architectural decisions, how memory is written and retrieved, the latency
work and its trade-offs, challenges, and any bonus features.

Before recording:

```bash
cd D:\calorai-agent
del calorai.db                    # start clean so the demo is reproducible
.venv\Scripts\activate
python -m calorai.cli --user demo
```

Keep a second terminal open with `pytest tests/ -q` and `python evals/run_evals.py` ready.

---

## 0:00 — What it is (30s)

> "CalorAI logs meals from plain text messages. The whole bet is that it should feel like texting a
> friend, not filling in a form. I'll show it working, then the three decisions I think actually
> matter, then the latency work."

---

## 0:30 — Demo (3 min)

Type these in order. Don't narrate every reply — let the speed show.

```
had 2 parathas and chai for breakfast
```
> "Normal case. Note it didn't ask me anything — it assumed a home portion and told me what it
> assumed, so I can correct it in three words if it's wrong."

```
leftover biryani, maybe two thirds of the box
```
> "Two thirds is 0.67 of one item, not two servings. That was a real bug I only found by running
> real models."

```
skipped lunch but grazed all afternoon
```
> "This is the one people get wrong. There's no right answer to 'how much did you graze'. Asking
> would be exactly the form-filling we're trying to avoid — so it logs an estimate and flags it."

**The correction case — the one the brief calls out:**
```
2 rotis with dal
actually that was 3 rotis not 2
```
> "Watch the total: it moves by one roti, not three. And this is structural, not prompt luck —
> I'll show why in a second."

```
i'm vegetarian btw
how much protein have I had today?
```
> "That protein answer came back in about 20 milliseconds with no model call at all."

**The image case:**
```
img:images/plate.jpg
img:images/plate.jpg half of this was my brother's
```
> "Two models, one meal. The caption doesn't create a second entry — it halves the first."

Then `/memory` to show what it retained, and `/debug` for per-stage timings.

---

## 3:30 — Architecture (1.5 min)

Open `src/calorai/graph.py`.

> "It's a LangGraph tool-calling loop. The model sees all six tool schemas and decides — it's not
> a classifier dispatching to handlers. A router would be faster but would only handle the
> phrasings I thought to write rules for."

> "`ingest` does the cheap deterministic work first — resolving 'my usual' to real food, loading
> memory. The model shouldn't pay for things a string match can do."

**The decision to dwell on** — open `src/calorai/tools.py`:

> "`correct_meal` is a *separate tool* from `log_meal`. That's the whole answer to 'totals must not
> double-count'. A single 'record what they said' tool lets the model handle 'actually that was 3
> rotis' by logging three more, and no prompt wording reliably stops that. Split, `correct_meal`
> has no INSERT path and `log_meal` has no UPDATE path — it's structurally impossible."

Then `src/calorai/db.py`:

> "And there's no stored total anywhere. Totals are a SUM at query time. There's no counter to
> drift, so corrections and deletes are correct by construction rather than by care."

---

## 5:00 — Memory (1.5 min)

Open `src/calorai/memory/`.

> "Three stores, and none of them is conversation history. Profile facts, learned aliases, and the
> meal log itself."

**Written:** `extractor.py`
> "Writes happen on a background thread *after* the reply is sent, so memory never shows up in the
> latency numbers. And extraction is three tiers — a regex gate that most messages exit having cost
> nothing, then rules, then the model. 'Had 2 rotis' is an event, not a fact about me, and it never
> reaches the model. There's a test asserting exactly that."

**Retrieved:** `render.py`
> "All the facts, every turn. That sounds naive until you notice facts are *keyed* and a
> contradiction supersedes rather than appends — so the store is bounded at a couple of dozen
> one-liners forever. The honest answer to 'how do you retrieve without bloating the prompt' is to
> make retrieval unnecessary."

> "That's also why there's no RAG. RAG solves corpus-bigger-than-context; this can't have that
> problem. The one unbounded corpus, meal history, is already behind the `find_meals` tool — which
> is better than embeddings, because 'same as yesterday' is a date predicate with an exact answer."

**Say this** — it's a deliberate disagreement:
> "The brief groups 'same as yesterday' with 'my usual' as memory problems. I think that's wrong.
> One is a database query, the other is learned shorthand. I built them differently."

**And the bit that ties memory to vision** — `render_vision_priors`:
> "Diet facts get injected into the *vision* prompt. If it knows I'm vegetarian, white cubes come
> back as paneer, not chicken. Memory improving multimodal accuracy, not just conversation."

---

## 6:30 — Multimodal (1 min)

Open `src/calorai/vision.py`.

> "I read the food-image estimation literature before writing this prompt, and it changed the
> design. The finding that mattered: portion is the error, not identification. Models name biryani
> fine; they can't tell 200g from 500g, because a photo has no absolute scale."

> "So confidence is two fields, not one — `id_confidence` and `portion_confidence`. They fail
> independently and deserve different questions: 'is that paneer or tofu' versus 'small katori or
> big bowl'. And the thresholds are asymmetric — low ID asks, low portion logs anyway and says what
> it assumed, because asking about portion on every photo is the over-asking failure."

> "The prompt also makes the model name its ruler — the plate, the katori — so the guess is
> auditable."

Show `images/ambiguous_plate.jpg` if it triggers the question branch.

---

## 7:30 — Latency (1.5 min)

Show `bench/results/latest.json` or re-run the benchmark.

> "Text p50 766ms, p95 1257ms. Image p50 around 4 seconds, dominated by the vision call."

The four things worth saying:

1. > "Biggest win was `reasoning_effort=low`. gpt-oss is a reasoning model and two-tool turns were
   > taking 12 to 20 seconds. Choosing between six tools doesn't need deliberation."
2. > "I originally picked the newest Gemini from the docs. Measured it: 8 seconds, because it's a
   > thinking model. The *older* 2.5-flash-lite does the same job in 429ms. The newest model was
   > the wrong model."
3. > "Both clients hid rate limits behind retry backoff. Gemini answered a throttled image in 228
   > seconds. Setting max_retries to zero turns that into a one-second failover."
4. > "And the honest part: the binding constraint isn't latency, it's tokens per minute. Groq's
   > free tier caps TPM and this agent spends about 1.1k tokens a turn. That's why the benchmark
   > paces requests and reports a throttled count — firing 30 turns back to back would measure the
   > rate limiter, not the agent."

---

## 9:00 — Testing, and what I'd change (1 min)

Run in the second terminal:
```
pytest tests/ -q
python evals/run_evals.py
```

> "53 tests, 19 eval cases, and both run on a clean clone with no API keys because there's a
> deterministic offline backend. The eval scores four axes per case, and the load-bearing one is
> the live row count — a correction done with the wrong tool still produces plausible calories, and
> only the row count catches it."

> "But the mock proves plumbing, not model behaviour. Running the real models found five bugs it
> couldn't have — including the model parroting an example from my own system prompt verbatim, and
> a portion getting halved twice because I had belt-and-braces logic that wasn't."

> "With more time: measure the image path with more samples, confirm-before-write on low-confidence
> photos, and prompt caching — which on a TPM-limited tier converts directly into more usable
> turns."

---

## Checklist

- [ ] Correction case shown, with the total moving by the right amount
- [ ] Image case shown, including caption → one meal
- [ ] Memory write *and* retrieve explained
- [ ] Latency numbers stated, with one thing you couldn't fix
- [ ] Tests/evals run on camera
- [ ] Bonus features named: evals, streaming, multi-user isolation, offline mode
