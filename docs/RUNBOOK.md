# Runbook — practice run, then record

Do **Part 1** first, on your own, with nothing recording. Then **Part 2** for the take.

---

# Part 1 — Practice run (~15 min)

## 1.1 Open these, in this order

| # | What | Command |
|---|---|---|
| 1 | Terminal A — the app | `cd D:\calorai-agent` then `.venv\Scripts\activate` |
| 2 | Terminal B — tests | same folder, same activate |
| 3 | Editor (VS Code) | `code D:\calorai-agent` |
| 4 | Browser | http://localhost:8501 (after step 1.3) |

## 1.2 Pre-flight — run every one of these

```bash
cd D:\calorai-agent
.venv\Scripts\activate

pytest tests/ -q                     # expect: 88 passed
python evals/run_evals.py            # expect: 21/21 cases, 77/77 assertions
python bench\_real_e2e.py --delay 7  # expect: p50 under ~1s, throttled=0
```

If `_real_e2e.py` shows throttling, your free-tier quota is tight today — raise `--delay` to 12
and rerun. **Do not record while throttled.**

## 1.3 Start the UI

```bash
streamlit run app.py
```

> **Restart this any time you change code.** Streamlit loads the code once at boot. I demoed a
> stale build this way — units showed "serving" because the server predated the fix.

Open http://localhost:8501. In the sidebar, set **user** to something fresh like `take1` and press
**Enter** (it won't apply until you do). Totals should read 0.

## 1.4 Practise the five turns

Type these in the chat box. Watch the **sidebar**, not the chat — that's where the story is.

| # | Type this | Watch for |
|---|---|---|
| 1 | `had 2 parathas and chai for breakfast` | 430 cal · rows read "2 piece paratha", "1 cup chai" |
| 2 | `2 rotis with dal` | 790 cal · 4 rows |
| 3 | `actually that was 3 rotis not 2` | **895 cal · still 4 rows** · roti row 2 → 3 |
| 4 | `i'm vegetarian and aiming for 140g of protein` | totals **unchanged** · "what it remembers" fills in |
| 5 | `how much protein have I had today?` | answers in **~30 ms**, "fast path, no model call" |

Then the photo:

6. Drag `D:\calorai-agent\images\plate.jpg` onto the uploader → press Enter with no text.
   → logs ~7 dishes as **one meal**.
7. Upload the same photo again, and this time type `half of this was my brother's`.
   → the whole plate halves.

## 1.5 If something goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| "i'm being rate limited" | Groq's free tier caps tokens/minute | wait 60s, carry on. It's honest — and worth mentioning on camera |
| Photo turn fails | Mistral or Gemini quota | try the other: set `CALORAI_VISION_BACKEND=gemini` in `.env`, restart Streamlit |
| Sidebar looks stale | Streamlit caches per user | switch the user field to a new name and press Enter |
| Units say "serving" | you're on old code | restart Streamlit |

---

# Part 2 — Recording (aim 7–9 min)

Recorder: **Snipping Tool** → video icon → New → drag a box round the window. It records silently;
you add voice afterwards. **Increase your terminal font first** (Ctrl+Shift+plus) — this is the
single most common reason demo videos are unwatchable.

Set the user to a fresh name before you start so totals begin at zero.

---

## 0:00 — What it is (~30s) · *browser*

> "CalorAI logs meals from plain text messages. The bet is that it should feel like texting a
> friend rather than filling in a form. I'll show it working, then the three decisions that
> actually carry it, then the latency work — including the parts I couldn't fix."

---

## 0:30 — Demo (~3 min) · *browser*

Run turns 1–7 from §1.4. Let the speed speak; don't narrate every reply.

**On turn 3, the correction — slow down. This is the most important thing in the video:**

> "Watch the sidebar. The total goes 790 to 895 — that's 105 calories, exactly one roti. The roti
> row changes from 2 to 3, and no second row appears. That's not the prompt getting lucky, and
> I'll show you why in a minute."

**On turn 4:**

> "That's a fact about me, not a meal. Nothing was logged — the total didn't move — but it now
> appears under 'what it remembers'."

**On turn 5:**

> "Thirty milliseconds. No model call at all. And it knows my target, because it learnt it one
> message ago."

**On the photo:**

> "Two different models, one meal. The photo goes to a vision model, the conversation runs on a
> different one, and the caption doesn't create a second entry — it halves the first."

---

## 3:30 — Architecture (~2 min) · *editor*

Open **`src/calorai/graph.py`** and show the diagram at the top.

```
message ──▶ ingest ──▶ [vision] ──▶ agent ◀────────┐
              │           │           │            │
       alias lookup,   Pixtral,   tool_calls? ──▶ tools
       memory load,  PlateAnalysis   │ no
       image detect       │          ▼
              │           │        reply
              │           ▼
              │      ask, if it can't identify the food
              └──▶ fast path ──▶ reply
                                    │
                                    ▼
                            memory extractor  ← after the reply, off the clock
```

> "It's a LangGraph tool-calling loop. The model sees all six tool schemas and decides which to
> call — it's genuinely an agent, not a classifier dispatching to handlers. A router would be
> faster, but it would only ever handle the phrasings I thought to write rules for."

> "`ingest` does the deterministic work first — resolving 'my usual' into real food, loading
> memory. The model shouldn't pay for anything a string match can do."

> "The loop is bounded at three rounds. Two is what 'same as yesterday' needs — find the meal, then
> log it. Past three the model is thrashing, and on a messaging surface a quick 'say that again?'
> beats a long silence."

**Now `src/calorai/tools.py` — the decision to dwell on:**

> "Six tools, split by what they do to the data: three writes, two reads, one lookup. The one that
> matters is that `correct_meal` is a *separate tool* from `log_meal`. A single 'record what they
> said' tool lets the model handle 'actually that was 3 rotis' by logging three more, and no prompt
> wording reliably stops that. Split them and `correct_meal` has no INSERT path while `log_meal`
> has no UPDATE path — double-counting becomes structurally impossible rather than something the
> prompt has to remember."

**Then `src/calorai/db.py`:**

> "And there's no stored total anywhere. Totals are a SUM over the items table at query time.
> There's no counter to drift, so corrections and deletes are correct by construction."

---

## 5:30 — Memory (~1.5 min) · *editor, `src/calorai/memory/`*

> "Three stores, and none of them is conversation history. Profile facts, learned aliases, and the
> meal log itself."

**`extractor.py` — how it's written:**

> "Writes happen on a background thread *after* the reply has gone out, so memory never appears in
> a latency number. And extraction is three tiers, cheapest first: a regex gate that most messages
> exit having cost nothing, then rules, then the model. 'Had 2 rotis' is an event, not a fact about
> me — it never reaches the model. There's a test that asserts exactly that."

**`render.py` — how it's read:**

> "All the facts, every turn. That sounds naive until you notice facts are *keyed* and a
> contradiction supersedes rather than appends — so the store stays at a couple of dozen one-liners
> no matter how long you use it. The honest answer to 'how do you retrieve without bloating the
> prompt' is to make retrieval unnecessary."

> "That's also why there's no RAG. RAG solves corpus-bigger-than-context, and this can't have that
> problem. The one genuinely unbounded corpus — meal history — is already retrieval-on-demand
> behind the `find_meals` tool, which beats embeddings here because 'same as yesterday' is a date
> predicate with an exact answer. When I measured where tokens actually go, memory was about 6%
> and tool schemas were 71%."

**Say this — it's a deliberate disagreement:**

> "The brief groups 'same as yesterday' with 'my usual' as memory problems. I think that's wrong.
> One is a database query, the other is learned shorthand, and I built them differently. Treating
> the first as a memory problem is what pushes people toward a vector store they don't need."

**And `render_vision_priors` — the bit that ties it together:**

> "Diet facts get injected into the *vision* prompt. If it knows I'm vegetarian, white cubes come
> back as paneer rather than chicken. Memory improving multimodal accuracy, not just conversation."

---

## 7:00 — Multimodal (~1 min) · *editor, `src/calorai/vision.py`*

> "I read the food-image estimation literature before writing this prompt, and it changed the
> design. The finding that mattered: portion is the error, not identification. Models name biryani
> fine — they can't tell 200 grams from 500, because a photo carries no absolute scale."

> "So confidence is two fields, not one. `id_confidence` and `portion_confidence` fail
> independently and deserve different questions — 'is that paneer or tofu' versus 'small katori or
> big bowl'. And the thresholds are deliberately asymmetric: low identification asks, low portion
> logs anyway and says what it assumed. Gating both the same way would make it ask about portions
> on nearly every photo, which is the over-asking failure the brief warns about."

> "The prompt also makes the model name its ruler — the plate, the katori — so the guess is
> auditable instead of invisible."

---

## 8:00 — Latency and testing (~1.5 min) · *Terminal B*

```bash
pytest tests/ -q
python evals/run_evals.py
```

> "88 tests and 19 eval cases, and both run on a clean clone with no API keys, because there's a
> deterministic offline backend. Eleven of those eval cases come straight from the brief's test
> conversation set — you can see them listed."

> "Each case scores four independent axes, and the load-bearing one is the live row count. A
> correction done with the wrong tool still produces plausible-looking calories — only the row
> count catches it."

Then the numbers:

| | p50 | p95 |
|---|---|---|
| text | 766 ms | 1257 ms |
| image | 6896 ms | 11259 ms |

The four things worth saying:

1. > "Biggest win was `reasoning_effort=low`. gpt-oss is a reasoning model, and two-tool turns were
   > taking twelve to twenty seconds. Choosing between six tools doesn't need deliberation."
2. > "I picked the newest Gemini from the docs, then measured it: eight seconds, because it's a
   > thinking model. The *older* 2.5-flash-lite does the same job in 429 milliseconds. The newest
   > model was the wrong model."
3. > "Both clients hid rate limits behind retry backoff. Gemini answered a throttled image in 228
   > seconds. Setting retries to zero turns that into a one-second failover to the other provider."
4. > "And the honest part: the binding constraint isn't latency, it's tokens per minute. That's why
   > the benchmark paces requests and reports a throttled count — firing thirty turns back to back
   > would be measuring the rate limiter, not the agent. The image p95 of eleven seconds is a
   > free-tier artifact, and I left it in the README with the explanation rather than dropping the
   > samples that made it ugly."

---

## 9:00 — Close (~30s)

> "What I'd do next: get the image path under three seconds — the vision call is six of the seven,
> and I've only taken the easy win of downscaling. Confirm-before-write on low-confidence photos.
> And prompt caching, which on a token-per-minute-limited tier converts directly into more usable
> turns."

> "The thing I'd most want to flag is that the mock backend proves plumbing, not model behaviour.
> Running the real models found bugs it couldn't have — including the model parroting an example
> from my own system prompt back at me, and a portion getting halved twice because I had
> belt-and-braces logic that wasn't."

---

## Model architecture — the one-paragraph version

If you only get one sentence out about models:

> "Two paths, two different models, and every id was measured rather than read off a docs page.
> The conversation runs on Groq's gpt-oss-20b, because the agent loop is tool calling and that's
> what it's good at — 230 milliseconds warm. Photos go to Mistral's Pixtral, a different model on a
> different provider. I picked that one by running it head to head against Gemini on the same
> plate: quality was a wash, both named their scale reference and both offered alternatives, but
> Gemini's free tier was exhausted by ten benchmark photos. A model you can't call isn't a fast
> model. Gemini stays wired in as failover for both paths."

| Path | Model | Provider | Warm |
|---|---|---|---|
| conversation / tools | `openai/gpt-oss-20b` | Groq | 230 ms |
| vision | `pixtral-12b-2409` | Mistral | ~5.7 s |
| failover, both paths | `gemini-2.5-flash-lite` | Google | ~5.0 s |
| offline / tests | rule-based mock | — | µs |

---

## Checklist before you stop recording

- [ ] Correction shown, total moving by exactly one roti
- [ ] Photo shown, **and** photo + caption halving the plate
- [ ] Memory written *and* retrieved (turns 4 and 5)
- [ ] `correct_meal` vs `log_meal` explained as *structural*
- [ ] "no RAG, and here's why" said out loud
- [ ] Latency numbers stated, **with one thing you couldn't fix**
- [ ] Tests and evals run on camera
- [ ] Bonuses named: evals, streaming, multi-user isolation, offline mode, Streamlit UI
