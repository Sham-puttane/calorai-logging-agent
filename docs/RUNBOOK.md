# Recording runbook

Do **Part 1** with nothing recording. Then **Part 2** is the take.

**Board:** https://claude.ai/code/artifact/3be0699f-b1bf-4ea4-9ef4-a67624675aea
One page carrying the graph, the decisions, the measured model choices and the bugs. Keep it in a
second tab — it covers the architecture and latency sections so you are not reading code aloud.

---

# Part 1 — Dry run

## What to open

| | |
|---|---|
| Terminal A | `cd D:\calorai-agent` → `.venv\Scripts\activate` → `streamlit run app.py` |
| Terminal B | same folder, activated, for `pytest` and evals |
| Browser tab 1 | http://localhost:8501 |
| Browser tab 2 | the board |
| Editor | `code D:\calorai-agent` |

**Bump the terminal font** (`Ctrl+Shift+plus`). Unreadable text is the commonest reason these
videos fail.

## Pre-flight — all four

```bash
pytest tests/ -q                  # 177 passed
python evals/run_evals.py         # 21/21 cases, 77/77 assertions
python bench\_real_e2e.py --delay 3
del calorai.db                    # start from zero
```

If `_real_e2e.py` prints `rate limited`, **fix that before recording** — see *Providers* below.

Start Streamlit, set the user field to something fresh like `take1`, and press **Enter** to apply.
Streamlit loads code once at boot, so **restart it after any change**.

## The exact sequence

Type these in order. Watch the **sidebar**, not the chat.

| # | Type | Watch for |
|---|---|---|
| 1 | `had 2 parathas and chai for breakfast` | 430 cal · rows read `2 piece paratha`, `1 cup chai` |
| 2 | `leftover biryani, maybe two thirds of the box` | **0.67**, not 2 servings · ~161 cal |
| 3 | `skipped lunch but grazed all afternoon` | logs an estimate, **asks nothing** |
| 4 | `2 rotis with dal` | 4 rows |
| 5 | `actually that was 3 rotis not 2` | **+105 cal, still 4 rows**, roti goes 2 → 3 |
| 6 | `i'm vegetarian and aiming for 140g of protein` | totals **do not move** · memory panel fills |
| 7 | `how much protein have I had today?` | ~30 ms · "fast path, no model call" · knows the 140 g target |
| 8 | upload `images/plate.jpg`, click **send photo** | preview first · **nothing logged** · it asks |
| 9 | `yes but it was 1 naan not 4` | logs the corrected list, not what it guessed |
| 10 | `remember this as my usual dinner` | no tool call · alias appears in the sidebar |
| 11 | `my usual` | logs that dinner back |

Then click **seed yesterday's meals** and send `same as yesterday` — it should find and log the
idli and sambar.

## Persistent memory — in the CLI, not the UI

Killing the process is the only proof that cannot be faked by history sitting in a variable.

```bash
python -m calorai.cli --user memtest
```
```
i'm vegetarian and aiming for 140g of protein
1 naan, 1 katori paneer and 1 cup daal for dinner
remember this dinner as my usual
/memory          → diet, target, and the whole dinner
/quit            ← actually exit the process
```
```bash
python -m calorai.cli --user memtest
```
```
/memory          → still there, from SQLite
my usual         → logs the dinner learned last session
/history         → meals live here; /memory holds only facts and the alias
```

## Tracing — worth 20 seconds on camera

The CLI banner and the sidebar both say whether tracing is really on:
`tracing on -> project 'calorai-agent'`. Open the public trace after the correction turn:

https://smith.langchain.com/public/de49189b-4278-407d-a884-6d296f55a787/r

> "That's the correction turn. You can see the agent node decide, then `correct_meal` fire — and
> `log_meal` not fire. The tool boundary that makes double counting impossible is a tree you can
> look at, not a claim I'm making."

## Multi-user, in two clicks

Change the sidebar user to any new name: zero calories, empty memory. Change it back: everything
returns. `user_id` is on every table and every query, and tools are built per session, so a tool
physically cannot address another user's rows.

## Providers — if you hit rate limits

```bash
CALORAI_TEXT_BACKEND=openrouter     # or groq | mistral
CALORAI_VISION_BACKEND=mistral-vision
```

| Symptom | Do this |
|---|---|
| `i'm being rate limited` on text | switch `CALORAI_TEXT_BACKEND` to `mistral`, restart Streamlit |
| `rate limited on photos` | wait ~30 s — Pixtral limits per second, not per day |
| Everything throttled | Groq is at its 200k tokens/day cap; use `openrouter` |

---

# Part 2 — The take (aim 7–9 min)

Record with **Snipping Tool** → video icon → drag a box round the window. Silent, so you add voice
afterwards.

## 0:00 — What it is · *the UI*

> "CalorAI logs meals from plain text messages. The bet is that it should feel like texting a
> friend rather than filling in a form. I'll show it working, then the decisions that actually
> carry it, then the latency work — including what I couldn't fix."

## 0:30 — Demo · *steps 1–11*

Let the speed talk. Four moments to slow down on.

**Step 5, the correction — the most important thing in the video:**
> "Watch the sidebar. 790 to 895 — that's 105 calories, exactly one roti. The roti row goes two to
> three, and no second row appears. That isn't the prompt getting lucky, and I'll show you why in a
> minute."

**Step 6:**
> "That's a fact about me, not a meal. The total didn't move — but it's now under what it
> remembers."

**Step 7:**
> "Thirty milliseconds, no model call at all. And it knows my target because I mentioned it one
> message ago."

**Step 8:**
> "It hasn't logged anything yet. A photo is the one input where you hand the whole description to
> a model, and models get plates wrong in ways you see instantly and the agent can't see at all. So
> it shows you what it found and waits."

## 3:30 — Architecture · *the board, then the editor*

Pan the **graph**.

> "It's a LangGraph tool-calling loop. The model sees all six tool schemas and decides — it's
> genuinely an agent, not a classifier dispatching to handlers. A router would be faster but would
> only handle the phrasings I thought to write rules for."

> "Ingest does the cheap deterministic work first. The two places the model is deliberately absent
> are the fast path and the memory write — and the memory write happens after you already have your
> reply, which is why memory costs nothing in p50."

**`src/calorai/tools.py`** — the one to dwell on:
> "`correct_meal` is a *separate tool* from `log_meal`. One 'record what they said' tool lets the
> model answer 'actually that was 3 rotis' by logging three more, and no prompt wording reliably
> stops that. Split, `correct_meal` has no INSERT path and `log_meal` has no UPDATE path — double
> counting becomes structurally impossible."

**`src/calorai/db.py`**:
> "And there's no stored total anywhere. Totals are a SUM at query time. There's no counter to
> drift."

## 5:30 — Memory · *`src/calorai/memory/`*

> "Three stores, and none of them is conversation history."

`extractor.py`:
> "Writes happen on a background thread after the reply goes out. Extraction is three tiers — a
> regex gate most messages exit having cost nothing, then rules, then the model. 'Had 2 rotis' is an
> event, not a fact about me, and it never reaches the model. There's a test asserting exactly
> that."

`render.py`:
> "All the facts, every turn. That sounds naive until you notice facts are *keyed* and a
> contradiction supersedes rather than appends — so the store stays at a couple of dozen one-liners
> forever. The honest answer to 'how do you retrieve without bloating the prompt' is to make
> retrieval unnecessary."

> "That's also why there's no RAG. It solves corpus-bigger-than-context, and this can't have that
> problem. When I measured where the tokens go, memory was about 6% and tool schemas were 71%."

**Say this — a deliberate disagreement:**
> "The brief groups 'same as yesterday' with 'my usual' as memory problems. I don't think they're
> the same thing. One's a date predicate with an exact answer; the other is learned shorthand. I
> built them differently."

`render_vision_priors`:
> "Diet facts go into the *vision* prompt too. If it knows I'm vegetarian, white cubes come back as
> paneer rather than chicken. Memory improving multimodal accuracy, not just conversation."

## 7:00 — Multimodal · *`src/calorai/vision.py`*

> "I read the food-image estimation literature before writing this prompt. The finding that
> mattered: portion is the error, not identification. Models name biryani fine — they can't tell
> 200 grams from 500, because a photo carries no absolute scale."

> "So confidence is two fields, not one. They fail independently and deserve different questions —
> 'is that paneer or tofu' versus 'small katori or big bowl'. And the thresholds are deliberately
> asymmetric: low identification asks, low portion logs anyway and says what it assumed. Gating both
> the same way would make it ask about portions on nearly every photo."

## 8:00 — Testing and latency · *Terminal B, then the board*

```bash
pytest tests/ -q
python evals/run_evals.py
```

> "177 tests and 21 eval cases, and both run on a clean clone with no API keys, because there's a
> deterministic offline backend. Eleven of those come straight from your test set — you can see
> them named as they pass."

> "Each case scores four axes, and the load-bearing one is the live row count. A correction done
> with the wrong tool still produces plausible-looking calories. Only the row count catches it."

Switch to the board's **latency** and **bugs** sections:

> "Text p50 766 milliseconds, p95 1257. The agent's own overhead is sub-millisecond — basically all
> of it is the model round trip."

> "Biggest win was `reasoning_effort=low`: gpt-oss is a reasoning model and two-tool turns were
> taking twelve to twenty seconds."

> "I also picked the newest Gemini from the docs, then measured it — eight seconds, because it's a
> thinking model. The older one does the same job in 429 milliseconds. The newest model was the
> wrong model."

> "And the honest part: the binding constraint isn't latency, it's tokens per day. That's why the
> benchmark paces requests and reports a throttled count — firing thirty turns back to back
> measures the rate limiter, not the agent."

## 9:00 — Close · *the board's bugs section*

> "The mock proves plumbing, not model behaviour. Running the real models found things it
> structurally couldn't — including the model parroting an example from my own system prompt back
> at me, and a portion getting halved twice because I had belt-and-braces logic that wasn't."

> "With more time: get the image path under three seconds, and prompt caching, which on a
> token-limited tier converts directly into more usable turns."

---

## Checklist

- [ ] Correction shown — total moving by exactly one roti, row count unchanged
- [ ] Photo shown, **including** the confirmation step and correcting it before it writes
- [ ] Memory written *and* retrieved
- [ ] `correct_meal` vs `log_meal` explained as **structural**
- [ ] "no RAG, and here's why" said out loud
- [ ] Latency stated, **with one thing you couldn't fix**
- [ ] Tests and evals run on camera
- [ ] Bonuses named: evals, streaming, multi-user isolation, offline mode, Streamlit UI
