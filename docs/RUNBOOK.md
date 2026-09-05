# Recording runbook

**Part 1** is the dry run — nothing recording. **Part 2** is the take, on a 10-minute clock.

Three surfaces, and each is in the video for a reason. Don't demo the same thing twice:

| Surface | What only it can prove |
|---|---|
| **Streamlit UI** | The product. Totals moving in the sidebar as you type, photo confirmation, the memory panel filling. |
| **CLI** | That the state is *real* — a different process, a different interface, same user, everything still there. Plus `/debug`, which shows the tool calls and per-stage timings the UI hides. |
| **Anatomy board** | The reasoning. The graph, the measured model choices, the latency work, the bugs — so you are not reading code aloud. |

**Board:** https://claude.ai/code/artifact/3be0699f-b1bf-4ea4-9ef4-a67624675aea

---

# Part 1 — Dry run

## What to open

| | |
|---|---|
| Terminal A | `cd D:\calorai-agent` → `.venv\Scripts\activate` → `streamlit run app.py` |
| Terminal B | same folder, activated — for the CLI, then tests |
| Browser tab 1 | http://localhost:8501 |
| Browser tab 2 | the board |
| Browser tab 3 | the [LangSmith trace](https://smith.langchain.com/public/de49189b-4278-407d-a884-6d296f55a787/r) |
| Editor | `code D:\calorai-agent` |

**Bump the terminal font** (`Ctrl+Shift+plus`). Unreadable text is the commonest reason these videos
fail. Put the eleven demo lines in a scratch file to paste — typing them live costs you 90 seconds
you do not have.

## Pre-flight — all four, in order

```bash
pytest tests/ -q                  # 178 passed
python evals/run_evals.py         # 21/21 cases, 77/77 assertions
python bench/_real_e2e.py --delay 5
del calorai.db                    # start from zero
```

The third one is the one that matters. **Read the `db:` line under every turn, not the replies.** A
reply that sounds right above an unchanged `db:` line is the failure worth catching now rather than
on camera — it is exactly how the lying failover model was found.

Then start Streamlit, set the user field to `take1`, press **Enter** to apply. Streamlit loads code
once at boot, so **restart it after any change**.

## Resetting between takes

You will run the sequence more than once. Pick the smallest reset that does the job:

| Want | Do | Keeps |
|---|---|---|
| A completely fresh person | type a new name in the sidebar **user** field | everything, under the old name |
| Wipe only what it remembers | sidebar **clear what it remembers**, or `/forget` in the CLI | the meals — which is the point: two stores, wipe one |
| Wipe only today's food | sidebar **clear this user's day** | the memory |
| Start the whole database over | `del calorai.db` | nothing, for every user |

**Changing the user name is the fastest and safest**, and it doubles as the isolation demo. Reach for
`del calorai.db` only in pre-flight.

## What the seed button does

**seed yesterday's meals** inserts 3 idli and 1 katori sambar dated *yesterday*, nothing else. It
exists because `same as yesterday` on a brand-new user correctly answers "nothing logged", which
reads as a bug on camera.

It deliberately does **not** plant a `my usual`. An alias the user never taught is a memory they did
not create, and showing it as though the agent learned something would be a lie — teaching it live
with `remember this as my usual` is the more convincing demonstration anyway. Worth one sentence out
loud if you press it on camera: *"that's seeding yesterday's food so there's a yesterday to refer
to — it doesn't seed any memory."*

## Rehearse the whole thing once, timed

Run Part 2 end to end with a stopwatch and nothing recording. You are looking for two numbers: total
length, and how long the photo turn takes — it is ~6 s and it will feel like thirty.

## Providers — if you hit rate limits

```bash
CALORAI_TEXT_BACKEND=openrouter     # or groq | mistral
CALORAI_VISION_BACKEND=mistral-vision
CALORAI_VISION_FALLBACK=none        # Gemini's daily quota is spent; leaving it
                                    # set pays for a dead attempt before Pixtral
```

**Check the model, not just the provider, before you switch.** `OPENROUTER_MODEL` must be
`inclusionai/ling-3.0-flash-fin:free`. On `llama-3.3-70b-instruct` the agent confirms meals it never
wrote — it is fast and it lies, which on camera looks like the product working right up until you
check the sidebar.

| Symptom | Do this |
|---|---|
| `i'm being rate limited` on text | switch `CALORAI_TEXT_BACKEND` to `openrouter`, restart Streamlit |
| `rate limited on photos` | wait ~30 s — Pixtral limits per second, not per day |
| Everything throttled | all three text providers are at their daily caps. `CALORAI_TEXT_BACKEND=mock` still demonstrates the graph, tools, memory and totals end to end with no network — say so on camera rather than fighting it |

After any switch, re-run `python bench/_real_e2e.py --delay 5` and read the `db:` lines. Do not
trust one good-looking reply.

---

# Part 2 — The take

**Hard cap 10:00. Target 9:45.** Record with **Snipping Tool** → video icon → drag a box round the
window. Silent, so you add voice afterwards.

| Clock | Surface | Segment |
|---|---|---|
| 0:00 | UI | What it is |
| 0:35 | UI | **Demo — the eleven steps** |
| 3:45 | CLI | **Persistence, tool calls, isolation** |
| 5:00 | Board + editor | Architecture, memory, multimodal |
| 7:45 | Terminal + board | Tests, evals, latency |
| 8:45 | Board | The bug, and close |

If you run behind, each segment names **what to cut first**. Cut it without hesitating — going long
is worse than dropping a point.

---

## 0:00 — What it is · *Streamlit, sidebar visible*

**Say:**
> "CalorAI logs meals from plain text messages. The bet is that it should feel like texting a friend
> rather than filling in a form. I'll show it working, then prove the state is real in a second
> process, then walk the decisions that actually carry it — including the bug I found last, which is
> the most interesting thing here."

---

## 0:35 — SURFACE 1: the UI · *the eleven steps*

**Test:** paste these in order. **Watch the sidebar, not the chat** — say that out loud once.

| # | Type | Watch for |
|---|---|---|
| 1 | `had 2 parathas and chai for breakfast` | 430 cal · rows read `2 piece paratha`, `1 cup chai` |
| 2 | `leftover biryani, maybe two thirds of the box` | **0.67**, not 2 servings · ~161 cal |
| 3 | `skipped lunch but grazed all afternoon` | logs an estimate, **asks nothing** |
| 4 | `2 rotis with dal` | 4 rows |
| 5 | `actually that was 3 rotis not 2` | **+105 cal, still 4 rows**, roti goes 2 → 3 |
| 6 | `i'm vegetarian and aiming for 140g of protein` | totals **do not move** · memory panel fills |
| 7 | `how much protein have I had today?` | ~30 ms · knows the 140 g target |
| 8 | upload `images/plate.jpg` → **send photo** | preview first · **nothing logged** · it asks |
| 9 | `yes but it was 1 naan not 4` | logs the corrected list, not what it guessed |
| 10 | `remember this as my usual dinner` | no tool call · alias appears in the sidebar **on the next interaction** — the write lands on a background thread, so don't stare at it waiting |
| 11 | `my usual` | logs that dinner back |

Let 1–4 run fast and quiet. **Four moments to slow down on:**

**Step 5 — the most important ten seconds in the video:**
> "Watch the sidebar. 790 to 895 — 105 calories, exactly one roti. The roti row goes two to three,
> and no second row appears. That is not the prompt getting lucky, and I'll show you why in a
> minute."

**Step 6:**
> "That's a fact about me, not a meal. The total didn't move — but it's now under what it
> remembers."

**Step 7:**
> "Thirty milliseconds, and no model call at all. And it knows my target because I mentioned it one
> message ago."

**Step 8 — the photo:**
> "It hasn't logged anything yet. A photo is the one input where you hand the whole description to a
> model, and models get plates wrong in ways you see instantly and the agent can't see at all. So it
> shows you what it found, and waits."

Then step 9, without pausing:
> "And I correct it before it ever writes. What gets logged is one naan, not the four it guessed."

**Cut first:** steps 10 and 11 — the alias gets proved again in the CLI segment, better.

---

## 3:45 — SURFACE 2: the CLI · *Terminal B*

This is the segment that proves the memory claim, and it takes about 75 seconds. **Use the same user
as the UI** — that is the whole trick.

**Test:**

```bash
python -m calorai.cli --user take1
```
```
/memory
```

*(Run `/memory` first, as written. It warms the process, so the timings you point at later are real
steady-state numbers rather than a cold first turn — which reads ~160 ms and undersells the point.)*

**Say — while the memory block is on screen:**
> "Different process. Different interface. Same user. Everything I taught the web app is here — the
> diet, the protein target, the learned shorthand — because memory lives in SQLite, not in a session
> variable. And none of this is conversation history. These are three purpose-built stores."

```
my usual
```
> "And the shorthand it learned two minutes ago in the browser resolves here."

```
/debug
```

**Say — this is why the CLI is in the video at all:**
> "This is the part the UI hides: per-turn stage timings, and the actual tool calls. Read the ratio
> — ingest is a few milliseconds, and that's alias resolution, memory load and routing all together.
> Essentially the entire turn is the model round trip, which is where it should be."

*(Say the numbers you can see, not memorised ones. On the real backend `ingest` reads well under a
millisecond against an `agent` of several hundred; on the mock it's a few ms against a few ms.)*

```
how much protein have I had today?
/debug
```
> "And that one: `tools` empty, `fastpath True`, about ten milliseconds. A totals question is a SQL
> query — there's no reason to pay a model for it."

Then isolation, in two commands:

```
/quit
python -m calorai.cli --user someone_else
```
```
/memory
```
> "New user, empty. `user_id` is on every table and every query, and tools are constructed per
> session, so a tool physically cannot address another user's rows."

**Cut first:** the `someone_else` isolation check — it's a bonus, not a core requirement.

---

## 5:00 — SURFACE 3: the board · *architecture, memory, multimodal*

Switch to the board tab. Pan the **graph**. You are talking over a diagram now, not reading code —
drop into the editor only for the two files named below.

> "It's a LangGraph tool-calling loop. The model sees all six tool schemas and decides — genuinely
> an agent, not a classifier dispatching to handlers. A router would be faster, but would only
> handle the phrasings I thought to write rules for."

> "Ingest does the cheap deterministic work first. The two places the model is deliberately absent
> are the fast path and the memory write — and the memory write happens after you already have your
> reply, which is why memory costs nothing in p50."

**Editor · `src/calorai/tools.py`** — the one file to dwell on:
> "`correct_meal` is a *separate tool* from `log_meal`. A single 'record what they said' tool lets
> the model answer 'actually that was 3 rotis' by logging three more, and no prompt wording reliably
> stops that. Split, `correct_meal` has no INSERT path and `log_meal` has no UPDATE path — double
> counting becomes structurally impossible. That's what you were watching in the sidebar earlier."

**Editor · `src/calorai/db.py`:**
> "And there's no stored total anywhere. Totals are a SUM at query time. There's no counter to
> drift."

**Back to the board — memory:**
> "Three stores, and none of them is conversation history. Writes happen on a background thread
> after the reply goes out, and extraction is three tiers — a regex gate most messages exit having
> cost nothing, then rules, then the model. 'Had 2 rotis' is an event, not a fact about me, and it
> never reaches the model."

> "Retrieval is: all the facts, every turn. That sounds naive until you notice facts are *keyed*, so
> a contradiction supersedes rather than appends, and the store stays at a couple of dozen
> one-liners forever. The honest answer to 'how do you retrieve without bloating the prompt' is to
> make retrieval unnecessary. That's also why there's no RAG — it solves corpus-bigger-than-context,
> and this can't have that problem. I measured it: memory is about 6% of tokens; tool schemas were
> 71%."

**Say this — a deliberate disagreement, and they asked for opinions:**
> "The brief groups 'same as yesterday' with 'my usual' as memory problems. I don't think they're
> the same thing. One's a date predicate with an exact answer; the other is learned shorthand. I
> built them differently."

**Multimodal:**
> "I read the food-image estimation literature before writing the vision prompt. The finding that
> mattered: portion is the error, not identification. Models name biryani fine — they can't tell 200
> grams from 500, because a photo carries no absolute scale."

> "So confidence is two fields, not one, and the thresholds are deliberately asymmetric: low
> identification asks, low portion logs anyway and says what it assumed. Gating both the same way
> would make it ask about portions on nearly every photo. And diet facts go into the *vision* prompt
> too — if it knows I'm vegetarian, white cubes come back as paneer rather than chicken."

**Cut first:** the token percentages, and the vision-priors point. Keep the disagreement — it is
worth more than either.

---

## 7:45 — Tests, evals, latency · *Terminal B, then the board*

```bash
pytest tests/ -q
python evals/run_evals.py
```

> "178 tests and 21 eval cases, and both run on a clean clone with no API keys, because there's a
> deterministic offline backend. Eleven of those come straight from your test set — you can see them
> named as they pass."

> "Each case scores four axes, and the load-bearing one is the live row count. A correction done
> with the wrong tool still produces plausible-looking calories. Only the row count catches it."

**Board → latency:**
> "Text p50 766 milliseconds, p95 1257 — and you saw in `/debug` that the agent's own overhead is
> sub-millisecond, so that is all model round trip. The image path is 5.9 seconds. The biggest
> single win was `reasoning_effort=low`: gpt-oss is a reasoning model, and two-tool turns were
> taking twelve to twenty seconds."

**The one worth telling as a story:**
> "The image p95 was 13.7 seconds and I assumed that was inference. It wasn't. I had a second vision
> provider configured as failover and its daily quota was gone — so every photo was paying for a
> dead provider's timeout before the working one was ever called. Deleting the fallback took p95 to
> 6.8. A fallback to something that's out of quota is worse than no fallback, and you can't reason
> your way to that. You have to look."

> "And the honest part: the binding constraint isn't latency, it's tokens per day. That's why the
> benchmark paces requests and reports a throttled count — firing thirty turns back to back measures
> the rate limiter, not the agent."

**Cut first:** the Gemini thinking-model story (429 ms vs 8.1 s) — it's on the board if they read it.

---

## 8:45 — The bug, and close · *the board's bugs section*

Do not rush this. It is the strongest minute in the video.

> "The mock proves plumbing, not model behaviour. Running real models found things it structurally
> couldn't — the model parroting an example out of my own system prompt, a portion getting halved
> twice because prompt and code were both being safe. But the one I found last is the one I'd lead
> with."

> "My failover provider was pointed at a model I'd verified with a single call. When I finally ran
> the whole conversation through it, it told me 'roughly 170 for assorted snacks' — and it had made
> no tool call at all. It was confirming meals it never wrote. That's the worst failure this product
> can have, because the user has no reason to check."

> "Notice what couldn't catch it. The mock can't, because the mock always calls the tool. A unit
> test can't, because the tool is correct. The latency benchmark can't — it passed, and that model
> was actually *faster* than the one I replaced it with, which is exactly how it became the default."

> "What caught it is a script that walks the whole conversation and prints the database after every
> turn, so a reply that sounds right sits directly above a row count that didn't move. The
> generalisation I'd defend: a health check has to spend what a real request spends. I got that
> lesson twice from opposite directions — a five-token 'say OK' probe returning 200 while every real
> turn was failing on the daily token cap, and a single-call probe passing on a model that couldn't
> hold a conversation."

**Close:**
> "With more time: the image path under three seconds, and prompt caching — which on a
> token-limited tier converts directly into more usable turns. Everything I've claimed is in the
> README, and the benchmark runs are committed, including the one I can't retake, and why."

---

## Checklist

Tick these watching it back, not while recording.

**Required by the brief**
- [ ] Correction shown — total moving by exactly one roti, **row count unchanged**
- [ ] Photo shown, **including** the confirmation step and correcting it before it writes
- [ ] Memory written *and* retrieved — and retrieved **in a second process**
- [ ] `correct_meal` vs `log_meal` explained as **structural**, not prompted
- [ ] Latency stated, **with one thing you couldn't fix**
- [ ] Tests and evals run on camera

**Worth the time**
- [ ] All three surfaces used, each for something only it proves
- [ ] "No RAG, and here's why" said out loud
- [ ] The disagreement with the brief stated
- [ ] The failover bug told as a story, with what couldn't have caught it
- [ ] Bonuses named: evals, streaming, multi-user isolation, offline mode, Streamlit UI

**Kill criteria — re-record if any of these**
- [ ] Over 10:00
- [ ] Terminal text unreadable at 1080p
- [ ] A `rate limited` reply appears without you naming it as the free-tier cap
