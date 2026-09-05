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
pytest tests/ -q                  # 182 passed
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
| 5:00 | Board + editor | Architecture, memory, ask-vs-log, multimodal, Supabase |
| 7:30 | Terminal + board | Tests, evals, **drift**, latency |
| 8:40 | Board | The bug, and close |

If you run behind, each segment names **what to cut first**. Cut it without hesitating — going long
is worse than dropping a point.

---

## 0:00 — What it is · *Streamlit, sidebar visible*

**Say:**
> "This is CalorAI. You text it what you ate, the way you'd text a friend, and it logs it. No forms,
> no dropdowns."

> "I'll show it working first. Then I'll prove the state is real by killing the process and opening
> it somewhere else. Then the decisions that actually carry it — including a bug I found in the last
> hour, which is honestly the most interesting thing in here."

---

## 0:35 — SURFACE 1: the UI · *the eleven steps*

**Test:** paste these in order. Say this once, up front:

> "Watch the sidebar, not the chat. The chat is easy to fake. The sidebar is the database."

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

Let 1, 2 and 4 run fast and quiet. **Five moments to slow down on:**

**Step 3 — say something here. The job description names this exact skill:**
> "Now watch what it does with that one. It logs an estimate, and it asks me nothing."

> "Knowing when to ask and when to just log it is the thing I spent longest on. Logging something
> imprecise beats interrogating someone about a snack they can't remember."

**Step 5 — the most important ten seconds in the video:**
> "Watch the sidebar. Seven ninety, to eight ninety-five. That's 105 calories. Exactly one roti."

> "The roti row goes from two to three. No second row appears. That's not the prompt getting lucky,
> and in a minute I'll show you why it can't be."

**Step 6:**
> "That one's a fact about me, not a meal. So the total doesn't move. But look — it's now under what
> it remembers."

**Step 7:**
> "Thirty milliseconds. No model call at all. And it knows my target, because I mentioned it one
> message ago."

**Step 8 — the photo:**
> "Notice it hasn't logged anything yet."

> "A photo is the one input where you hand the entire description over to a model. And models get
> plates wrong in ways you can see instantly and the agent can't see at all. So it shows me what it
> found, and it waits."

Then step 9, straight away, no pause:
> "And I correct it before it ever writes. What gets logged is one naan. Not the four it guessed."

**Cut first:** steps 10 and 11. The alias gets proved again in the CLI segment, and better.

---

## 3:45 — SURFACE 2: the CLI · *Terminal B*

This is the segment that proves the memory claim, and it takes about 75 seconds. **Use the same user
as the UI** — that's the whole trick.

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
> "So. Different process. Different interface. Same user."

> "Everything I taught the web app is here. The diet, the protein target, the shorthand it learned.
> Because memory lives in SQLite, not in a session variable."

> "And none of this is conversation history. That's the part I'd want to be judged on. These are
> three purpose-built stores."

```
my usual
```
> "And the shorthand it learned two minutes ago in the browser resolves in here."

```
/debug
```

**Say — this is why the CLI is in the video at all:**
> "This is the part the UI hides. Per-turn stage timings, and the actual tool calls."

> "Read the ratio. Ingest is a few milliseconds, and that's alias resolution, memory loading and
> routing, all together. Basically the whole turn is the model round trip. Which is where it should
> be."

*(Say the numbers you can see, not memorised ones. On the real backend `ingest` reads well under a
millisecond against an `agent` of several hundred; on the mock it's a few ms against a few ms.)*

```
how much protein have I had today?
/debug
```
> "And that one — tools empty, fast path true, about ten milliseconds. A totals question is a SQL
> query. There's no reason to pay a model for it."

Then isolation, in two commands:

```
/quit
python -m calorai.cli --user someone_else
```
```
/memory
```
> "New user. Empty. The user id is on every table and every query, and the tools are built per
> session — so a tool physically cannot reach another user's rows."

**Cut first:** the `someone_else` isolation check. It's a bonus, not a core requirement.

---

## 5:00 — SURFACE 3: the board · *architecture, memory, ask-vs-log, multimodal*

Switch to the board. Pan the **graph**. You're talking over a diagram now, not reading code. Drop
into the editor only for the two files named below.

> "So this is a LangGraph tool-calling loop. The model sees all six tool schemas, and it decides.
> That matters to me. It's genuinely an agent, not a classifier handing off to handlers. A router
> would be faster. But a router only handles the phrasings I thought to write rules for."

> "Ingest does the cheap deterministic work first. And there are two places I keep the model out on
> purpose. One is the fast path. The other is the memory write. The memory write runs after you
> already have your reply, so remembering things costs nothing in p50."

**Editor · `src/calorai/tools.py`** — the one file to dwell on:
> "Here's the decision I'd defend hardest. `correct_meal` is a separate tool from `log_meal`."

> "If you give a model one tool that means 'record what they said', then 'actually that was 3 rotis'
> gets answered by logging three more rotis. And no prompt wording reliably stops that. I tried."

> "So I split them. `correct_meal` has no INSERT path. `log_meal` has no UPDATE path. Double
> counting isn't unlikely now, it's impossible. That's what you were watching in the sidebar
> earlier."

**Editor · `src/calorai/db.py`:**
> "Same idea here. There's no stored total anywhere in the schema. Totals are a SUM at query time.
> There's no counter, so there's nothing to drift."

**Back to the board — memory:**
> "Three stores. And none of them is conversation history, which I think is the important part."

> "Writes happen on a background thread after the reply goes out. Extraction is three tiers. Most
> messages hit a regex gate and exit having cost nothing. Then rules. Then the model. 'Had 2 rotis'
> is an event, not a fact about me, so it never reaches the model at all."

**Say the word "weeks" — it's in the job description:**
> "Retrieval is: all the facts, every turn. That sounds naive. Here's why it holds up over weeks.
> Facts are *keyed*. So if I say I'm vegetarian and later say I eat fish, that supersedes, it
> doesn't append. The store stays at a couple of dozen one-liners no matter how long you use it."

> "So the honest answer to 'how do you retrieve without bloating the prompt' is that I made
> retrieval unnecessary. That's also why there's no RAG here. RAG solves corpus-bigger-than-context,
> and this can't have that problem."

**The disagreement — say it, they asked for opinions:**
> "One thing I'd push back on. The brief groups 'same as yesterday' with 'my usual' as memory
> problems. I don't think those are the same thing. One is a date predicate with an exact answer.
> The other is learned shorthand. So I built them differently."

**Ask-versus-log — this is a line straight out of the job description:**
> "And the thing I spent longest getting right isn't in any one file. It's knowing when to ask and
> when to just log it."

> "It's a written policy, not vibes. Log without asking when the items resolve. Ask exactly one
> question, batched, when something's unrecognisable or the swing is more than forty percent. And
> never ask about grams, or oil, or brands. Assume, and say what you assumed."

> "That's what 'skipped lunch but grazed all afternoon' was doing earlier. It logged an estimate and
> asked me nothing. Logging something imprecise beats interrogating someone. The confidence lives on
> the row, so the total hedges instead of the conversation."

**Multimodal:**
> "For the photo path, I read the food-image estimation literature before I wrote the prompt. The
> finding that mattered: portion is the error, not identification. Models name biryani fine. They
> cannot tell two hundred grams from five hundred, because a photo carries no absolute scale."

> "So confidence is two fields, not one. And the thresholds are deliberately asymmetric. Low
> identification asks. Low portion logs anyway and tells you what it assumed. If I gated both the
> same way it would ask about portions on nearly every photo, and that's the form-filling I was
> trying to get away from."

**Supabase — twenty seconds, and say it as a choice, not an apology:**
> "One last thing on the data layer, since you're on Supabase. This runs on SQLite, and that was
> deliberate. For an eight-hour build where the first red flag on your list is 'doesn't run from a
> clean clone', a database that needs no service and no keys is the right tool. My tests and my evals
> pass on a fresh clone with nothing installed. I wasn't going to trade that."

> "And the idea is the same either way. All the SQL lives in three files, nothing above them writes
> any, so the port is placeholders, serial ids, upserts. The thing that keeps totals correct is a
> schema decision, not a SQLite one, so it survives the move unchanged."

> "The one part I'd change rather than port is user scoping. Right now it's enforced by construction
> — the tools close over the user id, so no code path can take one from the model. On Supabase that
> should be row-level security as well, so the database enforces it even when the application is
> wrong."

**Cut first:** the RAG sentence and the multimodal asymmetry detail. **Do not cut** the disagreement,
the ask-versus-log policy, or Supabase — those are the three the job description names directly.

---

## 7:30 — Tests, evals, latency · *Terminal B, then the board*

```bash
pytest tests/ -q
python evals/run_evals.py
```

> "182 tests and 21 eval cases. Both run on a clean clone with no API keys, because there's a
> deterministic offline backend. Eleven of those cases come straight from your test set. You can
> watch them go past by name."

> "Each case scores four axes. The load-bearing one is the live row count. Because a correction done
> with the wrong tool still produces perfectly plausible-looking calories. Only the row count catches
> it."

**Then the drift check — this is the job description's own sentence, so land it:**
> "But pass-fail only tells you *does it work*. It doesn't tell you *did that change help*, which is
> the question you actually have after rewriting a prompt."

> "So there's a committed baseline with per-case scores. Every run compares against it and reports
> movement in both directions. If I fix the phrasing on corrections and quietly break the one about
> fractions, it says so, by name. Prompts regress sideways, and a green total will hide that from
> you."

**Board → latency:**
> "Text p50 is 766 milliseconds, p95 1257. And you saw in `/debug` that the agent's own overhead is
> under a millisecond. So that's basically all model round-trip, which is where it should be."

> "The image path is 5.9 seconds. The biggest single win anywhere was `reasoning_effort=low` —
> gpt-oss is a reasoning model, and two-tool turns were taking twelve to twenty seconds before I
> capped it."

**The one worth telling as a story:**
> "And here's the one I got wrong. Image p95 was 13.7 seconds, and I assumed that was inference. It
> wasn't."

> "I had a second vision provider set up as failover, and its daily quota was already gone. So every
> single photo was paying for a dead provider to time out before the working one got called.
> Deleting the fallback took p95 from 13.7 to 6.8 seconds."

> "A fallback to something that's out of quota is worse than no fallback at all. And you can't reason
> your way to that one. You have to go and look."

> "The honest headline, though: the binding constraint isn't latency. It's tokens per day. That's why
> the benchmark paces requests and reports a throttled count. Firing thirty turns back to back
> measures the rate limiter, not the agent."

**Cut first:** the Gemini thinking-model story. It's on the board if they want it.

---

## 8:40 — The bug, and close · *the board's bugs section*

Slow down here. Don't rush it. This is the strongest minute in the video, and it's the one they'll
remember.

> "Last thing. The mock proves the plumbing. It proves nothing about model behaviour. Running real
> models found things it structurally couldn't — the model parroting an example out of my own system
> prompt back at me, a portion getting halved twice because the prompt and the code were both being
> careful."

> "But the one I found last is the one I'd lead with."

> "My failover provider was pointed at a model I'd verified with a single call. When I finally ran
> the whole conversation through it, it told me 'roughly 170 for assorted snacks'. And it had made no
> tool call at all. It was confirming meals it never wrote."

*(beat)*

> "That's the worst failure this product can have. Because the user has no reason to check."

> "Now notice what couldn't have caught it. The mock can't — the mock always calls the tool. A unit
> test can't — the tool is correct. And the latency benchmark can't, because it passed. That model
> was actually *faster* than the one I replaced it with. That's exactly how it became the default."

> "What caught it is a script that walks the whole conversation and prints the database after every
> turn. So a reply that sounds right sits directly above a row count that didn't move."

> "The generalisation I'd defend from that: a health check has to spend what a real request spends.
> And I got that same lesson twice, from opposite directions. A five-token 'say OK' probe coming back
> 200 while every real turn was failing on the daily token cap. And a single-call probe passing on a
> model that couldn't hold a conversation."

**Close:**
> "With more time: the image path under three seconds, and prompt caching, which on a token-limited
> tier turns directly into more usable turns per day."

> "Everything I've claimed here is in the README, and the benchmark runs are committed — including
> the one I couldn't retake, and why. Thanks for reading it."

---

## Am I saying enough to get the job?

The posting names four things and a stack. Every one has a line in the script now — this is where to
check you actually said them, because these are the sentences they are listening for.

| They asked for | Where you say it | The line |
|---|---|---|
| **Tool calling** | 5:00, `tools.py` | `correct_meal` and `log_meal` split so double counting is *impossible*, not unlikely |
| **Memory across weeks** | 5:00, memory | facts are **keyed**, so contradictions supersede — the store stays two dozen one-liners no matter how long you use it |
| **Knowing when to ask vs just log** | **3:00 (step 3)** and 5:00 | it logged an estimate and asked nothing; the policy is written down, not vibes |
| **Evals that say whether it got better** | 7:30 | the committed baseline reports movement in both directions, by case name |
| **Python / LangGraph / Supabase** | 0:35 throughout, 5:00 | SQLite was the right tool for eight hours and a clean clone; scoping belongs in RLS on Supabase |

Three of those five had no line in an earlier draft. If you are cutting for time, cut the RAG
sentence, the multimodal asymmetry detail and the Gemini story — **never** these five.

Two more they will be listening for that aren't in the posting:

- **An opinion.** The brief says they would rather hire someone with opinions. The disagreement about
  "same as yesterday" versus "my usual" is the one — say it plainly, don't soften it.
- **Something you couldn't fix.** The tokens-per-day ceiling, and the failover p95 you still owe.
  Naming a limit is what makes the rest of the numbers believable.

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
