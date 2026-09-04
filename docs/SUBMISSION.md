# Submission checklist

Work top to bottom. The first section is the one that actually matters.

---

## 1. Rotate the API keys — do this first

Three keys were committed to `.env.example` in commit `d559782` (03 Sep) and removed in a later
commit. **The blob is still in git history**, so making the repo public exposes them. Rotating makes
that blob worthless.

Audited with `git log -p --all` against every secret pattern — these three, and only these three:

| Provider | Leaked prefix | Where to rotate |
|---|---|---|
| **Groq** | `***REMOVED-GROQ-KEY***…` | [console.groq.com/keys](https://console.groq.com/keys) — trash icon on the old key, then **Create API Key** |
| **Google** | `AQ.Ab8RN6…` | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) — delete, then **Create API key** |
| **Cerebras** | `***REMOVED-CEREBRAS-KEY***…` | [cloud.cerebras.ai](https://cloud.cerebras.ai) → API Keys — revoke (this one is unused; it returned 402 on every model, so just revoke it) |

**Mistral, OpenRouter and LangSmith were never exposed** — they were added to `.env.example` after
the leak, always with empty values. You do not need to touch them.

Then paste the two you actually use (Groq, Google) into `D:\calorai-agent\.env`. Nothing else
changes — `.env` is gitignored and the app reads it at import.

Verify the new keys work before you record:

```bash
cd D:\calorai-agent
.venv\Scripts\activate
python bench\_real_e2e.py --delay 5
```

Read the `db:` line under every turn, not just the replies.

**`.env` is gitignored. `.env.example` is not.** That is the mistake that caused this — never put a
real value in `.env.example`. Check before every push:

```bash
git status --short
git diff --cached --name-only | grep -i env    # must print nothing
```

### Optionally: scrub the history too

Rotating is sufficient for security. But a public repo whose history contains secrets is a bad look
in a hiring context, so if you want it clean:

```bash
pip install git-filter-repo
cd D:\calorai-agent
git filter-repo --path .env.example --invert-paths --force
git remote add origin <your repo url>    # filter-repo drops the remote
git push origin main --force
```

That drops `.env.example` from every commit. Re-add the current (empty) one afterwards and commit
normally. **Rotate regardless** — history scrubbing is cosmetic, and GitHub may retain the blob.

---

## 2. Make the repo public

Settings → General → Danger Zone → **Change repository visibility** → Public.

Only after step 1.

---

## 3. Record the video (5–10 min)

Follow [`video_script.md`](video_script.md). The brief explicitly asks for:

- [ ] Working demo — **including one image case and one correction case**
- [ ] Architectural decisions explained
- [ ] How memory is **written** and **retrieved**
- [ ] Latency work and the trade-offs
- [ ] Challenges and how you solved them
- [ ] Bonus features shown

Before you hit record:

```bash
cd D:\calorai-agent
del calorai.db
.venv\Scripts\activate
python bench\_real_e2e.py --delay 7    # confirm nothing is broken today
python -m calorai.cli --user demo      # then record from here
```

Have a second terminal ready with `pytest tests/ -q` and `python evals/run_evals.py`.

---

## 4. Send the progress note

[`progress_email.md`](progress_email.md) — check the numbers still match `bench/results/`
before sending.

---

## 5. Submit

- [ ] GitHub repo link (public)
- [ ] Video link (Loom / YouTube unlisted / Drive)
- [ ] Any notes for them
- [ ] LangSmith public trace link (already in the README)

---

## What's honest to say is incomplete

The README says all of this, but worth having it in your head on camera:

- **The image path is slow** — p50 5.9 s, and the vision call is 5.8 s of it. Resolution is
  tuned (512 px, A/B'd against 768 and 384); the next lever is a tighter output schema,
  since latency tracks generated JSON more than image size.
- **The failover's p95 is still owed.** Its p50 (2212 ms) comes from the end-to-end script
  rather than the percentile sweep, because every text provider hit its daily cap during
  the final measurement session. `bench/results/README.md` says so.
- **Free-tier rate limits are the binding constraint**, not the agent. Text p50 is 766 ms
  and the agent's own overhead is sub-millisecond.
- **The transcript table is written but not injected into prompts**, so cross-turn
  reference ("the first one") won't work.
- **Correction targeting is name-overlap matching** — right for what was tested, but it
  would mis-target "the second one".
- **Local inference was measured and rejected** on this hardware, not skipped.
