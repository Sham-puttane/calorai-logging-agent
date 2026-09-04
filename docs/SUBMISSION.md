# Submission checklist

Work top to bottom. The first section is the one that actually matters.

---

## 1. Rotate the API keys — do this first

Keys were briefly committed to `.env.example` and pushed. They were removed in a later
commit, but **the blob is still in git history**, so making the repo public exposes them.
Rotating makes that blob worthless.

| Provider | Where | What to do |
|---|---|---|
| **Groq** | [console.groq.com/keys](https://console.groq.com/keys) | Delete the old key (trash icon), then **Create API Key** |
| **Google** | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | Delete the key, then **Create API key** |
| **Mistral** | [console.mistral.ai](https://console.mistral.ai) → API Keys | Revoke, then create a new one |
| **OpenRouter** | [openrouter.ai/keys](https://openrouter.ai/keys) | Delete, then **Create Key** |

Then paste the new values into `D:\calorai-agent\.env`.

**`.env` is gitignored. `.env.example` is not.** That is the mistake that caused this —
never put a real value in `.env.example`.

Verify nothing is staged before every push:

```bash
git status --short
git diff --cached --name-only | grep -i env    # must print nothing
```

### Optionally: scrub the history too

Rotating is sufficient for security. But a public repo whose history contains secrets is
a bad look in a hiring context, so if you want it clean:

```bash
pip install git-filter-repo
cd D:\calorai-agent
git filter-repo --path .env.example --invert-paths --force
git push origin main --force
```

That drops `.env.example` from every commit. Re-add the current (empty) one afterwards
and commit normally. **Rotate the keys regardless** — history scrubbing is cosmetic, and
GitHub may retain the blob.

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

---

## What's honest to say is incomplete

The README says all of this, but worth having it in your head on camera:

- **The image path is slow** — p50 6.9 s, and the vision call is ~6 s of it. Downscaling
  was the easy win; a tighter output schema and a 512 px image are the next ones and I
  ran out of free-tier quota before measuring them.
- **Free-tier rate limits are the binding constraint**, not the agent. Text p50 is 766 ms
  and the agent's own overhead is sub-millisecond.
- **The transcript table is written but not injected into prompts**, so cross-turn
  reference ("the first one") won't work.
- **Correction targeting is name-overlap matching** — right for what was tested, but it
  would mis-target "the second one".
- **Local inference was measured and rejected** on this hardware, not skipped.
