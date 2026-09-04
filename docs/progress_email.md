# Draft progress note to Bhavesh

The brief invites a quick update when you start, if you hit a blocker, and when you submit, and
says questions are welcome. This is the mid-build one. Send from your own mail — edit freely, and
**delete the repo link if you'd rather share it only at submission** (the repo is currently
private).

---

**To:** bhavesh@calorai.ai
**Subject:** CalorAI test task — progress update + one question on latency measurement

Hi Bhavesh,

Quick update on the logging agent, plus one question you might have a view on.

**Where it is.** The six core features are built and working end to end: a LangGraph tool-calling
agent over six tools, SQLite persistence, running daily totals, a separate vision model for photos,
persistent memory across sessions, and a written-down policy for when the agent asks versus when it
just logs and states its assumption. There's an eval suite (21 cases covering all 11 messages from
your test set) and a latency benchmark. Both run on a clean clone with no API keys, via a
deterministic offline backend, so nothing about verifying it depends on my credentials.

**The question.** I've built this on free tiers — Groq for the agent loop, Gemini for vision. The
agent itself is fast: text p50 is 766 ms and p95 1257 ms, and the agent's own overhead outside the
model call is sub-millisecond. But the binding constraint turned out not to be latency at all, it's
tokens per minute: Groq's free tier caps TPM and the agent spends ~1.1k tokens a turn, so sustained
use throttles after a handful of turns. On the image path that shows up as a p50 of 5.8 s against a
p95 of 25 s, where the p95 is a rate-limit timeout rather than slow inference.

So my benchmark paces requests and reports a throttled count alongside the percentiles, on the
reasoning that firing 30 turns back-to-back would be measuring the rate limiter rather than the
agent. **Is that the reporting you'd want, or would you rather see unpaced numbers with the
throttling included in the distribution?** Happy to report both — I'd just rather ask than assume,
since you said latency reasoning is what you're reading for.

**One thing I'd flag.** Your test set groups "same as yesterday" with "my usual" as memory cases.
I've deliberately built them differently: the first is a query against the meal log, the second is
learned shorthand stored as an alias. I think collapsing them is what pushes people toward a vector
store they don't need — but it is a departure from the brief, so I wanted to name it rather than
have it look like I'd missed the requirement. It's written up in the README.

**Still to do:** more samples on the image path, and the walkthrough video.

Thanks — enjoying this one.

[Your name]

---

## Notes before sending

- Check the numbers still match `bench/results/` — don't send stale figures.
- **Rotate the API keys first** if you're linking the repo (they were briefly committed; see the
  README/commit history).
- Keep it this length. A progress note that reads like a status report is worse than none.
