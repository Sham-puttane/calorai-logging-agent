# Vision path: what the literature says, and what I changed because of it

Before writing the vision prompt I spent ~25 minutes on the food-image estimation
literature. It changed four decisions. Sources at the bottom.

## Finding 1 — Portion is the error, not identification

> "In food-image nutrient estimation with VLMs, portion size is a dominant source of
> error." — *A Japanese-Dietitian Prompt Systematically Shifts Portion Estimates*
> (Nutrients, 2026)

> "Scale ambiguity is the most persistent challenge in monocular estimation."
> — *Food Portion Estimation: From Pixels to Calories* (arXiv 2602.05078)

Modern VLMs identify *biryani* reliably. What they cannot do is tell you whether the
photo shows 200 g or 500 g of it, because **a single photo carries no absolute scale**.

**What I changed:** confidence is not one number. `PlateAnalysis` splits it into
`id_confidence` and `portion_confidence` per item, because they fail independently and
they warrant *different questions*:

| Low signal | Question the agent asks |
|---|---|
| `id_confidence` | "is that paneer or tofu?" |
| `portion_confidence` | "was that a small katori or a big bowl?" |

A single blended confidence would have produced vague questions for both failure modes.
This is the main design change the research bought.

## Finding 2 — Use dining vessels as the scale reference

Fiducial markers (a credit card in frame) work but "disrupt the user experience". The
literature's marker-less alternative is "learning the standard sizes of common dining
vessels" — plates, bowls, spoons as implicit rulers.

**What I changed:** the prompt names the reference set explicitly (dinner plate ~27 cm,
katori ~150 ml, tablespoon ~15 ml) and requires the model to **report which reference it
used** in a `scale_reference` field. When it can't find one, `portion_confidence` drops
and the agent asks. Making the model state its ruler turns an invisible guess into an
auditable one.

## Finding 3 — Locale priors systematically shift portion estimates

The Nutrients paper's whole result is that a dietitian persona with a *locale* framing
measurably moves portion estimates. "Locale-dependent priors about typical portion sizes
are what models must fall back on" when the image gives no scale.

**What I changed — this is the one I like most:** the vision prompt is **not static**.
The user's profile facts are injected into it. If memory knows the user is vegetarian
and eats North Indian food, the vision model is told so before it looks at the plate.

That means memory improves *multimodal accuracy*, not just conversation — a vegetarian's
white cubes are paneer, not chicken. Two rubric lines (memory design, multimodal
handling) get satisfied by one mechanism.

## Finding 4 — Reason in steps, but pay for only one call

> "Reasoning-driven food energy estimation via multimodal LLMs like CalorieLLaVA
> significantly outperforms direct, single-step predictions."

Naively that means two calls (identify, then quantify) — which doubles image latency.

**What I changed:** keep the *staged reasoning* but inside a single request. The prompt
walks identify → find scale reference → estimate household portions → emit JSON. One
round trip, staged reasoning.

**And: household units, never grams.** Research says models lean on locale priors for
portions anyway, our nutrition table is keyed by household unit, and users say
"two rotis", not "160 g of roti". Asking for grams forces a conversion the model is
measurably bad at.

## Model selection — measured, not read off a docs page

Final picks:

| Path | Choice | Measured |
|---|---|---|
| Text / agent | Groq `openai/gpt-oss-20b` | 230 ms warm; strong function calling, the axis the loop is graded on |
| Vision | Mistral `pixtral-12b-2409` | ~5.7 s warm on a real plate, with a workable quota |
| Vision failover | Gemini `gemini-2.5-flash-lite` | comparable quality, unworkable free quota |
| Text failover | Gemini `gemini-2.5-flash-lite` | different provider, so a Groq 429 costs a swap not a wait |

**Four of my initial picks were wrong, and only running them showed it.**

### Vision: Pixtral vs Gemini, same photo, same prompt

Both were handed the identical thali and the identical prompt:

| | Pixtral 12B | Gemini 2.5-flash-lite |
|---|---|---|
| Dishes identified | naan, rice, curry, yogurt sauce, chutney, salad, water | naan, rice, dal, paneer curry, raita, salad |
| Named a scale reference | yes — "dinner plate ~27cm" | yes — "dinner plate ~27cm" |
| Offered alternatives | yes — curry → `[dal, gravy]` | yes — paneer curry → `[vegetable curry]` |
| Warm latency | ~5.7 s | ~5.0 s |
| Free-tier quota | workable | **exhausted by 10 benchmark photos** |

Quality is a wash. Both do what the prompt asks, including the two things the research
said to demand: name your ruler, surface alternatives rather than guessing silently.
Gemini is marginally faster per call.

**Pixtral wins on the axis that actually decides it: being callable.** Gemini's free tier
allows so few images per model per day that one ten-photo benchmark exhausted it, which
showed up as a p95 of 25.1 s that was entirely rate-limit timeout rather than inference.
*A model you cannot call is not a fast model.* Gemini stays wired as failover, so the two
providers cover each other's limits.

### Rejected, with reasons

- **Groq for vision.** Its vision offering was Llama 4 Scout, preview-only and
  **deprecated on 17 June 2026** with a migration notice. An image path with a retirement
  date breaks between being written and being read.
- **`gemini-3.5-flash-lite`** — my first pick, straight from the docs. It is a *thinking*
  model: 8.1 s warm, 20 s cold, and it rejects `thinking_budget=0`. Nineteen times slower
  than its own older sibling for a job that is structured extraction, not reasoning.
  **The newest model was the wrong model.**
- **Cerebras** — the planned failover. Returns 402 Payment Required on every model its key
  can see; the free tier is gone.
- **Self-hosted `transformers`** — slower than the Ollama path, since raw PyTorch on CPU
  trails llama.cpp's quantised kernels. And the deciding argument: p50 is already dominated
  by network round-trip, so self-hosting swaps a 766 ms network call for a multi-second
  local one.
- **The "Nano Banana" Gemini models** (`gemini-3.1-flash-image`) are image *generation*.
  Wrong tool, easy to grab by mistake.

## Deliberately not done

Depth estimation, 3D reconstruction and amodal completion are where the accuracy
actually is, per the survey. All are far outside a 6–8 hour budget, and the brief
explicitly says nutrition accuracy is not being evaluated. Noted so it is clear this was
a scoping decision rather than an oversight.

## Sources

- [Food Portion Estimation: From Pixels to Calories](https://arxiv.org/pdf/2602.05078) — survey; scale ambiguity, vessel references
- [A Japanese-Dietitian Prompt Systematically Shifts Portion Estimates](https://www.mdpi.com/2072-6643/18/17/2892) — persona/locale priors move portion estimates
- [A Confidence-Aware Hybrid Vision–Language Framework for Food Recognition](https://doi.org/10.3390/nu18152449) — confidence gating
- [An Agentic Vision–Language Pipeline for Interactive Nutritional Estimation](https://openaccess.thecvf.com/content/CVPR2026W/MTF/papers/Bhatambarekar_An_Agentic_Vision-Language_Pipeline_for_Interactive_Nutritional_Estimation_from_Food_CVPRW_2026_paper.pdf) — interactive clarification
- [Groq model deprecations](https://console.groq.com/docs/deprecations) · [Groq models](https://console.groq.com/docs/models) · [Gemini models](https://ai.google.dev/gemini-api/docs/models)
