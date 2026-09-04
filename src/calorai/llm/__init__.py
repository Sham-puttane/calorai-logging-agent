"""Model selection.

Two paths, two different models. Every id below was **measured**, not taken
from a docs page -- see docs/RESEARCH.md for the numbers.

* **text / agent loop** -- Groq `openai/gpt-oss-20b`, 230 ms warm. The loop is
  tool calling, so function-calling reliability and throughput are the axes
  that matter.
* **vision** -- Mistral `pixtral-12b-2409`. Chosen over Gemini after running
  both against the same photo: comparable output (both named the dishes,
  reported a scale reference and offered alternatives) but a workable quota.
  Gemini's free tier allows so few images per model per day that ten benchmark
  photos exhausted it, which showed up as a p95 of 25 s that was pure rate-limit
  timeout. A model you cannot call is not a fast model.
* **text failover** -- Mistral `ministral-8b-latest`, 957 ms. A different
  provider on purpose. It was Cerebras (402 Payment Required on every model its
  key could see), then Gemini (daily quota spent).
* **also verified** -- OpenRouter `ling-3.0-flash-fin:free`, 793 ms, correct
  tool calls. What the demo runs on once Groq's daily budget is gone.

Failover is load-bearing rather than decorative, and the reason is a number:
Groq's free tier is **200,000 tokens per day** and this agent spends ~1.1k per
call, so a working session exhausts it. The per-minute headers stay healthy
while that happens, which is why a trivial "say OK" probe passes at the exact
moment every real turn is failing.

`gemini-3.5-flash-lite` was the first vision pick and is worth remembering as a
trap: it is a *thinking* model, measured 8.1 s warm against its older sibling's
429 ms, and it rejects `thinking_budget=0`.

Groq was rejected for vision on evidence: its vision model was Llama 4 Scout,
preview-only and deprecated in June 2026.

`mock` needs no key and no network, and is the default for tests and evals.
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()

CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"
MISTRAL_BASE_URL = "https://api.mistral.ai/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class BackendUnavailable(RuntimeError):
    """Raised when a backend is selected but its key is missing."""


def _require(var: str, backend: str) -> str:
    value = os.environ.get(var, "").strip()
    if not value:
        raise BackendUnavailable(
            f"{backend} needs {var}. Add it to .env, or set "
            f"CALORAI_TEXT_BACKEND=mock to run offline."
        )
    return value


def build_text_model(backend: str, *, streaming: bool = False) -> BaseChatModel:
    backend = (backend or "mock").lower()

    if backend == "mock":
        from .mock import MockChatModel

        return MockChatModel()

    if backend == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=os.environ.get("GROQ_TEXT_MODEL", "openai/gpt-oss-20b"),
            api_key=_require("GROQ_API_KEY", "groq"),
            temperature=0.3,
            streaming=streaming,
            # gpt-oss is a reasoning model, and reasoning is what made the
            # multi-round turns slow: measured 12-20s on two-tool turns at the
            # default effort, against ~900ms on single-tool ones. Deciding
            # between six tools does not need deliberation, so effort is
            # capped. This was the single biggest latency win in the project.
            reasoning_effort=os.environ.get("GROQ_REASONING_EFFORT", "low"),
            # Zero retries, deliberately. Groq's free tier limit is tokens per
            # minute (~8k), which this agent hits in roughly two turns, and the
            # SDK's default backoff sits on a 429 for 10-18 seconds before
            # giving up. Failing instantly into the Gemini fallback turns a
            # rate-limited turn from ~18s into ~1s. On a messaging surface a
            # fast answer from the second-choice model beats a slow one from
            # the first.
            max_retries=0,
            timeout=20,
        )

    if backend == "cerebras":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.environ.get("CEREBRAS_TEXT_MODEL", "llama-3.3-70b"),
            api_key=_require("CEREBRAS_API_KEY", "cerebras"),
            base_url=CEREBRAS_BASE_URL,
            temperature=0.3,
            streaming=streaming,
            max_retries=1,
            timeout=20,
        )

    if backend == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=os.environ.get("GEMINI_VISION_MODEL", "gemini-2.5-flash-lite"),
            google_api_key=_require("GOOGLE_API_KEY", "gemini"),
            temperature=0.2,
            # Same lesson as Groq, learned the same way. Left at its defaults
            # this client answered a rate-limited image in 228 SECONDS -- it
            # was not slow inference, it was backoff retrying behind a 429 with
            # no way for the caller to see it. A photo that cannot be read in
            # 25s is a photo the user should be told about, not one they wait
            # four minutes for.
            max_retries=0,
            timeout=25,
        )

    # Mistral and OpenRouter both speak the OpenAI wire protocol, so they need
    # a base_url rather than a new client library. Worth noting because it is
    # why adding a provider here costs about ten lines: the adapter is a thin
    # seam, and the graph never learns which vendor answered.
    if backend == "mistral":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.environ.get("MISTRAL_TEXT_MODEL", "ministral-8b-latest"),
            api_key=_require("MISTRAL_API_KEY", "mistral"),
            base_url=MISTRAL_BASE_URL,
            temperature=0.3,
            streaming=streaming,
            max_retries=0,
            timeout=25,
        )

    if backend == "mistral-vision":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.environ.get("MISTRAL_VISION_MODEL", "pixtral-12b-2409"),
            api_key=_require("MISTRAL_API_KEY", "mistral"),
            base_url=MISTRAL_BASE_URL,
            temperature=0.2,
            max_retries=0,
            timeout=25,
        )

    if backend == "openrouter":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct"),
            api_key=_require("OPENROUTER_API_KEY", "openrouter"),
            base_url=OPENROUTER_BASE_URL,
            temperature=0.3,
            streaming=streaming,
            max_retries=0,
            timeout=25,
        )

    if backend == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=os.environ.get("OLLAMA_TEXT_MODEL", "qwen2.5:3b"),
            temperature=0.3,
        )

    raise BackendUnavailable(f"unknown backend '{backend}'")


@lru_cache(maxsize=8)
def get_text_model(streaming: bool = False) -> BaseChatModel:
    """Primary conversational model. Cached: constructing a client per turn
    throws away the connection pool, which costs a TLS handshake we can see."""
    return build_text_model(
        os.environ.get("CALORAI_TEXT_BACKEND", "mock"), streaming=streaming
    )


@lru_cache(maxsize=4)
def get_fallback_text_model() -> BaseChatModel | None:
    """Optional second provider. Returns None when unset or unusable, so a
    missing Cerebras key degrades to 'no failover' rather than a crash."""
    name = os.environ.get("CALORAI_TEXT_FALLBACK", "").strip()
    if not name or name.lower() == "none":
        return None
    if name.lower() == os.environ.get("CALORAI_TEXT_BACKEND", "mock").lower():
        return None
    try:
        return build_text_model(name)
    except BackendUnavailable:
        return None


@lru_cache(maxsize=2)
def get_vision_model() -> BaseChatModel:
    """Deliberately a *different* model from the text path."""
    return build_text_model(os.environ.get("CALORAI_VISION_BACKEND", "mock"))


@lru_cache(maxsize=2)
def get_fallback_vision_model() -> BaseChatModel | None:
    """Second vision provider, for when the first is throttled.

    This is not belt-and-braces. Measured over 10 photos on Gemini's free tier
    the image path had a p50 of 5.8s but a p95 of 25.1s -- and that p95 is the
    client timeout firing on rate-limited calls, not slow inference. A second
    provider converts the worst case from "waited 25 seconds and got nothing"
    into "answered by the other model".
    """
    name = os.environ.get("CALORAI_VISION_FALLBACK", "").strip()
    if not name or name.lower() == "none":
        return None
    if name.lower() == os.environ.get("CALORAI_VISION_BACKEND", "mock").lower():
        return None
    try:
        return build_text_model(name)
    except BackendUnavailable:
        return None


def tracing_status() -> str:
    """Whether LangSmith is actually recording, and why not when it isn't.

    Worth surfacing because the failure mode is silence: tracing is configured
    entirely through environment variables that LangChain reads on its own, so
    a wrong variable name, a missing key, or a .env loaded too late all produce
    no traces and no error. "Off" is a fine answer; "you think it's on and it
    isn't" is not.
    """
    on = os.environ.get("LANGSMITH_TRACING", os.environ.get("LANGCHAIN_TRACING_V2", "")).lower()
    key = os.environ.get("LANGSMITH_API_KEY", os.environ.get("LANGCHAIN_API_KEY", "")).strip()
    project = os.environ.get("LANGSMITH_PROJECT", os.environ.get("LANGCHAIN_PROJECT", "default"))

    if on not in {"1", "true", "yes"}:
        return "off"
    if not key:
        return "ON but no API key -- nothing will be recorded"
    # Reports the environment rather than asking the langsmith client, whose
    # own tracing_is_enabled() is memoised: once evaluated it keeps answering
    # with whatever the environment looked like at first call, which is exactly
    # the stale reading this function exists to avoid.
    return f"on -> project '{project}'"


def active_backends() -> dict[str, str]:
    """For the CLI banner, benchmark headers and eval reports -- so a reported
    latency number always names the models it came from."""
    text = os.environ.get("CALORAI_TEXT_BACKEND", "mock")
    vision = os.environ.get("CALORAI_VISION_BACKEND", "mock")
    fallback = os.environ.get("CALORAI_TEXT_FALLBACK", "none")
    models = {
        "groq": os.environ.get("GROQ_TEXT_MODEL", "openai/gpt-oss-20b"),
        "cerebras": os.environ.get("CEREBRAS_TEXT_MODEL", "llama-3.3-70b"),
        "gemini": os.environ.get("GEMINI_VISION_MODEL", "gemini-2.5-flash-lite"),
        "mistral": os.environ.get("MISTRAL_TEXT_MODEL", "ministral-8b-latest"),
        "mistral-vision": os.environ.get("MISTRAL_VISION_MODEL", "pixtral-12b-2409"),
        "openrouter": os.environ.get("OPENROUTER_MODEL", "?"),
        "ollama": os.environ.get("OLLAMA_TEXT_MODEL", "qwen2.5:3b"),
        "mock": "rule-based stub",
    }
    return {
        "text": f"{text} ({models.get(text, '?')})",
        "vision": f"{vision} ({models.get(vision, '?')})",
        "fallback": fallback,
    }
