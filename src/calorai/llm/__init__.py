"""Model selection.

Two paths, two different models, chosen for different reasons:

* **text / agent loop** -- Groq `openai/gpt-oss-20b`. The loop is tool calling,
  so throughput and function-calling reliability are what matter, and Groq is
  roughly 1000 tok/s on a free tier.
* **vision** -- Gemini `gemini-3.5-flash-lite`. Groq's vision offering was
  Llama 4 Scout, which is preview-only and was deprecated in June 2026; an
  image path built on it has a retirement date. See docs/RESEARCH.md.

Cerebras sits behind Groq as a failover. That is not redundancy for its own
sake: the free tier is 30 requests/minute and the latency benchmark issues 30
requests in about a minute, so without a fallback the benchmark cannot finish.

`mock` needs no key and no network, and is the default for tests and evals.
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()

CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"


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
            # One retry only. On a messaging surface a fast failure that falls
            # over to Cerebras beats a slow success on the primary.
            max_retries=1,
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
            model=os.environ.get("GEMINI_VISION_MODEL", "gemini-3.5-flash-lite"),
            google_api_key=_require("GOOGLE_API_KEY", "gemini"),
            temperature=0.2,
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


def active_backends() -> dict[str, str]:
    """For the CLI banner, benchmark headers and eval reports -- so a reported
    latency number always names the models it came from."""
    text = os.environ.get("CALORAI_TEXT_BACKEND", "mock")
    vision = os.environ.get("CALORAI_VISION_BACKEND", "mock")
    fallback = os.environ.get("CALORAI_TEXT_FALLBACK", "none")
    models = {
        "groq": os.environ.get("GROQ_TEXT_MODEL", "openai/gpt-oss-20b"),
        "cerebras": os.environ.get("CEREBRAS_TEXT_MODEL", "llama-3.3-70b"),
        "gemini": os.environ.get("GEMINI_VISION_MODEL", "gemini-3.5-flash-lite"),
        "ollama": os.environ.get("OLLAMA_TEXT_MODEL", "qwen2.5:3b"),
        "mock": "rule-based stub",
    }
    return {
        "text": f"{text} ({models.get(text, '?')})",
        "vision": f"{vision} ({models.get(vision, '?')})",
        "fallback": fallback,
    }
