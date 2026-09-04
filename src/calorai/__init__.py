"""CalorAI logging agent.

`.env` is loaded here, at package import, rather than inside whichever module
first needs a key. LangSmith tracing is configured purely through environment
variables that LangChain reads on its own, so if the file were loaded later --
when the first model is constructed, say -- tracing would be silently off for
everything that ran before that. Loading once, first, makes
`LANGCHAIN_TRACING_V2=true` in .env actually take effect.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
