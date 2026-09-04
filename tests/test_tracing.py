"""LangSmith tracing is configured entirely through environment variables that
LangChain reads on its own, so every way of getting it wrong is silent: a
misspelled variable, a missing key, or a .env loaded after the client was built
all produce no traces and no error.

"Off" is a fine state. "You believe it is on and it is not" is the one worth
testing against, which is why the CLI banner and the Streamlit sidebar both
report what these functions return.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from calorai.llm import tracing_status  # noqa: E402

VARS = (
    "LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT",
    "LANGCHAIN_TRACING_V2", "LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT",
)


@pytest.fixture(autouse=True)
def clean_env():
    saved = {k: os.environ.get(k) for k in VARS}
    for k in VARS:
        os.environ.pop(k, None)
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def test_unconfigured_reports_off():
    assert tracing_status() == "off"


def test_enabled_without_a_key_says_so_loudly():
    """The worst state: you set the flag, saw no error, and assumed it worked."""
    os.environ["LANGSMITH_TRACING"] = "true"
    status = tracing_status()
    assert "no API key" in status
    assert "nothing will be recorded" in status


def test_enabled_with_a_key_names_the_project():
    os.environ.update(
        LANGSMITH_TRACING="true", LANGSMITH_API_KEY="lsv2_test", LANGSMITH_PROJECT="calorai-agent"
    )
    assert "calorai-agent" in tracing_status()


def test_the_legacy_variable_names_still_work():
    """LangChain renamed these; both are accepted, so a stale .env keeps working."""
    os.environ.update(LANGCHAIN_TRACING_V2="true", LANGCHAIN_API_KEY="legacy")
    assert tracing_status().startswith("on")


@pytest.mark.parametrize("value", ["false", "0", "no", "", "off"])
def test_falsey_values_are_off(value):
    os.environ["LANGSMITH_TRACING"] = value
    os.environ["LANGSMITH_API_KEY"] = "lsv2_test"
    assert tracing_status() == "off"


@pytest.mark.parametrize("value", ["true", "TRUE", "1", "yes"])
def test_truthy_values_are_on(value):
    os.environ["LANGSMITH_TRACING"] = value
    os.environ["LANGSMITH_API_KEY"] = "lsv2_test"
    assert tracing_status().startswith("on")


def test_the_status_string_is_ascii():
    """It is printed to a Windows terminal, which defaults to cp1252 -- a
    decorative arrow in here would crash the CLI banner."""
    os.environ.update(LANGSMITH_TRACING="true", LANGSMITH_API_KEY="k")
    tracing_status().encode("cp1252")
