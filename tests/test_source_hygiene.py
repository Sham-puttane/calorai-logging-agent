"""Guards against a silent-corruption class that bit this project three times.

Writing a regex through a shell heredoc turned `\\b` into a literal 0x08
backspace character. The file still parsed, the pattern still compiled, and the
regex matched nothing — a word-boundary assertion became "expect a backspace
character here", which no real input contains.

Every occurrence was invisible on inspection and produced a feature that
silently did nothing: an affirmative-confirmation matcher that never fired, and
an addition detector that classified every message as a correction.

A test is the right place for this because the failure mode is specifically
that reading the code does not reveal it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("src", "tests", "evals", "bench")

#: Characters that should never appear literally in source. Backspace is the
#: one that has actually happened; the others are the same accident with a
#: different escape (\f, \v, \a).
FORBIDDEN = {"\x08": r"\b", "\x0c": r"\f", "\x0b": r"\v", "\x07": r"\a"}


def python_files() -> list[Path]:
    found: list[Path] = []
    for directory in SOURCE_DIRS:
        found.extend(
            p for p in (ROOT / directory).rglob("*.py") if ".venv" not in p.parts
        )
    return found


@pytest.mark.parametrize("char,escape", list(FORBIDDEN.items()))
def test_no_control_characters_from_mangled_escapes(char, escape):
    offenders = []
    for path in python_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        if char in text:
            line = text[: text.index(char)].count("\n") + 1
            offenders.append(f"{path.relative_to(ROOT)}:{line}")
    assert not offenders, (
        f"literal {char!r} found where {escape!r} was meant — a shell heredoc ate "
        f"the backslash. Rewrite the line with an editor, not a heredoc: {offenders}"
    )


def test_every_source_file_compiles():
    """Cheap backstop: a patch applied by string replacement can leave a file
    that imports fine in one module and is broken in another."""
    import ast

    for path in python_files():
        try:
            ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:  # pragma: no cover - only on a real break
            pytest.fail(f"{path.relative_to(ROOT)} does not parse: {exc}")
