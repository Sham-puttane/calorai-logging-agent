"""Turning stored memory into prompt context.

The constraint: this block is prepended to every single turn, so it has to stay
small enough that it never becomes the reason a prompt is slow or expensive.
It is capped at MAX_CHARS and rendered as a few readable lines rather than JSON
-- models follow terse prose more reliably than a nested object, and it costs
fewer tokens.
"""

from __future__ import annotations

import sqlite3

from . import store

MAX_CHARS = 600

# How a stored key should read in the prompt. Keys without an entry here fall
# back to "key: value", so an unexpected fact still renders sensibly.
_LABELS = {
    "diet": "{value}",
    "allergy": "allergic to {value}",
    "avoids": "does not eat {value}",
    "protein_target_g": "protein target {value}g",
    "calorie_target": "calorie target {value}",
    "cuisine": "mostly eats {value} food",
    "name": "name is {value}",
}


def render_facts(conn: sqlite3.Connection, user_id: str) -> str:
    facts = store.get_facts(conn, user_id)
    if not facts:
        return ""
    parts = []
    for fact in facts:
        template = _LABELS.get(fact["key"], f"{fact['key']}: {{value}}")
        parts.append(template.format(value=fact["value"]))
    return " · ".join(parts)


def render_aliases(conn: sqlite3.Connection, user_id: str) -> str:
    aliases = store.get_aliases(conn, user_id)
    lines = []
    for alias in aliases[:3]:
        items = ", ".join(
            f"{i.get('qty', 1):g} {i.get('unit', '')} {i['name']}".strip()
            for i in alias["items"]
        )
        lines.append(f'"{alias["phrase"]}" = {items}')
    return "\n".join(lines)


def render_memory_block(conn: sqlite3.Connection, user_id: str) -> str:
    """The whole of what the agent knows about this person, every turn."""
    facts = render_facts(conn, user_id)
    aliases = render_aliases(conn, user_id)
    if not facts and not aliases:
        return ""

    body = "\n".join(part for part in (facts, aliases) if part)
    if len(body) > MAX_CHARS:
        body = body[: MAX_CHARS - 3].rstrip() + "..."
    return f"[what I know about you]\n{body}"


def render_vision_priors(conn: sqlite3.Connection, user_id: str) -> str:
    """A narrower slice, for the vision prompt.

    Research finding (docs/RESEARCH.md): locale and diet priors measurably shift
    a VLM's portion and identification estimates. Telling the vision model that
    this person is vegetarian is not a conversational nicety -- it changes
    whether white cubes come back as paneer or as chicken.

    Only the facts that could plausibly affect what is on a plate are included;
    a protein target has no bearing on reading an image.
    """
    relevant = {"diet", "allergy", "avoids", "cuisine"}
    facts = [f for f in store.get_facts(conn, user_id) if f["key"] in relevant]
    if not facts:
        return ""
    parts = []
    for fact in facts:
        template = _LABELS.get(fact["key"], f"{fact['key']}: {{value}}")
        parts.append(template.format(value=fact["value"]))
    return "Known about this person: " + "; ".join(parts) + "."
