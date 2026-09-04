"""Streamlit chat UI.

    streamlit run app.py

Deliberately thin -- the brief says the agent code is what gets read, so this
adds no logic of its own. It calls the same `stream_turn` the CLI does and
renders what comes back.

The one thing it earns its place with is the sidebar: the day's totals sit next
to the conversation and update after every turn, which makes "totals stay
correct through a correction" something you can watch rather than something you
have to take on trust.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from calorai import repository as repo  # noqa: E402
from calorai.db import connect  # noqa: E402
from calorai.graph import build_graph, stream_turn  # noqa: E402
from calorai.llm import active_backends  # noqa: E402
from calorai.memory import extractor, render, store  # noqa: E402
from calorai.schemas import FoodItem  # noqa: E402

st.set_page_config(page_title="CalorAI", page_icon="🍛", layout="wide")

SUGGESTIONS = [
    "had 2 parathas and chai for breakfast",
    "leftover biryani, maybe two thirds of the box",
    "skipped lunch but grazed all afternoon",
    "actually that was 3 rotis not 2",
    "how am I doing on calories?",
    "i'm vegetarian btw",
    "my usual",
]


@st.cache_resource
def get_session(user_id: str):
    """One connection and one compiled graph per user, cached across reruns.

    Streamlit re-executes this file on every interaction, so without the cache
    each keystroke would rebuild the graph and reconnect to SQLite.
    """
    conn = connect("calorai.db")
    return conn, build_graph(conn, user_id, streaming=True)


def main() -> None:
    with st.sidebar:
        st.title("🍛 CalorAI")

        # A user picker rather than a login: it makes session isolation
        # demonstrable in two clicks.
        user = st.text_input("user", value="demo", help="sessions are isolated per user")
        conn, graph = get_session(user)

        totals = repo.daily_totals(conn, user)
        st.subheader("today")
        c1, c2 = st.columns(2)
        c1.metric("calories", totals["kcal"])
        c2.metric("protein", f"{totals['protein_g']:g} g")
        c3, c4 = st.columns(2)
        c3.metric("carbs", f"{totals['carbs_g']:g} g")
        c4.metric("fat", f"{totals['fat_g']:g} g")
        if totals["items_estimated"]:
            st.caption(f"{totals['items_estimated']} of {totals['items_logged']} items are estimates")

        st.subheader("logged today")
        meals = repo.find_meals(conn, user, day="today", limit=30)["meals"]
        if meals:
            st.dataframe(
                [
                    {"food": m["name"], "qty": f"{m['qty']:g} {m['unit']}", "cal": m["kcal"]}
                    for m in meals
                ],
                hide_index=True, use_container_width=True,
            )
        else:
            st.caption("nothing yet")

        st.subheader("what it remembers")
        block = render.render_memory_block(conn, user)
        if block:
            st.code(block, language=None)
        else:
            # Empty here is correct, not broken: meals are events, not facts
            # about a person, so logging food deliberately writes nothing.
            st.caption(
                "nothing yet — this only fills up from durable facts like "
                "*i'm vegetarian* or *aiming for 140g protein*. Logging meals "
                "writes nothing here on purpose."
            )

        backends = active_backends()
        st.caption(f"text: {backends['text']}")
        st.caption(f"vision: {backends['vision']}")
        if backends["text"].startswith("mock"):
            st.warning("offline mock backend — set CALORAI_TEXT_BACKEND=groq in .env")

        # "same as yesterday" and "my usual" need a past to refer to. A brand new
        # user has neither, so the agent correctly answers "nothing logged
        # yesterday" -- which looks like a bug in a demo. This gives them one.
        if st.button("seed a returning user", use_container_width=True):
            repo.log_meal(
                conn, user,
                [FoodItem(name="idli", qty=3, unit="piece"),
                 FoodItem(name="sambar", qty=1, unit="katori")],
                slot="breakfast", day="yesterday",
            )
            store.put_alias(
                conn, user, "my usual",
                [{"name": "oats", "qty": 1, "unit": "katori"},
                 {"name": "banana", "qty": 1, "unit": "piece"}],
            )
            st.rerun()

        if st.button("clear this user's day", use_container_width=True):
            conn.execute(
                "UPDATE meal_items SET deleted_at=datetime('now') WHERE user_id=? AND local_date=date('now','localtime')",
                (user,),
            )
            conn.commit()
            st.session_state.pop(f"history_{user}", None)
            st.rerun()

    history_key = f"history_{user}"
    history = st.session_state.setdefault(history_key, [])

    for entry in history:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            if entry.get("image"):
                st.image(entry["image"], width=240)
            if entry.get("meta"):
                st.caption(entry["meta"])

    if not history:
        st.info("Text it like you'd text a friend. Try one of these, or upload a photo of a plate.")
        cols = st.columns(3)
        for i, suggestion in enumerate(SUGGESTIONS[:6]):
            if cols[i % 3].button(suggestion, use_container_width=True):
                st.session_state["pending"] = suggestion
                st.rerun()

    # Keyed by a counter so the widget resets after a send. Without this the
    # uploader keeps its file and silently re-attaches the same photo to the
    # next typed message.
    upload_round = st.session_state.setdefault("upload_round", 0)
    photo = st.file_uploader(
        "photo of a plate (optional)", type=["jpg", "jpeg", "png"],
        key=f"photo_{upload_round}",
    )
    typed = st.chat_input("what did you eat?")
    message = st.session_state.pop("pending", None) or typed

    if not message and not photo:
        return

    # A photo is STAGED, not sent. Streamlit reruns the moment a file is
    # uploaded, so firing the turn there sent the picture before the user could
    # type anything -- which made "[photo] half of this was my brother's", the
    # case the brief cares most about, unreachable from this UI. An upload now
    # shows a preview and waits for either a caption or an explicit send.
    if photo is not None and not message:
        preview, action = st.columns([3, 1])
        preview.image(photo, width=220)
        preview.caption(
            "add context below and press Enter — e.g. *half of this was my "
            "brother's*, or *this was lunch* — or send it as-is"
        )
        if not action.button("send photo", use_container_width=True):
            return
        message = ""

    image_path = None
    if photo is not None:
        # The graph takes a path, so an upload is spooled to a temp file. Kept
        # here rather than in the agent: file handling is a UI concern.
        suffix = Path(photo.name).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(photo.getbuffer())
            image_path = handle.name

    with st.chat_message("user"):
        st.markdown(message or "_sent a photo_")
        if image_path:
            st.image(image_path, width=240)
    history.append({"role": "user", "content": message or "_sent a photo_", "image": image_path})

    with st.chat_message("assistant"):
        placeholder = st.empty()
        started = time.perf_counter()
        pieces: list[str] = []
        result: dict = {}
        try:
            for kind, payload in stream_turn(conn, user, message, image_path=image_path, graph=graph):
                if kind == "status":
                    # A photo has ~6s of vision call before any token exists.
                    placeholder.markdown(f"_{payload}_")
                elif kind == "token":
                    pieces.append(payload)
                    placeholder.markdown("".join(pieces) + "▌")
                else:
                    result = payload
            reply = "".join(pieces) or result.get("reply", "")
            placeholder.markdown(reply)
        except Exception as exc:  # noqa: BLE001
            reply = f"that failed: {type(exc).__name__}"
            placeholder.error(reply)

        elapsed = (time.perf_counter() - started) * 1000
        meta = f"{elapsed:.0f} ms"
        if result.get("ttft"):
            meta = f"{result['ttft'] * 1000:.0f} ms to first word · {meta}"
        if result.get("used_fast_path"):
            meta += " · fast path, no model call"
        if result.get("tool_calls"):
            meta += f" · {', '.join(result['tool_calls'])}"
        st.caption(meta)

    history.append({"role": "assistant", "content": reply, "meta": meta})

    # Same as the CLI: memory is written after the reply, never before it.
    try:
        extractor.extract_and_store(conn, user, message)
        extractor.maybe_learn_alias(conn, user, message)
    except Exception:
        pass

    repo.transcript_append(conn, user, "user", message or "[photo]")
    repo.transcript_append(conn, user, "assistant", reply)
    if image_path:
        st.session_state["upload_round"] = upload_round + 1
    st.rerun()


if __name__ == "__main__":
    main()
