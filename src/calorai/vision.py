"""The image path: a photo (and optionally a caption) becomes one PlateAnalysis.

The prompt here is built on the findings in docs/RESEARCH.md rather than
guessed at. Four things it does deliberately:

1. **Stages the reasoning inside a single call.** Reasoning-driven estimation
   beats one-shot "how many calories is this", but two calls would double the
   latency of the slowest path in the product. So the model is walked through
   identify -> find a ruler -> size the portions -> emit JSON, and we pay for
   one round trip.

2. **Names the ruler.** Scale is the dominant error source, and a photo carries
   no absolute scale. The prompt supplies standard vessel sizes and requires
   the model to report which one it measured against, which turns an invisible
   guess into something the user can correct.

3. **Injects the user's priors.** Locale and diet priors measurably shift
   portion and identification estimates, so what memory knows about this person
   goes into the vision prompt.

4. **Asks for two confidences.** Identification and portion fail
   independently, and the agent asks a different question depending on which
   one is shaky.

The caption is folded into this same call. That is what makes
`[photo] "half of this was my brother's"` resolve to ONE meal at half portions
instead of a photo-meal plus a text-meal.
"""

from __future__ import annotations

import base64
import io
import mimetypes
from pathlib import Path

from langchain_core.messages import HumanMessage

from .schemas import DetectedItem, PlateAnalysis

MAX_IMAGE_BYTES = 8 * 1024 * 1024

_PROMPT = """\
You are a dietitian reading a photo of one person's meal. Work in household \
units, never grams.

{priors}

Work through this in order, then answer:

1. IDENTIFY every distinct food you can see. Use the everyday name a home cook \
would use.
2. FIND YOUR RULER. A photo has no absolute scale, so measure against something \
standard in the frame. Typical sizes: dinner plate ~27cm across, side plate \
~20cm, katori/small bowl ~150ml, mug or teacup ~200ml, tablespoon ~15ml. State \
which one you used.
3. SIZE each food against that ruler, in household units: pieces for breads and \
whole items, katori for curries, rice and dal, cup or glass for drinks, \
serving for anything else.
4. RATE YOUR CONFIDENCE TWICE, separately, and be honest:
   - id_confidence: how sure you are WHAT the food is.
   - portion_confidence: how sure you are HOW MUCH there is.
   These are usually different. Recognising biryani is easy; knowing whether \
that is one katori or two is not. A portion_confidence of 0.4 on a dish you \
can name confidently is a normal, useful answer -- do not inflate it.
5. If a food could plausibly be one of several things, put the others in \
`alternatives` rather than picking one silently.
{caption_rule}
Return only the structured result."""

_CAPTION_RULE = """\

6. THE CAPTION BELOW DESCRIBES THIS SAME PLATE. It is not a second meal.
   Use it to help you IDENTIFY the food -- if it names a dish, trust it over \
your own guess and raise id_confidence.
   Do NOT adjust the amounts for it. If it says the person only ate part of the \
plate, still report what is VISIBLE ON THE PLATE; the sharing fraction is \
applied afterwards.

   Caption: "{caption}"
"""

_NO_CAPTION_RULE = ""


#: Longest edge sent to the vision model. Phone photos are 3-12MP, and uploading
#: one is most of the image path's latency -- a 2.3MB plate measured ~11s
#: end to end, most of it transfer. Vision models tile images internally at a
#: few hundred pixels anyway, so past roughly this size you are paying to send
#: detail the model discards. Identifying dal in a bowl does not need 12MP.
MAX_EDGE_PX = 768
JPEG_QUALITY = 85


def _encode(path: str | Path) -> tuple[str, str]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"no image at {file_path}")
    size = file_path.stat().st_size
    if size > MAX_IMAGE_BYTES:
        raise ValueError(f"image is {size / 1e6:.1f}MB; limit is {MAX_IMAGE_BYTES / 1e6:.0f}MB")

    downscaled = _downscale(file_path)
    if downscaled is not None:
        return base64.b64encode(downscaled).decode("ascii"), "image/jpeg"

    mime = mimetypes.guess_type(str(file_path))[0] or "image/jpeg"
    return base64.b64encode(file_path.read_bytes()).decode("ascii"), mime


def _downscale(file_path: Path) -> bytes | None:
    """Shrink to MAX_EDGE_PX. Returns None if Pillow is missing or the image is
    already small, in which case the original bytes are sent unchanged -- this
    is an optimisation, never a hard dependency."""
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(file_path) as img:
            if max(img.size) <= MAX_EDGE_PX and file_path.stat().st_size < 400_000:
                return None
            img = img.convert("RGB")
            img.thumbnail((MAX_EDGE_PX, MAX_EDGE_PX), Image.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=True)
            return buffer.getvalue()
    except Exception:
        return None


def build_prompt(caption: str | None, priors: str | None) -> str:
    return _PROMPT.format(
        priors=priors or "You know nothing about this person's diet yet.",
        caption_rule=_CAPTION_RULE.format(caption=caption.strip())
        if caption and caption.strip()
        else _NO_CAPTION_RULE,
    )


def _mock_analysis(image_path: str, caption: str | None) -> PlateAnalysis:
    """Offline stand-in so the image path is exercisable with no key.

    Both branches need to be demonstrable without a key, so the filename
    selects which one: any path containing "ambiguous" comes back with an
    item the model cannot identify, and the agent asks. Everything else comes
    back confident enough to log, with portions still uncertain -- which is the
    realistic shape, since portion is the hard part.
    """
    ambiguous = "ambiguous" in Path(image_path).stem.lower()
    items = [
        DetectedItem(name="rice", qty=1.0, unit="katori",
                     id_confidence=0.95, portion_confidence=0.45),
        DetectedItem(name="dal", qty=1.0, unit="katori",
                     id_confidence=0.88, portion_confidence=0.50),
        DetectedItem(
            name="paneer", qty=0.5, unit="katori",
            id_confidence=0.35 if ambiguous else 0.82,
            portion_confidence=0.40,
            alternatives=["tofu"] if ambiguous else [],
        ),
    ]
    analysis = PlateAnalysis(
        items=items,
        scale_reference="dinner plate ~27cm (mock)",
        notes="deterministic mock analysis",
    )
    return apply_caption_multiplier(analysis, caption)


_PORTION_WORDS: list[tuple[str, float]] = [
    ("two thirds", 0.67), ("two-thirds", 0.67),
    ("three quarters", 0.75), ("three-quarters", 0.75),
    ("a third", 0.33), ("one third", 0.33),
    ("a quarter", 0.25), ("quarter", 0.25),
    ("half of this", 0.5), ("half of it", 0.5), ("half", 0.5),
]


def apply_caption_multiplier(analysis: PlateAnalysis, caption: str | None) -> PlateAnalysis:
    """Apply "half of this was my brother's" to the portions -- in code, once.

    Responsibility is split deliberately: the model reads the caption for
    *identification* only and reports what is visible on the plate, and the
    sharing fraction is applied here.

    An earlier version had the model apply the fraction *and* multiplied again
    here, on the theory that belt-and-braces was safe. It was not: against a
    real photo the model halved some items itself, so those got halved twice and
    came back at a quarter while others sat at a half. One meal, two different
    fractions, silently wrong. Arithmetic belongs in exactly one place.
    """
    if not caption:
        return analysis
    low = caption.lower()
    shared = any(
        word in low
        for word in ("brother", "shared", "split", "someone else", "sister", "friend", "didn't finish", "did not finish", "left")
    )
    fraction = next((value for phrase, value in _PORTION_WORDS if phrase in low), None)
    if fraction is None or not shared:
        return analysis

    analysis.notes = (analysis.notes + f" caption fraction {fraction:g} applied").strip()
    for item in analysis.items:
        item.qty = round(item.qty * fraction, 3)
        # a shared plate is a *less* certain portion, not a more certain one
        item.portion_confidence = min(item.portion_confidence, 0.5)
    return analysis


def analyse_plate(
    image_path: str, caption: str | None = None, priors: str | None = None
) -> PlateAnalysis:
    """One image (+ caption) -> one PlateAnalysis. Never raises: a failed
    analysis returns `failed=True` so the agent can ask instead of crashing."""
    from .llm import get_vision_model

    model = get_vision_model()
    if getattr(model, "_llm_type", "") == "calorai-mock":
        return _mock_analysis(image_path, caption)

    try:
        encoded, mime = _encode(image_path)
    except (FileNotFoundError, ValueError) as exc:
        return PlateAnalysis(failed=True, failure_reason=str(exc))

    message = HumanMessage(
        content=[
            {"type": "text", "text": build_prompt(caption, priors)},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
        ]
    )

    from .llm import get_fallback_vision_model

    # Try the primary, then a second provider if there is one. Both are given a
    # short timeout rather than a long retry: measured, the worst case on a
    # throttled free tier was the client sitting on a 429 until it timed out,
    # which is 25 seconds of nothing. Swapping providers is faster than waiting.
    candidates = [model]
    backup = get_fallback_vision_model()
    if backup is not None:
        candidates.append(backup)

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            structured = candidate.with_structured_output(PlateAnalysis)
            result = structured.invoke([message])
            analysis = (
                result if isinstance(result, PlateAnalysis) else PlateAnalysis(**dict(result))
            )
            return apply_caption_multiplier(analysis, caption)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
            last_error = exc

    return PlateAnalysis(failed=True, failure_reason=f"vision model error: {last_error}")
