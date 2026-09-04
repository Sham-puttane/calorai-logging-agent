"""Vision path: uncertainty routing, caption handling, and payload size.

The caption tests are regression tests for a real bug. An earlier version had
the model apply "half of this was my brother's" *and* multiplied again in code,
so against a real photo some items were halved twice and came back at a quarter
while others sat at a half. One meal, two fractions, silently wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from calorai import vision  # noqa: E402
from calorai.schemas import (  # noqa: E402
    ID_CONFIDENCE_THRESHOLD,
    PORTION_CONFIDENCE_THRESHOLD,
    DetectedItem,
    PlateAnalysis,
)

IMAGES = Path(__file__).resolve().parents[1] / "images"


def plate(**overrides) -> PlateAnalysis:
    return PlateAnalysis(
        items=[
            DetectedItem(name="rice", qty=1.0, unit="katori",
                         id_confidence=0.95, portion_confidence=0.7),
            DetectedItem(name="dal", qty=2.0, unit="katori",
                         id_confidence=0.9, portion_confidence=0.7),
        ],
        scale_reference="dinner plate ~27cm",
        **overrides,
    )


# ---------------------------------------------------------------------------
# caption -> portions, applied exactly once
# ---------------------------------------------------------------------------
def test_sharing_caption_halves_every_item_by_the_same_factor():
    result = vision.apply_caption_multiplier(plate(), "half of this was my brother's")
    assert [i.qty for i in result.items] == [0.5, 1.0]


def test_the_fraction_is_applied_once_not_per_item():
    """The bug: one caption producing two different fractions across one meal."""
    result = vision.apply_caption_multiplier(plate(), "half of this was my brother's")
    ratios = {round(after.qty / before.qty, 3)
              for before, after in zip(plate().items, result.items)}
    assert ratios == {0.5}, f"one caption must yield one factor, got {ratios}"


@pytest.mark.parametrize(
    "caption,factor",
    [
        ("half of this was my brother's", 0.5),
        ("my sister ate a third of it", 0.33),
        ("shared this, had about two thirds", 0.67),
        ("split a quarter of this with a friend", 0.25),
    ],
)
def test_fraction_words_are_read_from_the_caption(caption, factor):
    result = vision.apply_caption_multiplier(plate(), caption)
    assert result.items[0].qty == pytest.approx(factor, rel=0.01)


@pytest.mark.parametrize(
    "caption",
    [
        None,
        "",
        "this was lunch",
        "half of these were amazing",      # "half" with no sharing -> not a portion cut
        "had this after the gym",
    ],
)
def test_portions_are_untouched_without_an_actual_sharing_statement(caption):
    result = vision.apply_caption_multiplier(plate(), caption)
    assert [i.qty for i in result.items] == [1.0, 2.0]


def test_a_shared_plate_lowers_portion_confidence():
    """Sharing makes the amount *less* certain, never more."""
    result = vision.apply_caption_multiplier(plate(), "half of this was my brother's")
    assert all(i.portion_confidence <= 0.5 for i in result.items)


# ---------------------------------------------------------------------------
# uncertainty routing -- the asymmetric thresholds
# ---------------------------------------------------------------------------
def test_unidentifiable_food_asks_and_names_the_alternatives():
    analysis = PlateAnalysis(items=[
        DetectedItem(name="paneer", qty=1, unit="katori",
                     id_confidence=ID_CONFIDENCE_THRESHOLD - 0.1,
                     portion_confidence=0.8, alternatives=["tofu"]),
    ])
    assert analysis.needs_user_input()
    question = analysis.clarifying_question()
    assert "paneer" in question and "tofu" in question
    assert question.endswith("?")


def test_uncertain_portion_logs_anyway_and_states_the_assumption():
    """Asymmetric on purpose: scale is close to unknowable from one photo, so
    gating on portion the way we gate on identification would make the agent
    ask about nearly every image."""
    analysis = PlateAnalysis(items=[
        DetectedItem(name="rice", qty=1, unit="katori",
                     id_confidence=0.95,
                     portion_confidence=PORTION_CONFIDENCE_THRESHOLD - 0.1),
    ])
    assert not analysis.needs_user_input(), "a shaky portion must not block logging"
    assert analysis.clarifying_question() is None
    assert analysis.assumptions(), "but it must be admitted to in the reply"


def test_thresholds_are_asymmetric():
    assert PORTION_CONFIDENCE_THRESHOLD < ID_CONFIDENCE_THRESHOLD


def test_a_failed_analysis_asks_rather_than_raising():
    analysis = PlateAnalysis(failed=True, failure_reason="no image at nope.jpg")
    assert analysis.needs_user_input()
    assert analysis.clarifying_question().endswith("?")


def test_an_empty_plate_asks():
    assert PlateAnalysis(items=[]).needs_user_input()


def test_missing_file_returns_a_failure_not_an_exception():
    result = vision.analyse_plate("does/not/exist.jpg", None, None)
    assert isinstance(result, PlateAnalysis)


# ---------------------------------------------------------------------------
# prompt construction
# ---------------------------------------------------------------------------
def test_caption_prompt_forbids_the_model_adjusting_amounts():
    """The model identifies; the code does the arithmetic. If this instruction
    is lost, the double-application bug comes back."""
    prompt = vision.build_prompt("half of this was my brother's", None)
    assert "Do NOT adjust the amounts" in prompt
    assert "half of this was my brother's" in prompt


def test_memory_priors_reach_the_vision_prompt():
    prompt = vision.build_prompt(None, "Known about this person: vegetarian.")
    assert "vegetarian" in prompt


def test_prompt_asks_for_both_confidences_and_a_scale_reference():
    prompt = vision.build_prompt(None, None)
    assert "id_confidence" in prompt
    assert "portion_confidence" in prompt
    assert "RULER" in prompt


# ---------------------------------------------------------------------------
# payload size
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not (IMAGES / "plate.jpg").exists(), reason="no sample image")
def test_photos_are_downscaled_before_upload():
    original = (IMAGES / "plate.jpg").stat().st_size
    shrunk = vision._downscale(IMAGES / "plate.jpg")
    assert shrunk is not None
    assert len(shrunk) < original * 0.5, "upload is most of the image path's latency"


def test_downscaling_is_optional_not_required():
    """Pillow is an optimisation. Without it the original bytes still send."""
    assert vision._downscale(Path("does/not/exist.jpg")) is None


# ---------------------------------------------------------------------------
# failure messages -- blame the right thing
# ---------------------------------------------------------------------------
def test_a_rate_limit_does_not_get_blamed_on_the_photo():
    """Observed live: a perfectly readable plate of paneer came back as "I
    couldn't read that photo clearly" because both vision providers were
    throttled. That sends the user off to check an image that was fine."""
    analysis = PlateAnalysis(
        failed=True,
        failure_reason="vision model error: Error code: 429 RESOURCE_EXHAUSTED",
    )
    question = analysis.clarifying_question()
    assert "rate limited" in question
    assert "read that photo" not in question


def test_a_missing_file_says_so():
    analysis = PlateAnalysis(failed=True, failure_reason="no image at nope.jpg")
    assert "couldn't open that file" in analysis.clarifying_question()


def test_an_unexplained_failure_still_asks_for_the_food():
    analysis = PlateAnalysis(failed=True, failure_reason="something odd")
    question = analysis.clarifying_question()
    assert question.endswith("?")
    assert "on the plate" in question


@pytest.mark.parametrize(
    "message,throttled",
    [
        ("Error code: 429 - rate_limit_exceeded", True),
        ("RESOURCE_EXHAUSTED quota", True),
        ("Rate limit exceeded", True),
        ("no image at nope.jpg", False),
        ("Invalid API Key", False),
        ("connection reset", False),
    ],
)
def test_throttle_detection(message, throttled):
    """Only throttling earns a retry -- never a bad file or a bad key."""
    assert vision._is_throttle(RuntimeError(message)) is throttled


def test_no_exception_is_not_a_throttle():
    assert vision._is_throttle(None) is False
