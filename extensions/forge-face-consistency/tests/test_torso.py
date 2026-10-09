"""Offline tests for the torso/navel consistency helpers (pure logic)."""

import math

import numpy as np
import pytest

from face_consistency import torso


def _kp():
    """Shoulders y=210 (x 290/510), hips y=480 (x 310/490)."""
    return {
        "shoulder_r": (510.0, 210.0),
        "shoulder_l": (290.0, 210.0),
        "hip_r": (490.0, 480.0),
        "hip_l": (310.0, 480.0),
    }


# --- expected_navel_px ------------------------------------------------


def test_expected_navel_px_midline_and_fractions():
    # shoulder mid (400, 210), hip mid (400, 480).
    # waist_y = 210 + 0.5375 * 270 = 355.125
    # navel_y = 355.125 + 0.15 * (480 - 355.125) = 373.85625
    x, y = torso.expected_navel_px(_kp())
    assert x == pytest.approx(400.0)
    assert y == pytest.approx(373.85625)


def test_expected_navel_px_none_when_landmarks_missing():
    assert torso.expected_navel_px({}) is None
    kp = _kp()
    del kp["hip_l"]
    assert torso.expected_navel_px(kp) is None


def test_waist_fraction_matches_standard():
    assert torso.WAIST_SHOULDER_HIP_FRACTION == pytest.approx(
        (2.68 - 1.39) / (3.79 - 1.39))
    assert torso.NAVEL_WAIST_CROTCH_FRACTION == pytest.approx(0.15)


# --- square_box / crop_side_px / torso_box_for_crop --------------------


def test_square_box_centered_and_clamped():
    assert torso.square_box(1000, 800, 500, 400, 200) == (400, 300, 600, 500)
    # Near the corner the box is re-anchored inside the image.
    x0, y0, x1, y1 = torso.square_box(1000, 800, 30, 30, 200)
    assert (x0, y0) == (0, 0) and (x1 - x0, y1 - y0) == (200, 200)
    assert torso.square_box(100, 100, 50, 50, 0) is None


def test_crop_side_px_respects_minimum():
    assert torso.crop_side_px(768, 1152, 0.18, 128) == max(
        128, int(768 * 0.18))
    assert torso.crop_side_px(100, 100, 0.18, 128) == 128


def test_torso_box_for_crop_spans_shoulders_to_hips():
    box = torso.torso_box_for_crop(_kp(), 1000, 1000)
    assert box is not None
    x0, y0, x1, y1 = box
    assert x1 - x0 == y1 - y0  # square
    assert x0 <= 290 and x1 >= 510  # shoulders inside (with margin)
    assert y0 <= 210 and y1 >= 480  # hips inside (with margin)


def test_torso_box_for_crop_none_when_missing():
    assert torso.torso_box_for_crop({}, 1000, 1000) is None


# --- ncc_detect --------------------------------------------------------


def _synthetic_roi():
    rng = np.random.RandomState(7)
    roi = (rng.rand(120, 120).astype(np.float32) * 40 + 90)
    # Bright navel-like blob, top-left at (60, 70), 20x20.
    roi[70:90, 60:80] += 120.0
    template = roi[70:90, 60:80].copy()
    return roi, template


def test_ncc_detect_finds_template():
    roi, template = _synthetic_roi()
    cx, cy, score = torso.ncc_detect(roi, template)
    assert score == pytest.approx(1.0, abs=1e-3)
    assert cx == pytest.approx(70.0, abs=2.0)
    assert cy == pytest.approx(80.0, abs=2.0)


def test_ncc_detect_template_bigger_than_roi():
    roi = np.zeros((20, 20), dtype=np.float32)
    tpl = np.zeros((40, 40), dtype=np.float32)
    assert torso.ncc_detect(roi, tpl) == (None, None, 0.0)


def test_ncc_detect_flat_template():
    roi = np.random.RandomState(1).rand(60, 60).astype(np.float32)
    tpl = np.full((10, 10), 5.0, dtype=np.float32)
    assert torso.ncc_detect(roi, tpl) == (None, None, 0.0)


# --- assess_navel -------------------------------------------------------


def test_assess_navel_ok_when_close():
    dev, warn = torso.assess_navel((400.0, 374.0), (402.0, 376.0),
                                   0.9, head_h=126.0)
    assert warn is None
    assert dev == pytest.approx(math.hypot(2.0, 2.0) / 126.0)


def test_assess_navel_warns_on_deviation():
    dev, warn = torso.assess_navel((400.0, 374.0), (400.0, 474.0),
                                   0.9, head_h=126.0)
    assert dev == pytest.approx(100.0 / 126.0)
    assert warn is not None and "heads from expected" in warn


def test_assess_navel_low_confidence_not_a_failure():
    dev, warn = torso.assess_navel((400.0, 374.0), (400.0, 474.0),
                                   0.2, head_h=126.0)
    assert dev is None
    assert warn is not None and "low template-match confidence" in warn


def test_assess_navel_no_detection():
    dev, warn = torso.assess_navel((400.0, 374.0), None, 0.0, 126.0)
    assert dev is None
    assert warn is not None


# --- refine_navel_center ------------------------------------------------


def test_refine_navel_center_finds_dark_blob():
    gray = np.full((768, 512), 220, dtype=np.uint8)
    yy, xx = np.mgrid[0:768, 0:512]
    blob = 60 * np.exp(-((xx - 300) ** 2 + (yy - 660) ** 2) / (2 * 6.0 ** 2))
    gray = (gray - blob).astype(np.uint8)
    x, y = torso.refine_navel_center(gray, 286, 615)
    assert abs(x - 300) <= 6 and abs(y - 660) <= 6


def test_refine_navel_center_falls_back_on_flat():
    gray = np.full((768, 512), 220, dtype=np.uint8)
    assert torso.refine_navel_center(gray, 286, 615) == (286, 615)


def test_refine_navel_center_ignores_off_midline_darkness():
    # Dark blob 60px off the midline: not a navel candidate, keep estimate.
    gray = np.full((768, 512), 220, dtype=np.uint8)
    yy, xx = np.mgrid[0:768, 0:512]
    blob = 60 * np.exp(-((xx - 160) ** 2 + (yy - 615) ** 2) / (2 * 6.0 ** 2))
    gray = (gray - blob).astype(np.uint8)
    assert torso.refine_navel_center(gray, 286, 615) == (286, 615)
