"""Torso/navel consistency helpers (pure logic, no Forge imports).

Implements the measurable half of VJ_TORSO_STANDARD.md:

- expected navel position from OpenPose keypoints
- waist-ROI template matching (normalized cross-correlation, numpy only)
- navel-position assessment -- warn-only. The honest ceiling, confirmed
  by the 2026-10-08 research pass, is stable navel *position* plus clean
  rendering; no system preserves a pixel-identical navel.

Like body_gate.py, this module must import without Forge, gradio, torch,
cv2, or InsightFace so the offline test suite can exercise it. Only
stdlib and numpy are used here.
"""

from __future__ import annotations

import math

# Vertical landmarks from the torso standard (heads below crown):
# shoulders 1.39, waist 2.68, crotch 3.79. OpenPose has no waist
# keypoint, so the waist line is interpolated between the shoulder and
# hip lines at the fraction where the measured waist sits.
WAIST_SHOULDER_HIP_FRACTION = (2.68 - 1.39) / (3.79 - 1.39)

# Navel sits 15% of the way from the waist line to the crotch line, on
# the body midline (measured from the front full-body reference).
NAVEL_WAIST_CROTCH_FRACTION = 0.15

# Template-match confidence floor. Below this the "detection" is not
# trusted: the gate reports low confidence instead of a position, never
# a false failure.
NCC_SCORE_MIN = 0.35

# Warn when the detected navel is further than this (in head heights)
# from the expected position.
DEVIATION_WARN_HEADS = 0.20


def _mid(a, b):
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def expected_navel_px(kp):
    """Expected navel (x, y) in image pixels from normalized keypoints.

    ``kp`` is {name: (x, y)} as produced by
    ``body_gate.normalize_keypoints``. Returns None when the shoulders
    or hips are missing/unusable.
    """
    try:
        sh = _mid(kp["shoulder_r"], kp["shoulder_l"])
        hp = _mid(kp["hip_r"], kp["hip_l"])
    except (KeyError, TypeError, IndexError):
        return None
    try:
        waist_x = sh[0] + WAIST_SHOULDER_HIP_FRACTION * (hp[0] - sh[0])
        waist_y = sh[1] + WAIST_SHOULDER_HIP_FRACTION * (hp[1] - sh[1])
        navel_x = waist_x + NAVEL_WAIST_CROTCH_FRACTION * (hp[0] - waist_x)
        navel_y = waist_y + NAVEL_WAIST_CROTCH_FRACTION * (hp[1] - waist_y)
    except (TypeError, IndexError):
        return None
    if not all(math.isfinite(v) for v in (navel_x, navel_y)):
        return None
    return (navel_x, navel_y)


def crop_side_px(img_w, img_h, frac, min_px):
    """Side length for a square crop: frac of the smaller image side."""
    return max(int(min_px), int(min(img_w, img_h) * float(frac)))


def square_box(img_w, img_h, cx, cy, side):
    """Square (x0, y0, x1, y1) box of ``side`` px around (cx, cy).

    Clamped to the image; returns None when nothing usable remains.
    """
    side = int(side)
    if side <= 0:
        return None
    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))
    x0 = max(0, min(x0, img_w - 1))
    y0 = max(0, min(y0, img_h - 1))
    x1 = min(img_w, x0 + side)
    y1 = min(img_h, y0 + side)
    # Re-anchor after clamping so the box keeps its size when possible.
    x0 = max(0, x1 - side)
    y0 = max(0, y1 - side)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def torso_box_for_crop(kp, img_w, img_h, margin_frac=0.30):
    """Square torso crop box from keypoints, or None.

    Spans the shoulder line to the hip line with a margin, then squared.
    Used to build the canonical square torso crop that conditions the
    IP-Adapter / depth units.
    """
    try:
        xs = [kp["shoulder_r"][0], kp["shoulder_l"][0],
              kp["hip_r"][0], kp["hip_l"][0]]
        ys = [kp["shoulder_r"][1], kp["shoulder_l"][1],
              kp["hip_r"][1], kp["hip_l"][1]]
    except (KeyError, TypeError, IndexError):
        return None
    try:
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x0, x1, y0, y1)):
        return None
    mx = (x1 - x0) * float(margin_frac)
    my = (y1 - y0) * float(margin_frac)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    side = max(x1 - x0 + 2 * mx, y1 - y0 + 2 * my)
    return square_box(img_w, img_h, cx, cy, side)


def _to_gray(arr):
    """2D float32 grayscale from an RGB/RGBA/gray numpy array."""
    import numpy as np

    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 3:
        if a.shape[2] >= 3:
            a = (a[..., 0] * 0.299 + a[..., 1] * 0.587
                 + a[..., 2] * 0.114)
        else:
            a = a[..., 0]
    return np.ascontiguousarray(a)


def ncc_detect(roi, template, max_side=128):
    """Normalized cross-correlation template match (numpy only).

    ``roi`` and ``template`` are image arrays (grayscale or RGB).
    Returns ``(cx, cy, score)`` -- the template center in ROI pixel
    coordinates and the NCC score in [-1, 1] -- or
    ``(None, None, 0.0)`` when matching is not possible (template
    larger than the ROI, flat template, ...).
    """
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view

    r = _to_gray(roi)
    t = _to_gray(template)
    # Downscale so the window tensor stays small; coordinates are
    # mapped back to full ROI resolution at the end.
    factor = 1.0
    if max(r.shape) > max_side:
        factor = max_side / float(max(r.shape))
        r = r[:: int(round(1.0 / factor)), :: int(round(1.0 / factor))]
        t = t[:: int(round(1.0 / factor)), :: int(round(1.0 / factor))]
    rh, rw = r.shape
    th, tw = t.shape
    if th > rh or tw > rw or th < 2 or tw < 2:
        return None, None, 0.0
    t = t - t.mean()
    t_norm = math.sqrt(float((t ** 2).sum()))
    if t_norm < 1e-9:
        return None, None, 0.0
    wins = sliding_window_view(r, (th, tw))
    wmean = wins.mean(axis=(2, 3), keepdims=True)
    centered = wins - wmean
    num = (centered * t).sum(axis=(2, 3))
    den = np.sqrt((centered ** 2).sum(axis=(2, 3))) * t_norm
    ncc = np.divide(num, den, out=np.zeros_like(num),
                    where=den > 1e-9)
    iy, ix = (int(v) for v in np.unravel_index(int(np.argmax(ncc)),
                                              ncc.shape))
    score = float(ncc[iy, ix])
    up = 1.0 / factor
    return (ix + tw / 2.0) * up, (iy + th / 2.0) * up, score


def assess_navel(expected, detected, score, head_h):
    """(deviation_heads_or_None, warning_or_None) for the navel gate.

    ``expected`` / ``detected`` are (x, y) pixel tuples (detected may be
    None). ``head_h`` is the head height in px, or None. Below
    :data:`NCC_SCORE_MIN` the detection is not trusted: the deviation
    is not reported and the warning says so honestly. Otherwise a
    deviation beyond :data:`DEVIATION_WARN_HEADS` warns.
    """
    if detected is None or detected[0] is None:
        return None, "navel gate: navel not detected in waist ROI"
    if score < NCC_SCORE_MIN:
        return None, (
            "navel gate: low template-match confidence "
            f"(score={score:.2f}); navel position not verified")
    dev_px = math.hypot(detected[0] - expected[0],
                        detected[1] - expected[1])
    dev_heads = (dev_px / head_h) if head_h else None
    if dev_heads is not None and dev_heads > DEVIATION_WARN_HEADS:
        return dev_heads, (
            f"navel {dev_heads:.2f} heads from expected position "
            "(expected: midline, 15% waist->crotch)")
    return dev_heads, None
