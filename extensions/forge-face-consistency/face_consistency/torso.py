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

import numpy as np

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




def _box_blur_5(a):
    """5x5 mean blur, pure numpy (edge-padded), for navel refinement."""
    k, pad = 5, 2
    p = np.pad(a, pad, mode="edge")
    c = np.cumsum(p, axis=1)
    c = np.concatenate([np.zeros((c.shape[0], 1)), c], axis=1)
    s = c[:, k:] - c[:, :-k]
    c = np.cumsum(s, axis=0)
    c = np.concatenate([np.zeros((1, c.shape[1])), c], axis=0)
    s = c[k:, :] - c[:-k, :]
    return s / float(k * k)


def refine_navel_center(gray, cx, cy, search=64, band_half=24,
                        min_contrast=12.0):
    """Re-center a navel estimate on the darkest midline feature.

    The geometric estimate (shoulders/hips keypoints) can sit tens of
    pixels off from pose-keypoint noise; on bare skin the navel is the
    darkest small feature near the midline. ``gray`` is a 2D array.
    Returns (x, y, confirmed): ``confirmed`` is True when a feature
    darker than ``min_contrast`` below the local median was found.
    Falls back to (cx, cy, False) when nothing confident is found
    (uniform skin, clothing) -- never a wild guess. Callers must not
    repair at an unconfirmed center: the img2img "navel close-up"
    prompt paints a navel where the box is centered, so an off-center
    repair creates a SECOND navel (observed live 2026-10-09).
    """
    h, w = gray.shape[:2]
    cx_i, cy_i = int(round(cx)), int(round(cy))
    x0, x1 = max(0, cx_i - search), min(w, cx_i + search)
    y0, y1 = max(0, cy_i - search), min(h, cy_i + search)
    if x1 <= x0 or y1 <= y0:
        return (cx_i, cy_i, False)
    win = np.asarray(gray[y0:y1, x0:x1], dtype=np.float32)
    blurred = _box_blur_5(win)
    bx = cx_i - x0
    b0, b1 = max(0, bx - band_half), min(win.shape[1], bx + band_half)
    if b1 <= b0:
        return (cx_i, cy_i, False)
    band = blurred[:, b0:b1]
    jy, jx = np.unravel_index(int(np.argmin(band)), band.shape)
    contrast = float(np.median(win) - band[jy, jx])
    if contrast < min_contrast:
        return (cx_i, cy_i, False)
    return (x0 + b0 + int(jx), y0 + int(jy), True)


def skin_fraction(rgb, cx, cy, half=10):
    """Fraction of skin-like pixels in the (2*half+1)^2 patch at (cx, cy).

    Skin heuristic: bright and red-dominant (R > 95, R > G >= B,
    R - B > 15). Used by the navel detailer to avoid "repairing"
    clothing -- a covered navel must skip, not repaint denim.
    """
    h, w = rgb.shape[:2]
    cx_i, cy_i = int(round(cx)), int(round(cy))
    x0, x1 = max(0, cx_i - half), min(w, cx_i + half + 1)
    y0, y1 = max(0, cy_i - half), min(h, cy_i + half + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    px = np.asarray(rgb[y0:y1, x0:x1]).astype(int)
    r, g, b = px[..., 0], px[..., 1], px[..., 2]
    skin = (r > 95) & (r > g) & (g >= b) & ((r - b) > 15)
    return float(skin.mean())


def pick_repair_center(rgb, refined, confirmed, threshold=0.45):
    """Choose the navel-detailer repair center, or None to skip.

    Repairs ONLY at a confirmed, on-skin refined point. There is no
    estimate fallback: the geometric estimate runs ~40-50px high in
    practice, and repairing at an unconfirmed center makes the img2img
    "navel close-up" prompt paint a navel where the box is centered --
    a second navel next to the real one (observed live 2026-10-09).
    Skipping leaves the base render, which is always safer than a
    hallucinated one. Returns (x, y) or None.
    """
    if not confirmed:
        return None
    cx, cy = refined
    if skin_fraction(rgb, cx, cy) >= threshold:
        return (int(round(cx)), int(round(cy)))
    return None


# --- Anatomical plausibility zone for the navel detailer ----------------

# A real navel sits on the torso midline, ~61% of the way from the
# shoulder line to the hip line (torso standard: navel 2.85 heads below
# crown; shoulders 1.39, crotch 3.79 -> (2.85-1.39)/(3.79-1.39) = 0.608).
# refine_navel_center() confirms the darkest feature within +-24px of the
# keypoint midline estimate -- but keypoint noise can shift that band onto
# a lateral dark feature (mole/shadow/fold). Observed 2026-10-09
# (seed 20262003): the detailer "confirmed" a mole at (285,680) and
# painted a navel on her SIDE, off-midline. This guard rejects any
# refined point outside the anatomical zone before a repair is allowed.
NAVEL_ZONE_LATERAL_FRAC = 0.12  # of shoulder width, from the midline segment
NAVEL_ZONE_TOP_FRAC = 0.45      # of shoulder->hip distance
NAVEL_ZONE_BOTTOM_FRAC = 0.80   # of shoulder->hip distance


def _point_segment_dist(px, py, ax, ay, bx, by):
    """Perpendicular distance from point (px, py) to segment AB."""
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom <= 0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / denom
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def navel_in_anatomical_zone(kp, x, y):
    """(ok, reason): is (x, y) a plausible navel position?

    ``kp`` is {name: (x, y)} as produced by
    ``body_gate.normalize_keypoints`` (pixel space). A navel must lie
    close to the shoulder-mid -> hip-mid segment (laterally) and between
    45% and 80% of the way from the shoulder line to the hip line
    (vertically). Anything else -- a mole on the side, a shadow on the
    waistband, a fold -- is not a navel and must never be repaired.
    Fail-open when landmarks are unusable: the detailer already requires
    shoulders+hips for its geometric estimate before this is consulted.
    """
    try:
        sh_r, sh_l = kp["shoulder_r"], kp["shoulder_l"]
        hip_r, hip_l = kp["hip_r"], kp["hip_l"]
        sh_m = _mid(sh_r, sh_l)
        hip_m = _mid(hip_r, hip_l)
        shoulder_w = math.hypot(sh_r[0] - sh_l[0], sh_r[1] - sh_l[1])
    except (KeyError, TypeError, IndexError):
        return True, "landmarks missing; zone check skipped"
    if not (math.isfinite(shoulder_w) and shoulder_w > 0):
        return True, "landmarks missing; zone check skipped"
    if not (math.isfinite(sh_m[1]) and math.isfinite(hip_m[1])
            and hip_m[1] > sh_m[1]):
        return True, "landmarks missing; zone check skipped"
    try:
        x, y = float(x), float(y)
    except (TypeError, ValueError):
        return False, "non-numeric point"
    if not (math.isfinite(x) and math.isfinite(y)):
        return False, "non-finite point"
    lateral = _point_segment_dist(x, y, sh_m[0], sh_m[1],
                                  hip_m[0], hip_m[1])
    limit = NAVEL_ZONE_LATERAL_FRAC * shoulder_w
    if lateral > limit:
        return False, (f"off-midline ({lateral:.0f}px lateral, "
                       f"limit {limit:.0f}px)")
    span = hip_m[1] - sh_m[1]
    top = sh_m[1] + NAVEL_ZONE_TOP_FRAC * span
    bottom = sh_m[1] + NAVEL_ZONE_BOTTOM_FRAC * span
    if not (top <= y <= bottom):
        return False, (f"outside vertical navel band (y={y:.0f}, "
                       f"band {top:.0f}-{bottom:.0f})")
    return True, "in anatomical navel zone"


# --- Pose frontality gate for the navel detailer ------------------------

# Maximum allowed horizontal offset of the nose from the shoulder
# midline, as a fraction of shoulder width. The navel detailer's
# geometric estimate assumes the 2D keypoint midline tracks the
# anatomical midline. That assumption breaks in 3/4 poses: the head and
# torso turn puts the projected midline somewhere that is NOT the
# anatomical midline, so refine() "confirms" a dark feature on the side
# of the torso and the detailer paints a navel there. Observed
# 2026-10-09 (seed 20262003): nose 30% of shoulder width off the
# shoulder midline; the keypoint midline ran through the sheer side
# panel and a navel was painted "on her side", while the anatomical
# zone guard (same bad keypoints) passed it. Beyond this threshold the
# detailer skips -- the base render is always safer than a repair from
# untrustworthy geometry.
POSE_FRONTAL_NOSE_FRAC = 0.25


def pose_frontal_enough(kp):
    """(ok, reason): is the pose frontal enough for the navel detailer?

    Checks that the nose sits above the shoulder midline (within
    POSE_FRONTAL_NOSE_FRAC of shoulder width). A large offset means the
    head/torso are turned enough that the 2D keypoint midline no longer
    tracks the anatomical midline, and any "confirmed" navel position is
    untrustworthy. Fail-open when landmarks are missing: the detailer
    already requires shoulders+hips before this is consulted.
    """
    try:
        nose = kp["nose"]
        sh_m = _mid(kp["shoulder_r"], kp["shoulder_l"])
        shoulder_w = math.hypot(kp["shoulder_r"][0] - kp["shoulder_l"][0],
                                kp["shoulder_r"][1] - kp["shoulder_l"][1])
    except (KeyError, TypeError, IndexError):
        return True, "landmarks missing; frontality check skipped"
    if not (math.isfinite(shoulder_w) and shoulder_w > 0):
        return True, "landmarks missing; frontality check skipped"
    try:
        offset = abs(float(nose[0]) - float(sh_m[0]))
    except (TypeError, ValueError, IndexError):
        return True, "landmarks missing; frontality check skipped"
    if not math.isfinite(offset):
        return True, "landmarks missing; frontality check skipped"
    limit = POSE_FRONTAL_NOSE_FRAC * shoulder_w
    if offset > limit:
        return False, (f"head {offset:.0f}px off shoulder midline "
                       f"(limit {limit:.0f}px); 3/4 pose, "
                       f"midline untrustworthy")
    return True, "pose frontal enough"


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
