"""Automated body-proportion gate (pure logic, no Forge imports).

The face-similarity verify gate only checks the *face*. This module
checks the *body*: pose keypoints from the generated image are reduced
to a handful of proportions (heads-tall, shoulder:hip, widths in head
units, leg fraction) and compared against a reference profile measured
from the person (Settings -> Forge Face Consistency -> reference
proportions). Gross deviations are warned on, or -- in "reject" mode --
abort the generation before the image is presented.

Waist:hip is deliberately NOT measured: OpenPose-style keypoints carry
no waist landmark, and estimating it from keypoints alone would be
guesswork. This gate is a coarse screen for "that body is not theirs",
not a body-identity verifier.
"""

from __future__ import annotations

import math

# OpenPose keypoint indices for the two layouts the ControlNet
# annotators emit. Only the landmarks the gate needs are mapped.
COCO_18 = {
    "nose": 0, "neck": 1,
    "shoulder_r": 2, "shoulder_l": 5,
    "hip_r": 8, "hip_l": 11,
    "knee_r": 9, "knee_l": 12,
    "ankle_r": 10, "ankle_l": 13,
}
BODY_25 = {
    "nose": 0, "neck": 1,
    "shoulder_r": 2, "shoulder_l": 5,
    "hip_r": 9, "hip_l": 12,
    "knee_r": 10, "knee_l": 13,
    "ankle_r": 11, "ankle_l": 14,
}

# Keypoints below this confidence are treated as missing.
SCORE_MIN = 0.2


class BodyProportionError(RuntimeError):
    """Raised in "reject" mode when the body fails the proportion gate."""


def _point_of(item):
    """(x, y, score_or_None) from a tuple/list or a keypoint object."""
    if item is None:
        return None
    if hasattr(item, "x") and hasattr(item, "y"):
        score = getattr(item, "score", None)
        try:
            x, y = float(item.x), float(item.y)
            s = float(score) if score is not None else None
        except (TypeError, ValueError):
            return None
        return x, y, s
    try:
        seq = list(item)
    except TypeError:
        return None
    if len(seq) < 2:
        return None
    try:
        x, y = float(seq[0]), float(seq[1])
    except (TypeError, ValueError):
        return None
    score = None
    if len(seq) > 2:
        try:
            score = float(seq[2])
        except (TypeError, ValueError):
            score = None
    return x, y, score


def _valid(pt):
    if pt is None:
        return False
    x, y, score = pt
    if not (math.isfinite(x) and math.isfinite(y)):
        return False
    if score is not None and score < SCORE_MIN:
        return False
    return True


def normalize_keypoints(keypoints):
    """Named landmark dict from one pose's raw keypoint list.

    Accepts COCO-18 (18 points) or BODY_25 (25 points) layouts, chosen
    by length; anything else falls back to the COCO map with bounds
    checks. Returns {name: (x, y)} with only confident landmarks kept.
    """
    if not keypoints:
        return {}
    try:
        n = len(keypoints)
    except TypeError:
        return {}
    layout = BODY_25 if n == 25 else COCO_18
    out = {}
    for name, idx in layout.items():
        if 0 <= idx < n:
            pt = _point_of(keypoints[idx])
            if _valid(pt):
                out[name] = (pt[0], pt[1])
    return out


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def measure_ratios(kp, head_h=None):
    """Body proportions from normalized keypoints.

    ``kp``: {name: (x, y)} from :func:`normalize_keypoints`.
    ``head_h``: head height in px (face bbox height), or None.
    "Heads" as a unit means head *heights* (shoulders 2.2 heads =
    2.2 * head height). Returns {ratio_name: value}; a ratio is simply
    absent when its inputs are missing -- the caller assesses only
    what is measurable.
    """
    measured = {}
    sh_r, sh_l = kp.get("shoulder_r"), kp.get("shoulder_l")
    hip_r, hip_l = kp.get("hip_r"), kp.get("hip_l")
    shoulder_w = _dist(sh_r, sh_l) if sh_r and sh_l else 0.0
    hip_w = _dist(hip_r, hip_l) if hip_r and hip_l else 0.0
    if shoulder_w > 0 and hip_w > 0:
        measured["shoulder_hip"] = shoulder_w / hip_w
    try:
        hh = float(head_h)
    except (TypeError, ValueError):
        hh = 0.0
    if hh > 0:
        if shoulder_w > 0:
            measured["shoulder_heads"] = shoulder_w / hh
        if hip_w > 0:
            measured["hip_heads"] = hip_w / hh
    # Body height: crown (nose minus half a head) to lowest ankle.
    nose = kp.get("nose")
    ankles = [a for a in (kp.get("ankle_r"), kp.get("ankle_l")) if a]
    body_h = 0.0
    if nose and ankles and hh > 0:
        ankle_y = max(a[1] for a in ankles)
        body_h = ankle_y - (nose[1] - 0.5 * hh)
    if body_h > 0 and hh > 0:
        measured["heads_tall"] = body_h / hh
    if body_h > 0:
        legs = []
        for hkey, akey in (("hip_r", "ankle_r"), ("hip_l", "ankle_l")):
            h, a = kp.get(hkey), kp.get(akey)
            if h and a:
                legs.append(_dist(h, a))
        if legs:
            measured["leg_fraction"] = max(legs) / body_h
    return measured


def assess(measured, reference, tolerance):
    """Compare measured ratios against the reference profile.

    Returns (verdict, details): verdict is "pass", "fail", or
    "unmeasured" (nothing comparable); details maps each compared
    ratio to {"measured", "reference", "rel_dev", "ok"}.
    """
    try:
        tol = float(tolerance)
    except (TypeError, ValueError):
        tol = 0.15
    details = {}
    for name, m in (measured or {}).items():
        try:
            r = float((reference or {}).get(name))
            m = float(m)
        except (TypeError, ValueError):
            continue
        if r == 0 or m <= 0:
            continue
        rel = (m - r) / r
        details[name] = {"measured": m, "reference": r,
                         "rel_dev": rel, "ok": abs(rel) <= tol}
    if not details:
        return "unmeasured", {}
    verdict = "pass" if all(d["ok"] for d in details.values()) else "fail"
    return verdict, details


def decide_action(verdict, mode):
    """Gate action for a verdict under a mode ("off"/"warn"/"reject").

    Returns "ok", "warn", or "reject". Unknown modes fall back to
    "warn" (the safe default, matching DEFAULT_BODY_GATE).
    """
    m = str(mode or "warn").strip().lower()
    if m not in ("off", "warn", "reject"):
        m = "warn"
    if m == "off" or verdict in ("pass", "unmeasured"):
        return "ok"
    return m  # "warn" or "reject" on a "fail"


_RATIO_LABELS = {
    "heads_tall": "heads tall",
    "shoulder_hip": "shoulder:hip",
    "shoulder_heads": "shoulder width",
    "hip_heads": "hip width",
    "leg_fraction": "leg fraction",
}
_RATIO_UNITS = {
    "heads_tall": "",
    "shoulder_hip": "",
    "shoulder_heads": " heads",
    "hip_heads": " heads",
    "leg_fraction": "",
}


def format_assessment(details):
    """One-line infotext summary of the measured proportions."""
    parts = []
    for name, d in details.items():
        label = _RATIO_LABELS.get(name, name)
        unit = _RATIO_UNITS.get(name, "")
        parts.append(
            f"{d['measured']:.2f}{unit} {label} "
            f"(ref {d['reference']:.2f}{unit})")
    return ", ".join(parts)


def format_failures(details, tolerance):
    """One-line description of every ratio outside tolerance."""
    try:
        tol = float(tolerance)
    except (TypeError, ValueError):
        tol = 0.15
    return "; ".join(
        f"{_RATIO_LABELS.get(n, n)} {d['measured']:.2f} vs ref "
        f"{d['reference']:.2f} ({d['rel_dev']:+.1%}, tolerance {tol:.0%})"
        for n, d in details.items() if not d["ok"])
