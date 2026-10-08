"""Pure decision logic for Forge Face Consistency.

This module must import without Forge, gradio, torch, or InsightFace so the
offline test suite can exercise it in a bare python environment. Only stdlib
and (optionally) numpy are used here; numpy is only needed by the template
helpers in swap_engine, not by the functions in this file.
"""

from __future__ import annotations

MAX_STRENGTH = 1.0
EPS = 1e-6

# Preprocessor names exactly as registered by
# extensions-builtin/sd_forge_ipadapter/scripts/forge_ipadapter.py
PREPROCESSOR_FACEID = "InsightFace+CLIP-H (IPAdapter)"
PREPROCESSOR_INSTANTID = "InsightFace (InstantID)"

# Same-person verify threshold (ArcFace cosine). Kept in sync with
# settings.DEFAULT_VERIFY_THRESHOLD; consumed by the script shell to warn
# when a swap lands below it (see is_below_threshold).
VERIFY_THRESHOLD = 0.55
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def is_max_strength(strength: float) -> bool:
    """True when the slider is at (or effectively at) the swap position."""
    try:
        s = float(strength)
    except (TypeError, ValueError):
        return False
    return s >= MAX_STRENGTH - 1e-9


def clamp_strength(strength: float) -> float:
    try:
        s = float(strength)
    except (TypeError, ValueError):
        return 0.0
    if s < 0.0:
        return 0.0
    if s > MAX_STRENGTH:
        return MAX_STRENGTH
    return s


def detect_family(is_sdxl: bool = False, checkpoint_hint: str = "",
                  class_name: str = "") -> str:
    """Classify the loaded checkpoint family.

    Returns 'flux', 'sdxl', or 'other'. Flux is checked first so a Flux
    checkpoint is never offered SDXL adapters, even if a stale flag says
    otherwise. SDXL detection mirrors Forge's engine flags
    (backend/diffusion_engine/*: only the SDXL engines set is_sdxl=True).
    """
    hint = (checkpoint_hint or "").lower()
    cls = (class_name or "").lower()
    if "flux" in cls or "flux" in hint or "chroma" in cls:
        return "flux"
    if is_sdxl:
        return "sdxl"
    if "sdxl" in hint or "xl" in cls:
        # Class-name hint only; filename hints alone are not trusted.
        if "sdxl" in cls or "xl" in cls:
            return "sdxl"
    return "other"


def find_adapter_model(controlnet_names):
    """Pick a FaceID / InstantID ControlNet model from available names.

    ``controlnet_names`` is the list from
    ``lib_controlnet.global_state.controlnet_names`` (display names, 'None'
    first). Returns ``(kind, preprocessor_name, model_name)`` where kind is
    'faceid' or 'instantid', or ``None`` when no matching model exists.
    FaceID (IP-Adapter) is preferred over InstantID when both are present,
    matching the research note's route ordering.
    """
    names = [n for n in (controlnet_names or []) if n and n != "None"]
    faceid = None
    instantid = None
    for name in names:
        low = name.lower()
        if faceid is None and "faceid" in low:
            faceid = name
        if instantid is None and ("instantid" in low or "instant_id" in low
                                 or "instant-id" in low):
            instantid = name
    if faceid is not None:
        return ("faceid", PREPROCESSOR_FACEID, faceid)
    if instantid is not None:
        return ("instantid", PREPROCESSOR_INSTANTID, instantid)
    return None


def decide_mode(strength: float, family: str, adapter_available: bool):
    """Map (strength, family, adapter availability) to an execution mode.

    Returns ``(mode, downgraded, reason)``:

    - mode 'swap': full swap + restore (strength at max, any family).
    - mode 'faceid' / 'instantid': ControlNet injection (SDXL, below max,
      adapter present). The caller resolves which of the two via
      :func:`find_adapter_model`; here both collapse to 'controlnet' intent,
      so this function returns 'faceid' as the representative and the caller
      refines it. To keep this function total and testable it returns:
    - mode 'controlnet': inject a unit at weight=strength.
    - mode 'blended-swap': blended swap at blend=strength (fallback).
    - mode 'disabled': strength <= 0.

    ``downgraded`` is True when the requested reference path could not be
    honoured and a fallback runs instead; ``reason`` explains why and must
    be surfaced in logs/infotext (never silent).
    """
    s = clamp_strength(strength)
    if s <= 0.0:
        return ("disabled", False, "strength is 0")
    if is_max_strength(s):
        return ("swap", False, "strength at max: full swap + restore")
    # Below max: a face reference is wanted.
    if family == "flux":
        return ("blended-swap", True,
                "Flux has no SDXL FaceID/InstantID adapter in Forge; "
                "using blended swap")
    if family == "sdxl":
        if adapter_available:
            return ("controlnet", False,
                    "SDXL reference via ControlNet FaceID/InstantID")
        return ("blended-swap", True,
                "no FaceID/InstantID ControlNet model found; "
                "using blended swap")
    # SD1.5 / SD3 / unknown: no vetted reference adapter either.
    return ("blended-swap", True,
            "no reference adapter for this model family; using blended swap")


def is_below_threshold(similarity, threshold: float = VERIFY_THRESHOLD) -> bool:
    """True when a measured ArcFace similarity is below the verify bar.

    None (no face measured) is not "below" — the caller reports it as
    n/a instead of a verify warning.
    """
    if similarity is None:
        return False
    try:
        return float(similarity) < float(threshold)
    except (TypeError, ValueError):
        return False


def list_ref_candidates(refs_dir: str):
    """All image files (sorted) inside ``refs_dir`` as joined paths.

    Pure os listing. The ControlNet unit needs one concrete image even
    when the user supplied a whole folder; callers try these in order
    until one decodes (repair round 3 / N2: a corrupt first file must
    not discard an otherwise usable folder).
    """
    import os

    if not refs_dir or not str(refs_dir).strip():
        return []
    folder = str(refs_dir).strip()
    if not os.path.isdir(folder):
        return []
    names = sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith(IMAGE_EXTENSIONS))
    return [os.path.join(folder, name) for name in names]


def pick_representative_ref(refs_dir: str):
    """First image file (sorted) inside ``refs_dir``, or None.

    Pure os listing: the ControlNet unit needs one concrete image even
    when the user supplied a whole folder. The blended/swap template
    path still uses the full folder; this only picks the unit image.
    Returns the absolute-ish path as given (joined), never decodes.
    """
    candidates = list_ref_candidates(refs_dir)
    return candidates[0] if candidates else None
