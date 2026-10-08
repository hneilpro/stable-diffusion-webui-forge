"""Forge Face Consistency — reference strength with max = face swap."""

from face_consistency.logic import (
    MAX_STRENGTH,
    VERIFY_THRESHOLD,
    clamp_strength,
    decide_mode,
    detect_family,
    find_adapter_model,
    is_below_threshold,
    is_max_strength,
    list_ref_candidates,
    pick_representative_ref,
)

__all__ = [
    "MAX_STRENGTH",
    "VERIFY_THRESHOLD",
    "clamp_strength",
    "decide_mode",
    "detect_family",
    "find_adapter_model",
    "is_below_threshold",
    "is_max_strength",
    "list_ref_candidates",
    "pick_representative_ref",
]
