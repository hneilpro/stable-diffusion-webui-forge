"""Offline tests: pure logic must import and behave without Forge."""

import numpy as np
import pytest

from face_consistency import logic
from face_consistency.blend import blend_images, feather_mask
from face_consistency.swap_engine import template_embedding


# --- strength -> mode mapping ---------------------------------------------

def test_max_strength_is_swap_for_every_family():
    for family in ("sdxl", "flux", "other"):
        mode, downgraded, _ = logic.decide_mode(1.0, family, adapter_available=False)
        assert mode == "swap"
        assert downgraded is False


def test_sdxl_below_max_with_adapter_uses_controlnet():
    mode, downgraded, _ = logic.decide_mode(0.85, "sdxl", adapter_available=True)
    assert mode == "controlnet"
    assert downgraded is False


def test_sdxl_below_max_without_adapter_downgrades_loudly():
    mode, downgraded, reason = logic.decide_mode(0.85, "sdxl", adapter_available=False)
    assert mode == "blended-swap"
    assert downgraded is True
    assert "blended swap" in reason


def test_flux_never_gets_sdxl_adapter():
    mode, downgraded, reason = logic.decide_mode(0.5, "flux", adapter_available=True)
    assert mode == "blended-swap"
    assert downgraded is True
    assert "Flux" in reason


def test_zero_strength_is_disabled():
    mode, _, _ = logic.decide_mode(0.0, "sdxl", adapter_available=True)
    assert mode == "disabled"


def test_strength_helpers():
    assert logic.is_max_strength(1.0)
    assert not logic.is_max_strength(0.99)
    assert logic.clamp_strength(1.7) == 1.0
    assert logic.clamp_strength(-0.2) == 0.0


# --- family detection -------------------------------------------------------

def test_detect_family():
    assert logic.detect_family(is_sdxl=True) == "sdxl"
    # Class-name substring matching is deliberately NOT trusted for
    # "flux": this Forge build reports a flux-flavoured class name for
    # SDXL checkpoints (live 2026-10-08: iniverseMix SDXL detected as
    # flux). The checkpoint FILENAME is the primary signal.
    assert logic.detect_family(class_name="Flux") == "other"
    assert logic.detect_family(checkpoint_hint="flux1-dev.safetensors") == "flux"
    assert logic.detect_family(checkpoint_hint="chroma-v1.safetensors") == "flux"
    # The is_sdxl flag beats a stale flux-flavoured class name.
    assert logic.detect_family(is_sdxl=True, class_name="Flux") == "sdxl"
    # Regression: a flux-flavoured class name on an SDXL checkpoint must
    # not disable the SDXL path; "other" is the safe fallback.
    assert logic.detect_family(
        checkpoint_hint="iniverseMixSFWNSFW_f1dRealnsfwGuofengV2.safetensors",
        class_name="FluxEngine") == "other"
    assert logic.detect_family() == "other"


# --- adapter model picking ----------------------------------------------------

def test_find_adapter_model_prefers_faceid():
    names = ["None", "ip-adapter-faceid-plusv2_sdxl [abcd]",
             "instantid-controlnet [efgh]"]
    kind, preprocessor, model = logic.find_adapter_model(names)
    assert kind == "faceid"
    assert preprocessor == logic.PREPROCESSOR_FACEID
    assert model == names[1]


def test_find_adapter_model_instantid_fallback_and_none():
    kind, preprocessor, _ = logic.find_adapter_model(["None", "InstantID [1234]"])
    assert kind == "instantid"
    assert preprocessor == logic.PREPROCESSOR_INSTANTID
    assert logic.find_adapter_model(["None", "control_v11p_sd15_canny"]) is None
    assert logic.find_adapter_model([]) is None


# --- blend math -----------------------------------------------------------------

def test_blend_endpoints_and_midpoint():
    original = np.full((8, 8, 3), 0, dtype=np.uint8)
    swapped = np.full((8, 8, 3), 200, dtype=np.uint8)
    mask = np.ones((8, 8), dtype=np.float32)
    assert blend_images(original, swapped, mask, 0.0).max() == 0
    assert blend_images(original, swapped, mask, 1.0).min() == 200
    mid = blend_images(original, swapped, mask, 0.5)
    assert mid[0, 0, 0] == 100


def test_blend_respects_mask_and_clamps_strength():
    original = np.full((4, 4, 3), 10, dtype=np.uint8)
    swapped = np.full((4, 4, 3), 250, dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=np.float32)
    out = blend_images(original, swapped, mask, 5.0)  # clamped, mask empty
    assert (out == 10).all()


def test_feather_mask_soft_edges():
    mask = feather_mask((64, 64), (16, 16, 48, 48), feather=4.0)
    assert mask[32, 32] == pytest.approx(1.0, abs=0.05)
    assert mask[0, 0] == 0.0
    assert 0.0 < mask[16, 32] < 1.0  # feathered border


# --- template averaging -----------------------------------------------------------

def test_template_drops_outlier():
    rng = np.random.default_rng(0)
    base = rng.normal(size=512)
    base /= np.linalg.norm(base)
    near = [base + 0.01 * rng.normal(size=512) for _ in range(3)]
    near = [v / np.linalg.norm(v) for v in near]
    far = -base + 0.01 * rng.normal(size=512)  # opposing identity: must drop
    far /= np.linalg.norm(far)
    mean, keep = template_embedding(near + [far])
    assert len(keep) == 3
    assert logic_cos(mean, base) > 0.9


def test_template_single_and_identical():
    v = np.array([1.0, 0.0, 0.0])
    mean, keep = template_embedding([v])
    assert keep == [0]
    assert np.allclose(mean, v)
    mean, keep = template_embedding([v, v, v])
    assert keep == [0, 1, 2]


def logic_cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# --- onnxruntime provider selection -------------------------------------------

def test_ort_providers_cpu_is_always_last():
    from face_consistency.swap_engine import _ort_providers

    provs = _ort_providers()
    assert len(provs) >= 1
    assert provs[-1] == "CPUExecutionProvider"


# --- general IP-Adapter model picking ----------------------------------------

def test_find_ipadapter_model_skips_face_variants():
    names = ["None", "ip-adapter-faceid-plusv2_sdxl [a]",
             "ip-adapter_sdxl [b]", "instantid-controlnet [c]"]
    assert logic.find_ipadapter_model(names) == "ip-adapter_sdxl [b]"


def test_find_ipadapter_model_none_when_absent():
    assert logic.find_ipadapter_model(
        ["None", "control_v11p_sd15_canny"]) is None
    assert logic.find_ipadapter_model([]) is None
    assert logic.find_ipadapter_model(None) is None


def test_find_ipadapter_model_deterministic_first_sorted():
    names = ["None", "ip-adapter-plus_sdxl [z]", "ip-adapter_sdxl [a]"]
    # "-" (0x2D) sorts before "_" (0x5F): the plus build wins deterministically
    assert logic.find_ipadapter_model(names) == "ip-adapter-plus_sdxl [z]"


# --- IP-Adapter preprocessor picking ------------------------------------------

def test_pick_ipadapter_preprocessor():
    assert logic.pick_ipadapter_preprocessor(
        "ip-adapter_sdxl [b]", "sdxl") == logic.PREPROCESSOR_IPADAPTER_BIGG
    assert logic.pick_ipadapter_preprocessor(
        "ip-adapter-plus_sdxl_vit-h [c]",
        "sdxl") == logic.PREPROCESSOR_IPADAPTER_H
    assert logic.pick_ipadapter_preprocessor(
        "ip-adapter_sd15 [d]", "other") == logic.PREPROCESSOR_IPADAPTER_H
    assert logic.pick_ipadapter_preprocessor(
        "ip-adapter_sdxl [b]", "flux") is None


# --- head-height proportion gate ------------------------------------------------

def test_head_height_assessment_normal():
    ratio, warn = logic.head_height_assessment(115, 1024)
    assert warn is None
    assert ratio == pytest.approx(115 / 1024)


def test_head_height_assessment_warns_when_tiny():
    ratio, warn = logic.head_height_assessment(50, 1024)
    assert warn is not None
    assert "closer crop" in warn
    assert ratio == pytest.approx(50 / 1024)


def test_head_height_assessment_degenerate_input():
    assert logic.head_height_assessment(0, 1024) == (0.0, None)
    assert logic.head_height_assessment(100, 0) == (0.0, None)
    assert logic.head_height_assessment(None, 1024) == (0.0, None)


def test_find_ipadapter_model_skips_plus_face():
    # Live 2026-10-08: the box's only IP-Adapter was a plus-face build;
    # it must not be picked for outfit duty (face-only conditioning).
    names = ["None", "ip-adapter-plus-face_sdxl_vit-h [368cf551]"]
    assert logic.find_ipadapter_model(names) is None
    assert logic.list_face_ipadapter_models(names) == [
        "ip-adapter-plus-face_sdxl_vit-h [368cf551]"]


def test_list_face_ipadapter_models_empty_for_general():
    assert logic.list_face_ipadapter_models(
        ["None", "ip-adapter_sdxl [b]"]) == []


def test_detect_family_xl_in_filename():
    # Bare "XL" in the checkpoint filename marks SDXL even without the
    # "sd" prefix (live 2026-10-08: epicrealismXL misdetected as "other").
    assert logic.detect_family(
        checkpoint_hint="epicrealismXL_vxviiCrystalclear.safetensors") == "sdxl"
