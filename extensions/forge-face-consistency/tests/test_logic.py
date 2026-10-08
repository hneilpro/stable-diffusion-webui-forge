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
    assert logic.detect_family(class_name="Flux") == "flux"
    assert logic.detect_family(checkpoint_hint="flux1-dev.safetensors") == "flux"
    # A Flux checkpoint must win over a stale sdxl flag.
    assert logic.detect_family(is_sdxl=True, class_name="Flux") == "flux"
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
