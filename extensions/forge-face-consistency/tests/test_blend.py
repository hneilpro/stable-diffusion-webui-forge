"""Offline tests for face_consistency.blend tone matching."""

import numpy as np

from face_consistency.blend import match_tone_to


def test_match_tone_to_transfers_mean_and_std():
    rng = np.random.default_rng(0)
    ref = np.clip(rng.normal(180, 10, (32, 32, 3)), 0, 255).astype(np.uint8)
    src = np.clip(rng.normal(100, 25, (32, 32, 3)), 0, 255).astype(np.uint8)
    out = match_tone_to(src, ref)
    assert out.shape == src.shape and out.dtype == np.uint8
    for c in range(3):
        assert abs(float(out[..., c].mean()) - float(ref[..., c].mean())) < 3.0
        assert abs(float(out[..., c].std()) - float(ref[..., c].std())) < 3.0


def test_match_tone_to_constant_source_uses_ref_mean():
    ref = np.full((16, 16, 3), 200, dtype=np.uint8)
    src = np.full((16, 16, 3), 50, dtype=np.uint8)
    out = match_tone_to(src, ref)
    assert abs(float(out.mean()) - 200) < 2.0


def test_match_tone_to_identical_is_identity():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 256, (16, 16, 3)).astype(np.uint8)
    out = match_tone_to(a, a)
    assert np.abs(out.astype(int) - a.astype(int)).max() <= 1
