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


def test_texture_preserving_roi_repairs_center():
    import numpy as np
    from face_consistency import blend
    rep = np.full((40, 40, 3), 200, dtype=np.uint8)
    orig = np.full((40, 40, 3), 100, dtype=np.uint8)
    out = blend.texture_preserving_roi(rep, orig, (20, 20))
    assert out.shape == (40, 40, 3) and out.dtype == np.uint8
    assert out[20, 20].tolist() == [200, 200, 200]


def test_texture_preserving_roi_keeps_original_grain_outside():
    import numpy as np
    from face_consistency import blend
    rng = np.random.default_rng(0)
    orig = (100 + 20 * rng.standard_normal((40, 40, 3))).clip(0, 255).astype(np.uint8)
    rep = np.full((40, 40, 3), 100, dtype=np.uint8)  # smooth repair
    out = blend.texture_preserving_roi(rep, orig, (20, 20))
    # far corner: grain should follow the original, not the flat repair
    assert abs(float(out[:8, :8].std()) - float(orig[:8, :8].std())) < 8.0
    assert float(out[:8, :8].std()) > float(rep[:8, :8].std()) + 5.0


def test_texture_preserving_roi_identical_is_identity():
    import numpy as np
    from face_consistency import blend
    rng = np.random.default_rng(1)
    img = (120 + 15 * rng.standard_normal((32, 32, 3))).clip(0, 255).astype(np.uint8)
    out = blend.texture_preserving_roi(img, img, (16, 16))
    assert np.abs(out.astype(int) - img.astype(int)).max() <= 2
