"""Feathered-mask blending for the below-max-strength fallback path.

Pure numpy (plus optional cv2 for the blur, with a numpy fallback) so the
blend math can be unit-tested offline without Forge or InsightFace.
"""

from __future__ import annotations

import numpy as np


def feather_mask(shape_hw, bbox, feather: float = 8.0):
    """Binary face-box mask softened with a gaussian feather.

    ``bbox`` is (x0, y0, x1, y1) in pixel coordinates. Returns a float32
    array of shape (H, W) with values in [0, 1].
    """
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.float32)
    x0, y0, x1, y1 = (int(round(v)) for v in bbox)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return mask
    mask[y0:y1, x0:x1] = 1.0
    if feather and feather > 0:
        mask = gaussian_blur(mask, feather)
        mask = np.clip(mask, 0.0, 1.0)
    return mask


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Separable gaussian blur for a 2D float array (cv2 if present)."""
    try:
        import cv2  # type: ignore

        return cv2.GaussianBlur(img, (0, 0), float(sigma))
    except Exception:
        # Fallback: small separable kernel built from sigma.
        radius = max(1, int(round(sigma * 3)))
        xs = np.arange(-radius, radius + 1, dtype=np.float32)
        kernel = np.exp(-(xs ** 2) / (2.0 * float(sigma) ** 2))
        kernel /= kernel.sum()

        def _conv_axis(a, axis):
            pad = [(0, 0)] * a.ndim
            pad[axis] = (radius, radius)
            padded = np.pad(a, pad, mode="edge")
            out = np.zeros_like(a)
            for i, wgt in enumerate(kernel):
                sl = [slice(None)] * a.ndim
                sl[axis] = slice(i, i + a.shape[axis])
                out += wgt * padded[tuple(sl)]
            return out

        return _conv_axis(_conv_axis(img, 1), 0)


def blend_images(original: np.ndarray, swapped: np.ndarray,
                 mask: np.ndarray, strength: float) -> np.ndarray:
    """Blend ``swapped`` over ``original`` inside ``mask`` at ``strength``.

    alpha = clip(strength, 0, 1) * mask, broadcast over channels. Strength
    0 returns the original, 1 returns the swapped face inside the mask.
    Inputs are uint8 (H, W, 3); the output is uint8.
    """
    s = float(strength)
    if s < 0.0:
        s = 0.0
    if s > 1.0:
        s = 1.0
    alpha = (s * mask.astype(np.float32))[..., None]
    out = (original.astype(np.float32) * (1.0 - alpha)
           + swapped.astype(np.float32) * alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def match_tone_to(src, ref):
    """Match per-channel mean/std of ``src`` to ``ref`` (uint8 HxWx3).

    A regenerated ROI drifts in overall tone from the img2img pass; the
    drift reads as a visible patch boundary after pasting. Matching the
    repair's tone to the original crop kills the boundary while keeping
    the new detail (only the first two moments move, not structure).
    """
    src_f = np.asarray(src, dtype=np.float32)
    ref_f = np.asarray(ref, dtype=np.float32)
    out = np.empty_like(src_f)
    for c in range(src_f.shape[2]):
        s, r = src_f[..., c], ref_f[..., c]
        s_std = float(s.std())
        r_mean, r_std = float(r.mean()), float(r.std())
        if s_std < 1e-3:
            out[..., c] = r_mean
        else:
            out[..., c] = (s - float(s.mean())) * (r_std / s_std) + r_mean
    return np.clip(out, 0, 255).astype(np.uint8)


def texture_preserving_roi(repaired, original, center_xy, protect_r=10.0,
                           blend_r=14.0, sigma=2.0):
    """Merge a regenerated ROI so its skin texture matches the original.

    The img2img repair smooths fine skin grain; even after tone matching,
    the pasted ROI reads as a visible patch at 200% zoom. This keeps the
    repair's low frequencies (tone + regenerated feature shape) but
    transplants the original's high frequencies (pores/grain) everywhere
    except a small protected radius around ``center_xy`` -- the
    regenerated feature itself -- where the repair wins fully, blending
    out over ``blend_r`` pixels. Inputs are uint8 (H, W, 3) of equal
    shape; the output is uint8.
    """
    rep = np.asarray(repaired, dtype=np.float32)
    orig = np.asarray(original, dtype=np.float32)
    low_rep = np.stack(
        [gaussian_blur(rep[..., c], sigma) for c in range(rep.shape[2])],
        axis=-1)
    low_orig = np.stack(
        [gaussian_blur(orig[..., c], sigma) for c in range(orig.shape[2])],
        axis=-1)
    high = orig - low_orig
    h, w = rep.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.sqrt((xx - float(center_xy[0])) ** 2
                   + (yy - float(center_xy[1])) ** 2)
    protect = np.clip((dist - float(protect_r)) / float(blend_r),
                      0.0, 1.0)[..., None]
    out = protect * (low_rep + high) + (1.0 - protect) * rep
    return np.clip(out, 0, 255).astype(np.uint8)
