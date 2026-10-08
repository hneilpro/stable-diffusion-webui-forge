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
