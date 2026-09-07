"""Color space conversions.

Two invariants hold throughout the pipeline:

* Anything that *averages* pixels (blur, downsample, temporal blend) runs in
  **linear light**. Averaging gamma-encoded sRGB darkens edges and muddies the
  flat regions we are trying to produce.
* Anything that measures *perceptual distance* (palette k-means, nearest-color
  quantization, hysteresis thresholds) runs in **Oklab**, where Euclidean
  distance approximates perceived difference far better than RGB.

Arrays are float32 in [0, 1] for RGB and unbounded float32 for Oklab, shaped
(..., 3). All functions are vectorized over leading dimensions.
"""

from __future__ import annotations

import cv2
import numpy as np

# Björn Ottosson's Oklab matrices (linear sRGB <-> LMS <-> Oklab).
_RGB_TO_LMS = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ],
    dtype=np.float32,
)

_LMS_TO_OKLAB = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ],
    dtype=np.float32,
)

_OKLAB_TO_LMS = np.array(
    [
        [1.0, 0.3963377774, 0.2158037573],
        [1.0, -0.1055613458, -0.0638541728],
        [1.0, -0.0894841775, -1.2914855480],
    ],
    dtype=np.float32,
)

_LMS_TO_RGB = np.array(
    [
        [4.0767416621, -3.3077115913, 0.2309699292],
        [-1.2684380046, 2.6097574011, -0.3413193965],
        [-0.0041960863, -0.7034186147, 1.7076147010],
    ],
    dtype=np.float32,
)


def _apply_matrix(arr: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """Apply a 3x3 matrix to the last axis of `arr`."""
    return arr @ mat.T


def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    """sRGB electro-optical transfer function. Input/output in [0, 1].

    Uses cv2.pow rather than np.power: it is float32-native and threaded, and
    profiling put the transfer functions at roughly 70% of total render time.
    """
    x = np.ascontiguousarray(srgb, dtype=np.float32)
    hi = cv2.pow((np.maximum(x, 0.0) + np.float32(0.055)) * np.float32(1.0 / 1.055), 2.4)
    return np.where(x <= np.float32(0.04045), x * np.float32(1.0 / 12.92), hi)


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """Inverse sRGB transfer function. Input/output in [0, 1]."""
    x = np.ascontiguousarray(linear, dtype=np.float32)
    hi = cv2.pow(np.maximum(x, 0.0), 1.0 / 2.4) * np.float32(1.055) - np.float32(0.055)
    out = np.where(x <= np.float32(0.0031308), x * np.float32(12.92), hi)
    return np.clip(out, 0.0, 1.0, out=out)


def linear_to_oklab(linear_rgb: np.ndarray) -> np.ndarray:
    """Linear sRGB -> Oklab."""
    lms = _apply_matrix(np.asarray(linear_rgb, dtype=np.float32), _RGB_TO_LMS)
    # np.cbrt is signed, so slightly out-of-gamut negatives stay well behaved.
    return _apply_matrix(np.cbrt(lms), _LMS_TO_OKLAB)


def oklab_to_linear(oklab: np.ndarray) -> np.ndarray:
    """Oklab -> linear sRGB. May return out-of-gamut values; caller clips."""
    lms_root = _apply_matrix(np.asarray(oklab, dtype=np.float32), _OKLAB_TO_LMS)
    return _apply_matrix(lms_root**3, _LMS_TO_RGB)


def srgb_to_oklab(srgb: np.ndarray) -> np.ndarray:
    return linear_to_oklab(srgb_to_linear(srgb))


def oklab_to_srgb(oklab: np.ndarray) -> np.ndarray:
    return linear_to_srgb(np.clip(oklab_to_linear(oklab), 0.0, 1.0))


def u8_to_float(img: np.ndarray) -> np.ndarray:
    """uint8 [0, 255] -> float32 [0, 1]."""
    return img.astype(np.float32) / 255.0


def float_to_u8(img: np.ndarray) -> np.ndarray:
    """float32 [0, 1] -> uint8 [0, 255], rounded and clipped."""
    return np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)


def oklab_distance_sq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance in Oklab over the last axis."""
    diff = a - b
    return np.einsum("...i,...i->...", diff, diff)
