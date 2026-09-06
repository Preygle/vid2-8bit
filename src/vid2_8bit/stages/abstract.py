"""Stage 1 -- pre-abstraction.

This is where the noise problem is actually solved, and it runs at *source*
resolution, before anything is downsampled.

A building facade carries detail (brick courses, window mullions, railings) at
4-10 source pixels. Downsampling by 6x puts all of it far below the output
grid's Nyquist limit, where it cannot be represented -- only aliased. Averaging
turns it into mid-grey mush that then quantizes semi-randomly per pixel, which
is the salt-and-pepper "noise" that makes naive pixelation look wrong.

The fix is to remove that detail *deliberately and edge-preservingly*, so what
reaches the sampler is already flat regions bounded by clean edges:

1. **Structure-texture separation** (L0) -- strip texture, keep and re-sharpen
   region boundaries. This does the heavy lifting.
2. **Optional Nyquist low-pass** -- an extra dial for genuinely noisy footage.
   Off by default; see nyquist_lowpass() for why it must come *after* L0.
3. **Luminance banding** -- collapse smooth gradients into discrete shading
   steps, which is how a pixel artist renders form.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..color.quantize import posterize_luma
from ..color.spaces import (
    linear_to_srgb,
    oklab_to_srgb,
    srgb_to_linear,
    srgb_to_oklab,
)


def box_filter(img: np.ndarray, r: int) -> np.ndarray:
    """Fast box blur with radius r, edge-replicated."""
    k = 2 * int(r) + 1
    return cv2.blur(img, (k, k), borderType=cv2.BORDER_REPLICATE)


def guided_filter(src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Self-guided filter -- edge-preserving smoothing, O(1) in radius.

    The fast-tier structure-texture separator. Unlike a bilateral filter it has
    no gradient reversal artifacts near strong edges, which matters here because
    those edges become the 1-pixel outlines of the final image.
    """
    radius = max(1, int(radius))
    src = src.astype(np.float32)
    mean_i = box_filter(src, radius)
    corr_i = box_filter(src * src, radius)
    var_i = np.maximum(corr_i - mean_i * mean_i, 0.0)
    a = var_i / (var_i + eps)
    b = mean_i - a * mean_i
    return (box_filter(a, radius) * src + box_filter(b, radius)).astype(np.float32)


def l0_smooth(
    img: np.ndarray,
    lam: float = 0.015,
    kappa: float = 2.0,
    beta_max: float = 1e5,
) -> np.ndarray:
    """L0 gradient minimization (Xu et al., 2011), solved in the frequency domain.

    The quality-tier structure-texture separator. It penalizes the *count* of
    non-zero gradients rather than their magnitude, so it produces genuinely
    piecewise-constant regions with knife-sharp boundaries -- exactly the
    "flat areas bounded by hard edges" that pixel art is made of. Filters that
    penalize gradient magnitude instead always leave a soft ramp behind.
    """
    from scipy import fft as sfft

    img = img.astype(np.float32)
    h, w = img.shape[:2]
    channels = img.shape[2] if img.ndim == 3 else 1
    s = img.reshape(h, w, channels).copy()

    # Real-input FFTs: the data is real, so rfft2 halves both the transform cost
    # and the working set, and `workers=-1` spreads it over all cores. This is
    # the difference between roughly 5s and well under 1s per HD frame.
    fftargs = dict(axes=(0, 1), workers=-1)

    # Transfer functions of the forward difference operators. The kernels are
    # written so that rfft2(fx) is exactly the operator applied by
    # `np.diff(s, append=s[:, :1])` -- i.e. h[x] = s[x+1] - s[x] as a circular
    # convolution, which needs k[-1] = +1 and k[0] = -1. Getting this sign
    # backwards makes the adjoint term below oppose the data term, and the
    # solve returns a hazy low-contrast image instead of flat regions.
    fx = np.zeros((h, w), dtype=np.float32)
    fy = np.zeros((h, w), dtype=np.float32)
    fx[0, 0], fx[0, -1] = -1, 1
    fy[0, 0], fy[-1, 0] = -1, 1
    otf_x = sfft.rfft2(fx, workers=-1)
    otf_y = sfft.rfft2(fy, workers=-1)
    denom_const = (np.abs(otf_x) ** 2 + np.abs(otf_y) ** 2)[..., None]
    conj_x = np.conj(otf_x)[..., None]
    conj_y = np.conj(otf_y)[..., None]

    normin1 = sfft.rfft2(s, **fftargs)
    beta = 2.0 * lam

    while beta < beta_max:
        # h/v subproblem: hard-threshold gradients with too little energy.
        hgrad = np.diff(s, axis=1, append=s[:, :1])
        vgrad = np.diff(s, axis=0, append=s[:1, :])
        energy = (hgrad**2 + vgrad**2).sum(axis=2, keepdims=True)
        mask = energy >= (lam / beta)
        hgrad *= mask
        vgrad *= mask

        # S subproblem: closed-form least squares, solved directly in the
        # frequency domain as
        #     S = F^-1[ (F(I) + b(conj(Dx)F(h) + conj(Dy)F(v))) / (1 + b|D|^2) ]
        # Applying the adjoint as a multiplication by conj(D) avoids having to
        # express it as a spatial finite difference, where the wraparound term
        # and the sign are both easy to get wrong.
        num = normin1 + beta * (
            conj_x * sfft.rfft2(hgrad, **fftargs)
            + conj_y * sfft.rfft2(vgrad, **fftargs)
        )
        s = sfft.irfft2(num / (1.0 + beta * denom_const), s=(h, w), **fftargs).astype(
            np.float32
        )
        beta *= kappa

    out = np.clip(s, 0.0, 1.0)
    return out if img.ndim == 3 else out[..., 0]


def nyquist_lowpass(img_srgb: np.ndarray, cell: float, strength: float = 1.0) -> np.ndarray:
    """Optional low-pass for detail the output grid cannot represent.

    Off by default, and it must run *after* L0, not before. Low-passing first
    seems principled -- sub-cell detail can only alias, so remove it -- but it
    measurably makes things worse: the blur turns building silhouettes into soft
    ramps, and L0 then has no sharp edge to snap to, so it invents smooth curved
    region boundaries where a straight roofline should be. A/B renders show
    buildings dissolving into haze.

    Applied afterwards it is a mild denoiser for footage with real sensor grain
    or compression mosquito noise, where a little pre-filtering helps and the
    structure has already been fixed by L0. Sigma is 0.25 cells, deliberately
    just under the cell scale.
    """
    sigma = 0.25 * float(cell) * float(strength)
    if sigma < 0.3:
        return img_srgb
    # Blur in linear light; averaging gamma-encoded values darkens edges.
    lin = srgb_to_linear(img_srgb)
    blurred = cv2.GaussianBlur(lin, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return linear_to_srgb(blurred)


def adjust_saturation(img_srgb: np.ndarray, factor: float) -> np.ndarray:
    """Scale chroma in Oklab, leaving lightness untouched.

    Done in Oklab rather than HSV so boosting saturation does not also shift
    perceived brightness -- which would fight the luminance banding step.
    """
    if abs(factor - 1.0) < 1e-3:
        return img_srgb
    lab = srgb_to_oklab(img_srgb)
    lab[..., 1:] *= float(factor)
    return oklab_to_srgb(lab)


def abstract_frame(
    img_srgb: np.ndarray,
    cfg,
    cell: float,
    tier: str = "quality",
) -> np.ndarray:
    """Run the full abstraction stage on one source-resolution frame.

    `cell` is the number of source pixels per output pixel; the smoothing radius
    is derived from it so the stage automatically scales with output chunkiness.
    """
    acfg = cfg.abstract
    if not acfg.enabled:
        return img_srgb

    out = img_srgb.astype(np.float32)

    # 1. Structure-texture separation: flattens regions while re-sharpening
    #    the boundaries between them. This is the stage that matters.
    if acfg.texture_removal > 0.0:
        if tier == "quality":
            # Mapped so the 0..1 strength dial spans L0's useful range. Below
            # ~0.01 the solve barely touches texture; past ~0.1 whole regions
            # start merging and building silhouettes dissolve.
            lam = 0.008 + 0.09 * float(acfg.texture_removal)
            out = l0_smooth(out, lam=lam)
        else:
            radius = max(1, int(round(cell * 0.5 * acfg.texture_removal)) )
            eps = max(0.01, float(acfg.edge_threshold)) ** 2
            out = guided_filter(out, radius, eps)

    # 2. Optional extra low-pass, only for noisy source footage. See
    #    nyquist_lowpass() -- it runs after L0 deliberately.
    if acfg.smooth_radius_cells > 0.0:
        out = nyquist_lowpass(out, cell, acfg.smooth_radius_cells)

    # 3. Saturation, then luminance banding.
    out = adjust_saturation(out, acfg.saturation)
    if acfg.luma_bands >= 2:
        out = posterize_luma(out, int(acfg.luma_bands))

    return np.clip(out, 0.0, 1.0).astype(np.float32)
