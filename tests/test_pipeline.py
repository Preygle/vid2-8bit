"""Unit tests.

These focus on the invariants that are easy to break silently and hard to spot
by eye -- colour-space round trips, solver correctness, and the temporal
stability mechanisms. The L0 tests in particular exist because that solver was
wrong twice during development in ways that produced plausible-looking but
degraded output.
"""

from __future__ import annotations

import numpy as np
import pytest

from vid2_8bit.color import despeckle, dither, palette, spaces, tiles
from vid2_8bit.color.quantize import Quantizer, posterize_luma, snap_image_bit_depth
from vid2_8bit.config import available_presets, build_config
from vid2_8bit.metrics import isolated_pixel_ratio, temporal_churn
from vid2_8bit.stages import abstract, output, sample, structure, tone


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def image(rng):
    """A small image with flat regions, an edge and a gradient."""
    img = np.zeros((64, 96, 3), dtype=np.float32)
    img[:, :48] = [0.7, 0.3, 0.25]
    img[:, 48:] = [0.2, 0.4, 0.7]
    img[:16] = np.linspace(0.1, 0.9, 96, dtype=np.float32)[None, :, None]
    return img


# -- colour spaces ---------------------------------------------------------


class TestSpaces:
    def test_srgb_linear_roundtrip(self, rng):
        img = rng.random((32, 32, 3), dtype=np.float32)
        back = spaces.linear_to_srgb(spaces.srgb_to_linear(img))
        assert np.allclose(img, back, atol=1e-5)

    def test_oklab_roundtrip(self, rng):
        img = rng.random((32, 32, 3), dtype=np.float32)
        back = spaces.oklab_to_srgb(spaces.srgb_to_oklab(img))
        assert np.allclose(img, back, atol=1e-4)

    def test_white_maps_to_l_one(self):
        lab = spaces.srgb_to_oklab(np.array([[1.0, 1.0, 1.0]], dtype=np.float32))
        assert lab[0, 0] == pytest.approx(1.0, abs=1e-4)
        assert lab[0, 1] == pytest.approx(0.0, abs=1e-4)
        assert lab[0, 2] == pytest.approx(0.0, abs=1e-4)

    def test_greys_have_zero_chroma(self):
        greys = np.linspace(0, 1, 16, dtype=np.float32)[:, None].repeat(3, axis=1)
        lab = spaces.srgb_to_oklab(greys)
        assert np.abs(lab[:, 1:]).max() < 1e-4

    def test_lightness_is_monotonic(self):
        greys = np.linspace(0, 1, 32, dtype=np.float32)[:, None].repeat(3, axis=1)
        L = spaces.srgb_to_oklab(greys)[:, 0]
        assert np.all(np.diff(L) > 0)


# -- L0 solver -------------------------------------------------------------


class TestL0:
    def test_tiny_lambda_is_identity(self, image):
        """The data term dominates as lambda vanishes, so output must match input.

        This is the check that catches sign errors in the difference operators:
        a wrong sign still converges, just to a smoothed-out image.
        """
        out = abstract.l0_smooth(image, lam=1e-7)
        assert np.abs(out - image).max() < 1e-4

    def test_preserves_dynamic_range(self, image):
        """A correct solve keeps contrast; a broken one returns a hazy mean."""
        out = abstract.l0_smooth(image, lam=0.03)
        assert out.std() > image.std() * 0.75

    def test_flattens_texture_but_keeps_the_edge(self, rng):
        base = np.zeros((64, 64, 3), dtype=np.float32)
        base[:, :32] = 0.25
        base[:, 32:] = 0.75
        noisy = np.clip(base + rng.normal(0, 0.03, base.shape), 0, 1).astype(np.float32)
        out = abstract.l0_smooth(noisy, lam=0.02)
        # Texture inside each flat region is reduced...
        assert out[:, 4:28].std() < noisy[:, 4:28].std() * 0.6
        # ...while the step across the middle survives.
        assert abs(float(out[:, 40:60].mean() - out[:, 4:24].mean())) > 0.4

    def test_output_is_bounded(self, image):
        out = abstract.l0_smooth(image, lam=0.05)
        assert out.min() >= 0.0 and out.max() <= 1.0


# -- palette ---------------------------------------------------------------


class TestPalette:
    def test_hardware_palettes_load(self):
        for name in palette.HARDWARE_PALETTES:
            pal = palette.hardware_palette(name)
            assert pal.ndim == 2 and pal.shape[1] == 3
            assert len(pal) >= 4
            assert pal.min() >= 0.0 and pal.max() <= 1.0

    def test_nes_has_54_unique_colors(self):
        assert len(palette.dedupe(palette.hardware_palette("nes"))) == 54

    def test_gameboy_has_four(self):
        assert len(palette.hardware_palette("gameboy")) == 4

    def test_unknown_palette_raises(self):
        with pytest.raises(ValueError):
            palette.hardware_palette("megadrive64")

    def test_kmeans_is_deterministic(self, rng):
        samples = rng.random((4000, 3)).astype(np.float32)
        a = palette.fit_palette(samples, 12)
        b = palette.fit_palette(samples, 12)
        assert np.array_equal(a, b)

    def test_palette_is_sorted_by_lightness(self, rng):
        pal = palette.fit_palette(rng.random((4000, 3)).astype(np.float32), 10)
        L = spaces.srgb_to_oklab(pal)[:, 0]
        assert np.all(np.diff(L) >= -1e-6)

    def test_rare_vivid_accent_survives(self, rng):
        """A small saturated region must get its own entry.

        Pixel art reads by hue contrast, so losing the one red object in a grey
        scene is a worse failure than a slightly wrong grey.
        """
        bulk = np.tile([0.5, 0.5, 0.5], (9800, 1)) + rng.normal(0, 0.01, (9800, 3))
        accent = np.tile([0.9, 0.1, 0.1], (200, 1))
        samples = np.clip(np.concatenate([bulk, accent]), 0, 1).astype(np.float32)
        pal = palette.fit_palette(samples, 8, chroma_weight=1.5)
        lab = spaces.srgb_to_oklab(pal)
        assert np.abs(lab[:, 1]).max() > 0.1

    def test_bit_depth_snapping(self):
        pal = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
        snapped = palette.snap_to_bit_depth(pal, [1, 1, 1])
        assert set(np.unique(snapped).tolist()) <= {0.0, 1.0}

    def test_select_subset_size(self, rng):
        master = palette.hardware_palette("nes")
        samples = rng.random((3000, 3)).astype(np.float32)
        assert len(palette.select_subset(master, samples, 8)) == 8


# -- quantization ----------------------------------------------------------


class TestQuantize:
    def test_indices_in_range(self, image):
        q = Quantizer(palette.hardware_palette("pico8"))
        idx = q.quantize(image)
        assert idx.shape == image.shape[:2]
        assert idx.min() >= 0 and idx.max() < len(q)

    def test_exact_palette_colors_are_preserved(self):
        pal = palette.hardware_palette("pico8")
        img = pal.reshape(4, 4, 3)
        q = Quantizer(pal)
        assert np.allclose(q.to_rgb(q.quantize(img)), img, atol=1e-5)

    def test_lut_approximates_exact_search(self, rng):
        pal = palette.hardware_palette("c64")
        img = rng.random((48, 48, 3)).astype(np.float32)
        exact = Quantizer(pal).quantize(img)
        approx = Quantizer(pal, lut_bits=6).quantize(img)
        assert (exact == approx).mean() > 0.95

    def test_hysteresis_suppresses_flicker(self, rng):
        """The core anti-boiling mechanism: sub-threshold noise must not flip."""
        pal = palette.hardware_palette("pico8")
        q = Quantizer(pal)
        img = rng.random((64, 64, 3)).astype(np.float32)
        first = q.quantize(img)
        jittered = np.clip(img + rng.normal(0, 0.003, img.shape), 0, 1).astype(np.float32)
        without = temporal_churn(first, q.quantize(jittered))
        with_hyst = temporal_churn(
            first, q.quantize(jittered, prev_idx=first, hysteresis=0.05)
        )
        assert without > 0.0
        assert with_hyst < without * 0.25

    def test_hysteresis_still_tracks_real_change(self):
        pal = palette.hardware_palette("gameboy")
        q = Quantizer(pal)
        dark = np.zeros((16, 16, 3), dtype=np.float32)
        light = np.ones((16, 16, 3), dtype=np.float32)
        prev = q.quantize(dark)
        assert not np.array_equal(
            q.quantize(light, prev_idx=prev, hysteresis=0.02), prev
        )

    def test_posterize_reduces_levels(self, image):
        out = posterize_luma(image, 4)
        assert len(np.unique(np.round(spaces.srgb_to_oklab(out)[..., 0], 4))) <= 6

    def test_bit_depth_snap_image(self, image):
        assert len(np.unique(snap_image_bit_depth(image, [2, 2, 2]))) <= 4


# -- dithering -------------------------------------------------------------


class TestDither:
    @pytest.mark.parametrize("n", [2, 4, 8])
    def test_bayer_is_zero_mean(self, n):
        """A non-zero mean would shift overall image brightness."""
        assert abs(float(dither.bayer_matrix(n).mean())) < 1e-6

    @pytest.mark.parametrize("n", [2, 4, 8])
    def test_bayer_values_unique_and_bounded(self, n):
        m = dither.bayer_matrix(n)
        assert m.shape == (n, n)
        assert len(np.unique(m)) == n * n
        assert m.min() >= -0.5 and m.max() < 0.5

    def test_bayer_rejects_non_power_of_two(self):
        with pytest.raises(ValueError):
            dither.bayer_matrix(6)

    def test_blue_noise_is_zero_mean(self):
        assert abs(float(dither.blue_noise_matrix(32).mean())) < 1e-6

    def test_ordered_dither_is_deterministic(self, image):
        q = Quantizer(palette.hardware_palette("pico8"))
        a = dither.apply_ordered(image, q, "bayer4", 0.6)
        b = dither.apply_ordered(image, q, "bayer4", 0.6)
        assert np.array_equal(a, b)

    def test_selective_dither_leaves_exact_colors_alone(self):
        """Flat regions already representable must not be speckled."""
        pal = palette.hardware_palette("pico8")
        img = np.tile(pal[3], (32, 32, 1)).astype(np.float32)
        q = Quantizer(pal)
        out = dither.apply_ordered(img, q, "bayer4", 0.8, selective_threshold=0.02)
        assert np.allclose(out, img, atol=1e-6)

    def test_none_is_a_passthrough(self, image):
        q = Quantizer(palette.hardware_palette("pico8"))
        assert np.array_equal(dither.apply_ordered(image, q, "none", 0.5), image)


# -- tiles -----------------------------------------------------------------


class TestTiles:
    def test_respects_colors_per_tile(self):
        rng = np.random.default_rng(0)
        img = rng.random((32, 32, 3)).astype(np.float32)
        idx = tiles.apply_tile_constraints(
            img, palette.hardware_palette("nes"), tile_size=8, colors_per_tile=4
        )
        assert idx.shape == (32, 32)
        for ty in range(4):
            for tx in range(4):
                block = idx[ty * 8 : (ty + 1) * 8, tx * 8 : (tx + 1) * 8]
                assert len(np.unique(block)) <= 4

    def test_handles_non_multiple_sizes(self):
        img = np.random.default_rng(0).random((30, 45, 3)).astype(np.float32)
        idx = tiles.apply_tile_constraints(img, palette.hardware_palette("c64"))
        assert idx.shape == (30, 45)


# -- stages ----------------------------------------------------------------


class TestStages:
    def test_xdog_responds_to_edges(self):
        img = np.zeros((64, 64), dtype=np.float32)
        img[:, 32:] = 1.0
        lines = structure.xdog(img)
        assert lines.shape == img.shape
        assert lines[:, 28:36].max() > lines[:, :16].max()

    def test_outline_expansion_preserves_shape(self, image):
        out = sample.outline_expansion(image, cell=6.0)
        assert out.shape == image.shape
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_sampling_hits_target_grid(self, image):
        cfg = build_config("modern-hifi")
        out = sample.sample_frame(image, cfg, (24, 16), tier="fast")
        assert out.shape[:2] == (16, 24)

    def test_thin_dark_line_survives_downsampling(self):
        """The failure mode outline expansion exists to prevent."""
        img = np.ones((60, 60, 3), dtype=np.float32) * 0.9
        img[:, 29:31] = 0.05
        cfg = build_config("modern-hifi")
        plain = sample.area_downsample(img, (10, 10))
        expanded = sample.sample_frame(img, cfg, (10, 10), tier="fast")
        assert expanded.min() < plain.min()

    def test_downsample_does_not_replicate_edge_rows(self):
        """Edge padding fabricated 13 duplicate rows at the bottom of every frame.

        A 608-row working image onto a 101-row grid padded 99 replicated rows,
        which then dominated the medians of the final logical rows and rendered
        the bottom 13% of the picture as identical vertical streaks.
        """
        rng = np.random.default_rng(0)
        img = rng.random((608, 1080, 3)).astype(np.float32)
        out = sample.median_downsample(img, (180, 101))
        assert out.shape[:2] == (101, 180)
        row_diff = np.abs(np.diff(out, axis=0)).mean(axis=(1, 2))
        assert (row_diff[-15:] < 1e-6).sum() == 0

    def test_maxpool_does_not_replicate_edge_rows(self):
        from vid2_8bit.pipeline import maxpool_to

        rng = np.random.default_rng(1)
        mask = rng.random((608, 1080)).astype(np.float32)
        out = maxpool_to(mask, (180, 101))
        assert out.shape == (101, 180)
        assert (np.abs(np.diff(out, axis=0)).mean(axis=1)[-15:] < 1e-6).sum() == 0

    def test_downsample_uses_the_whole_frame(self):
        """The bottom of the source must reach the bottom of the output."""
        img = np.zeros((600, 900, 3), dtype=np.float32)
        img[-30:] = 1.0                       # bright band only at the very bottom
        out = sample.median_downsample(img, (90, 60))
        assert out[-1].mean() > 0.5

    def test_integer_scale_and_upscale(self):
        assert output.integer_scale(213, 120, 1280, 720) == 6
        img = np.zeros((10, 20, 3), dtype=np.float32)
        assert output.upscale(img, 4).shape == (40, 80, 3)

    def test_upscale_is_nearest_neighbour(self):
        """Interpolation would invent colours outside the palette."""
        img = np.array([[[1.0, 0, 0], [0, 0, 1.0]]], dtype=np.float32)
        up = output.upscale(img, 3)
        assert len(np.unique(up.reshape(-1, 3), axis=0)) == 2

    def test_pad_to_letterboxes(self):
        img = np.ones((10, 20, 3), dtype=np.float32)
        assert output.pad_to(img, 30, 20).shape == (20, 30, 3)


# -- config ----------------------------------------------------------------


class TestConfig:
    def test_all_presets_load_and_validate(self):
        names = available_presets()
        assert "tetris-movie" in names and "nes" in names
        for name in names:
            build_config(name).validate()

    def test_preset_inheritance_applies(self):
        cfg = build_config("gameboy")
        assert cfg.palette.hardware == "gameboy"
        assert cfg.output.pix_fmt == "yuv444p"  # inherited from base

    def test_cli_overrides_beat_preset(self):
        cfg = build_config("nes", overrides={"sample": {"cell_size": 3}})
        assert cfg.sample.cell_size == 3

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError):
            build_config(overrides={"sample": {"nonexistent": 1}})

    def test_logical_size_and_cell(self):
        cfg = build_config(overrides={"sample": {"cell_size": 6}})
        assert cfg.logical_size(1280, 720) == (213, 120)
        assert cfg.effective_cell(1280, 720) == pytest.approx(6.0, abs=0.05)

    def test_target_width_overrides_cell(self):
        cfg = build_config(overrides={"sample": {"target_width": 160}})
        assert cfg.logical_size(1280, 720) == (160, 90)

    def test_invalid_values_rejected(self):
        for bad in (
            {"tier": "turbo"},
            {"palette": {"size": 1}},
            {"dither": {"mode": "swirl"}},
            {"palette": {"mode": "hardware"}},
            {"palette": {"bits_per_channel": [9, 9, 9]}},
        ):
            with pytest.raises(ValueError):
                build_config(overrides=bad)



class TestTone:
    """Stage 0.5. Its absence was the single largest quality defect found."""

    def test_auto_levels_expands_a_compressed_range(self):
        """The failure that made every night shot render as a muddy blob."""
        img = (np.random.default_rng(0).random((64, 64, 3)) * 0.2 + 0.15).astype(np.float32)
        spread = lambda i: float(
            np.percentile(spaces.srgb_to_oklab(i)[..., 0], 99)
            - np.percentile(spaces.srgb_to_oklab(i)[..., 0], 1)
        )
        assert spread(tone.auto_levels(img)) > spread(img) * 1.4

    def test_auto_levels_leaves_a_full_range_image_alone(self):
        """Forcing the same stretch on daylight blew an orange car to white.

        The fixture is built as a ramp in Oklab *lightness*, not uniform RGB:
        uniform RGB only spans about 0.66 of the L range because lightness is
        nonlinear, so it is not actually a full-range image and a lift on it is
        correct behaviour.
        """
        L = np.linspace(0.02, 0.99, 64, dtype=np.float32)
        lab = np.zeros((64, 64, 3), dtype=np.float32)
        lab[..., 0] = L[:, None]
        img = spaces.oklab_to_srgb(lab)
        span = lambda i: float(
            np.percentile(spaces.srgb_to_oklab(i)[..., 0], 99)
            - np.percentile(spaces.srgb_to_oklab(i)[..., 0], 1)
        )
        assert span(img) > 0.9, "fixture must actually span the range"
        assert np.abs(tone.auto_levels(img) - img).max() < 0.05

    def test_two_pole_contrast_increases_spread(self):
        img = (np.random.default_rng(1).random((48, 48, 3)) * 0.4 + 0.3).astype(np.float32)
        out = tone.two_pole_contrast(img, 1.0)
        assert spaces.srgb_to_oklab(out)[..., 0].std() > spaces.srgb_to_oklab(img)[..., 0].std()

    def test_contrast_zero_is_a_passthrough(self):
        img = np.random.default_rng(3).random((16, 16, 3)).astype(np.float32)
        assert np.array_equal(tone.two_pole_contrast(img, 0.0), img)

    def test_color_transfer_moves_toward_reference(self):
        img = np.tile([0.2, 0.3, 0.6], (32, 32, 1)).astype(np.float32)
        ref = np.tile([0.6, 0.3, 0.2], (32, 32, 1)).astype(np.float32)
        mean, std = tone.image_stats(ref)
        out = tone.color_transfer(img, mean, std, 1.0)
        before = np.abs(spaces.srgb_to_oklab(img).mean((0, 1)) - mean).sum()
        after = np.abs(spaces.srgb_to_oklab(out).mean((0, 1)) - mean).sum()
        assert after < before


class TestPaletteRamp:
    def test_hue_takes_the_short_arc(self):
        """Linear degree interpolation puts a cyan band in a warm ramp."""
        pal = palette.ramp_palette(12, shadow_hue=210, mid_hue=350, highlight_hue=45)
        lab = spaces.srgb_to_oklab(pal)
        hue = (np.degrees(np.arctan2(lab[:, 2], lab[:, 1])) + 360) % 360
        # The upper half must not detour through green (roughly 90-180 deg).
        assert not ((hue[6:] > 90) & (hue[6:] < 180)).any()

    def test_lightness_is_evenly_spaced_and_monotonic(self):
        L = spaces.srgb_to_oklab(palette.ramp_palette(12))[:, 0]
        gaps = np.diff(L)
        assert np.all(gaps > 0)
        assert gaps.std() < 0.02


class TestPaletteRedistribution:
    def test_entries_get_comparable_pixel_share(self):
        """The 72%-on-one-colour collapse a blind critic measured."""
        rng = np.random.default_rng(0)
        src = np.clip(rng.normal(0.16, 0.04, 60000), 0, 1).astype(np.float32)
        pal = np.linspace(0.05, 0.95, 16, dtype=np.float32)[:, None].repeat(3, 1)
        out = palette.redistribute_lightness(pal, src, 1.0, anchor_ink=True)
        L = spaces.srgb_to_oklab(out)[:, 0]
        share = np.bincount(
            np.abs(src[:, None] - L[None, :]).argmin(1), minlength=16
        ) / len(src)
        assert share.max() < 0.30

    def test_even_spacing_removes_invisible_duplicates(self):
        """Pure percentile placement buries the palette in the dominant tone.

        On a mostly-dark frame most percentiles are dark, so most entries land
        in the shadows: measured 26 pairs closer than 0.03 in Oklab, which the
        eye reads as one colour, plus a large empty gap in the mid-tones.
        """
        rng = np.random.default_rng(0)
        # 80% of pixels dark, 20% spread bright: the shape that breaks it.
        src = np.concatenate([
            np.clip(rng.normal(0.12, 0.03, 40000), 0, 1),
            np.clip(rng.uniform(0.3, 0.95, 10000), 0, 1),
        ]).astype(np.float32)
        pal = np.linspace(0.05, 0.95, 16, dtype=np.float32)[:, None].repeat(3, 1)

        def stats(p):
            lab = spaces.srgb_to_oklab(p)
            d = np.sqrt(((lab[:, None, :] - lab[None, :, :]) ** 2).sum(-1))
            np.fill_diagonal(d, 9.0)
            return d.min(), np.diff(np.sort(lab[:, 0])).max()

        pure_min, pure_gap = stats(
            palette.redistribute_lightness(pal, src, 1.0, even_spacing=0.0))
        even_min, even_gap = stats(
            palette.redistribute_lightness(pal, src, 1.0, even_spacing=0.65))
        assert even_min > pure_min * 1.5
        assert even_gap < pure_gap

    def test_even_spacing_stays_inside_the_content_range(self):
        """Spacing across 0..1 instead described tones the shot did not contain."""
        rng = np.random.default_rng(1)
        src = np.clip(rng.normal(0.2, 0.05, 20000), 0, 1).astype(np.float32)
        pal = np.linspace(0.05, 0.95, 12, dtype=np.float32)[:, None].repeat(3, 1)
        out = palette.redistribute_lightness(pal, src, 1.0, even_spacing=1.0)
        L = spaces.srgb_to_oklab(out)[:, 0]
        assert L.max() <= float(np.percentile(src, 99)) + 0.05

    def test_order_is_preserved(self):
        rng = np.random.default_rng(2)
        L = spaces.srgb_to_oklab(
            palette.redistribute_lightness(
                rng.random((12, 3)).astype(np.float32),
                rng.random(5000).astype(np.float32), 1.0,
            )
        )[:, 0]
        assert np.all(np.diff(L) >= -1e-4)

    def test_warm_highlights_raises_chroma_at_the_top(self):
        pal = np.linspace(0.05, 0.95, 8, dtype=np.float32)[:, None].repeat(3, 1)
        chroma = lambda p: np.hypot(
            spaces.srgb_to_oklab(p)[:, 1], spaces.srgb_to_oklab(p)[:, 2]
        )
        assert chroma(palette.warm_highlights(pal, 0.5))[-1] > chroma(pal)[-1]


class TestDespeckle:
    @staticmethod
    def _pal():
        pal = np.linspace(0, 1, 8, dtype=np.float32)[:, None].repeat(3, 1)
        return pal, spaces.srgb_to_oklab(pal)

    def test_removes_low_contrast_island(self):
        _, lab = self._pal()
        idx = np.full((9, 9), 2, np.int32)
        idx[4, 4] = 3
        assert despeckle.despeckle_indices(idx, lab, 0.15)[4, 4] == 2

    def test_keeps_high_contrast_island(self):
        """Lit windows are isolated cells too and must survive."""
        _, lab = self._pal()
        idx = np.full((9, 9), 2, np.int32)
        idx[4, 4] = 7
        assert despeckle.despeckle_indices(idx, lab, 0.15)[4, 4] == 7

    def test_keeps_run_ends(self):
        """min_neighbours=7 ate run ends, shortening architectural lines."""
        _, lab = self._pal()
        idx = np.full((9, 9), 2, np.int32)
        idx[4, 2:7] = 3
        assert (despeckle.despeckle_indices(idx, lab, 0.9)[4, 2:7] == 3).all()

    def test_zero_threshold_is_a_passthrough(self):
        _, lab = self._pal()
        idx = np.full((9, 9), 2, np.int32)
        idx[4, 4] = 5
        assert np.array_equal(despeckle.despeckle_indices(idx, lab, 0.0), idx)


class TestGeneralisation:
    """Settings must hold across very different footage, not just one clip."""

    def test_min_cell_caps_the_grid_on_small_sources(self):
        """180 logical px from a 360px clip is a cell of 2 - barely pixelated."""
        cfg = build_config(overrides={
            "sample": {"cell_size": None, "target_width": 180, "min_cell": 4.0}
        })
        assert cfg.effective_cell(1920, 1080) > 10      # HD unaffected
        assert cfg.effective_cell(360, 640) == pytest.approx(4.0, abs=0.2)

    def test_min_cell_never_collapses_the_grid(self):
        cfg = build_config(overrides={
            "sample": {"cell_size": None, "target_width": 180, "min_cell": 999.0}
        })
        w, _ = cfg.logical_size(360, 640)
        assert w >= 8

    def test_target_chroma_boosts_muted_and_tames_vivid(self):
        """A fixed multiplier over-saturates footage that is already vivid."""
        from vid2_8bit.stages.tone import _saturate_in_place

        def mean_chroma(lab):
            return float(np.hypot(lab[..., 1], lab[..., 2]).mean())

        muted = np.zeros((32, 32, 3), dtype=np.float32)
        muted[..., 0] = 0.5
        muted[..., 1] = 0.01
        vivid = muted.copy()
        vivid[..., 1] = 0.12

        _saturate_in_place(muted, 1.0, 0.04, 2.5)
        _saturate_in_place(vivid, 1.0, 0.04, 2.5)
        assert mean_chroma(muted) > 0.02          # lifted toward target
        assert mean_chroma(vivid) < 0.06          # pulled down toward target

    def test_target_chroma_respects_the_boost_ceiling(self):
        """A near-monochrome frame must not be amplified into false colour."""
        from vid2_8bit.stages.tone import _saturate_in_place

        grey = np.zeros((16, 16, 3), dtype=np.float32)
        grey[..., 0] = 0.5
        grey[..., 1] = 0.001
        _saturate_in_place(grey, 1.0, 0.04, 2.5)
        assert float(np.abs(grey[..., 1]).mean()) <= 0.001 * 2.5 + 1e-6


class TestPerformance:
    def test_prescale_reduces_resolution_but_not_output_size(self):
        """Output size must not depend on the speed setting."""
        from vid2_8bit.pipeline import Converter

        cfg = build_config(overrides={
            "sample": {"cell_size": None, "target_width": 100},
            "performance": {"oversample": 4.0},
        })
        conv = Converter(cfg, collect_metrics=False)
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        assert conv._prescale(frame, 100).shape[1] == 400
        # oversample 0 disables it entirely.
        cfg2 = build_config(overrides={"performance": {"oversample": 0.0}})
        assert Converter(cfg2)._prescale(frame, 100).shape[1] == 1920

    def test_prescale_skips_when_there_is_nothing_to_save(self):
        from vid2_8bit.pipeline import Converter

        cfg = build_config(overrides={"performance": {"oversample": 6.0}})
        conv = Converter(cfg, collect_metrics=False)
        small = np.zeros((90, 160, 3), dtype=np.uint8)
        assert conv._prescale(small, 100).shape[1] == 160

    def test_fps_config_accepts_both_rates(self):
        cfg = build_config(overrides={
            "temporal": {"decimate_fps": 12.0}, "output": {"fps": 12.0},
        })
        cfg.validate()
        assert cfg.temporal.decimate_fps == 12.0 and cfg.output.fps == 12.0


class TestGenAssist:
    """The generative path must stay strictly optional."""

    def test_importing_gen_does_not_pull_in_torch(self):
        """A plain render must never depend on a multi-gigabyte stack."""
        import subprocess
        import sys as _sys

        code = (
            "import sys; import vid2_8bit.gen; "
            "assert 'torch' not in sys.modules, sorted(sys.modules)[:0] or 'torch imported'; "
            "print('ok')"
        )
        r = subprocess.run([_sys.executable, "-c", code], capture_output=True,
                           text=True, cwd="src")
        assert r.returncode == 0, r.stderr[-500:]

    def test_unreachable_comfy_reports_cleanly(self):
        """No stack trace when ComfyUI is simply not running."""
        from vid2_8bit.gen import ComfyClient

        client = ComfyClient(host="127.0.0.1:59999")
        assert client.available() is False

    def test_img2img_graph_is_well_formed(self):
        from vid2_8bit.gen import img2img_graph

        g = img2img_graph("x.png", "model.safetensors", "pixel art")
        assert all("class_type" in n and "inputs" in n for n in g.values())
        # Every node reference must point at a node that exists.
        for node in g.values():
            for v in node["inputs"].values():
                if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                    assert v[0] in g, f"dangling reference to {v[0]}"

    def test_lora_is_spliced_between_checkpoint_and_sampler(self):
        from vid2_8bit.gen import img2img_graph

        plain = img2img_graph("x.png", "m.safetensors", "p")
        withl = img2img_graph("x.png", "m.safetensors", "p", lora="l.safetensors")
        assert plain["7"]["inputs"]["model"][0] == "1"
        assert withl["7"]["inputs"]["model"][0] == "4"

    def test_keyframe_indices_are_spread_and_in_range(self):
        from vid2_8bit.gen import keyframe_indices

        idx = keyframe_indices(600, 6)
        assert len(idx) == 6
        assert all(0 <= i < 600 for i in idx)
        assert idx == sorted(idx)
        assert keyframe_indices(3, 10) == [0, 1, 2]

    def test_measure_emits_a_loadable_preset(self):
        """The distilled preset must survive a real config load."""
        from vid2_8bit.gen import measure

        rng = np.random.default_rng(0)
        frames = []
        for _ in range(2):
            small = (rng.random((40, 60, 3)) * 255).astype(np.uint8)
            frames.append(np.repeat(np.repeat(small, 8, 0), 8, 1))
        fit = measure(frames, palette_size=12)
        assert fit.palette.shape[1] == 3
        assert 60 <= fit.target_width <= 480
        assert 0.0 <= fit.contrast <= 1.0
        doc = fit.to_preset()
        cfg = build_config(overrides={k: v for k, v in doc.items()
                                      if k not in ("description", "extends")})
        cfg.validate()


# -- metrics ---------------------------------------------------------------


class TestMetrics:
    def test_flat_image_has_no_isolated_pixels(self):
        assert isolated_pixel_ratio(np.zeros((32, 32), dtype=np.int32)) == 0.0

    def test_checkerboard_is_all_isolated(self):
        idx = np.indices((32, 32)).sum(axis=0) % 2
        assert isolated_pixel_ratio(idx.astype(np.int32)) > 0.95

    def test_churn_is_zero_for_identical_frames(self):
        idx = np.random.default_rng(0).integers(0, 8, (32, 32)).astype(np.int32)
        assert temporal_churn(idx, idx) == 0.0
