"""Configuration model.

One dataclass tree describes an entire render. Era presets (``presets/*.yaml``)
are partial overlays merged over the defaults, and CLI flags are a final overlay
on top of that, so precedence is:

    dataclass defaults  <  preset YAML  <  user YAML  <  CLI flags

The ``tier`` field ("fast" or "quality") selects between implementations of the
same stage rather than between different pipelines -- see docs/PLAN.md.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

PRESET_DIR = Path(__file__).parent / "presets"

TIERS = ("fast", "quality")


@dataclass
class ToneConfig:
    """Stage 0.5 -- tone mapping and colour grading, before stylization.

    Cinematic footage is graded for a display, not for quantization: a night
    exterior may use only a third of the lightness range, all bunched in the
    shadows. Quantizing that directly produces a muddy blob. See stages/tone.py.
    """

    enabled: bool = True
    # Stretch lightness percentiles across the full range.
    auto_levels: bool = True
    black_point: float = 1.0
    white_point: float = 99.0
    gamma: float = 1.0
    saturation: float = 1.0
    # Seek this mean Oklab chroma instead of multiplying blindly. Measured at
    # about 0.042 across the film reference frames. 0 disables and `saturation`
    # acts as a plain multiplier. Adapts to the source: muted footage gets the
    # full boost, already-vivid footage gets little.
    target_chroma: float = 0.0
    max_chroma_boost: float = 2.5
    # Ceiling on the auto-levels lift. The applied amount scales down when the
    # source already uses most of the range, so daylight shots are not blown out.
    levels_strength: float = 1.0
    target_range: float = 0.85
    # Two-pole S-curve strength. The decisive control for night exteriors.
    contrast: float = 0.0
    # Style reference image; its colour statistics are transferred onto frames.
    reference: str | None = None
    transfer_strength: float = 0.0


@dataclass
class AbstractConfig:
    """Stage 1 -- pre-abstraction. The stage that removes pixelation noise."""

    enabled: bool = True
    # Structure-texture separation strength. Higher strips more texture.
    texture_removal: float = 0.35
    # Optional extra low-pass, as a multiple of the output cell size. Off by
    # default: it runs after L0 and is only useful for grainy or heavily
    # compressed source footage.
    smooth_radius_cells: float = 0.0
    # How strongly edges resist smoothing (sigma_r of the edge-aware filter).
    edge_threshold: float = 0.09
    # Quantize luminance into N plateaus (0 = off). Makes shading read as bands.
    luma_bands: int = 0
    # Extra saturation applied before quantization; pixel art tends punchy.
    saturation: float = 1.15


@dataclass
class StructureConfig:
    """Stage 2 -- line and silhouette extraction."""

    outlines: bool = True
    # XDoG parameters.
    xdog_sigma: float = 0.8
    xdog_k: float = 1.6
    xdog_tau: float = 0.98
    xdog_phi: float = 12.0
    xdog_epsilon: float = 0.05
    # Blend weight of extracted lines over the sampled image.
    outline_strength: float = 0.75
    # Darken factor applied along outlines (1.0 = black, 0.0 = no darkening).
    outline_darken: float = 0.55
    # Snap near-axis-aligned line segments to the pixel grid.
    snap_lines: bool = False
    # Use the depth assist (if cached) to outline silhouettes only.
    depth_gated: bool = False


@dataclass
class SampleConfig:
    """Stage 3 -- source resolution to logical pixel grid."""

    # Exactly one of cell_size / target_width drives the grid.
    cell_size: int | None = 6
    target_width: int | None = None
    # Floor on source-pixels-per-output-pixel. A fixed target_width is the right
    # model for HD footage but breaks on small sources: 180 logical px from a
    # 360px-wide clip is a cell of 2, which barely reads as pixel art at all.
    # This caps the grid so low-resolution input still looks deliberate.
    min_cell: float = 0.0
    # area | median | outline_expand | superpixel
    method: str = "outline_expand"
    # Outline-expansion window as a multiple of cell size (PixelOE-style).
    expand_radius: float = 1.0
    # Weight of contrast-aware selection vs plain area mean, 0..1.
    contrast_weight: float = 0.7


@dataclass
class PaletteConfig:
    """Stage 4 -- palette derivation."""

    # auto | hardware | custom | reference | ramp
    mode: str = "auto"
    size: int = 16
    hardware: str | None = None
    custom_path: str | None = None
    # For mode="reference": fit the palette to this image's actual colours.
    reference: str | None = None
    # Snap final colors to a reduced channel depth, e.g. [5, 5, 5] for SNES.
    bits_per_channel: list[int] | None = None
    # Frames sampled across a shot when fitting an auto palette.
    sample_frames: int = 16
    # Pixels sampled per frame when fitting.
    sample_pixels: int = 20000
    # Weight given to saturated colors so small vivid accents survive k-means.
    chroma_weight: float = 1.0
    # Place palette entries at the source's own luminance percentiles, so each
    # gets a roughly equal share of pixels. Fixes both the "16 colours that read
    # as one" collapse and the opposite failure of spreading a ramp uniformly
    # across tones the shot does not contain. See palette.redistribute_lightness.
    redistribute: float = 0.0
    # Hold chroma toward the bright end of the ramp instead of greying out.
    warm_highlights: float = 0.0
    # For mode="ramp": the authored hue path, in degrees around Oklab.
    # Cool saturated shadow -> warm mid -> near-neutral highlight.
    shadow_hue: float = 210.0
    mid_hue: float = 350.0
    highlight_hue: float = 45.0
    # Remove isolated single cells that sit close to their neighbours (blurred
    # gradient wobbling across a threshold), while keeping high-contrast islands
    # such as lit windows. Oklab distance; 0 disables.
    despeckle: float = 0.0
    # Force the darkest entry to a true ink and the lightest to a specular.
    anchor_ink: bool = False


@dataclass
class DitherConfig:
    """Stage 4 -- dithering."""

    # none | bayer2 | bayer4 | bayer8 | bluenoise | floyd
    mode: str = "none"
    amount: float = 0.6
    # Only dither where quantization error exceeds this Oklab distance.
    # Keeps flat facades clean while letting skies band smoothly.
    selective_threshold: float = 0.02


@dataclass
class TilesConfig:
    """Stage 4 -- hardware tile color constraints (NES-style)."""

    enabled: bool = False
    tile_size: int = 8
    colors_per_tile: int = 4
    # Number of distinct sub-palettes the hardware could hold.
    subpalettes: int = 4
    shared_backdrop: bool = True


@dataclass
class TemporalConfig:
    """Stage 5 -- temporal stabilization."""

    enabled: bool = True
    # none | dis | raft
    flow: str = "dis"
    # Blend weight for the flow-warped previous logical frame.
    blend: float = 0.5
    # Palette index only changes if the new color wins by this Oklab margin.
    hysteresis: float = 0.015
    # Snap the pixel grid to integer offsets tracking camera motion.
    grid_anchor: bool = True
    # Animation rate: how often the picture actually changes, in fps.
    # 0 keeps the source rate. 8-15 is the period-accurate range -- 8-bit games
    # could not animate faster, and stepping the motion is a large part of why
    # pixel art reads as pixel art rather than as a filtered video.
    decimate_fps: float = 0.0


@dataclass
class OutputConfig:
    """Stage 6 -- upscale, post, encode."""

    # Integer upscale factor, or None to restore approximately source size.
    scale: int | None = None
    crt: bool = False
    scanline_strength: float = 0.25
    # Container frame rate. 0 keeps the source rate and duplicates held frames,
    # so the file is e.g. 24fps showing 12 distinct images per second. Setting
    # it writes a genuinely low-rate file instead -- smaller, and what most
    # people mean by "export at 12fps". Pair it with temporal.decimate_fps.
    fps: float = 0.0
    pix_fmt: str = "yuv444p"
    crf: int = 12
    codec: str = "libx264"
    preset: str = "slow"


@dataclass
class PerformanceConfig:
    """Speed controls that trade a little fidelity for a lot of time."""

    # Work at (logical_width * oversample) rather than full source resolution.
    # A 1920-wide source rendering to a 180-wide grid spends 99% of its pixel
    # budget on detail that cannot survive sampling; profiling showed tone and
    # outline expansion dominating purely because they ran at 1920x1080.
    # 0 disables and uses the full source. 4-8 is visually indistinguishable.
    oversample: float = 6.0
    # Worker processes for the parallelizable per-frame stages. 0 = auto,
    # 1 = sequential. Temporal state stays in the main process and is applied
    # in order, so results do not depend on worker count.
    workers: int = 0


@dataclass
class Config:
    """Root configuration for one render."""

    tier: str = "quality"
    preset_name: str | None = None
    tone: ToneConfig = field(default_factory=ToneConfig)
    abstract: AbstractConfig = field(default_factory=AbstractConfig)
    structure: StructureConfig = field(default_factory=StructureConfig)
    sample: SampleConfig = field(default_factory=SampleConfig)
    palette: PaletteConfig = field(default_factory=PaletteConfig)
    dither: DitherConfig = field(default_factory=DitherConfig)
    tiles: TilesConfig = field(default_factory=TilesConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    # Directory holding assist sidecar caches, if any were produced elsewhere.
    assist_cache: str | None = None

    # -- derived -----------------------------------------------------------

    def logical_size(self, src_w: int, src_h: int) -> tuple[int, int]:
        """Logical (pixel-art) resolution for a given source resolution."""
        if self.sample.target_width:
            w = int(self.sample.target_width)
            if self.sample.min_cell > 0:
                w = max(8, min(w, int(src_w / float(self.sample.min_cell))))
            h = max(1, int(round(src_h * w / src_w)))
        else:
            cell = max(1, int(self.sample.cell_size or 6))
            w = max(1, src_w // cell)
            h = max(1, src_h // cell)
        return w, h

    def effective_cell(self, src_w: int, src_h: int) -> float:
        """Average source pixels per logical pixel."""
        w, _ = self.logical_size(src_w, src_h)
        return src_w / float(w)

    def validate(self) -> None:
        if self.tier not in TIERS:
            raise ValueError(f"tier must be one of {TIERS}, got {self.tier!r}")
        if self.palette.mode not in ("auto", "hardware", "custom", "reference", "ramp"):
            raise ValueError(f"unknown palette mode {self.palette.mode!r}")
        if self.palette.mode == "hardware" and not self.palette.hardware:
            raise ValueError("palette.mode='hardware' requires palette.hardware")
        if self.palette.mode == "custom" and not self.palette.custom_path:
            raise ValueError("palette.mode='custom' requires palette.custom_path")
        if self.palette.mode == "reference" and not self.palette.reference:
            raise ValueError("palette.mode='reference' requires palette.reference")
        if self.sample.cell_size is None and self.sample.target_width is None:
            raise ValueError("set one of sample.cell_size or sample.target_width")
        if self.palette.size < 2:
            raise ValueError("palette.size must be >= 2")
        if self.sample.method not in ("area", "median", "outline_expand", "superpixel"):
            raise ValueError(f"unknown sample method {self.sample.method!r}")
        if self.dither.mode not in (
            "none", "bayer2", "bayer4", "bayer8", "bluenoise", "floyd"
        ):
            raise ValueError(f"unknown dither mode {self.dither.mode!r}")
        if self.temporal.flow not in ("none", "dis", "raft"):
            raise ValueError(f"unknown flow {self.temporal.flow!r}")
        if self.palette.bits_per_channel is not None:
            bpc = self.palette.bits_per_channel
            if len(bpc) != 3 or any(not (1 <= b <= 8) for b in bpc):
                raise ValueError("bits_per_channel must be three ints in 1..8")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# -- merging ---------------------------------------------------------------


def _merge_into(obj: Any, overlay: dict[str, Any], path: str = "") -> None:
    """Recursively apply a nested dict overlay onto a dataclass instance."""
    valid = {f.name: f for f in fields(obj)}
    for key, value in overlay.items():
        if key not in valid:
            raise ValueError(f"unknown config key {path + key!r}")
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge_into(current, value, f"{path}{key}.")
        else:
            setattr(obj, key, value)


def available_presets() -> list[str]:
    if not PRESET_DIR.is_dir():
        return []
    return sorted(p.stem for p in PRESET_DIR.glob("*.yaml"))


def load_preset_dict(name: str) -> dict[str, Any]:
    path = PRESET_DIR / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"unknown preset {name!r}; available: {', '.join(available_presets())}"
        )
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def build_config(
    preset: str | None = None,
    user_yaml: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Config:
    """Assemble a Config from preset, user file and CLI overrides."""
    cfg = Config()
    if preset:
        data = load_preset_dict(preset)
        # A preset may inherit from another via `extends:`.
        chain: list[dict[str, Any]] = [data]
        seen = {preset}
        while "extends" in chain[-1]:
            parent = chain[-1].pop("extends")
            if parent in seen:
                raise ValueError(f"circular preset inheritance at {parent!r}")
            seen.add(parent)
            chain.append(load_preset_dict(parent))
        for layer in reversed(chain):
            layer = {k: v for k, v in layer.items() if k != "description"}
            _merge_into(cfg, layer)
        cfg.preset_name = preset
    if user_yaml:
        with Path(user_yaml).open("r", encoding="utf-8") as fh:
            _merge_into(cfg, yaml.safe_load(fh) or {})
    if overrides:
        _merge_into(cfg, overrides)
    cfg.validate()
    return cfg
