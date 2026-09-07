"""Pipeline orchestration.

Ties the stages together and owns the two pieces of state that must be scoped to
a shot rather than a frame: the palette, and the temporal history. Getting that
scoping right is most of what separates this from a per-frame filter.
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .assist import NULL_CACHE, AssistCache
from .metrics import isolated_pixel_ratio, temporal_churn
from .color import dither as dither_mod
from .color import palette as palette_mod
from .color import despeckle as despeckle_mod
from .color import tiles as tiles_mod
from .color.quantize import Quantizer, snap_image_bit_depth
from .color.spaces import float_to_u8, u8_to_float
from .config import Config
from .io import decode, encode, shots as shots_mod
from .stages import abstract as abstract_mod
from .stages import output as output_mod
from .stages import sample as sample_mod
from .stages import structure as structure_mod
from .stages import tone as tone_mod
from .stages.temporal import TemporalState, decimate_plan

log = logging.getLogger(__name__)


def maxpool_to(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Downsample a line mask by MAX, not by average.

    Averaging a 1-pixel line over a 6x6 cell reduces it to 1/36 strength and it
    disappears. Max-pooling asks "was there a line anywhere in this cell", which
    is the question that matters -- it is how a 1-pixel source line becomes a
    1-pixel output line instead of a faint grey smudge.
    """
    w_out, h_out = size
    h, w = mask.shape
    # Pad up to an exact multiple so the reshape trick is valid.
    ph, pw = (-h) % h_out, (-w) % w_out
    if ph or pw:
        mask = np.pad(mask, ((0, ph), (0, pw)), mode="edge")
    bh, bw = mask.shape[0] // h_out, mask.shape[1] // w_out
    if bh < 1 or bw < 1:
        return cv2.resize(mask, size, interpolation=cv2.INTER_LINEAR)
    trimmed = mask[: bh * h_out, : bw * w_out]
    return trimmed.reshape(h_out, bh, w_out, bw).max(axis=(1, 3)).astype(np.float32)


@dataclass
class ShotRender:
    """Per-shot state: the palette and temporal history."""

    quantizer: Quantizer
    temporal: TemporalState
    frames: int = 0


@dataclass
class Stats:
    frames: int = 0
    seconds: float = 0.0
    shots: int = 0
    palette_sizes: list[int] = field(default_factory=list)
    # Quality metrics accumulated from palette index maps.
    noise_samples: list[float] = field(default_factory=list)
    churn_samples: list[float] = field(default_factory=list)

    @property
    def noise(self) -> float:
        return float(np.mean(self.noise_samples)) if self.noise_samples else 0.0

    @property
    def churn(self) -> float:
        return float(np.mean(self.churn_samples)) if self.churn_samples else 0.0

    def report(self) -> str:
        fps = self.frames / self.seconds if self.seconds > 0 else 0.0
        pal = (
            f"{min(self.palette_sizes)}-{max(self.palette_sizes)}"
            if self.palette_sizes
            else "n/a"
        )
        line = (
            f"{self.frames} frames in {self.seconds:.1f}s ({fps:.2f} fps), "
            f"{self.shots} shot(s), palette {pal} colors"
        )
        if self.noise_samples:
            line += (
                f"\nspatial noise {self.noise * 100:.2f}%  |  "
                f"temporal churn {self.churn * 100:.2f}%"
            )
        return line


class Converter:
    """Converts a video (or a single frame) to pixel art."""

    def __init__(
        self,
        cfg: Config,
        assist: AssistCache | None = None,
        collect_metrics: bool = True,
    ):
        cfg.validate()
        self.cfg = cfg
        self.assist = assist or NULL_CACHE
        self.stats = Stats()
        # Metrics are gathered from index maps here rather than by re-reading
        # the encoded file. Video compression perturbs pixel values, so a churn
        # measured on an h.264 output reports ~98% no matter how stable the
        # render actually is -- it is measuring the codec, not the pipeline.
        self.collect_metrics = collect_metrics

    # -- palette -----------------------------------------------------------

    def fit_palette(self, source: str | Path, shot: "shots_mod.Shot") -> np.ndarray:
        """Fit one palette for an entire shot.

        Sampling across the whole shot rather than from a single frame is what
        keeps the palette from lurching when the content changes -- a palette
        fitted to frame 0 of a shot that pans from sky to street will have no
        colors left for the street.
        """
        pcfg = self.cfg.palette
        if pcfg.mode != "auto":
            return palette_mod.build_palette(self.cfg, self._sample_pixels(source, shot))

        pixels = self._sample_pixels(source, shot)
        return palette_mod.build_palette(self.cfg, pixels)

    def _sample_pixels(self, source: str | Path, shot: "shots_mod.Shot") -> np.ndarray:
        frames = decode.sample_frames(
            source,
            n=self.cfg.palette.sample_frames,
            start=shot.start,
            end=shot.end,
            scale_width=480,
        )
        per_frame = max(256, self.cfg.palette.sample_pixels // max(1, len(frames)))
        rng = np.random.default_rng(0)
        chunks = []
        for frame in frames:
            img = u8_to_float(frame)
            # Fit to *abstracted* colors: that is what will actually be
            # quantized, and raw footage contains texture colours that the
            # abstraction stage removes before quantization ever sees them.
            #
            # The cell must match what the render will actually use, scaled for
            # the reduced sampling resolution. Fitting at a hardcoded cell fitted
            # the palette to colours the renderer never produces, which showed up
            # as hue contamination on flat regions.
            cell = self.cfg.effective_cell(frame.shape[1], frame.shape[0])
            img = tone_mod.apply_tone(img, self.cfg)
            img = abstract_mod.abstract_frame(img, self.cfg, cell, tier=self.cfg.tier)
            flat = img.reshape(-1, 3)
            n = min(per_frame, len(flat))
            chunks.append(flat[rng.choice(len(flat), size=n, replace=False)])
        return np.concatenate(chunks, axis=0)

    # -- per frame ---------------------------------------------------------

    def render_frame(
        self,
        frame_u8: np.ndarray,
        state: ShotRender,
        frame_index: int = 0,
        target_size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Convert one source frame. Returns uint8 RGB at delivery resolution."""
        cfg = self.cfg
        src_h_full, src_w_full = frame_u8.shape[:2]
        size = cfg.logical_size(src_w_full, src_h_full)
        offset = state.temporal.grid_offset(u8_to_float(self._prescale(frame_u8, size[0])))
        logical = self.render_logical(frame_u8, size, offset, frame_index)
        return self.finish_frame(
            logical, state, offset, (src_w_full, src_h_full), target_size
        )

    def render_logical(
        self,
        frame_u8: np.ndarray,
        size: tuple[int, int],
        offset: tuple[int, int],
        frame_index: int = 0,
    ) -> np.ndarray:
        """Everything with no dependency on other frames, down to the logical grid.

        Split out from render_frame so it can run on a worker thread. It is also
        the expensive part -- roughly 95% of per-frame cost -- while the
        sequential remainder (temporal blend, quantize, upscale) works on a tiny
        180x101 image and costs almost nothing.
        """
        cfg = self.cfg

        # Work at a modest multiple of the output grid rather than at full
        # source resolution. Everything below roughly a quarter of a cell is
        # removed by sampling regardless, so rendering a 180-wide grid from a
        # 1920-wide frame spends ~99% of the pixel budget on detail that is
        # discarded. Costs a little fine texture, saves most of the runtime.
        frame_u8 = self._prescale(frame_u8, size[0])
        img = u8_to_float(frame_u8)
        cell = img.shape[1] / float(size[0])

        depth = self.assist.depth(frame_index) if cfg.structure.depth_gated else None

        # Stage 0.5 -- tone and grade. Must precede abstraction: L0 needs real
        # contrast to find edges in, and k-means needs a spread of colours.
        img = tone_mod.apply_tone(img, cfg)

        # Stage 1 -- abstraction at working resolution.
        abstracted = abstract_mod.abstract_frame(img, cfg, cell, tier=cfg.tier)

        # Stage 2 -- line extraction, also before downsampling.
        lines = structure_mod.extract_lines(abstracted, cfg, depth)

        # Stage 3 -- down to the logical grid.
        logical = sample_mod.sample_frame(abstracted, cfg, size, cfg.tier, offset)

        # Stage 2b -- composite lines, max-pooled so 1px lines survive.
        if cfg.structure.outlines and lines.max() > 0:
            lines_small = maxpool_to(lines, size)
            if cfg.structure.snap_lines:
                lines_small = structure_mod.snap_axis_lines(lines_small)
            logical = structure_mod.composite_outlines(
                logical,
                lines_small,
                cfg.structure.outline_strength,
                cfg.structure.outline_darken,
            )
        return logical

    def finish_frame(
        self,
        logical: np.ndarray,
        state: ShotRender,
        offset: tuple[int, int],
        full_size: tuple[int, int],
        target_size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """The sequential remainder: temporal state, quantize, upscale.

        Must run in frame order -- every step here reads state left by the
        previous frame, which is exactly what makes the output stable.
        """
        cfg = self.cfg
        src_w_full, src_h_full = full_size

        # Stage 5b -- temporal blending before quantization, so the quantizer
        # sees an already-stable image.
        logical = state.temporal.stabilize(logical)

        if cfg.palette.bits_per_channel:
            logical = snap_image_bit_depth(logical, cfg.palette.bits_per_channel)

        # Stage 4 -- dither then quantize.
        idx = self._quantize(logical, state, offset)
        if cfg.palette.despeckle > 0.0:
            idx = despeckle_mod.despeckle_indices(
                idx, state.quantizer.palette_lab, cfg.palette.despeckle
            )
        if self.collect_metrics:
            self.stats.noise_samples.append(isolated_pixel_ratio(idx))
            prev = state.temporal.prev_idx
            if prev is not None and prev.shape == idx.shape:
                self.stats.churn_samples.append(temporal_churn(prev, idx))
        state.temporal.note_indices(idx)
        out = state.quantizer.to_rgb(idx)

        # Stage 6 -- upscale. Sized from the ORIGINAL frame, so the delivery
        # resolution does not shift when the performance prescale changes.
        tw, th = target_size or (src_w_full, src_h_full)
        final = output_mod.finalize(out, cfg, tw, th)
        return float_to_u8(final)

    def _prescale(self, frame_u8: np.ndarray, logical_w: int) -> np.ndarray:
        """Downscale toward the working resolution before the expensive stages."""
        over = float(self.cfg.performance.oversample)
        if over <= 0:
            return frame_u8
        target_w = int(max(logical_w * over, logical_w))
        src_h, src_w = frame_u8.shape[:2]
        # Only bother when there is real work to save.
        if target_w >= src_w * 0.9:
            return frame_u8
        target_h = max(2, int(round(src_h * target_w / src_w)))
        return cv2.resize(
            frame_u8, (target_w, target_h), interpolation=cv2.INTER_AREA
        )

    def _quantize(
        self, logical: np.ndarray, state: ShotRender, offset: tuple[int, int]
    ) -> np.ndarray:
        cfg = self.cfg
        q = state.quantizer

        if cfg.tiles.enabled:
            # Tile constraints subsume plain quantization: each tile is solved
            # against its own restricted sub-palette.
            return tiles_mod.apply_tile_constraints(
                logical,
                q.palette,
                tile_size=cfg.tiles.tile_size,
                colors_per_tile=cfg.tiles.colors_per_tile,
                subpalettes=cfg.tiles.subpalettes,
                shared_backdrop=cfg.tiles.shared_backdrop,
            )

        if cfg.dither.mode == "floyd":
            return dither_mod.apply_floyd_steinberg(logical, q)

        if cfg.dither.mode != "none":
            # Offset the threshold matrix by the grid anchor so the dither
            # pattern stays locked to the scene during a pan.
            cell = max(1, int(round(self.cfg.sample.cell_size or 6)))
            dither_offset = (offset[0] // cell, offset[1] // cell)
            logical = dither_mod.apply_ordered(
                logical,
                q,
                cfg.dither.mode,
                amount=cfg.dither.amount,
                selective_threshold=cfg.dither.selective_threshold,
                offset=dither_offset,
            )

        return q.quantize(logical, **state.temporal.hysteresis_args())

    # -- whole clip --------------------------------------------------------

    def new_shot_state(self, palette: np.ndarray, cell: float) -> ShotRender:
        lut_bits = 6 if self.cfg.tier == "fast" else 0
        return ShotRender(
            quantizer=Quantizer(palette, lut_bits=lut_bits),
            temporal=TemporalState(self.cfg, cell),
        )

    def convert(
        self,
        source: str | Path,
        dest: str | Path,
        detect_cuts: bool = True,
        max_frames: int | None = None,
        progress=None,
        keep_audio: bool = True,
    ) -> Stats:
        """Convert a whole video."""
        cfg = self.cfg
        info = decode.probe(source)
        size = cfg.logical_size(info.width, info.height)
        cell = cfg.effective_cell(info.width, info.height)
        scale = cfg.output.scale or output_mod.integer_scale(
            size[0], size[1], info.width, info.height
        )
        out_w, out_h = size[0] * scale, size[1] * scale

        # Animation rate (how often the picture changes) and container rate are
        # separate. Holding at the source rate duplicates frames; setting an
        # output fps writes a genuinely low-rate file instead.
        anim_fps = float(cfg.temporal.decimate_fps)
        out_fps = float(cfg.output.fps)
        hold_global = 1
        if anim_fps > 0 and anim_fps < info.fps:
            hold_global = max(1, int(round(info.fps / anim_fps)))
        if out_fps > 0:
            # One written frame per rendered frame; the container carries the rate.
            writer_fps = out_fps
            if anim_fps <= 0:
                hold_global = max(1, int(round(info.fps / out_fps)))
            duplicate = 1
        else:
            writer_fps = info.fps
            duplicate = hold_global

        shot_list = (
            shots_mod.detect_shots(source) if detect_cuts else shots_mod.single_shot(source)
        )
        self.stats.shots = len(shot_list)
        log.info(
            "%s: %dx%d @ %.2ffps -> logical %dx%d (cell %.2f), out %dx%d, %d shot(s)",
            Path(source).name, info.width, info.height, info.fps,
            size[0], size[1], cell, out_w, out_h, len(shot_list),
        )

        total = info.n_frames or 0
        if max_frames:
            total = min(total, max_frames) if total else max_frames

        # 6 measured as the plateau on a 16-thread machine: beyond that this
        # pool contends with OpenCV's own internal threading and throughput
        # stops improving (8 was marginally slower than 4 in benchmarks).
        workers = int(cfg.performance.workers) or min(6, (os.cpu_count() or 4))
        workers = max(1, workers)
        depth = max(2, workers * 2)
        log.info("rendering with %d worker thread(s), oversample %.1f",
                 workers, cfg.performance.oversample)

        started = time.perf_counter()
        with ThreadPoolExecutor(workers) as pool, encode.VideoWriter(
            dest,
            out_w,
            out_h,
            writer_fps,
            pix_fmt=cfg.output.pix_fmt,
            codec=cfg.output.codec,
            crf=cfg.output.crf,
            preset=cfg.output.preset,
            audio_from=source if (keep_audio and out_fps <= 0) else None,
        ) as writer:
            written = 0
            for shot in shot_list:
                if max_frames and written >= max_frames:
                    break
                palette = self.fit_palette(source, shot)
                self.stats.palette_sizes.append(len(palette))
                state = self.new_shot_state(palette, cell)
                log.info("shot %d [%d:%d] palette=%d colors",
                         shot.index, shot.start, shot.end, len(palette))

                count = shot.length
                if max_frames:
                    count = min(count, max_frames - written)

                # Animating "on twos" -- rendering at 12-15fps and holding each
                # frame -- is period-accurate for 8-bit games and also hides
                # whatever flicker survives Stage 5. Held frames are written
                # again rather than re-rendered, so it is a speedup too.
                hold = hold_global

                held: np.ndarray | None = None
                full_size = (info.width, info.height)

                # Pipelined: the cheap sequential steps stay on this thread and
                # the expensive per-frame work (tone, abstraction, sampling --
                # about 95% of the cost) runs on a pool. numpy and OpenCV both
                # release the GIL, so threads give real parallelism here without
                # the cost of pickling HD frames to worker processes.
                #
                # Grid anchoring is computed here, in order, because it
                # accumulates camera motion across frames; the resulting offset
                # is handed to the worker. Everything order-dependent therefore
                # still happens in order, and the output does not depend on how
                # many workers are used.
                inflight: deque = deque()

                def drain_one() -> bool:
                    nonlocal held, written
                    idx_f, fut = inflight.popleft()
                    logical = fut.result()
                    held = self.finish_frame(
                        logical, state, offsets[idx_f], full_size, (out_w, out_h)
                    )
                    self.stats.frames += 1
                    for _ in range(holds[idx_f]):
                        writer.write(held)
                        written += 1
                        if progress:
                            progress(written, total)
                        if max_frames and written >= max_frames:
                            return False
                    return True

                offsets: dict[int, tuple[int, int]] = {}
                holds: dict[int, int] = {}
                stop = False
                rendered = 0

                for i, frame in enumerate(
                    decode.read_frames(source, start=shot.start, count=count)
                ):
                    if i % hold != 0:
                        continue          # held frames are written, not rendered
                    remaining = count - i
                    holds[rendered] = 1 if duplicate == 1 else min(hold, remaining)
                    offsets[rendered] = state.temporal.grid_offset(
                        u8_to_float(self._prescale(frame, size[0]))
                    )
                    inflight.append((
                        rendered,
                        pool.submit(self.render_logical, frame, size,
                                    offsets[rendered], shot.start + i),
                    ))
                    rendered += 1
                    # Bounded look-ahead keeps memory flat on long clips.
                    while len(inflight) >= depth:
                        if not drain_one():
                            stop = True
                            break
                    if stop:
                        break

                while inflight and not stop:
                    if not drain_one():
                        break

        self.stats.seconds = time.perf_counter() - started
        return self.stats

    def convert_frame(self, source: str | Path, index: int) -> np.ndarray:
        """Convert a single frame -- used by the contact sheet and preview."""
        info = decode.probe(source)
        frame = decode.read_frame(source, index)
        cell = self.cfg.effective_cell(info.width, info.height)
        shot = shots_mod.Shot(0, max(0, index - 24), index + 24)
        palette = self.fit_palette(source, shot)
        state = self.new_shot_state(palette, cell)
        size = self.cfg.logical_size(info.width, info.height)
        scale = self.cfg.output.scale or output_mod.integer_scale(
            size[0], size[1], info.width, info.height
        )
        return self.render_frame(
            frame, state, frame_index=index,
            target_size=(size[0] * scale, size[1] * scale),
        )

    def convert_image(self, img_u8: np.ndarray) -> np.ndarray:
        """Convert a standalone still image."""
        h, w = img_u8.shape[:2]
        cell = self.cfg.effective_cell(w, h)
        pixels = u8_to_float(img_u8).reshape(-1, 3)
        rng = np.random.default_rng(0)
        n = min(self.cfg.palette.sample_pixels, len(pixels))
        sub = pixels[rng.choice(len(pixels), size=n, replace=False)]
        palette = palette_mod.build_palette(self.cfg, sub)
        state = self.new_shot_state(palette, cell)
        size = self.cfg.logical_size(w, h)
        scale = self.cfg.output.scale or output_mod.integer_scale(size[0], size[1], w, h)
        return self.render_frame(
            img_u8, state, target_size=(size[0] * scale, size[1] * scale)
        )
