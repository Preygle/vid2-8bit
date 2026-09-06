# vid2-8bit — Video → Pixel Art Converter

## Context

Convert real footage (cinematic shots, buildings, cars) into pixel art video in the style of the
*Tetris* (2023) pixel-art sequences — clean, readable, hand-drawn-looking pixel art that moves, not
a "pixelate" filter. Palette size, pixel/cell size and bit depth must all be configurable.

The naive approach —

```
resize(area) → quantize(palette) → resize(nearest)
```

— produces **noise** on detailed subjects (facades, foliage, grilles). Four compounding reasons:

1. **Aliasing.** A facade's window grid sits at 4–10 source px. At 8× downscale that's far below
   the target Nyquist limit. Area-averaging turns it to gray mush; the mush then quantizes
   semi-randomly per pixel → salt-and-pepper.
2. **No structure preservation.** A pixel artist doesn't *sample* a building, they *re-author* it:
   flat facade color, a regular 1px window grid, a 1px silhouette outline, 2–3 shading bands.
   Information is reconstructed, not averaged.
3. **Independent per-pixel quantization.** Neighbouring near-identical colors snap to different
   palette entries.
4. **Temporal.** All of the above re-rolls every frame → boiling, crawling, palette flicker.

The pipeline must therefore be **abstract → restructure → sample → quantize → stabilize**, not
sample-then-quantize.

**Honest framing:** the *Tetris* sequences were substantially hand-authored/rotoscoped by artists,
not produced by an automatic filter. The target is the closest automatable approximation. Nearly
all the distance between "pixelate filter" and "that look" lives in Stages 1–2 below.

## Decisions taken

| Question | Decision |
|---|---|
| Speed | **Two tiers** — fast preview + quality final, sharing one config |
| AI | **Structure-assist only** — nets supply segmentation/depth/lines, never pixels. Generative options parked (see *Parked approaches*) |
| Form factor | **Python CLI + local preview GUI** |
| Reference | User supplying *Tetris* movie frames → drop in `references/`, used to derive the `tetris-movie` preset |

---

## The pipeline

All color math in **Oklab**; all averaging/filtering in **linear light** (decode sRGB first).
Everything is scoped per-shot, so palettes and temporal state reset at cuts.

### Stage 0 — Ingest & shot segmentation
- ffmpeg decode → linear RGB float frames over a raw stdout pipe (don't round-trip PNGs).
- Shot detection via PySceneDetect or ffmpeg `scdet`. Defines palette scope + temporal-state scope.

### Stage 1 — Pre-abstraction ← **the single most important stage**
At source resolution, before any downsampling:
1. **Structure–texture separation** — RTV (Relative Total Variation, Xu 2012) or L0 gradient
   minimization (Xu 2011). Strips brick/stucco/sensor noise while keeping window edges and
   silhouettes.
2. **Nyquist-matched edge-preserving low-pass** — remove the spatial frequencies just above the
   target grid's Nyquist (precisely the ones that *will* alias) while preserving edges. This is
   the principled fix for the noise problem.
3. **Luminance flattening** — mean-shift / bilateral in Oklab, collapsing gradients into plateaus
   so shading reads as discrete bands rather than ramps.

### Stage 2 — Structure extraction (what must survive the downscale)
- **Stylized line extraction** — XDoG (extended difference-of-Gaussians): deterministic, fast,
  purpose-built for art lines.
- **Line-segment snapping** — LSD / EDLines, then snap near-horizontal/vertical segments to the
  pixel grid. This is what makes buildings look *drawn* rather than *sampled*.
- **Selective outlining** — outline at depth/occlusion discontinuities (silhouettes) only, not at
  interior texture. Depth from the optional assist module makes this markedly better and gives
  exactly the clean building silhouettes in the reference.

### Stage 3 — Grid sampling
- **Fast:** area downsample of the *abstracted* image + outline expansion. Works now, because
  Stage 1 removed the aliasing energy.
- **Quality:** **PixelOE**-style contrast-aware outline expansion — expand thin dark outlines
  *before* downsampling so they survive as crisp 1px lines. Purpose-built for the detail-loss
  problem; deterministic. See [PixelOE](https://github.com/KohakuBlueleaf/PixelOE).
- **Max (M6 tier):** **Gerstner et al., "Pixelated Image Abstraction" (NPAR 2012)** — each output
  pixel owns a spatially-constrained superpixel (modified SLIC/SNIC), palette co-optimized by
  simulated annealing with palette-size annealing. Academic gold standard for photo → pixel art;
  seconds per frame.
- Composite Stage-2 structure back on top at logical resolution (1px outlines, snapped lines).

### Stage 4 — Palette & quantization
- **Palette modes:**
  - `auto-N` — k-means in Oklab over frames sampled across the whole shot, area/saliency weighted.
  - `hardware` — Game Boy (4), NES (54, 25 on-screen), C64 (16), CGA, PICO-8 (16), Master System
    (64), Amiga OCS/EHB.
  - `custom` — user `.hex` / `.gpl` (Lospec-compatible).
- **Tile constraints** — what "8-bit" *actually* meant, and a large authenticity lever: NES allows
  max 4 colors per 8×8 tile with one shared backdrop. Per-tile constrained palette assignment
  (greedy + local search; small ILP for the quality tier). Optional per preset.
- **Dithering** — none / ordered Bayer 2×2·4×4·8×8 / blue-noise mask / Floyd–Steinberg.
  Default **selective Bayer**: applied only where quantization error exceeds a threshold, so skies
  and gradients stay smooth without speckling flat facades. Floyd–Steinberg is temporally unstable
  — offer it, but not for video defaults.

### Stage 5 — Temporal stabilization ← **make-or-break for video**
1. **Per-shot fixed palette.** Never re-derive per frame. Cached to disk, shared by both tiers.
2. **Grid anchoring via global motion.** Estimate camera motion; snap the pixel grid to *integer*
   logical offsets tracking the camera. Removes the "image sliding under a static grid" crawl and
   replaces it with clean stepped motion — which is the pixel-art-in-motion feel.
3. **Flow-guided temporal filter** on the pre-quantization logical-res frame: warp the previous
   logical frame by optical flow, blend under a forward–backward-consistency occlusion mask.
   DIS flow (fast tier) / RAFT (quality tier).
4. **Index hysteresis.** Keep the previous palette index unless another beats it by margin τ in
   Oklab. Kills residual flicker.
5. **Optional frame-rate decimation** — render at 12/15fps and hold. Period-authentic, and masks
   whatever flicker remains.

### Stage 6 — Output
- Nearest-neighbour upscale at an **integer** factor; pad/letterbox rather than scale fractionally.
- Optional CRT post: scanlines, slot mask, mild bloom, NTSC composite artifacts.
- **Encoding gotcha:** `yuv420p` chroma subsampling smears hard pixel edges. Encode `yuv444p`, or
  upscale enough that subsampling lands sub-cell. Lossless/near-lossless intermediate.

---

## Two-tier architecture

Each stage exposes one interface with `fast` and `quality` implementations; tier is a config axis
from day one. The governing principle: **the fast tier is a cheaper approximation of the same
algorithm, not a different algorithm** — otherwise the preview lies to you.

| Stage | Fast (preview) | Quality (final) |
|---|---|---|
| 1 Abstraction | Domain transform / guided filter | RTV or L0 gradient minimization |
| 2 Structure | XDoG | XDoG + LSD snapping + depth-gated outlines |
| 3 Sampling | Area + outline expansion | PixelOE quality → Gerstner superpixel |
| 4 Palette | **Shared cached palette**, 3D-LUT quantize | Same palette + tile-constraint solve |
| 5 Temporal | DIS flow, hysteresis | RAFT, fwd-bwd occlusion, grid anchoring |
| 6 Output | same | same |

**The palette is computed once per shot and shared by both tiers** — color is the most visible
property, so preview and final must never disagree on it.

**Sequencing note:** do *not* write two full parallel implementations up front. Build the CPU path
first (M1–M4) and have the preview GUI run that same path at reduced resolution with fast-tier
algorithm choices. Port hot stages to GLSL only in M6, once the look is locked. Otherwise you
maintain two implementations of an algorithm you haven't finalized.

---

## Configuration model — "8 bit / 10 bit / 16 bit"

These are three *independent* axes that get conflated in casual usage. Expose them separately,
then wrap in presets:

| Axis | Knob | Notes |
|---|---|---|
| Chunkiness | `cell_size` / `target_resolution` | cell=6 on 1080p → 320×180 logical |
| Palette size | `palette_size` | "8-bit era" ≈ 4–64 colors; literal 8-bit color = 256 |
| Channel depth | `bits_per_channel` | 3-3-2 (VGA 256), 5-5-5 (SNES), 8-8-8 |

**Era presets** set all three together with tile constraints, dither mode and shading-band count:
`gameboy`, `nes`, `c64`, `pico8`, `snes`, `genesis`, `vga256`, `modern-hifi`, plus a derived
`tetris-movie`.

---

## Project layout

```
vid2-8bit/
  pyproject.toml
  references/                 # Tetris frames → tetris-movie preset derivation
  src/vid2_8bit/
    cli.py                    # entrypoint
    config.py                 # dataclasses, YAML load, preset resolution, tier flags
    pipeline.py               # stage graph, tier dispatch, per-shot state
    presets/                  # nes.yaml, gameboy.yaml, snes.yaml, tetris-movie.yaml …
    io/       decode.py  encode.py  shots.py
    color/    spaces.py  palette.py  quantize.py  dither.py  tiles.py
    stages/   abstract.py  structure.py  sample.py  temporal.py  output.py
    assist/   depth.py  segment.py     # optional ONNX; graceful no-op if models absent
    gui/      preview.py               # moderngl + imgui viewer
  tools/
    contact_sheet.py          # A/B comparison harness
    metrics.py                # noise + temporal-churn metrics
```

### Stack
- **Python 3.13** + numpy / OpenCV / scipy; **numba** for hot loops (palette assignment, tile solve).
- **ffmpeg** via raw stdin/stdout pipes (already installed).
- **moderngl + imgui + glfw** for the preview GUI — real GPU sliders on AMD via OpenGL, and it
  double-serves as the shader host for the M6 fast tier.
- **onnxruntime-directml** for the assist modules.

### Hardware note
The RX 6700M is RDNA2. ROCm-on-Windows is preview-only for RX 7000/9000 series, so there is no
practical local PyTorch-CUDA path. Assist models run through ONNX Runtime + DirectML, with CPU
fallback (Depth Anything V2 Small is tolerable on CPU at reduced resolution). Verify DirectML
actually accelerates the chosen models before depending on it — treat the assist path as optional
throughout, which the `graceful no-op` design already enforces.

---

## Build order

| Milestone | Deliverable |
|---|---|
| **M0** | ffmpeg raw-pipe IO, config + preset system, naive pipeline, **A/B contact-sheet harness** — you cannot tune this without one |
| **M1** | Stage 1 abstraction + Stage 4 Oklab per-shot palette → the single biggest visual jump |
| **M2** | Stage 5 temporal stabilization (fixed palette, grid anchoring, flow filter, hysteresis) |
| **M3** | Stage 2 structure/outlines + PixelOE-style outline expansion |
| **M4** | Presets, tile constraints, dither modes, bit-depth snapping; derive `tetris-movie` from `references/` |
| **M5** | Preview GUI (moderngl + imgui), live sliders on a scrubbed frame, CPU path at reduced res |
| **M6** | Neural assists behind a flag; GLSL fast tier; Gerstner quality tier |

---

## Verification

- **Contact-sheet harness (M0)** — render N stills through each variant/preset side by side.
- **Noise metric** — count isolated single-pixel palette outliers per frame; should trend toward 0
  on flat regions (facades) as Stage 1 improves.
- **Temporal metric** — mean per-pixel palette-index change rate between frames. On a locked-off
  shot it should approach 0; on a pan it should show clean stepped motion, not per-pixel churn.
- **Eyeball tests**, each a distinct failure mode: (1) locked-off building, (2) slow pan across a
  facade, (3) a car crossing frame, (4) a face.
- **Tier agreement** — preview and final on the same frame must match in palette and read as the
  same image; regression-test this, it's the thing that silently rots.
- Encode a clip and confirm hard edges survive at `yuv444p`.

---

## Parked approaches (kept for future use)

Not being built now, but the stage-graph + `assist/` plugin design leaves room for each. Recorded
here so the decision context isn't lost.

| # | Approach | Quality | Temporal | Cost | Runs on RX 6700M |
|---|---|---|---|---|---|
| **D** | Feed-forward pixelization net — [Pixelization, SIGGRAPH Asia '22](https://github.com/WuZongWei6/Pixelization) | Good, purpose-built | Needs flow help | Low | ⚠️ DirectML / CPU |
| **E** | Per-frame diffusion img2img (SDXL + pixel-art LoRA + ControlNet lineart/depth/tile) | Gorgeous stills | **Poor** — boils badly | High | ❌ cloud |
| **F** | Video diffusion v2v (Runway Aleph, Wan VACE, LTX) | "Pixel-art-ish" | Good | High $/sec | ❌ cloud |
| **G** | **Distillation hybrid** — diffusion on 3–10 keyframes to art-direct the look, extract palette + style params, then render the full video deterministically | High | Excellent | Low, one-time | ✅ |

**Why they're parked.** E and F produce something that *looks like* pixel art but has no true cell
grid, no exact palette, no hardware tile constraints, and sub-cell noise. You cannot dial
"16 colors, 8px cells, NES tile rules" — which is a stated requirement. They are art-direction
tools, not renderers.

**G is the natural future addition** and needs no architectural change: it only writes a palette +
preset file that the deterministic pipeline already consumes. Worth revisiting once the
`tetris-movie` preset exists and you want variations on it.

**D** would slot in as an alternative Stage 3 implementation.
