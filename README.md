# vid2-8bit

Convert real footage — cinematic shots, buildings, cars — into **pixel art video** in the style of
the *Tetris* (2023) pixel sequences. Palette size, pixel/cell size and bit depth are all
configurable.

Not a "pixelate" filter. See [Why naive pixelation fails](#why-naive-pixelation-fails).

---

## Status

Working end to end. `convert`, `frame`, `contact-sheet`, `preview`, `presets` and `metrics` all
run; 55 tests pass. See [`docs/PLAN.md`](docs/PLAN.md) for the full design.

| Milestone | Scope | State |
|---|---|---|
| M0 | ffmpeg IO, config/presets, pipeline, A/B contact-sheet harness | Done |
| M1 | L0 pre-abstraction + Oklab per-shot palette | Done |
| M2 | Temporal stabilization | Done |
| M3 | Structure extraction + outline expansion | Done |
| M4 | 10 presets, tile constraints, dithering, bit-depth snap | Done |
| M5 | Preview GUI (OpenCV highgui, live sliders) | Done |
| M6 | Neural assists, GLSL fast tier, Gerstner superpixel tier | Not started |

### Measured results

On the synthetic test clip (`python tools/make_testclip.py`), measured on palette index maps
during rendering rather than on the encoded file:

**Spatial noise** — isolated single-pixel outliers, the salt-and-pepper artifact, on a panning shot:

| Configuration | noise | churn |
|---|---|---|
| Naive (`downsample → quantize`) | 5.65% | 32.3% |
| + L0 abstraction | 5.11% | 35.0% |
| + abstraction and temporal | **2.73%** | **25.2%** |

**Temporal stability** — a locked-off shot with realistic sensor grain, which is what actually
causes boiling:

| Configuration | noise | churn |
|---|---|---|
| Naive, no temporal | 5.30% | 1.66% |
| + L0 abstraction | 4.10% | 1.31% |
| + index hysteresis | 4.07% | **0.01%** |
| + full temporal stack | 4.07% | 0.02% |

Index hysteresis is the decisive mechanism for flicker: it cuts churn by more than 100x, because
it stops pixels sitting near a palette boundary from flipping between two colours every frame.

> **Caveat on the noise metric:** it counts *every* isolated pixel, so it cannot distinguish
> moire speckle from a deliberate 1-pixel outline or an ordered dither. Presets that use both
> (like `tetris-movie`) score *worse* on it while looking better. Treat it as a regression gate
> for a fixed configuration, not as a cross-preset quality score. Visual A/B via
> `contact-sheet` remains the primary instrument.

---

## Why naive pixelation fails

The obvious approach —

```
resize(area) -> quantize(palette) -> resize(nearest)
```

— produces **noise** on detailed subjects. Four compounding causes:

1. **Aliasing.** A building facade's window grid sits at 4–10 source px. At 8x downscale that is
   far below the target grid's Nyquist limit. Area-averaging turns it into gray mush, and the mush
   then quantizes semi-randomly per pixel → salt-and-pepper.
2. **No structure preservation.** A pixel artist does not *sample* a building, they *re-author* it:
   flat facade color, a regular 1px window grid, a 1px silhouette outline, 2–3 shading bands.
3. **Independent per-pixel quantization.** Neighbouring near-identical colors snap to different
   palette entries.
4. **No temporal model.** All of the above re-rolls every frame → boiling, crawling, palette flicker.

So the pipeline is **abstract → restructure → sample → quantize → stabilize**, not
sample-then-quantize. The abstraction and restructuring stages are where nearly all the quality
lives.

**One counter-intuitive result worth recording.** It seems principled to low-pass sub-cell detail
*before* structure-texture separation - detail below the cell scale can only alias, so remove it
first. Measured in isolation that looks like a huge win (isolated-pixel noise 5.2% → 0.2%). It is
a trap: the blur turns building silhouettes into soft ramps, L0 then has no sharp edge to snap to,
and it invents smooth curved region boundaries where a straight roofline belongs. A/B renders show
buildings dissolving into haze while the metric improves. L0 runs first; the low-pass is an
optional post-step, off by default. This is why `contact-sheet` exists.

> **Framing:** the *Tetris* sequences were substantially hand-authored and rotoscoped by artists,
> not made by an automatic filter. This targets the closest automatable approximation.

---

## Architecture

Seven stages, all scoped **per shot** so palettes and temporal state reset at cuts. All averaging
happens in **linear light**; all perceptual distance is measured in **Oklab**.

| Stage | Module | Purpose |
|---|---|---|
| 0 Ingest | `io/decode.py`, `io/shots.py` | ffmpeg raw pipe → linear RGB; scene detection |
| 1 Abstraction | `stages/abstract.py` | L0 structure–texture separation, optional Nyquist low-pass, luminance banding |
| 2 Structure | `stages/structure.py` | XDoG lines, axis-run snapping, depth-gated silhouette outlines |
| 3 Sampling | `stages/sample.py` | Outline-expanded downsample to logical resolution |
| 4 Quantize | `color/palette.py`, `quantize.py`, `dither.py`, `tiles.py` | Oklab palette, tile constraints, selective dithering |
| 5 Temporal | `stages/temporal.py` | Fixed palette, grid anchoring, flow-guided filter, index hysteresis |
| 6 Output | `stages/output.py`, `io/encode.py` | Integer nearest upscale, CRT post, `yuv444p` encode |

Every stage has **`fast`** and **`quality`** implementations selected by one config axis. Governing
rule: *the fast tier is a cheaper approximation of the same algorithm, not a different algorithm* —
otherwise the preview lies to you. The palette is computed once per shot and **shared by both
tiers**, since color is the most visible property.

### The three config axes people conflate

"8-bit / 10-bit / 16-bit" means three independent things. They are exposed separately, then bundled
into era presets:

| Axis | Knob | Notes |
|---|---|---|
| Chunkiness | `cell_size` / `target_resolution` | cell=6 on 1080p → 320x180 logical |
| Palette size | `palette_size` | "8-bit era" is 4–64 colors; literal 8-bit color is 256 |
| Channel depth | `bits_per_channel` | 3-3-2 (VGA 256), 5-5-5 (SNES), 8-8-8 |

Presets: `gameboy`, `nes`, `c64`, `pico8`, `snes`, `genesis`, `vga256`, `modern-hifi`,
plus `tetris-movie` (derived from reference frames).

---

## Install

```bash
python -m pip install -e .
```

Core deps are CPU-only (`numpy`, `opencv-python`, `scipy`, `PyYAML`) and **ffmpeg must be on PATH**.

Optional extras:

```bash
pip install -e ".[fast]"    # numba - JIT for palette assignment / tile solve
pip install -e ".[gui]"     # moderngl + glfw - reserved for the future GLSL fast tier
```

The preview GUI needs **no extras** - it is built on OpenCV's highgui, which is already a core
dependency. The neural assist modules are deliberately **not** installed here; see below.

---

## Compute topology

This project runs across two machines. **Read this section before assuming what is available.**

### 1. Windows (primary dev box)

- **GPU:** AMD Radeon RX 6700M — RDNA2, `gfx1031`, ~10 GB VRAM.
- **System Python:** 3.13, CPU-only. Has `numpy`, `cv2`, `scipy`, `yaml`, `sklearn`,
  `onnxruntime`, `PIL`. **No torch.**
- **Working GPU PyTorch exists**, inside the ComfyUI-ROCm install — *not* in system Python:

  ```
  C:\Users\moham\Documents\PROGRAMMING\comfyui-rocm\python_env\python.exe
  ```

  Verified: `torch 2.12.0+rocm7.14.0a`, HIP 7.14, `torch.cuda.is_available() == True`,
  device `AMD Radeon RX 6700M`, `gfx1031`, 9.98 GB.

  > Note `gfx1031` is flagged `IS_LEGACY_GPU` by ComfyUI's launcher; some kernels fall back or
  > need `HSA_OVERRIDE_GFX_VERSION=10.3.0`. Treat GPU speedups as *measure, do not assume*.

- **ComfyUI** is installed at `.../PROGRAMMING/comfyui-rocm` (launch via `comfyui-rocm.bat`,
  default API on `http://127.0.0.1:8188`). There is also a
  `.../PROGRAMMING/comfyui-mcp-server` for driving it programmatically.
  ComfyUI is the intended host for the **parked generative approaches** (see PLAN.md) — SDXL-class
  img2img fits in 10 GB.

### 2. Linux (secondary — PyTorch environment)

A working PyTorch setup lives on a separate Linux OS, **not reachable from the Windows session**.
It is the preferred host for heavy neural work.

### Consequence for the design

The core pipeline is **pure CPU numpy/OpenCV** and must stay that way — it has to run anywhere.
Neural work is isolated behind `src/vid2_8bit/assist/`, which:

- never imports torch at module import time,
- degrades to a **graceful no-op** when models or torch are unavailable,
- communicates by **cached `.npy` sidecar files**, not in-process tensors.

That last point is the key portability decision: an assist pass can be produced on *any* machine
with torch and consumed on *any* machine without it.

---

## For a Claude session on the Linux box

You are picking up a project whose main pipeline was developed on Windows without torch. Your job
is the parts that need a GPU. Read [`docs/PLAN.md`](docs/PLAN.md) first, then:

### What to build here

The assist passes in `src/vid2_8bit/assist/`. Each is a **standalone script** that reads frames and
writes a `.npy` sidecar cache — it must not become an import-time dependency of the core pipeline.

| Pass | Model | Consumed by | Feeds |
|---|---|---|---|
| `depth.py` | Depth Anything V2 (Small/Base) | `stages/structure.py` | Depth-gated silhouette outlining, shading bands |
| `segment.py` | SAM 2 (video mode — tracks across frames, giving temporal coherence for free) | `stages/abstract.py` | Flat semantic regions — the fix for noisy facades |
| `flow.py` | RAFT | `stages/temporal.py` | Quality-tier optical flow (CPU fallback: OpenCV DIS) |

### Contract every assist pass must satisfy

```
python -m vid2_8bit.assist.<pass> --input <frames_dir|video> --out cache/<pass>/ [--fp16]
```

- Output: one `.npy` per frame named `%08d.npy`, plus a `meta.json` recording model name, version,
  input resolution and any preprocessing — so the consuming machine can validate cache freshness.
- Arrays are float32 `(H, W)` for depth, float32 `(H, W, 2)` for flow, int32 `(H, W)` label maps
  for segmentation.
- Resolution must match the **source** frames, not the logical pixel grid. Downsampling to the
  logical grid is the core pipeline's job, not the assist's.
- Deterministic: fix seeds, disable nondeterministic kernels. Temporal stability depends on it.

### Environment setup on Linux

```bash
python -m pip install -e ".[assist-torch]"   # add this extra to pyproject if missing
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If the Linux side is also ROCm on the same RX 6700M, expect `gfx1031` quirks; you may need
`HSA_OVERRIDE_GFX_VERSION=10.3.0`. Benchmark before committing to GPU — for Depth Anything V2 Small
at reduced resolution, CPU is often adequate.

### Ground rules

- **Do not** move core pipeline stages onto torch. The CPU path is the product; assists are optional.
- **Do not** let an assist failure fail a render — the pipeline must fall back and log.
- **Do not** commit sidecar caches or model weights. They are gitignored. Commit code, presets, and
  `meta.json` schemas only.

---

## Web UI

```
run-webapp.bat
```

Double-click it. The script finds Python, checks ffmpeg, installs any missing packages, starts a
local server and opens your browser. Leave the console window open; Ctrl+C there stops it.

Pick a file three ways: **Browse** opens this PC's own file dialog (no copying &mdash; best for
large video), **Upload** uses the browser's file picker, or drag a file onto the drop zone. Then
drag sliders — the pixel-art preview re-renders
next to the source as you go (~0.4s per update at the fast tier). Every parameter is exposed: cell
size, sampling method, palette mode/size, hardware palettes, bits-per-channel, NES tile limits,
texture removal, shading bands, saturation, outlines, dithering, CRT.

**Export preset** dumps YAML you can drop straight into `src/vid2_8bit/presets/`.
**Render full video** runs the whole clip at the quality tier with a progress bar.

The preview keeps the same *logical* resolution as the final render, so what you tune is what you
get — it works from a downscaled source and scales the cell size to match, rather than applying the
full cell size to a smaller image and showing you something twice as chunky as reality.

Options: `run-webapp.bat --port 9000`, `run-webapp.bat --no-browser`. Binds `127.0.0.1` only; it
takes a filesystem path and starts renders, so it must not be exposed to a network.

Equivalent without the .bat:

```bash
vid2-8bit web --port 8750
```

---

## Usage (CLI)

```bash
vid2-8bit convert in.mp4 -o out.mp4 --preset nes
vid2-8bit convert in.mp4 -o out.mp4 --cell 6 --palette-size 16 --dither bayer4
vid2-8bit convert in.mp4 -o out.mp4 --preset tetris-movie --tier quality

vid2-8bit preview in.mp4 --frame 120          # GUI, live sliders (M5)
vid2-8bit contact-sheet in.mp4 --frame 120 \
    --presets nes,gameboy,snes,tetris-movie   # A/B comparison grid
```

### Encoding gotcha

`yuv420p` chroma subsampling smears hard pixel edges — the exact thing this tool produces. Output
defaults to `yuv444p`. If a target platform demands 4:2:0, upscale enough that subsampling lands
sub-cell.

---

## Development

```bash
python -m pytest                          # 55 unit tests
python tools/make_testclip.py -o test.mp4 # synthetic clip exercising the hard cases
python tools/contact_sheet.py --help      # A/B harness - the main tuning instrument
python tools/derive_preset.py --help      # fit a preset to reference frames
python tools/metrics.py --help            # measure an existing (lossless) render
```

`tools/make_testclip.py` generates a scene built specifically to break naive pixelation: a facade
window grid at the aliasing threshold, long straight architectural edges, a smooth sky gradient, a
camera pan, and an independently moving car. Each targets a different failure mode.

Two quantitative gates guard what is hard to eyeball, both printed by `convert`:

- **Noise** — isolated single-pixel palette outliers per frame. See the caveat above.
- **Temporal churn** — per-pixel palette-index change rate between frames. On a locked-off shot it
  should approach 0.

Both are computed from index maps during rendering. Do **not** measure them by re-reading a lossy
output file: h.264 perturbs pixel values everywhere, which pins churn near 100% regardless of how
stable the render is.

## License

TBD.
