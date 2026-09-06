# vid2-8bit

Convert real footage — cinematic shots, buildings, cars — into **pixel art video** in the style of
the *Tetris* (2023) pixel sequences. Palette size, pixel/cell size and bit depth are all
configurable.

Not a "pixelate" filter. See [Why naive pixelation fails](#why-naive-pixelation-fails).

---

## Status

Early. `M0` scaffolding in progress. See [`docs/PLAN.md`](docs/PLAN.md) for the full design and
build order.

| Milestone | Scope | State |
|---|---|---|
| M0 | ffmpeg IO, config/presets, naive pipeline, A/B contact-sheet harness | 🚧 in progress |
| M1 | Pre-abstraction + Oklab per-shot palette | ☐ |
| M2 | Temporal stabilization | ☐ |
| M3 | Structure extraction + outline expansion | ☐ |
| M4 | Presets, tile constraints, dithering, bit-depth snap | ☐ |
| M5 | Preview GUI | ☐ |
| M6 | Neural assists, GLSL fast tier, superpixel quality tier | ☐ |

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

> **Framing:** the *Tetris* sequences were substantially hand-authored and rotoscoped by artists,
> not made by an automatic filter. This targets the closest automatable approximation.

---

## Architecture

Seven stages, all scoped **per shot** so palettes and temporal state reset at cuts. All averaging
happens in **linear light**; all perceptual distance is measured in **Oklab**.

| Stage | Module | Purpose |
|---|---|---|
| 0 Ingest | `io/decode.py`, `io/shots.py` | ffmpeg raw pipe → linear RGB; scene detection |
| 1 Abstraction | `stages/abstract.py` | Structure–texture separation, Nyquist-matched edge-preserving low-pass, luminance flattening |
| 2 Structure | `stages/structure.py` | XDoG lines, LSD grid snapping, depth-gated silhouette outlines |
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
pip install -e ".[gui]"     # moderngl + glfw + imgui - preview GUI (M5)
pip install -e ".[fast]"    # numba - JIT for palette assignment / tile solve
```

The neural assist modules are deliberately **not** installed into this environment. See below.

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

## Usage (target CLI — M0)

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
python -m pytest                       # unit tests
python tools/contact_sheet.py --help   # A/B harness - the main tuning instrument
python tools/metrics.py --help         # noise + temporal-churn metrics
```

Two quantitative gates guard what is hard to eyeball:

- **Noise metric** — count of isolated single-pixel palette outliers per frame. Should trend to ~0
  on flat regions as Stage 1 improves.
- **Temporal churn** — mean per-pixel palette-index change rate between frames. On a locked-off
  shot it should approach 0; on a pan it should show clean stepped motion, not per-pixel churn.

## License

TBD.
