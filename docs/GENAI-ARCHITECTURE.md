# Generative architecture: Unwrap → Author → Replay

A design for producing **asset-ready** pixel art video. Supersedes the
`gen distill` / `gen render` approach in `src/vid2_8bit/gen/`, which stays as a
useful stopgap but is not the destination.

Video is the priority. Images fall out as the single-frame case.

---

## 1. Why the current generative plan is a dead end

The plan was: pixel-art LoRA + SDXL img2img, either per frame (`render`) or on
keyframes to distil a preset (`distill`). Both are wrong, for reasons this
project has already measured rather than guessed.

**Diffusion cannot emit asset-ready pixels.** It outputs continuous RGB with
soft edges. "Pixel art" from SDXL is a *picture of* pixel art: blocks with
anti-aliased boundaries, thousands of colours, no exact grid. Quantizing it
afterwards reintroduces every failure catalogued in the README. The constraint
has to be inside the solver, not applied after it.

**Per-frame generation destroys the one thing that took real work.** Index
hysteresis got temporal churn to 0.01% on a locked-off shot. Diffusion re-rolls
the image every frame; two adjacent frames of a static shot come back different.
No amount of post-hoc flow warping recovers that.

**Distillation only transfers statistics.** It measures palette, contrast and
chroma from generated keyframes and hands them to the deterministic renderer.
That is real but shallow: it cannot fix what three independent blind critics
converged on. Quoting the sharpest of them:

> it is a 9-pixel mosaic filter over a dark video frame with a colour-count cap,
> not pixel art. The tell is not the noise — it is that there is no drawing
> underneath the blocks to look at.

And concretely: **5.1% of our cells sit in horizontal runs of ≥10, against 27%
in the film frame.** Long architectural lines do not survive. Our grid's edge
autocorrelation is perfectly isotropic (0.92 / 0.84) — the signature of
downsampling — where the film's is anisotropic (0.47 / 0.80) because horizontal
floor courses drive the structure, not the block size.

A LoRA does not fix any of that. The gap is not style, it is that **we sample
where an artist authors**.

---

## 2. What "asset-ready" actually demands

These are hard constraints, and they are what rules out diffusion output
directly:

| Requirement | Consequence |
|---|---|
| Indexed colour, not RGB | Output is a palette + an index map. Ship indexed PNG (PLTE), openable in Aseprite. |
| Exact integer grid, zero anti-aliasing | Every logical pixel is one palette index. No blending anywhere in the chain. |
| One palette for the whole shot | A sprite sheet with a per-frame palette is not a sprite sheet. |
| Structure is drawn, not sampled | Lines are runs. Regions are flat fills. Repeats are lattices. |
| Temporally locked | An unchanged part of the scene must be *byte-identical* frame to frame. |
| Editable | The artist gets layers, sprites and a palette — not a flattened video. |

The last two are why this is a video architecture and not an image one.

---

## 3. The architecture

Three stages. The load-bearing idea is in the names: the pixel art is
**authored once, then replayed**, rather than computed per frame.

```
    ┌─────────────┐     ┌──────────────┐     ┌────────────┐
    │  1 UNWRAP   │────▶│   2 AUTHOR   │────▶│  3 REPLAY  │
    │  (neural)   │     │  (hybrid)    │     │(determin.) │
    └─────────────┘     └──────────────┘     └────────────┘
     shot → layered      atlas → indexed      atlas + tracks
     canonical atlases   pixel art, once      → every frame
```

### Stage 1 — UNWRAP (neural, once per shot)

Decompose the shot into **canonical spaces**: a flat 2D image per layer in
which every frame's contribution lands at the same place.

- **Background**: a mosaic built by chaining frame-to-frame homographies. For
  drone and pan footage — which is most of the test material — this is exact,
  takes seconds, and needs no training. *Not* a learned atlas: Layered Neural
  Atlases needs hours of per-shot optimisation, which is the wrong trade here.
- **Moving objects**: SAM 2 video mode gives persistent tracks. Each track gets
  its own small canonical sprite space, aligned by its own similarity transform.
- **Depth** (Depth Anything V2) orders the layers and marks true silhouettes.

Output: `N` atlases, a per-frame warp per atlas, an occlusion mask, a depth
order. Frames are now *views* onto atlases.

**This is where the temporal guarantee comes from, and it is structural rather
than a filter.** If the art lives in the atlas and the frame is a window onto
it, an unchanged region cannot flicker — there is nothing to re-roll.

### Stage 2 — AUTHOR (the hybrid core, once per atlas)

Solve the pixel art **once**, on the atlas, in discrete index space. Three
mechanisms, in order of how much they matter.

#### 2a. Lattice regularisation — the fix for the measured failure

Man-made structure is periodic: window grids, floor courses, balconies, railings.
Our sampler *moirés* it, because a 14 px window pitch on a 6 px cell beats. An
artist draws a perfect lattice. So:

1. Detect the 2D lattice per region — period, phase and basis — by Fourier peak
   / autocorrelation analysis on the atlas (which is static, so this is a clean
   measurement, not a per-frame guess).
2. **Snap the period to an integer number of logical cells.** This is the whole
   trick: a 2.33-cell pitch becomes exactly 2, and the beat disappears.
3. Synthesise one small motif (a 3×4 px window) — either by median-averaging
   every lattice cell in the atlas, or by generating it.
4. Stamp the motif on the regularised lattice.

Result: a perfectly regular window grid, no moiré, and it *reads as drawn*
because it was. This directly attacks the run-length deficit — a stamped lattice
row is an unbroken run by construction. I have found no prior art applying
lattice regularisation to pixel-art conversion; this is the piece I would defend
as novel.

#### 2b. Structure as primitives, not samples

Fit the atlas's region boundaries to polygons at logical resolution, snap
near-axis edges to exact rows and columns, then **rasterise with pixel-art
rules**: Bresenham runs, no anti-aliasing, consistent stair patterns, 1 px
outlines only where depth says silhouette.

The output of this stage is a **drawing program** — a list of primitives —
not an image:

```
palette: 16 entries
layers:
  sky      : region(poly=[...], fill=gradient(3→5, bayer4))
  facade_A : region(poly=[...], fill=flat(7), outline=ink 1px)
             lattice(origin=(4,9), period=(2,3), motif=win_a)
  car      : sprite(atlas=car, frames=[...])
```

That representation is what makes the output asset-ready and editable, and it is
what "authored" concretely means.

#### 2c. Discrete solve with a generative prior

For everything the primitives do not cover — organic texture, foliage, cloth —
optimise the index map directly under a style prior. This is
[SD-πXL](https://arxiv.org/pdf/2410.06236)'s score-distillation formulation:
the optimisation variable *is* the palette-index assignment, so the hard
constraints hold by construction rather than being applied afterwards.

Two adaptations:

- It runs on the **atlas**, not on frames. The temporal loss term that would
  otherwise be needed disappears, because there is only one canvas.
- The prior is whichever pixel-art model is available — a Retro-Diffusion-class
  model, or the existing Krea 2 + `krea2_retroanime` LoRA. The prior supplies
  *taste*; the solver supplies *legality*.

Cost is trivial: a 1000×200 atlas at 16 colours is a 3.2 M-float logit tensor.
The expensive part is the neural feature extractors, and those run once.

### Stage 3 — REPLAY (deterministic, per frame)

Per frame, per layer: take the authored atlas, warp by that frame's inverse
transform **snapped to integer logical pixels**, composite by depth order,
resolve disocclusions, write the index map.

No quantization happens here. No palette decisions. No filtering. The frame is a
lookup. Churn is zero wherever the warp is integer-stable — by construction, not
by hysteresis.

---

## 4. Division of labour

The split is the point of the design: **generative models make decisions,
deterministic code places pixels.** A model never emits a final pixel.

| Generative | Deterministic |
|---|---|
| Segmentation and tracking (SAM 2) | Mosaic / homography chaining |
| Depth and silhouettes (Depth Anything V2) | Lattice detection and integer snapping |
| Optical flow (RAFT) | Polygon fitting, Bresenham rasterisation |
| Style prior for the discrete solve | Palette optimisation (existing Oklab work) |
| Motif synthesis (one window, one brick) | Index assignment, hysteresis, despeckle |
| Atlas inpainting for disocclusions | Warping, compositing, encoding |

Every existing deterministic component survives. Tone, palette redistribution,
even spacing, despeckle, median sampling — all of it moves from operating on
frames to operating on atlases, which is strictly better because the atlas is
static and larger.

---

## 5. Output format

```
out/shot_003/
  palette.hex              one palette, whole shot
  atlas_bg.png             indexed PNG, the authored background
  atlas_bg.json            lattices, region polygons, primitive program
  sprites/car.png          indexed sprite sheet
  sprites/car.json         per-frame placement
  frames/%06d.png          indexed PNG per frame
  shot.mp4                 yuv444p, integer upscale
```

Indexed PNG throughout. Aseprite opens all of it.

---

## 6. Honest failure modes

Naming these now, because the architecture should be judged on where it breaks.

- **Non-rigid chaos does not atlas.** `test3` — churning water — has no
  canonical space. Every wave is new. This is a hard fallback to the existing
  per-frame path, and it will still shimmer. No architecture fixes water;
  measured churn there is 63% and that is close to irreducible.
- **Heavy parallax breaks a homography mosaic.** A single plane cannot model a
  dolly past foreground objects. Mitigation is per-layer mosaics from the depth
  split; beyond that, fall back.
- **Lattice regularisation only helps man-made periodic structure.** Useless on
  landscape, decisive on the buildings that are the actual target.
- **Disocclusion needs inpainting**, and inpainted atlas regions are the one
  place a generative model's instability can leak into the output. Constrain it
  to the existing palette.
- **A quality gate is mandatory.** Fit residual per shot decides atlas path vs
  per-frame path. Without it this silently produces garbage on the shots it
  cannot model.

---

## 7. Build order

Each step is independently testable and independently useful. Nothing here needs
a model download to start.

| Step | What | Proves |
|---|---|---|
| 1 | Background mosaic from existing phase-correlation motion; render frames as views onto it | The temporal claim. Churn should collapse on `real1` and `test1`. |
| 2 | Lattice detection + integer snapping + motif stamping on the mosaic | The structure claim. Run-length ≥10 should move from 5% toward 27%. |
| 3 | Quality gate: fit residual → atlas path or per-frame fallback | Safety, before anything neural |
| 4 | SAM 2 tracks → sprite atlases for moving objects | Foreground handling, sprite assets |
| 5 | Indexed-PNG asset export + primitive program JSON | "Asset-ready" becomes literal |
| 6 | Score-distilled discrete solve with a pixel-art prior | Organic regions; the last quality step |

**Step 1 and step 2 are the whole bet.** They need no generative model at all,
they attack the two defects the critics actually measured, and if they do not
move those numbers the rest is not worth building.

---

## 8. What is genuinely new here

Being precise, since the ask was for a unique architecture rather than a
combination of papers:

- **Not new:** score-distilled palette-constrained generation (SD-πXL);
  canonical-space video editing (Layered Neural Atlases); pixel-art diffusion
  models (Retro Diffusion, PixDiff-PIG).
- **New in combination:** authoring pixel art *in* a canonical atlas so that
  temporal coherence is structural rather than filtered, and so the atlas is
  simultaneously the temporal guarantee and the shippable asset.
- **New outright, as far as I can find:** lattice regularisation as a
  pixel-art conversion primitive — detecting periodic man-made structure and
  re-emitting it on an integer-snapped lattice with a synthesised motif, instead
  of sampling and moiréing it.

The second and third are what turn "a mosaic filter with a colour cap" into
something with a drawing underneath.
