# Per-input tuned presets

One preset per source, tuned individually rather than for generality. Use with:

```bash
vid2-8bit convert in/test1.mp4 -o out.mp4 --config presets-tuned/test1.yaml --fps 12
vid2-8bit frame  in/real1.png  -o out.png --config presets-tuned/real1.yaml
```

Each `extends: tetris-movie` and overrides only what that source needs.

## What the sweep found

Six candidate styles were rendered for every input. The winner almost
everywhere was **32 colours with contrast around 0.5**, not the 16-colour /
contrast-0.9 combination in the shared `tetris-movie` preset — that is too
aggressive for most footage and crushes mid-tones.

| Preset | Source | The adjustment that mattered |
|---|---|---|
| `real1` | night exterior | contrast 0.65 to hold the dark, 32 colours to keep facade structure |
| `real2` | daylight close-up | target_chroma 0.050 so the car commits to orange instead of drifting to cream |
| `test1` | daylight cityscape | baseline 32/0.5 |
| `test2` | aerial landscape | coarser grid (160) and despeckle 0.14 to calm the rock texture |
| `test3` | vertical water | much coarser (110 wide, min_cell 6) — chaotic high-frequency texture needs a big cell or it reads as noise |
| `shot1-3` | already pixel art | barely touch it: 48 colours, contrast 0.15, levels_strength 0.3, fine grid |

## A note on the screenshots

`shot1`, `shot2` and `shot3` are frames of *hand-drawn* pixel art from the film,
not rotoscoped footage. They are already the target style, so the job is to pass
them through intact rather than re-abstract them — hence the very light
settings. They are useful as a fidelity check: if the pipeline mangles input
that is already correct, it is over-processing.
