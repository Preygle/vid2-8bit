# References

Drop *Tetris* (2023) pixel-sequence frames here as `tetris/*.png`.

They are used by `tools/derive_preset.py` to fit the `tetris-movie` preset:
palette size and colors, cell size, outline weight/darkness, and the number of
shading bands per material region.

Image files in this directory are **gitignored** — they are copyrighted film
frames kept locally for style analysis. Only the derived preset YAML (which
contains numbers, not imagery) is committed.
