"""Thin wrapper around `vid2-8bit contact-sheet`, for running from the repo."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from vid2_8bit.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["contact-sheet", *sys.argv[1:]]))
