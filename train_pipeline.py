"""Workspace-level compatibility launcher for the training CLI."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    """Run the maintained training entry point from the `app` directory."""

    app_root = Path(__file__).resolve().parent / "app"
    entry_point = app_root / "train_pipeline.py"
    if not entry_point.is_file():
        raise FileNotFoundError(f"Training entry point not found: {entry_point}")

    sys.path.insert(0, str(app_root))
    runpy.run_path(str(entry_point), run_name="__main__")


if __name__ == "__main__":
    main()
