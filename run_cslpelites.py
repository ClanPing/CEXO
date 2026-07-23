#!/usr/bin/env python3
"""Compatibility entry point for the historical runner filename.

The official project entry point is now ``run_cexo.py``. This wrapper remains
so existing scripts, notebooks, and documentation links can continue to run.
"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("run_cexo.py")), run_name="__main__")
