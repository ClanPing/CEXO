#!/usr/bin/env python3
"""Convenience runner for the CEXO Bulleen case study."""

from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    user_args = sys.argv[1:]
    user_selected_facility_mode = any(
        arg in user_args for arg in ("--practical-bulleen", "--sample-bulleen")
    )
    facility_mode = [] if user_selected_facility_mode else ["--practical-bulleen"]
    default_args = [
        *facility_mode,
        "--bulleen-boundary",
        "--bulleen-entrances",
        "--bulleen-roads",
        "--output",
        str(ROOT / "output" / "bulleen"),
    ]

    sys.path.insert(0, str(ROOT))
    sys.argv = [str(ROOT / "run_cexo.py"), *default_args, *user_args]
    runpy.run_path(str(ROOT / "run_cexo.py"), run_name="__main__")


if __name__ == "__main__":
    main()
