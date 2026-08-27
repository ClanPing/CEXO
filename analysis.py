#!/usr/bin/env python3
"""Unified launcher for CEXO manuscript analysis workflows.

Run this file from the repository root, for example:

    python analysis.py --reproducibility
    python analysis.py --scalability
    python analysis.py --sensitivity
    python analysis.py --ablation --facilities 5 --iterations 1000

The individual analysis scripts remain in the analysis/ folder for readability,
while this launcher keeps a stable public command.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent


WORKFLOWS: dict[str, list[tuple[str, list[str]]]] = {
    "ablation": [
        ("main.py", ["--ablation"]),
    ],
    "comparison": [
        ("analysis/comparison/algorithm_comparison.py", []),
    ],
    "reproducibility": [
        ("analysis/reproducibility/reproducibility_analysis.py", []),
        ("analysis/reproducibility/visualize_reproducibility.py", []),
    ],
    "scalability": [
        ("analysis/scalability/run_scalability_analysis.py", []),
        ("analysis/scalability/visualize_scalability_results.py", []),
    ],
    "sensitivity": [
        ("analysis/sensitivity/run_sensitivity_analysis.py", []),
        ("analysis/sensitivity/visualize_sensitivity_results.py", []),
        ("analysis/sensitivity/visualize_all_tradeoff_plots.py", []),
        ("analysis/sensitivity/generate_sensitivity_table.py", []),
    ],
    "figures": [
        ("analysis/reproducibility/visualize_reproducibility.py", []),
        ("analysis/scalability/visualize_scalability_results.py", []),
        ("analysis/sensitivity/visualize_sensitivity_results.py", []),
        ("analysis/sensitivity/visualize_all_tradeoff_plots.py", []),
        ("analysis/sensitivity/generate_sensitivity_table.py", []),
    ],
}

SEED_AWARE_SCRIPTS = {
    "main.py",
    "analysis/comparison/algorithm_comparison.py",
    "analysis/reproducibility/reproducibility_analysis.py",
    "analysis/scalability/run_scalability_analysis.py",
    "analysis/sensitivity/run_sensitivity_analysis.py",
}


def run_script(relative_path: str, script_args: Iterable[str]) -> None:
    """Run one script with repo-root imports and repo-root working directory."""
    script_path = REPO_ROOT / relative_path
    if not script_path.exists():
        raise FileNotFoundError(f"Analysis script not found: {script_path}")

    old_cwd = Path.cwd()
    old_argv = sys.argv[:]
    old_path = sys.path[:]

    try:
        os.chdir(REPO_ROOT)
        sys.path.insert(0, str(REPO_ROOT))
        sys.argv = [str(script_path), *script_args]
        print("\n" + "=" * 80)
        print(f"Running: {relative_path}")
        if script_args:
            print(f"Args: {' '.join(script_args)}")
        print("=" * 80)
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        os.chdir(old_cwd)
        sys.argv = old_argv
        sys.path = old_path


def selected_workflows(args: argparse.Namespace) -> list[str]:
    if args.all:
        return ["ablation", "comparison", "reproducibility", "scalability", "sensitivity"]

    selections = []
    for name in ["ablation", "comparison", "reproducibility", "scalability", "sensitivity", "figures"]:
        if getattr(args, name):
            selections.append(name)
    return selections


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CEXO analysis workflows from one command.",
        epilog=(
            "Extra arguments are forwarded only when a single workflow is selected. "
            "This is most useful for --ablation, which reuses main.py arguments."
        ),
    )
    parser.add_argument("--ablation", action="store_true", help="Run CEXO vs MAP-Elites vs NSGA-II ablation.")
    parser.add_argument("--comparison", action="store_true", help="Run the manuscript algorithm comparison.")
    parser.add_argument("--reproducibility", action="store_true", help="Run reproducibility study and plot.")
    parser.add_argument("--scalability", action="store_true", help="Run scalability study and plot.")
    parser.add_argument("--sensitivity", action="store_true", help="Run sensitivity study and figures/tables.")
    parser.add_argument("--figures", action="store_true", help="Rebuild figures/tables from existing analysis outputs.")
    parser.add_argument("--all", action="store_true", help="Run all major analysis workflows. This can take a long time.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for reproducible analysis runs.")

    args, extra_args = parser.parse_known_args()
    workflows = selected_workflows(args)

    if not workflows:
        parser.print_help()
        return

    if extra_args and len(workflows) > 1:
        parser.error("Extra arguments can only be forwarded when one workflow is selected.")

    for workflow in workflows:
        for script, default_args in WORKFLOWS[workflow]:
            script_args = [*default_args]
            if script in SEED_AWARE_SCRIPTS:
                script_args.extend(["--seed", str(args.seed)])
            if len(workflows) == 1:
                script_args.extend(extra_args)
            run_script(script, script_args)


if __name__ == "__main__":
    main()
