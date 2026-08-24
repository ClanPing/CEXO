#!/usr/bin/env python3
"""Run Bulleen CEXO and regenerate the styled manuscript visual outputs.

This is the end-to-end script path for fresh Bulleen results:
1. run ``main.py`` with the Bulleen practical-case geometry and core settings;
2. analyze the generated layout JSON archive;
3. regenerate selected annotated 2D previews;
4. export standalone Three.js 3D screenshots without starting Streamlit;
5. compose the 2D-to-3D manuscript-style figure.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
from PIL import Image

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from scripts.generate_2d_previews import draw_layout


PROJECT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_DIR / "scripts"
DEFAULT_OUTPUT = PROJECT_DIR / "results" / f"cexo_bulleen_pipeline_{datetime.now():%Y%m%d_%H%M%S}"


def resolve_results_dir(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_DIR / path


def run_command(args: list[str], cwd: Path = PROJECT_DIR) -> None:
    print("\n$ " + " ".join(str(arg) for arg in args), flush=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    result = subprocess.run(args, cwd=cwd, env=env)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def selected_layout_ids(results_dir: Path, limit: int) -> list[str]:
    table_path = results_dir / "paper_analysis" / "table_showcase_layouts.csv"
    if table_path.exists():
        rows = list(csv.DictReader(table_path.open("r", encoding="utf-8")))
        ids = [row["layout_id"] for row in rows if row.get("layout_id")]
        if ids:
            return ids[:limit]

    return [path.stem for path in sorted(results_dir.glob("cslpelite_layout_*.json"))[:limit]]


def showcase_layout_ids(results_dir: Path, limit: int) -> list[str]:
    table_path = results_dir / "paper_analysis" / "table_showcase_layouts.csv"
    ids: list[str] = []
    if table_path.exists():
        rows = list(csv.DictReader(table_path.open("r", encoding="utf-8")))
        for row in rows:
            layout_id = row.get("layout_id")
            if layout_id and layout_id not in ids:
                ids.append(layout_id)
            if len(ids) >= limit:
                return ids

    for path in sorted(results_dir.glob("cslpelite_layout_*.json")):
        if path.stem not in ids:
            ids.append(path.stem)
        if len(ids) >= limit:
            break
    return ids


def generate_selected_2d_previews(results_dir: Path, layout_ids: list[str], dpi: int) -> None:
    for layout_id in layout_ids:
        layout_path = results_dir / f"{layout_id}.json"
        if not layout_path.exists():
            print(f"Skipping missing layout JSON: {layout_path}")
            continue

        layout = json.loads(layout_path.read_text(encoding="utf-8"))
        output_path = results_dir / f"{layout_id}.png"
        draw_layout(layout, output_path, dpi=dpi)
        print(f"Generated styled 2D preview: {output_path}")


def generate_styled_main_layouts(results_dir: Path, best_layout_id: str, diverse_layout_ids: list[str], dpi: int) -> None:
    best_json = results_dir / f"{best_layout_id}.json"
    if best_json.exists():
        best_layout = json.loads(best_json.read_text(encoding="utf-8"))
        draw_layout(best_layout, results_dir / "best_layout.png", dpi=dpi)
        print(f"Replaced best_layout.png with styled Bulleen preview from {best_layout_id}")

    if not diverse_layout_ids:
        return

    temp_dir = results_dir / "paper_analysis" / "_styled_diverse_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        preview_paths: list[Path] = []
        for layout_id in diverse_layout_ids:
            layout_json = results_dir / f"{layout_id}.json"
            if not layout_json.exists():
                continue
            layout = json.loads(layout_json.read_text(encoding="utf-8"))
            preview_path = temp_dir / f"{layout_id}.png"
            draw_layout(layout, preview_path, dpi=dpi)
            preview_paths.append(preview_path)

        if not preview_paths:
            return

        cols = min(3, len(preview_paths))
        rows = (len(preview_paths) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows))
        if rows * cols == 1:
            axes_list = [axes]
        else:
            axes_list = list(axes.flatten())

        for ax, preview_path in zip(axes_list, preview_paths):
            image = Image.open(preview_path)
            ax.imshow(image)
            ax.axis("off")
        for ax in axes_list[len(preview_paths):]:
            ax.axis("off")

        fig.suptitle("Representative Bulleen CEXO Layouts", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(results_dir / "diverse_layouts.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Replaced diverse_layouts.png with {len(preview_paths)} styled Bulleen previews")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def run_optimizer(args: argparse.Namespace, results_dir: Path) -> None:
    command = [
        sys.executable,
        "-X",
        "utf8",
        "main.py",
        "--practical-bulleen",
        "--bulleen-boundary",
        "--bulleen-entrances",
        "--bulleen-roads",
        "--iterations",
        str(args.iterations),
        "--initial-pop",
        str(args.initial_pop),
        "--pretrain",
        str(args.pretrain),
        "--train-freq",
        str(args.train_freq),
        "--latent-dim",
        str(args.latent_dim),
        "--site-width-m",
        str(args.site_width_m),
        "--site-length-m",
        str(args.site_length_m),
        "--seed",
        str(args.seed),
        "--output",
        str(results_dir),
    ]
    run_command(command)


def run_visual_pipeline(args: argparse.Namespace, results_dir: Path) -> None:
    run_command([sys.executable, str(SCRIPTS_DIR / "analyse_cexo_results.py"), "--results-dir", str(results_dir)])

    if not args.skip_case_figure:
        run_command([sys.executable, str(SCRIPTS_DIR / "generate_academic_case_results_figure.py"), "--results-dir", str(results_dir)])

    if not args.skip_gallery:
        gallery_command = [sys.executable, str(SCRIPTS_DIR / "generate_layout_gallery_figure.py"), "--results-dir", str(results_dir)]
        if args.plain_gallery:
            gallery_command.append("--plain")
        run_command(gallery_command)

    layout_ids = selected_layout_ids(results_dir, args.visual_limit)
    generate_selected_2d_previews(results_dir, layout_ids, dpi=args.preview_dpi)
    diverse_ids = showcase_layout_ids(results_dir, args.diverse_limit)
    best_id = selected_layout_ids(results_dir, 1)[0]
    generate_styled_main_layouts(results_dir, best_id, diverse_ids, dpi=args.preview_dpi)

    run_command(
        [
            sys.executable,
            str(SCRIPTS_DIR / "export_streamlit_3d_screenshots.py"),
            "--results-dir",
            str(results_dir),
            "--limit",
            str(args.visual_limit),
            "--camera",
            args.camera,
        ]
    )
    run_command(
        [
            sys.executable,
            str(SCRIPTS_DIR / "generate_streamlit_2d_to_3d_figure.py"),
            "--results-dir",
            str(results_dir),
            "--limit",
            str(args.visual_limit),
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Fresh results directory.")
    parser.add_argument("--skip-optimizer", action="store_true", help="Only regenerate visual outputs for an existing results directory.")
    parser.add_argument("--iterations", type=int, default=15000)
    parser.add_argument("--initial-pop", type=int, default=500)
    parser.add_argument("--pretrain", type=int, default=0)
    parser.add_argument("--train-freq", type=int, default=1000)
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--site-width-m", type=float, default=300.0)
    parser.add_argument("--site-length-m", type=float, default=250.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--visual-limit", type=int, default=3)
    parser.add_argument("--diverse-limit", type=int, default=9)
    parser.add_argument("--preview-dpi", type=int, default=110)
    parser.add_argument("--camera", choices=["reset", "low", "top"], default="low")
    parser.add_argument("--skip-case-figure", action="store_true")
    parser.add_argument("--skip-gallery", action="store_true")
    parser.add_argument("--plain-gallery", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = resolve_results_dir(args.output)
    results_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_optimizer:
        run_optimizer(args, results_dir)

    run_visual_pipeline(args, results_dir)

    print("\nPipeline complete.")
    print(f"Results directory: {results_dir}")
    print(f"2D-to-3D figure: {results_dir / 'paper_analysis' / 'figure_2d_to_3d_streamlit.png'}")
    print(f"3D screenshots: {results_dir / 'paper_analysis' / 'streamlit_3d_exports'}")


if __name__ == "__main__":
    main()
