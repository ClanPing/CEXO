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
import textwrap
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
VISUAL_LIMIT = 3
DIVERSE_LIMIT = 9
PREVIEW_DPI = 110
CAMERA = "low"


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
    if args.export_count is not None:
        command.extend(["--export-count", str(args.export_count)])
    if args.export_all:
        command.append("--export-all")
    if args.no_export_pngs:
        command.append("--no-export-pngs")
    if args.export_unsafe:
        command.append("--export-unsafe")
    run_command(command)


def run_visual_pipeline(args: argparse.Namespace, results_dir: Path) -> None:
    run_command([sys.executable, str(SCRIPTS_DIR / "analyse_cexo_results.py"), "--results-dir", str(results_dir)])

    run_command([sys.executable, str(SCRIPTS_DIR / "generate_academic_case_results_figure.py"), "--results-dir", str(results_dir)])

    run_command([
        sys.executable,
        str(SCRIPTS_DIR / "generate_layout_gallery_figure.py"),
        "--results-dir",
        str(results_dir),
        "--plain",
    ])

    layout_ids = selected_layout_ids(results_dir, VISUAL_LIMIT)
    generate_selected_2d_previews(results_dir, layout_ids, dpi=PREVIEW_DPI)
    diverse_ids = showcase_layout_ids(results_dir, DIVERSE_LIMIT)
    best_id = selected_layout_ids(results_dir, 1)[0]
    generate_styled_main_layouts(results_dir, best_id, diverse_ids, dpi=PREVIEW_DPI)

    run_command(
        [
            sys.executable,
            str(SCRIPTS_DIR / "export_streamlit_3d_screenshots.py"),
            "--results-dir",
            str(results_dir),
            "--limit",
            str(VISUAL_LIMIT),
            "--camera",
            CAMERA,
        ]
    )
    run_command(
        [
            sys.executable,
            str(SCRIPTS_DIR / "generate_streamlit_2d_to_3d_figure.py"),
            "--results-dir",
            str(results_dir),
            "--limit",
            str(VISUAL_LIMIT),
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(
            """\
            Run the Bulleen CEXO case-study pipeline.

            The pipeline runs the Bulleen-specific CEXO optimiser, then prepares the
            case-study visual outputs from the exported layout JSON files.
            """
        ),
        epilog=textwrap.dedent(
            """\
            Examples:
              python run_bulleen_cexo_pipeline.py
              python run_bulleen_cexo_pipeline.py --output results\\cexo_bulleen_quick_review --iterations 1000 --initial-pop 100 --export-count 50
              python run_bulleen_cexo_pipeline.py --skip-optimizer --output results\\cexo_bulleen_15000_full_fg
            """
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    general = parser.add_argument_group("general")
    general.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        metavar="DIR",
        help="Output folder for the Bulleen run. Defaults to a timestamped folder under results/.",
    )
    general.add_argument(
        "--skip-optimizer",
        action="store_true",
        help="Skip the long optimisation step and regenerate figures from an existing --output folder.",
    )

    optimisation = parser.add_argument_group("optimisation")
    optimisation.add_argument(
        "--iterations",
        type=int,
        default=15000,
        metavar="N",
        help="Number of CEXO optimisation iterations. Lower this for quick review runs.",
    )
    optimisation.add_argument(
        "--initial-pop",
        type=int,
        default=500,
        metavar="N",
        help="Number of initial layouts used to seed the behavioural archive.",
    )
    optimisation.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="N",
        help="Random seed used for reproducible Bulleen optimisation runs.",
    )

    export = parser.add_argument_group("layout export")
    export.add_argument(
        "--export-count",
        type=int,
        default=None,
        metavar="N",
        help="Export only the top N layout JSON/PNG pairs. Omit this to export all safety-threshold layouts.",
    )
    export.add_argument("--export-all", action="store_true", help=argparse.SUPPRESS)
    export.add_argument(
        "--no-export-pngs",
        action="store_true",
        help="Export layout JSON files only, without matching individual PNG previews.",
    )
    export.add_argument(
        "--export-unsafe",
        action="store_true",
        help="Also export layouts below the safety threshold. Default export keeps safety-threshold layouts only.",
    )

    advanced = parser.add_argument_group("advanced")
    advanced.add_argument(
        "--pretrain",
        type=int,
        default=0,
        metavar="N",
        help="Iterations before the first autoencoder descriptor training step.",
    )
    advanced.add_argument(
        "--train-freq",
        type=int,
        default=1000,
        metavar="N",
        help="Autoencoder retraining interval during optimisation.",
    )
    advanced.add_argument(
        "--latent-dim",
        type=int,
        default=2,
        metavar="N",
        help="Latent dimension used for learned behavioural descriptors.",
    )
    advanced.add_argument(
        "--site-width-m",
        type=float,
        default=300.0,
        metavar="M",
        help="Bulleen site width used when exporting scaled layout JSON and 3D visuals.",
    )
    advanced.add_argument(
        "--site-length-m",
        type=float,
        default=250.0,
        metavar="M",
        help="Bulleen site length used when exporting scaled layout JSON and 3D visuals.",
    )
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
