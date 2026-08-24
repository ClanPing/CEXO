#!/usr/bin/env python3
"""Combine exported 2D layouts and Streamlit/Three.js 3D screenshots."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageChops
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_DIR / "results" / "cexo_bulleen_15000_full_fg"


def trim(image: Image.Image, pad: int = 8) -> Image.Image:
    rgb = image.convert("RGB")
    bg = Image.new("RGB", rgb.size, rgb.getpixel((0, 0)))
    diff = ImageChops.difference(rgb, bg)
    bbox = diff.getbbox()
    if bbox is None:
        return rgb
    left, top, right, bottom = bbox
    return rgb.crop((max(0, left - pad), max(0, top - pad), min(rgb.width, right + pad), min(rgb.height, bottom + pad)))


def load_showcase_rows(results_dir: Path, limit: int) -> list[dict]:
    table = results_dir / "paper_analysis" / "table_showcase_layouts.csv"
    rows = list(csv.DictReader(table.open("r", encoding="utf-8")))
    return rows[:limit]


def screenshot_path(results_dir: Path, index: int, row: dict) -> Path:
    role = row["showcase_role"]
    layout_id = row["layout_id"]
    safe_role = "".join(ch.lower() if ch.isalnum() else "_" for ch in role).strip("_")
    return results_dir / "paper_analysis" / "streamlit_3d_exports" / f"{index:02d}_{safe_role}_{layout_id}_clean.png"


def crop_3d(image: Image.Image) -> Image.Image:
    # Remove excess sky/ground margin from the headless viewer capture while
    # preserving the same camera rendering.
    w, h = image.size
    return image.crop((0, int(h * 0.22), w, int(h * 0.95)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    output_dir = results_dir / "paper_analysis"
    rows = load_showcase_rows(results_dir, args.limit)

    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 450,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )

    fig, axes = plt.subplots(len(rows), 2, figsize=(7.4, 2.25 * len(rows)))
    if len(rows) == 1:
        axes = [axes]

    for idx, row in enumerate(rows, start=1):
        ax2d, ax3d = axes[idx - 1]
        image_2d = trim(Image.open(results_dir / f"{row['layout_id']}.png"), pad=14)
        image_3d = crop_3d(Image.open(screenshot_path(results_dir, idx, row)))

        ax2d.imshow(image_2d)
        ax3d.imshow(image_3d)
        for ax in (ax2d, ax3d):
            ax.axis("off")

        if idx == 1:
            ax2d.text(0.5, 1.02, "(a) 2D optimized layout", transform=ax2d.transAxes,
                      ha="center", va="bottom", fontsize=8.5, fontweight="bold")
            ax3d.text(0.5, 1.02, "(b) 3D Streamlit transformation", transform=ax3d.transAxes,
                      ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    fig.subplots_adjust(wspace=0.02, hspace=0.10)

    stem = output_dir / "figure_2d_to_3d_streamlit"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{stem}.{ext}")
    plt.close(fig)

    caption = (
        "Figure X. Two-dimensional to three-dimensional transformation of selected Bulleen CEXO layouts. "
        "Each row shows an optimized 2D layout exported by CEXO and the corresponding Streamlit/Three.js "
        "3D site model generated from the same layout JSON. The transformation preserves facility positions, "
        "footprint dimensions, road-exclusion corridors, entrances, and crane operating zones while adding "
        "facility-type-specific 3D geometry for visual inspection."
    )
    (output_dir / "figure_2d_to_3d_streamlit_caption.txt").write_text(caption + "\n", encoding="utf-8")
    print(f"Saved {stem}.png/.pdf/.svg")


if __name__ == "__main__":
    main()
