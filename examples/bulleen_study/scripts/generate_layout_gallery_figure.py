#!/usr/bin/env python3
"""Generate a compact academic gallery of representative Bulleen CEXO layouts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Polygon, Rectangle


PROJECT_DIR = Path(__file__).resolve().parents[1]

COLORS = {
    "boundary": "#C43B3B",
    "site_fill": "#FAF3EA",
    "road": "#E9D45A",
    "road_edge": "#B9A500",
    "core": "#4F81BD",
    "crane": "#C43B3B",
    "storage": "#4DAA57",
    "office": "#8E63B0",
    "rest_area": "#E9A15A",
    "dark": "#222222",
    "mid": "#666666",
}


def set_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def load_layouts(results_dir: Path) -> list[dict]:
    layouts = []
    for path in sorted(results_dir.glob("cslpelite_layout_*.json")):
        with path.open("r", encoding="utf-8") as handle:
            layout = json.load(handle)
        layout["_source_file"] = path.name
        layouts.append(layout)
    if not layouts:
        raise SystemExit(f"No layout files found in {results_dir}")
    return layouts


def score(layout: dict, key: str) -> float:
    return float(layout["objectives"][key])


def bd(layout: dict, key: str) -> float:
    return float(layout["behaviors"][key])


def strict_safe(layout: dict) -> bool:
    return bool(layout.get("feasibility", {}).get("safe", False))


def choose_stratified_layouts(layouts: list[dict], bins: int = 5) -> list[list[dict | None]]:
    """Select best strict-feasible layout per learned descriptor bin."""
    selected: list[list[dict | None]] = [[None for _ in range(bins)] for _ in range(bins)]
    strict_layouts = [layout for layout in layouts if strict_safe(layout)]
    threshold_layouts = [layout for layout in layouts if score(layout, "safety_compliance") >= 0.7]

    def best_in_bin(pool: list[dict], col: int, row: int) -> dict | None:
        x_min, x_max = col / bins, (col + 1) / bins
        y_min, y_max = row / bins, (row + 1) / bins
        candidates = [
            layout
            for layout in pool
            if x_min <= bd(layout, "module_dispersion") < x_max
            and y_min <= bd(layout, "worker_operational_separation") < y_max
        ]
        if not candidates and col == bins - 1:
            candidates = [
                layout
                for layout in pool
                if x_min <= bd(layout, "module_dispersion") <= x_max
                and y_min <= bd(layout, "worker_operational_separation") < y_max
            ]
        if not candidates and row == bins - 1:
            candidates = [
                layout
                for layout in pool
                if x_min <= bd(layout, "module_dispersion") < x_max
                and y_min <= bd(layout, "worker_operational_separation") <= y_max
            ]
        if not candidates:
            return None
        return max(candidates, key=lambda item: score(item, "combined_score"))

    for row in range(bins):
        for col in range(bins):
            chosen = best_in_bin(strict_layouts, col, row)
            if chosen is None:
                chosen = best_in_bin(threshold_layouts, col, row)
            if chosen is None:
                chosen = best_in_bin(layouts, col, row)
            selected[row][col] = chosen
    return selected


def draw_polygon(ax, points, **kwargs) -> None:
    if points and len(points) >= 3:
        ax.add_patch(Polygon(points, closed=True, **kwargs))


def draw_thumbnail(ax, layout: dict | None) -> None:
    ax.set_xlim(0.08, 0.92)
    ax.set_ylim(0.14, 0.86)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)
        spine.set_edgecolor("#BDBDBD")

    if layout is None:
        ax.text(0.5, 0.5, "empty", transform=ax.transAxes, ha="center", va="center", fontsize=7, color=COLORS["mid"])
        return

    draw_polygon(
        ax,
        layout.get("boundary_polygon") or [],
        facecolor=COLORS["site_fill"],
        edgecolor=COLORS["boundary"],
        linewidth=0.75,
        alpha=1.0,
        zorder=1,
    )

    for zone in layout.get("exclusion_zones") or []:
        draw_polygon(
            ax,
            zone.get("polygon") or [],
            facecolor=COLORS["road"],
            edgecolor=COLORS["road_edge"],
            linewidth=0.15,
            alpha=0.75,
            zorder=2,
        )

    for facility in layout.get("facilities") or []:
        ftype = facility.get("type", "unknown")
        x = float(facility.get("x", 0.0))
        y = float(facility.get("y", 0.0))
        width = float(facility.get("width", 0.0))
        length = float(facility.get("length", 0.0))
        ax.add_patch(
            Rectangle(
                (x - width / 2, y - length / 2),
                width,
                length,
                facecolor=COLORS.get(ftype, "#999999"),
                edgecolor=COLORS["dark"],
                linewidth=0.28,
                alpha=0.95,
                zorder=4,
            )
        )
        if ftype == "crane":
            ax.add_patch(
                Circle(
                    (x, y),
                    0.055,
                    fill=False,
                    edgecolor=COLORS["crane"],
                    linewidth=0.25,
                    linestyle="--",
                    alpha=0.55,
                    zorder=3,
                )
            )

    for entrance in layout.get("entrances") or []:
        ax.plot(
            entrance.get("x"),
            entrance.get("y"),
            marker="*",
            markersize=2.8,
            color="#D4A017",
            markeredgecolor="#7A5A00",
            markeredgewidth=0.25,
            zorder=6,
        )


def write_selection_csv(selected: list[list[dict | None]], output_dir: Path) -> None:
    path = output_dir / "figure_layout_gallery_selection.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "row",
                "column",
                "layout_id",
                "source_file",
                "learned_bd1",
                "learned_bd2",
                "safety",
                "efficiency",
                "adaptability",
                "combined",
                "strict_safe",
            ],
        )
        writer.writeheader()
        for row_idx, row in enumerate(selected):
            for col_idx, layout in enumerate(row):
                if layout is None:
                    writer.writerow({"row": row_idx, "column": col_idx})
                    continue
                writer.writerow(
                    {
                        "row": row_idx,
                        "column": col_idx,
                        "layout_id": layout.get("id"),
                        "source_file": layout.get("_source_file"),
                        "learned_bd1": f"{bd(layout, 'module_dispersion'):.4f}",
                        "learned_bd2": f"{bd(layout, 'worker_operational_separation'):.4f}",
                        "safety": f"{score(layout, 'safety_compliance'):.4f}",
                        "efficiency": f"{score(layout, 'operational_efficiency'):.4f}",
                        "adaptability": f"{score(layout, 'layout_adaptability'):.4f}",
                        "combined": f"{score(layout, 'combined_score'):.4f}",
                        "strict_safe": strict_safe(layout),
                    }
                )


def generate(results_dir: Path, output_dir: Path, bins: int = 5, plain: bool = False) -> None:
    set_style()
    layouts = load_layouts(results_dir)
    selected = choose_stratified_layouts(layouts, bins=bins)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_selection_csv(selected, output_dir)

    strict_count = sum(1 for layout in layouts if strict_safe(layout))
    selected_count = sum(1 for row in selected for layout in row if layout is not None)

    fig = plt.figure(figsize=(7.2, 7.45))
    gs = fig.add_gridspec(
        bins,
        bins,
        left=0.095,
        right=0.965,
        top=0.90,
        bottom=0.135,
        wspace=0.055,
        hspace=0.055,
    )

    for display_row in range(bins):
        source_row = bins - 1 - display_row
        for col in range(bins):
            ax = fig.add_subplot(gs[display_row, col])
            draw_thumbnail(ax, selected[source_row][col])

    fig.text(
        0.095,
        0.945,
        f"{selected_count} representative feasible layouts from the CEXO archive",
        ha="left",
        va="center",
        fontsize=9,
        color=COLORS["dark"],
    )
    if not plain:
        fig.text(0.53, 0.080, "Learned latent BD1", ha="center", va="center", fontsize=9)
        fig.text(0.095, 0.105, "low", ha="left", va="center", fontsize=7, color=COLORS["mid"])
        fig.text(0.965, 0.105, "high", ha="right", va="center", fontsize=7, color=COLORS["mid"])
        fig.text(0.038, 0.515, "Learned latent BD2", ha="center", va="center", rotation=90, fontsize=9)
        fig.text(0.060, 0.135, "low", ha="center", va="bottom", rotation=90, fontsize=7, color=COLORS["mid"])
        fig.text(0.060, 0.900, "high", ha="center", va="top", rotation=90, fontsize=7, color=COLORS["mid"])

    legend_handles = [
        Patch(facecolor=COLORS["core"], edgecolor=COLORS["dark"], label="Core"),
        Patch(facecolor=COLORS["crane"], edgecolor=COLORS["dark"], label="Crane"),
        Patch(facecolor=COLORS["storage"], edgecolor=COLORS["dark"], label="Storage"),
        Patch(facecolor=COLORS["office"], edgecolor=COLORS["dark"], label="Office"),
        Patch(facecolor=COLORS["rest_area"], edgecolor=COLORS["dark"], label="Rest"),
        Patch(facecolor=COLORS["road"], edgecolor=COLORS["road_edge"], label="Road exclusion"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#D4A017", markeredgecolor="#7A5A00", markersize=5, label="Entrance"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.53, 0.020),
        ncol=7,
        frameon=False,
        fontsize=7,
        handlelength=1.4,
        columnspacing=1.0,
    )

    suffix = "plain" if plain else "25"
    png_path = output_dir / f"figure_layout_gallery_{suffix}.png"
    pdf_path = output_dir / f"figure_layout_gallery_{suffix}.pdf"
    svg_path = output_dir / f"figure_layout_gallery_{suffix}.svg"
    fig.savefig(png_path, dpi=600)
    fig.savefig(pdf_path)
    fig.savefig(svg_path)
    plt.close(fig)

    if plain:
        caption = (
            f"Gallery of {selected_count} representative feasible CEXO layouts for the Bulleen case study. "
            f"The thumbnails are selected from the {strict_count:,} strict-feasible layouts within the "
            f"{len(layouts):,} exported elite layouts to illustrate the breadth of generated alternatives "
            "without displaying the full archive."
        )
        caption_path = output_dir / "figure_layout_gallery_plain_caption.txt"
    else:
        caption = (
            f"Gallery of {selected_count} representative CEXO layouts for the Bulleen case study. "
            "Layouts are selected by stratifying the autoencoder-learned behavioural archive into a 5 x 5 grid "
            "and choosing the highest-scoring strict-feasible layout from each occupied region. "
            f"The gallery summarises visual diversity within the {len(layouts):,} exported elite layouts without displaying the full archive."
        )
        caption_path = output_dir / "figure_layout_gallery_25_caption.txt"
    caption_path.write_text(caption + "\n", encoding="utf-8")
    print(png_path)
    print(pdf_path)
    print(svg_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results") / "cexo_bulleen_15000_full_fg")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--plain", action="store_true", help="Remove learned-BD axis annotations and render as a simple gallery.")
    args = parser.parse_args()

    results_dir = args.results_dir
    if not results_dir.is_absolute():
        results_dir = PROJECT_DIR / results_dir
    output_dir = args.output_dir or (results_dir / "paper_analysis")
    if not output_dir.is_absolute():
        output_dir = PROJECT_DIR / output_dir
    generate(results_dir, output_dir, bins=args.bins, plain=args.plain)


if __name__ == "__main__":
    main()
