#!/usr/bin/env python3
"""Generate a paper figure showing 2D-to-3D layout transformation."""

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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


PROJECT_DIR = Path(__file__).resolve().parents[1]

COLORS = {
    "boundary": "#C43B3B",
    "site_fill": "#F7E8D6",
    "site_3d": "#B9895B",
    "road": "#E9D45A",
    "road_edge": "#A38E00",
    "core": "#4F81BD",
    "crane": "#C43B3B",
    "storage": "#4DAA57",
    "office": "#8E63B0",
    "rest_area": "#E9A15A",
    "dark": "#1F2933",
    "fence": "#6B7280",
    "entrance": "#D4A017",
}

LABELS = {
    "core": "Core",
    "crane": "Crane",
    "storage": "Storage",
    "office": "Office",
    "rest_area": "Rest",
}

HEIGHTS = {
    "core": 0.080,
    "crane": 0.145,
    "storage": 0.030,
    "office": 0.040,
    "rest_area": 0.035,
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


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_showcase_layouts(results_dir: Path, limit: int) -> list[tuple[str, dict]]:
    table_path = results_dir / "paper_analysis" / "table_showcase_layouts.csv"
    rows = []
    if table_path.exists():
        with table_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

    preferred_roles = [
        "Best overall",
        "Best efficiency",
        "Best adaptability",
        "Balanced trade-off",
        "Latent BD2 extreme",
        "Latent BD1 extreme",
    ]

    selected: list[tuple[str, dict]] = []
    for role in preferred_roles:
        match = next((row for row in rows if row.get("showcase_role") == role), None)
        if not match:
            continue
        json_file = match.get("json_file") or f"{match['layout_id']}.json"
        path = results_dir / json_file
        if path.exists():
            selected.append((role, load_json(path)))
        if len(selected) >= limit:
            break

    if selected:
        return selected

    layouts = []
    for path in sorted(results_dir.glob("cslpelite_layout_*.json"))[:limit]:
        layout = load_json(path)
        layouts.append((layout.get("id", path.stem), layout))
    if not layouts:
        raise SystemExit(f"No cslpelite_layout_*.json files found in {results_dir}")
    return layouts


def objective_text(layout: dict) -> str:
    obj = layout["objectives"]
    return (
        f"S {obj['safety_compliance']:.2f} | "
        f"E {obj['operational_efficiency']:.2f} | "
        f"A {obj['layout_adaptability']:.2f}"
    )


def draw_polygon_2d(ax, points, **kwargs) -> None:
    if points and len(points) >= 3:
        ax.add_patch(Polygon(points, closed=True, **kwargs))


def draw_2d(ax, layout: dict) -> None:
    ax.set_xlim(0.08, 0.92)
    ax.set_ylim(0.14, 0.86)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    draw_polygon_2d(
        ax,
        layout.get("boundary_polygon") or [],
        facecolor=COLORS["site_fill"],
        edgecolor=COLORS["boundary"],
        linewidth=1.0,
        zorder=1,
    )

    for zone in layout.get("exclusion_zones") or []:
        draw_polygon_2d(
            ax,
            zone.get("polygon") or [],
            facecolor=COLORS["road"],
            edgecolor=COLORS["road_edge"],
            linewidth=0.2,
            alpha=0.82,
            zorder=2,
        )

    for facility in layout.get("facilities") or []:
        ftype = facility.get("type")
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
                linewidth=0.35,
                alpha=0.96,
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
                    linewidth=0.35,
                    linestyle="--",
                    alpha=0.65,
                    zorder=3,
                )
            )

    for entrance in layout.get("entrances") or []:
        ax.plot(
            entrance.get("x"),
            entrance.get("y"),
            marker="*",
            markersize=4.0,
            color=COLORS["entrance"],
            markeredgecolor="#7A5A00",
            markeredgewidth=0.35,
            zorder=6,
        )


def prism_faces(x: float, y: float, w: float, d: float, h: float):
    x0, x1 = x - w / 2, x + w / 2
    y0, y1 = y - d / 2, y + d / 2
    return [
        [(x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0)],
        [(x0, y0, h), (x1, y0, h), (x1, y1, h), (x0, y1, h)],
        [(x0, y0, 0), (x1, y0, 0), (x1, y0, h), (x0, y0, h)],
        [(x1, y0, 0), (x1, y1, 0), (x1, y1, h), (x1, y0, h)],
        [(x1, y1, 0), (x0, y1, 0), (x0, y1, h), (x1, y1, h)],
        [(x0, y1, 0), (x0, y0, 0), (x0, y0, h), (x0, y1, h)],
    ]


def add_prism(ax, x: float, y: float, w: float, d: float, h: float, color: str, alpha: float = 0.94) -> None:
    collection = Poly3DCollection(
        prism_faces(x, y, w, d, h),
        facecolor=color,
        edgecolor="#263238",
        linewidth=0.25,
        alpha=alpha,
    )
    ax.add_collection3d(collection)


def add_ground_polygon(ax, points, color: str, edge: str, z: float, alpha: float, lw: float) -> None:
    if not points or len(points) < 3:
        return
    verts = [[(float(x), float(y), z) for x, y in points]]
    collection = Poly3DCollection(verts, facecolor=color, edgecolor=edge, linewidth=lw, alpha=alpha)
    ax.add_collection3d(collection)


def add_crane(ax, x: float, y: float, footprint: float, height: float) -> None:
    add_prism(ax, x, y, footprint, footprint, height * 0.72, COLORS["crane"], alpha=0.98)
    jib = footprint * 7.5
    z = height * 0.80
    ax.plot([x - jib * 0.28, x + jib * 0.72], [y, y], [z, z], color=COLORS["crane"], linewidth=2.0)
    ax.plot([x + jib * 0.72, x + jib * 0.72], [y, y], [z, z * 0.55], color="#9B1C1C", linewidth=1.0)
    theta = np.linspace(0, 2 * np.pi, 96)
    radius = 0.055
    ax.plot(x + radius * np.cos(theta), y + radius * np.sin(theta), np.zeros_like(theta) + 0.004,
            color=COLORS["crane"], linewidth=0.5, linestyle="--", alpha=0.65)


def draw_3d(ax, layout: dict) -> None:
    ax.set_xlim(0.08, 0.92)
    ax.set_ylim(0.14, 0.86)
    ax.set_zlim(0.0, 0.18)
    ax.set_box_aspect((0.84, 0.72, 0.26))
    ax.view_init(elev=34, azim=-58)
    ax.set_proj_type("ortho")
    ax.set_axis_off()

    add_ground_polygon(ax, layout.get("boundary_polygon") or [], COLORS["site_3d"], COLORS["boundary"], 0.0, 0.68, 0.8)
    for zone in layout.get("exclusion_zones") or []:
        add_ground_polygon(ax, zone.get("polygon") or [], COLORS["road"], COLORS["road_edge"], 0.002, 0.88, 0.18)

    # Lightweight perimeter fence to make the 3D view read as a site model.
    boundary = layout.get("boundary_polygon") or []
    if len(boundary) > 2:
        xs = [p[0] for p in boundary] + [boundary[0][0]]
        ys = [p[1] for p in boundary] + [boundary[0][1]]
        ax.plot(xs, ys, [0.012] * len(xs), color=COLORS["fence"], linewidth=0.8)

    for facility in layout.get("facilities") or []:
        ftype = facility.get("type")
        x = float(facility.get("x", 0.0))
        y = float(facility.get("y", 0.0))
        width = float(facility.get("width", 0.0))
        length = float(facility.get("length", 0.0))
        if ftype == "crane":
            add_crane(ax, x, y, max(width, length), HEIGHTS["crane"])
        else:
            add_prism(ax, x, y, width, length, HEIGHTS.get(ftype, 0.035), COLORS.get(ftype, "#999999"))

    for entrance in layout.get("entrances") or []:
        x = float(entrance.get("x", 0.0))
        y = float(entrance.get("y", 0.0))
        ax.scatter([x], [y], [0.018], marker="*", s=28, color=COLORS["entrance"], edgecolor="#7A5A00", linewidth=0.35)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results/cexo_bulleen_15000_full_fg"))
    parser.add_argument("--limit", type=int, default=4)
    args = parser.parse_args()

    set_style()
    results_dir = args.results_dir
    if not results_dir.is_absolute():
        results_dir = PROJECT_DIR / results_dir
    results_dir = results_dir.resolve()
    output_dir = results_dir / "paper_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = load_showcase_layouts(results_dir, args.limit)

    fig = plt.figure(figsize=(7.35, 1.62 * len(selected)))
    grid = fig.add_gridspec(len(selected), 2, width_ratios=[1.0, 1.08], wspace=0.00, hspace=0.12)

    for row, (role, layout) in enumerate(selected):
        ax_2d = fig.add_subplot(grid[row, 0])
        ax_3d = fig.add_subplot(grid[row, 1], projection="3d")
        draw_2d(ax_2d, layout)
        draw_3d(ax_3d, layout)
        ax_2d.text(
            -0.115,
            0.5,
            f"{role}\n{objective_text(layout)}",
            transform=ax_2d.transAxes,
            ha="center",
            va="center",
            fontsize=6.7,
            fontweight="bold",
            rotation=90,
        )
        if row == 0:
            ax_2d.text(0.5, 1.02, "(a) 2D optimized layout", transform=ax_2d.transAxes,
                       ha="center", va="bottom", fontsize=8.1, fontweight="bold")
            ax_3d.text2D(0.5, 1.02, "(b) 3D transformed layout", transform=ax_3d.transAxes,
                         ha="center", va="bottom", fontsize=8.1, fontweight="bold")

    legend_handles = [
        Patch(facecolor=COLORS["core"], edgecolor=COLORS["dark"], label="Core"),
        Patch(facecolor=COLORS["crane"], edgecolor=COLORS["dark"], label="Crane"),
        Patch(facecolor=COLORS["storage"], edgecolor=COLORS["dark"], label="Storage"),
        Patch(facecolor=COLORS["office"], edgecolor=COLORS["dark"], label="Office"),
        Patch(facecolor=COLORS["rest_area"], edgecolor=COLORS["dark"], label="Rest area"),
        Patch(facecolor=COLORS["road"], edgecolor=COLORS["road_edge"], label="Road exclusion"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor=COLORS["entrance"],
               markeredgecolor="#7A5A00", markersize=6, label="Entrance"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(0.5, -0.006), fontsize=7.5)

    stem = output_dir / "figure_2d_to_3d_transformation"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{stem}.{ext}")
    plt.close(fig)

    caption = (
        "Figure X. Two-dimensional to three-dimensional transformation of selected Bulleen CEXO layouts. "
        "Each row shows the same generated layout as a 2D planning footprint and as a 3D construction-site "
        "model. The transformation preserves the optimized facility coordinates, footprint dimensions, "
        "irregular boundary, road-exclusion corridors, and entrance locations, while assigning facility-type "
        "specific 3D representations for visual inspection."
    )
    (output_dir / "figure_2d_to_3d_transformation_caption.txt").write_text(caption + "\n", encoding="utf-8")
    print(f"Saved {stem}.png/.pdf/.svg")


if __name__ == "__main__":
    main()
