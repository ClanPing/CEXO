#!/usr/bin/env python3
"""Generate 2D PNG previews for exported Bulleen layout JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle


PROJECT_DIR = Path(__file__).resolve().parents[1]

FACILITY_COLORS = {
    "core": "#4b82c8",
    "crane": "#e63232",
    "storage": "#32b44b",
    "office": "#b482d7",
    "rest_area": "#f0a55a",
}

FACILITY_LABELS = {
    "core": "CORE",
    "crane": "CRANE",
    "storage": "STORAGE",
    "office": "OFFICE",
    "rest_area": "REST",
}


def facility_base_type(facility_type: str) -> str:
    if facility_type.startswith("rest"):
        return "rest_area"
    return facility_type.split("_")[0]


def draw_polygon(ax, points, **kwargs) -> None:
    if not points or len(points) < 3:
        return
    ax.add_patch(Polygon(points, closed=True, **kwargs))


def draw_layout(layout: dict, output_path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))

    boundary = layout.get("boundary_polygon") or []
    draw_polygon(
        ax,
        boundary,
        facecolor="#f7eadc",
        edgecolor="#d9272e",
        linewidth=2.4,
        alpha=0.55,
        zorder=1,
    )

    for zone in layout.get("exclusion_zones") or []:
        draw_polygon(
            ax,
            zone.get("polygon") or [],
            facecolor="#ffe45c",
            edgecolor="#d3bd00",
            linewidth=1.0,
            alpha=0.85,
            zorder=2,
        )

    for entrance_idx, entrance in enumerate(layout.get("entrances") or [], start=1):
        x = entrance.get("x")
        y = entrance.get("y")
        if x is None or y is None:
            continue
        ax.plot(x, y, marker="*", markersize=13, color="#f4c430", markeredgecolor="#a36b00", zorder=6)
        ax.text(x, y - 0.025, f"E{entrance_idx}", ha="center", va="top", fontsize=7, color="#8a5a00")

    for facility in layout.get("facilities") or []:
        facility_type = facility.get("type", "unknown")
        base_type = facility_base_type(facility_type)
        x = facility.get("x")
        y = facility.get("y")
        width = facility.get("width")
        length = facility.get("length")
        if None in (x, y, width, length):
            continue

        rect = Rectangle(
            (x - width / 2, y - length / 2),
            width,
            length,
            facecolor=FACILITY_COLORS.get(base_type, "#999999"),
            edgecolor="#1f2933",
            linewidth=1.0,
            alpha=0.86,
            zorder=4,
        )
        ax.add_patch(rect)

        label = FACILITY_LABELS.get(base_type, base_type.upper())
        ax.text(x, y, label, ha="center", va="center", fontsize=5.5, fontweight="bold", color="white", zorder=5)

        if base_type == "crane":
            jib_radius = facility.get("jib_length_m")
            site_width = layout.get("site_width_m") or 100.0
            if jib_radius:
                radius = float(jib_radius) / float(site_width)
            else:
                radius = 0.10
            ax.add_patch(
                Circle((x, y), radius, fill=False, edgecolor="#e63232", linewidth=1.0, linestyle="--", alpha=0.7, zorder=3)
            )

    objectives = layout.get("objectives") or {}
    title = (
        f"{layout.get('id', output_path.stem)} | "
        f"S {objectives.get('safety_compliance', objectives.get('safety', 0)):.3f} | "
        f"E {objectives.get('operational_efficiency', objectives.get('efficiency', 0)):.3f} | "
        f"A {objectives.get('layout_adaptability', objectives.get('adaptability', 0)):.3f}"
    )

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlim(0.08, 0.92)
    ax.set_ylim(0.14, 0.86)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Normalized X")
    ax.set_ylabel("Normalized Y")
    ax.grid(True, linewidth=0.5, alpha=0.28)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results") / "cexo_bulleen_15000_full_fg",
        help="Folder containing cslpelite_layout_*.json files.",
    )
    parser.add_argument("--max", type=int, default=None, help="Optional maximum number of previews to generate.")
    parser.add_argument("--dpi", type=int, default=110, help="PNG resolution.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate PNGs that already exist.")
    args = parser.parse_args()

    results_dir = args.results_dir
    if not results_dir.is_absolute():
        results_dir = PROJECT_DIR / results_dir

    layout_files = sorted(results_dir.glob("cslpelite_layout_*.json"))
    if args.max is not None:
        layout_files = layout_files[: args.max]

    if not layout_files:
        raise SystemExit(f"No cslpelite_layout_*.json files found in {results_dir}")

    generated = 0
    skipped = 0
    for idx, layout_file in enumerate(layout_files, start=1):
        output_path = layout_file.with_suffix(".png")
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue

        with layout_file.open("r", encoding="utf-8") as handle:
            layout = json.load(handle)
        draw_layout(layout, output_path, dpi=args.dpi)
        generated += 1

        if idx % 100 == 0:
            print(f"Processed {idx}/{len(layout_files)} layouts...")

    print(f"Done. Generated {generated} PNG previews, skipped {skipped}.")


if __name__ == "__main__":
    main()
