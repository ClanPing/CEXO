from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "results" / "cexo_bulleen_15000_full_fg"
ANALYSIS_DIR = RESULTS_DIR / "paper_analysis"
OUT_STEM = ANALYSIS_DIR / "figure10_archive_objectives"
GRID_SIZE = 20


OBJECTIVE_KEYS = [
    ("safety_compliance", "Safety"),
    ("operational_efficiency", "Efficiency"),
    ("layout_adaptability", "Adaptability"),
    ("combined_score", "Combined"),
]


def load_layouts() -> list[dict]:
    layouts: list[dict] = []
    for path in sorted(RESULTS_DIR.glob("cslpelite_layout_*.json")):
        with path.open("r", encoding="utf-8") as handle:
            layouts.append(json.load(handle))
    if not layouts:
        raise SystemExit(f"No exported layout JSON files found in {RESULTS_DIR}")
    return layouts


def obj(layout: dict, key: str) -> float:
    return float(layout.get("objectives", {}).get(key, 0.0))


def behaviour_index(value: float) -> int:
    return max(0, min(GRID_SIZE - 1, int(float(value) * GRID_SIZE)))


def build_archive_grid(layouts: list[dict]) -> np.ndarray:
    grid = np.full((GRID_SIZE, GRID_SIZE), np.nan, dtype=float)
    for layout in layouts:
        behaviours = layout.get("behaviors", {})
        bx = behaviour_index(behaviours.get("module_dispersion", 0.0))
        by = behaviour_index(behaviours.get("worker_operational_separation", 0.0))
        combined = obj(layout, "combined_score")
        if np.isnan(grid[by, bx]) or combined > grid[by, bx]:
            grid[by, bx] = combined
    return grid


def plot_archive(ax: plt.Axes, layouts: list[dict]) -> matplotlib.image.AxesImage:
    grid = build_archive_grid(layouts)
    cmap = plt.colormaps["viridis"].copy()
    cmap.set_bad(color="white")

    image = ax.imshow(grid, origin="lower", vmin=0.0, vmax=1.0, cmap=cmap)
    ax.set_title("Behavioural archive for Bulleen case", fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Behavioural descriptor 1", fontsize=13)
    ax.set_ylabel("Behavioural descriptor 2", fontsize=13)
    ax.set_xticks(np.arange(0, GRID_SIZE, 2))
    ax.set_yticks(np.arange(0, GRID_SIZE, 2))
    ax.tick_params(labelsize=11)
    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(-0.5, GRID_SIZE - 0.5)

    # Thin cell grid keeps the archive readable without overpowering the data.
    ax.set_xticks(np.arange(-0.5, GRID_SIZE, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, GRID_SIZE, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.35, alpha=0.18)
    ax.tick_params(which="minor", bottom=False, left=False)

    occupied = int(np.sum(~np.isnan(grid)))
    coverage = occupied / float(GRID_SIZE * GRID_SIZE) * 100.0
    feasible = sum(1 for layout in layouts if layout.get("feasibility", {}).get("safe", False))
    best = max(obj(layout, "combined_score") for layout in layouts)
    note = (
        f"Coverage: {occupied}/400 ({coverage:.1f}%)\n"
        f"Feasible: {feasible:,}/{len(layouts):,}\n"
        f"Best fitness: {best:.3f}"
    )
    ax.text(
        0.035,
        0.955,
        note,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        linespacing=1.25,
        bbox={
            "boxstyle": "round,pad=0.38",
            "facecolor": "white",
            "edgecolor": "#444444",
            "linewidth": 0.7,
            "alpha": 0.94,
        },
    )
    return image


def plot_objective_distributions(fig: plt.Figure, spec, layouts: list[dict]) -> None:
    right = spec.subgridspec(3, 2, height_ratios=[0.18, 1, 1], hspace=0.62, wspace=0.34)
    title_ax = fig.add_subplot(right[0, :])
    title_ax.axis("off")
    title_ax.text(
        0.5,
        0.3,
        "Distribution of CEXO elite objective scores",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
    )

    axes = [
        fig.add_subplot(right[1, 0]),
        fig.add_subplot(right[1, 1]),
        fig.add_subplot(right[2, 0]),
        fig.add_subplot(right[2, 1]),
    ]

    for idx, (ax, (key, label)) in enumerate(zip(axes, OBJECTIVE_KEYS)):
        values = np.asarray([obj(layout, key) for layout in layouts], dtype=float)
        ax.hist(values, bins=28, color="#4b82c8", alpha=0.82, edgecolor="white", linewidth=0.8)
        ax.axvline(np.mean(values), color="#d9272e", linewidth=2.0, label="Mean")
        ax.axvline(np.median(values), color="#263238", linewidth=1.8, linestyle="--", label="Median")
        ax.set_title(label, fontsize=13, fontweight="bold", pad=7)
        ax.set_xlabel("Score" if idx >= 2 else "", fontsize=11.5)
        ax.set_ylabel("Elite layouts", fontsize=11.5)
        ax.tick_params(labelsize=10)
        ax.grid(True, axis="y", alpha=0.22, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].legend(frameon=False, fontsize=10, loc="upper left")


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    layouts = load_layouts()

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 12,
            "axes.linewidth": 0.9,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )

    fig = plt.figure(figsize=(12.4, 5.15), constrained_layout=False)
    outer = fig.add_gridspec(1, 2, width_ratios=[1.02, 1.28], wspace=0.22)

    ax_archive = fig.add_subplot(outer[0, 0])
    image = plot_archive(ax_archive, layouts)
    cbar = fig.colorbar(image, ax=ax_archive, fraction=0.047, pad=0.035)
    cbar.set_label("Scalar fitness", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    plot_objective_distributions(fig, outer[0, 1], layouts)

    fig.text(0.012, 0.982, "(a)", fontsize=15, fontweight="bold", ha="left", va="top")
    fig.text(0.455, 0.982, "(b)", fontsize=15, fontweight="bold", ha="left", va="top")

    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{OUT_STEM}.{ext}")
    plt.close(fig)

    caption = (
        "Figure 10. CEXO archive coverage and objective-score distribution for the "
        "Bulleen practical case study. (a) Occupied cells in the 20 x 20 learned "
        "behavioural archive, coloured by the best scalar fitness retained in each "
        "cell. (b) Distribution of exported elite scores for safety, operational "
        "efficiency, adaptability, and the combined objective."
    )
    (ANALYSIS_DIR / "figure10_archive_objectives_caption.txt").write_text(
        caption + "\n", encoding="utf-8"
    )

    print(f"Saved {OUT_STEM}.png/.pdf/.svg")


if __name__ == "__main__":
    main()
