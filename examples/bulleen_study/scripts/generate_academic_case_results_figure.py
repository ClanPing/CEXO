#!/usr/bin/env python3
"""Generate a publication-style CEXO Bulleen case-study result figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


PROJECT_DIR = Path(__file__).resolve().parents[1]

PALETTE = {
    "blue": "#0F4D92",
    "blue_light": "#D6E6F5",
    "green": "#3B8E5E",
    "green_light": "#A9D18E",
    "red": "#B64342",
    "orange": "#D99035",
    "grey": "#D9D9D9",
    "dark": "#222222",
    "mid": "#666666",
}


def set_publication_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
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
        layouts.append(layout)
    if not layouts:
        raise SystemExit(f"No layout JSON files found in {results_dir}")
    return layouts


def objective(layout: dict, key: str) -> float:
    return float(layout["objectives"][key])


def behavior(layout: dict, key: str) -> float:
    return float(layout["behaviors"][key])


def strict_safe(layout: dict) -> bool:
    return bool(layout.get("feasibility", {}).get("safe", False))


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.13,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
        color=PALETTE["dark"],
    )


def plot_archive(ax, layouts: list[dict], grid_size: int = 20) -> None:
    grid = np.full((grid_size, grid_size), np.nan)
    for layout in layouts:
        bd1 = behavior(layout, "module_dispersion")
        bd2 = behavior(layout, "worker_operational_separation")
        i = int(np.clip(bd1 * grid_size, 0, grid_size - 1))
        j = int(np.clip(bd2 * grid_size, 0, grid_size - 1))
        combined = objective(layout, "combined_score")
        if np.isnan(grid[j, i]) or combined > grid[j, i]:
            grid[j, i] = combined

    cmap = colors.LinearSegmentedColormap.from_list(
        "case_quality",
        ["#F0F0F0", "#D6E6F5", "#7AA6D6", "#0F4D92"],
    )
    cmap.set_bad("#F2F2F2")
    im = ax.imshow(
        grid,
        origin="lower",
        extent=(0, 1, 0, 1),
        vmin=0.50,
        vmax=0.87,
        cmap=cmap,
        interpolation="nearest",
        aspect="equal",
    )
    occupied = int(np.count_nonzero(~np.isnan(grid)))
    ax.set_xlabel("Learned latent BD1")
    ax.set_ylabel("Learned latent BD2")
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.tick_params(length=3)
    ax.text(
        0.03,
        0.97,
        f"{occupied}/400 cells\n{100 * occupied / 400:.1f}% coverage",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7.5,
        bbox={"facecolor": "white", "edgecolor": PALETTE["mid"], "linewidth": 0.6, "boxstyle": "round,pad=0.25"},
    )
    return im


def plot_feasibility(ax, layouts: list[dict]) -> None:
    n_total = len(layouts)
    n_strict = sum(1 for layout in layouts if strict_safe(layout))
    n_threshold = sum(1 for layout in layouts if objective(layout, "safety_compliance") >= 0.7)
    counts = [n_strict, n_threshold - n_strict, n_total - n_threshold]
    labels = ["Strictly feasible", "Safety-threshold only", "Below threshold"]
    colors_ = [PALETTE["green"], PALETTE["green_light"], PALETTE["grey"]]

    left = 0
    for count, label, color_ in zip(counts, labels, colors_):
        ax.barh(0, count, left=left, color=color_, edgecolor="white", linewidth=1.0, height=0.42)
        if count > 80:
            ax.text(
                left + count / 2,
                0,
                f"{count:,}\n({100 * count / n_total:.1f}%)",
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold",
                color="white" if color_ == PALETTE["green"] else PALETTE["dark"],
            )
        left += count

    ax.set_xlim(0, n_total)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel("Exported elite layouts")
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.6)
    ax.set_axisbelow(True)
    handles = [Patch(facecolor=color_, edgecolor="none", label=label) for label, color_ in zip(labels, colors_)]
    for idx, (label, color_) in enumerate(zip(labels, colors_)):
        ax.text(
            0.02,
            0.94 - idx * 0.10,
            label.replace("\n", " "),
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=6.8,
            color=PALETTE["dark"],
            bbox={"facecolor": color_, "edgecolor": "none", "alpha": 0.35, "pad": 1.2},
        )


def plot_objective_distributions(ax, layouts: list[dict]) -> None:
    keys = ["safety_compliance", "operational_efficiency", "layout_adaptability"]
    labels = ["Safety", "Efficiency", "Adaptability"]
    values = [[objective(layout, key) for layout in layouts] for key in keys]
    best = max(layouts, key=lambda layout: objective(layout, "combined_score"))
    best_values = [objective(best, key) for key in keys]

    box = ax.boxplot(
        values,
        patch_artist=True,
        widths=0.55,
        showfliers=False,
        medianprops={"color": PALETTE["dark"], "linewidth": 1.2},
        whiskerprops={"color": PALETTE["dark"], "linewidth": 0.9},
        capprops={"color": PALETTE["dark"], "linewidth": 0.9},
        boxprops={"edgecolor": PALETTE["dark"], "linewidth": 0.9},
    )
    for patch in box["boxes"]:
        patch.set_facecolor(PALETTE["blue_light"])

    x = np.arange(1, len(labels) + 1)
    ax.scatter(x, best_values, marker="D", s=34, color=PALETTE["red"], edgecolor="white", linewidth=0.5, zorder=3)
    for xi, value in zip(x, best_values):
        ax.text(xi + 0.08, value + 0.012, f"{value:.3f}", fontsize=7.5, color=PALETTE["red"], ha="left", va="bottom")

    ax.set_xticks(x, labels)
    ax.set_ylabel("Objective score")
    ax.set_ylim(0.45, 1.05)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.text(
        0.03,
        0.05,
        "red diamond: best exported layout",
        transform=ax.transAxes,
        fontsize=7,
        color=PALETTE["red"],
        ha="left",
        va="bottom",
    )


def plot_tradeoff(ax, layouts: list[dict]) -> None:
    safe = np.array([strict_safe(layout) for layout in layouts])
    efficiency = np.array([objective(layout, "operational_efficiency") for layout in layouts])
    adaptability = np.array([objective(layout, "layout_adaptability") for layout in layouts])
    safety = np.array([objective(layout, "safety_compliance") for layout in layouts])
    combined = np.array([objective(layout, "combined_score") for layout in layouts])
    best_idx = int(np.argmax(combined))

    ax.scatter(
        efficiency[~safe],
        adaptability[~safe],
        s=10,
        color="#BDBDBD",
        alpha=0.40,
        linewidth=0,
        rasterized=True,
        label="Non-strict layouts",
    )
    sc = ax.scatter(
        efficiency[safe],
        adaptability[safe],
        c=safety[safe],
        s=13,
        cmap="Blues",
        vmin=0.70,
        vmax=1.00,
        alpha=0.75,
        linewidth=0,
        rasterized=True,
        label="Strictly feasible layouts",
    )
    ax.scatter(
        efficiency[best_idx],
        adaptability[best_idx],
        marker="*",
        s=130,
        color=PALETTE["red"],
        edgecolor="white",
        linewidth=0.7,
        zorder=5,
        label="Best exported layout",
    )
    ax.set_xlabel("Operational efficiency")
    ax.set_ylabel("Layout adaptability")
    ax.set_xlim(0.25, 0.88)
    ax.set_ylim(0.635, 0.725)
    ax.grid(color="#E6E6E6", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.text(
        0.03,
        0.08,
        "grey: non-strict\nblue: strictly feasible\nstar: best exported",
        transform=ax.transAxes,
        fontsize=6.8,
        ha="left",
        va="bottom",
        color=PALETTE["dark"],
        bbox={"facecolor": "white", "edgecolor": "#CCCCCC", "linewidth": 0.4, "alpha": 0.85, "pad": 2.0},
    )
    return sc


def generate(results_dir: Path, output_dir: Path) -> None:
    set_publication_style()
    layouts = load_layouts(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7.20, 5.80))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.08], height_ratios=[1.0, 1.0], wspace=0.42, hspace=0.46)

    ax_a = fig.add_subplot(gs[0, 0])
    im = plot_archive(ax_a, layouts)
    add_panel_label(ax_a, "a")
    ax_a.text(0.5, 1.055, "Learned archive quality", transform=ax_a.transAxes, ha="center", va="bottom", fontsize=8.3)

    ax_b = fig.add_subplot(gs[0, 1])
    plot_feasibility(ax_b, layouts)
    add_panel_label(ax_b, "b")
    ax_b.text(0.5, 1.055, "Feasibility composition", transform=ax_b.transAxes, ha="center", va="bottom", fontsize=8.3)

    ax_c = fig.add_subplot(gs[1, 0])
    plot_objective_distributions(ax_c, layouts)
    add_panel_label(ax_c, "c")
    ax_c.text(0.5, 1.055, "Objective score distributions", transform=ax_c.transAxes, ha="center", va="bottom", fontsize=8.3)

    ax_d = fig.add_subplot(gs[1, 1])
    sc = plot_tradeoff(ax_d, layouts)
    add_panel_label(ax_d, "d")
    ax_d.text(0.5, 1.055, "Efficiency-adaptability trade-off", transform=ax_d.transAxes, ha="center", va="bottom", fontsize=8.3)

    cbar = fig.colorbar(im, ax=ax_a, fraction=0.046, pad=0.035)
    cbar.set_label("Best combined score per cell", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5)

    cbar2 = fig.colorbar(sc, ax=ax_d, fraction=0.046, pad=0.035)
    cbar2.set_label("Safety score", fontsize=7)
    cbar2.ax.tick_params(labelsize=6.5)

    fig.text(
        0.01,
        0.01,
        "Archive axes are autoencoder-learned latent behavioural descriptors; empty archive cells are shown in light grey.",
        fontsize=6.8,
        color=PALETTE["mid"],
    )

    png_path = output_dir / "figure_bulleen_case_results_academic.png"
    pdf_path = output_dir / "figure_bulleen_case_results_academic.pdf"
    svg_path = output_dir / "figure_bulleen_case_results_academic.svg"
    fig.savefig(png_path, dpi=600)
    fig.savefig(pdf_path)
    fig.savefig(svg_path)
    plt.close(fig)

    caption = (
        "Publication-style summary of CEXO results for the Bulleen practical case. "
        "(a) Learned behavioural archive, where each occupied cell reports the best exported combined objective score. "
        "(b) Feasibility composition of the exported elite layouts. "
        "(c) Objective score distributions across all exported elite layouts; diamonds mark the best exported layout. "
        "(d) Efficiency-adaptability trade-off, with point colour indicating safety compliance."
    )
    (output_dir / "figure_bulleen_case_results_academic_caption.txt").write_text(caption + "\n", encoding="utf-8")
    print(png_path)
    print(pdf_path)
    print(svg_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results") / "cexo_bulleen_15000_full_fg")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    results_dir = args.results_dir
    if not results_dir.is_absolute():
        results_dir = PROJECT_DIR / results_dir
    output_dir = args.output_dir or (results_dir / "paper_analysis")
    if not output_dir.is_absolute():
        output_dir = PROJECT_DIR / output_dir
    generate(results_dir, output_dir)


if __name__ == "__main__":
    main()
