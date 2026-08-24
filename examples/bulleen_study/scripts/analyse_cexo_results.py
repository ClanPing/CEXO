#!/usr/bin/env python3
"""Create paper-facing tables and figures from a CEXO Bulleen result folder."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon, Rectangle


PROJECT_DIR = Path(__file__).resolve().parents[1]

FACILITY_COLORS = {
    "core": "#4b82c8",
    "crane": "#e63232",
    "storage": "#32b44b",
    "office": "#9d6ac6",
    "rest_area": "#f0a55a",
}

FACILITY_LABELS = {
    "core": "C",
    "crane": "CR",
    "storage": "S",
    "office": "O",
    "rest_area": "R",
}

OBJECTIVE_KEYS = [
    ("safety_compliance", "Safety"),
    ("operational_efficiency", "Efficiency"),
    ("layout_adaptability", "Adaptability"),
    ("combined_score", "Combined"),
]

BEHAVIOR_KEYS = [
    ("module_dispersion", "Learned latent BD1"),
    ("worker_operational_separation", "Learned latent BD2"),
]


@dataclass
class LayoutRecord:
    layout_id: str
    path: Path
    layout: dict
    safety: float
    efficiency: float
    adaptability: float
    combined: float
    dispersion: float
    separation: float
    strict_safe: bool
    violation_count: int

    @property
    def min_objective(self) -> float:
        return min(self.safety, self.efficiency, self.adaptability)


def load_records(results_dir: Path) -> list[LayoutRecord]:
    records: list[LayoutRecord] = []
    for path in sorted(results_dir.glob("cslpelite_layout_*.json")):
        with path.open("r", encoding="utf-8") as handle:
            layout = json.load(handle)
        objectives = layout.get("objectives", {})
        behaviors = layout.get("behaviors", {})
        feasibility = layout.get("feasibility", {})
        records.append(
            LayoutRecord(
                layout_id=layout.get("id", path.stem),
                path=path,
                layout=layout,
                safety=float(objectives.get("safety_compliance", 0.0)),
                efficiency=float(objectives.get("operational_efficiency", 0.0)),
                adaptability=float(objectives.get("layout_adaptability", 0.0)),
                combined=float(objectives.get("combined_score", 0.0)),
                dispersion=float(behaviors.get("module_dispersion", 0.0)),
                separation=float(behaviors.get("worker_operational_separation", 0.0)),
                strict_safe=bool(feasibility.get("safe", False)),
                violation_count=len(feasibility.get("violations", []) or []),
            )
        )
    if not records:
        raise SystemExit(f"No cslpelite_layout_*.json files found in {results_dir}")
    return records


def percentile_stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "mean": float(np.mean(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_tables(records: list[LayoutRecord], results_dir: Path, output_dir: Path) -> dict[str, Path]:
    with (results_dir / "results.json").open("r", encoding="utf-8") as handle:
        results = json.load(handle)
    with (results_dir / "cslpelite_summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    strict_safe = sum(1 for record in records if record.strict_safe)
    safe_threshold = sum(1 for record in records if record.safety >= 0.7)
    runtime_seconds = float(results["statistics"]["runtime_seconds"])

    overview_rows = [
        {"metric": "Facilities", "value": str(results["configuration"]["facility_count"])},
        {"metric": "Initial population", "value": str(results["configuration"]["initial_population"])},
        {"metric": "Generations / iterations", "value": str(results["configuration"]["iterations"])},
        {"metric": "Archive grid", "value": "20 x 20"},
        {"metric": "Max elites per cell", "value": "12"},
        {"metric": "Occupied archive cells", "value": f"{summary['archive_performance']['coverage']} / 400"},
        {"metric": "Archive coverage", "value": f"{summary['archive_performance']['coverage_percentage']:.2f}%"},
        {"metric": "Exported elite layouts", "value": str(len(records))},
        {"metric": "Strictly feasible layouts", "value": f"{strict_safe} ({100 * strict_safe / len(records):.2f}%)"},
        {"metric": "Safety-threshold layouts", "value": f"{safe_threshold} ({100 * safe_threshold / len(records):.2f}%)"},
        {"metric": "Best scalar fitness", "value": f"{results['statistics']['best_scalar_fitness']:.4f}"},
        {"metric": "Best exported combined objective", "value": f"{max(record.combined for record in records):.4f}"},
        {"metric": "Runtime", "value": f"{runtime_seconds / 60:.1f} min"},
    ]
    overview_path = output_dir / "table_cexo_overview.csv"
    write_csv(overview_path, overview_rows, ["metric", "value"])

    range_rows = []
    for key, label in OBJECTIVE_KEYS:
        values = [getattr(record, key.replace("safety_compliance", "safety")
                          .replace("operational_efficiency", "efficiency")
                          .replace("layout_adaptability", "adaptability")
                          .replace("combined_score", "combined")) for record in records]
        stats = percentile_stats(values)
        range_rows.append({"metric": label, **{name: f"{value:.4f}" for name, value in stats.items()}})
    for key, label in BEHAVIOR_KEYS:
        attr = "dispersion" if key == "module_dispersion" else "separation"
        stats = percentile_stats([getattr(record, attr) for record in records])
        range_rows.append({"metric": label, **{name: f"{value:.4f}" for name, value in stats.items()}})
    ranges_path = output_dir / "table_objective_behavior_ranges.csv"
    write_csv(ranges_path, range_rows, ["metric", "min", "p05", "p25", "median", "mean", "p75", "p95", "max"])

    violation_categories = {
        "crane_collision": 0,
        "entrance_clearance": 0,
        "crane_danger": 0,
        "boundary": 0,
        "overlap": 0,
        "other": 0,
    }
    layouts_with_category = {category: set() for category in violation_categories}
    for record in records:
        for violation in record.layout.get("feasibility", {}).get("violations", []) or []:
            if violation.startswith("crane_collision"):
                category = "crane_collision"
            elif violation.startswith("entrance_clearance"):
                category = "entrance_clearance"
            elif violation.startswith("crane_danger"):
                category = "crane_danger"
            elif violation.startswith("boundary"):
                category = "boundary"
            elif violation.startswith("overlap"):
                category = "overlap"
            else:
                category = "other"
            violation_categories[category] += 1
            layouts_with_category[category].add(record.layout_id)

    violation_rows = [
        {
            "violation_category": category,
            "violation_instances": str(count),
            "affected_layouts": str(len(layouts_with_category[category])),
        }
        for category, count in violation_categories.items()
        if count > 0
    ]
    violations_path = output_dir / "table_violation_categories.csv"
    write_csv(violations_path, violation_rows, ["violation_category", "violation_instances", "affected_layouts"])

    md_rows = [
        [row["metric"], row["min"], row["p25"], row["median"], row["mean"], row["p75"], row["max"]]
        for row in range_rows
    ]
    md = [
        "# CEXO Bulleen Result Tables",
        "",
        "## Run Overview",
        "",
        markdown_table(["Metric", "Value"], [[row["metric"], row["value"]] for row in overview_rows]),
        "",
        "## Objective And Behaviour Ranges",
        "",
        markdown_table(["Metric", "Min", "P25", "Median", "Mean", "P75", "Max"], md_rows),
        "",
        "## Constraint Violation Categories",
        "",
        markdown_table(
            ["Category", "Violation Instances", "Affected Layouts"],
            [[row["violation_category"], row["violation_instances"], row["affected_layouts"]] for row in violation_rows],
        ),
        "",
    ]
    md_path = output_dir / "paper_tables.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    return {"overview": overview_path, "ranges": ranges_path, "violations": violations_path, "markdown": md_path}


def unique_select(records: list[LayoutRecord]) -> list[tuple[str, LayoutRecord, str]]:
    safe_records = [record for record in records if record.strict_safe]
    if not safe_records:
        safe_records = records[:]
    median_combined = float(np.median([record.combined for record in safe_records]))
    strong = [record for record in safe_records if record.combined >= median_combined]

    criteria = [
        ("Best overall", lambda rs: max(rs, key=lambda r: r.combined), "Highest combined objective score"),
        ("Best efficiency", lambda rs: max(rs, key=lambda r: (r.efficiency, r.combined)), "Strongest operational efficiency"),
        ("Best adaptability", lambda rs: max(rs, key=lambda r: (r.adaptability, r.combined)), "Strongest layout adaptability"),
        ("Balanced trade-off", lambda rs: max(rs, key=lambda r: (r.min_objective, r.combined)), "Highest worst-objective score"),
        ("Latent BD2 extreme", lambda rs: max(strong or rs, key=lambda r: (r.separation, r.combined)), "High-performing layout from an extreme region of learned latent BD2"),
        ("Latent BD1 extreme", lambda rs: max(strong or rs, key=lambda r: (r.dispersion, r.combined)), "High-performing layout from an extreme region of learned latent BD1"),
    ]

    selected: list[tuple[str, LayoutRecord, str]] = []
    used: set[str] = set()
    for label, chooser, reason in criteria:
        candidate_pool = [record for record in safe_records if record.layout_id not in used] or safe_records
        chosen = chooser(candidate_pool)
        selected.append((label, chosen, reason))
        used.add(chosen.layout_id)
    return selected


def write_showcase_table(selected: list[tuple[str, LayoutRecord, str]], output_dir: Path) -> Path:
    rows = []
    for label, record, reason in selected:
        rows.append(
            {
                "showcase_role": label,
                "layout_id": record.layout_id,
                "reason": reason,
                "safety": f"{record.safety:.4f}",
                "efficiency": f"{record.efficiency:.4f}",
                "adaptability": f"{record.adaptability:.4f}",
                "combined": f"{record.combined:.4f}",
                "learned_latent_bd1": f"{record.dispersion:.4f}",
                "learned_latent_bd2": f"{record.separation:.4f}",
                "strict_safe": str(record.strict_safe),
                "json_file": record.path.name,
            }
        )
    path = output_dir / "table_showcase_layouts.csv"
    write_csv(
        path,
        rows,
        [
            "showcase_role",
            "layout_id",
            "reason",
            "safety",
            "efficiency",
            "adaptability",
            "combined",
            "learned_latent_bd1",
            "learned_latent_bd2",
            "strict_safe",
            "json_file",
        ],
    )

    md_rows = [
        [
            row["showcase_role"],
            row["layout_id"],
            row["safety"],
            row["efficiency"],
            row["adaptability"],
            row["combined"],
            row["reason"],
        ]
        for row in rows
    ]
    md = markdown_table(
        ["Role", "Layout", "Safety", "Efficiency", "Adaptability", "Combined", "Reason"],
        md_rows,
    )
    (output_dir / "table_showcase_layouts.md").write_text(md + "\n", encoding="utf-8")
    return path


def draw_polygon(ax, points, **kwargs) -> None:
    if points and len(points) >= 3:
        ax.add_patch(Polygon(points, closed=True, **kwargs))


def draw_layout(ax, record: LayoutRecord, title: str) -> None:
    layout = record.layout
    draw_polygon(
        ax,
        layout.get("boundary_polygon") or [],
        facecolor="#f7eadc",
        edgecolor="#d9272e",
        linewidth=1.8,
        alpha=0.6,
        zorder=1,
    )
    for zone in layout.get("exclusion_zones") or []:
        draw_polygon(
            ax,
            zone.get("polygon") or [],
            facecolor="#ffe45c",
            edgecolor="#c7aa00",
            linewidth=0.45,
            alpha=0.78,
            zorder=2,
        )
    for facility in layout.get("facilities") or []:
        base_type = facility.get("type", "unknown")
        x = float(facility.get("x", 0.0))
        y = float(facility.get("y", 0.0))
        width = float(facility.get("width", 0.0))
        length = float(facility.get("length", 0.0))
        ax.add_patch(
            Rectangle(
                (x - width / 2, y - length / 2),
                width,
                length,
                facecolor=FACILITY_COLORS.get(base_type, "#999999"),
                edgecolor="#1f2933",
                linewidth=0.6,
                alpha=0.9,
                zorder=4,
            )
        )
        ax.text(
            x,
            y,
            FACILITY_LABELS.get(base_type, "?"),
            ha="center",
            va="center",
            fontsize=4.5,
            fontweight="bold",
            color="white",
            zorder=5,
        )
        if base_type == "crane":
            ax.add_patch(
                Circle(
                    (x, y),
                    0.06,
                    fill=False,
                    edgecolor="#e63232",
                    linewidth=0.6,
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
            markersize=7,
            color="#f4c430",
            markeredgecolor="#8c5a00",
            zorder=6,
        )
    ax.set_title(title, fontsize=8.5, fontweight="bold")
    ax.set_xlim(0.08, 0.92)
    ax.set_ylim(0.14, 0.86)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])


def save_run_summary_figure(records: list[LayoutRecord], results_dir: Path, output_dir: Path) -> Path:
    with (results_dir / "results.json").open("r", encoding="utf-8") as handle:
        results = json.load(handle)
    with (results_dir / "cslpelite_summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    archive_stats = summary["archive_performance"]
    occupied_cells = int(archive_stats["coverage"])
    total_cells = 400
    empty_cells = total_cells - occupied_cells
    exported = len(records)
    strict_safe = sum(1 for record in records if record.strict_safe)
    threshold_safe = sum(1 for record in records if record.safety >= 0.7)
    threshold_only = threshold_safe - strict_safe
    below_threshold = exported - threshold_safe
    best = max(records, key=lambda record: record.combined)

    objective_names = ["Safety", "Efficiency", "Adaptability"]
    objective_values = {
        "Safety": [record.safety for record in records],
        "Efficiency": [record.efficiency for record in records],
        "Adaptability": [record.adaptability for record in records],
    }
    medians = [np.median(objective_values[name]) for name in objective_names]
    p25 = [np.percentile(objective_values[name], 25) for name in objective_names]
    p75 = [np.percentile(objective_values[name], 75) for name in objective_names]
    best_values = [best.safety, best.efficiency, best.adaptability]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    fig = plt.figure(figsize=(11.5, 7.2), facecolor="white")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], wspace=0.32, hspace=0.42)

    ax_coverage = fig.add_subplot(grid[0, 0])
    ax_coverage.pie(
        [occupied_cells, empty_cells],
        startangle=90,
        counterclock=False,
        colors=["#2f6f9f", "#e8edf2"],
        wedgeprops={"width": 0.34, "edgecolor": "white", "linewidth": 2},
    )
    ax_coverage.text(0, 0.08, f"{100 * occupied_cells / total_cells:.1f}%", ha="center", va="center", fontsize=24, fontweight="bold", color="#17324d")
    ax_coverage.text(0, -0.18, f"{occupied_cells}/{total_cells} cells", ha="center", va="center", fontsize=10, color="#4b5563")
    ax_coverage.set_title("Archive coverage")
    ax_coverage.set_aspect("equal")

    ax_feasible = fig.add_subplot(grid[0, 1])
    feasibility_labels = ["Strictly\nfeasible", "Safety-threshold\nonly", "Below safety\nthreshold"]
    feasibility_values = [strict_safe, threshold_only, below_threshold]
    feasibility_colors = ["#2f855a", "#82c91e", "#d0d7de"]
    x_feasible = np.arange(len(feasibility_labels))
    bars = ax_feasible.bar(x_feasible, feasibility_values, color=feasibility_colors, edgecolor="#243447", linewidth=0.9)
    for bar, value in zip(bars, feasibility_values):
        label_y = value + exported * 0.025 if value > exported * 0.04 else exported * 0.055
        ax_feasible.text(
            bar.get_x() + bar.get_width() / 2,
            label_y,
            f"{value:,}\n{100 * value / exported:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#243447",
        )
    ax_feasible.set_xticks(x_feasible, feasibility_labels)
    ax_feasible.set_ylim(0, exported * 1.14)
    ax_feasible.set_ylabel("Exported elite layouts")
    ax_feasible.set_title("Feasibility breakdown")
    ax_feasible.grid(True, axis="y", alpha=0.25)

    ax_objectives = fig.add_subplot(grid[1, 0])
    x = np.arange(len(objective_names))
    lower_err = np.array(medians) - np.array(p25)
    upper_err = np.array(p75) - np.array(medians)
    ax_objectives.bar(x, medians, yerr=[lower_err, upper_err], capsize=5, color="#4b82c8", alpha=0.82, edgecolor="#17324d", label="Median with IQR")
    ax_objectives.scatter(x, best_values, marker="D", s=72, color="#d9272e", edgecolor="white", linewidth=0.8, zorder=4, label="Best exported layout")
    for idx, value in enumerate(best_values):
        ax_objectives.text(idx, value + 0.025, f"{value:.3f}", ha="center", va="bottom", fontsize=8, color="#7f1d1d", fontweight="bold")
    ax_objectives.set_xticks(x, objective_names)
    ax_objectives.set_ylim(0, 1.08)
    ax_objectives.set_ylabel("Objective score")
    ax_objectives.set_title("Objective performance")
    ax_objectives.grid(True, axis="y", alpha=0.25)
    ax_objectives.legend(frameon=False, fontsize=8, loc="lower right")

    ax_cards = fig.add_subplot(grid[1, 1])
    ax_cards.axis("off")
    card_data = [
        ("Elite layouts", f"{exported:,}"),
        ("Best scalar fitness", f"{results['statistics']['best_scalar_fitness']:.3f}"),
        ("Best combined objective", f"{best.combined:.3f}"),
        ("Runtime", f"{results['statistics']['runtime_seconds'] / 60:.1f} min"),
    ]
    card_positions = [(0.03, 0.55), (0.53, 0.55), (0.03, 0.08), (0.53, 0.08)]
    for (label, value), (x0, y0) in zip(card_data, card_positions):
        rect = Rectangle((x0, y0), 0.43, 0.34, transform=ax_cards.transAxes, facecolor="#f6f8fa", edgecolor="#d0d7de", linewidth=1.2)
        ax_cards.add_patch(rect)
        ax_cards.text(x0 + 0.04, y0 + 0.22, value, transform=ax_cards.transAxes, fontsize=20, fontweight="bold", color="#17324d", va="center")
        ax_cards.text(x0 + 0.04, y0 + 0.10, label, transform=ax_cards.transAxes, fontsize=9.5, color="#4b5563", va="center")
    ax_cards.set_title("Headline run metrics")

    fig.suptitle("CEXO Bulleen Practical Case Study: Run Summary", fontsize=15, fontweight="bold", y=0.985)
    fig.text(
        0.5,
        0.018,
        "Archive uses two autoencoder-learned latent behavioural descriptors; feasibility categories are mutually exclusive.",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    path = output_dir / "figure_run_summary.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def save_figures(records: list[LayoutRecord], selected: list[tuple[str, LayoutRecord, str]], results_dir: Path, output_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    safe_mask = np.array([record.strict_safe for record in records], dtype=bool)
    paths["run_summary"] = save_run_summary_figure(records, results_dir, output_dir)

    objective_arrays = {
        "Safety": np.array([record.safety for record in records]),
        "Efficiency": np.array([record.efficiency for record in records]),
        "Adaptability": np.array([record.adaptability for record in records]),
        "Combined": np.array([record.combined for record in records]),
    }

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
    axes = axes.flatten()
    for ax, (label, values) in zip(axes, objective_arrays.items()):
        ax.hist(values, bins=28, color="#4b82c8", alpha=0.82, edgecolor="white")
        ax.axvline(np.mean(values), color="#d9272e", linewidth=1.5, label="Mean")
        ax.axvline(np.median(values), color="#263238", linewidth=1.2, linestyle="--", label="Median")
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Score")
        ax.set_ylabel("Elite layouts")
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Distribution of CEXO Elite Objective Scores", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = output_dir / "figure_objective_distributions.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths["objective_distributions"] = path

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    x = np.array([record.dispersion for record in records])
    y = np.array([record.separation for record in records])
    c = np.array([record.combined for record in records])
    ax.scatter(x[~safe_mask], y[~safe_mask], s=13, color="#b0b0b0", alpha=0.35, label="Constraint-violating")
    sc = ax.scatter(x[safe_mask], y[safe_mask], c=c[safe_mask], s=18, cmap="viridis", alpha=0.8, label="Strictly feasible")
    for idx, (label, record, _) in enumerate(selected, start=1):
        ax.scatter(record.dispersion, record.separation, s=110, marker="*", color="#d9272e", edgecolor="white", linewidth=0.8, zorder=5)
        ax.text(record.dispersion + 0.012, record.separation + 0.012, str(idx), fontsize=8, fontweight="bold", color="#7f1d1d")
    ax.set_xlabel("Learned latent BD1")
    ax.set_ylabel("Learned latent BD2")
    ax.set_title("CEXO Behavioural Archive Coverage", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.25)
    fig.colorbar(sc, ax=ax, label="Combined objective score")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    path = output_dir / "figure_behavior_space.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths["behavior_space"] = path

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    eff = np.array([record.efficiency for record in records])
    adapt = np.array([record.adaptability for record in records])
    safety = np.array([record.safety for record in records])
    ax.scatter(eff[~safe_mask], adapt[~safe_mask], s=12, color="#b0b0b0", alpha=0.35, label="Constraint-violating")
    sc = ax.scatter(eff[safe_mask], adapt[safe_mask], c=safety[safe_mask], s=18, cmap="plasma", alpha=0.75, label="Strictly feasible")
    for idx, (_, record, _) in enumerate(selected, start=1):
        ax.scatter(record.efficiency, record.adaptability, s=105, marker="*", color="#174ea6", edgecolor="white", linewidth=0.8, zorder=5)
        ax.text(record.efficiency + 0.004, record.adaptability + 0.004, str(idx), fontsize=8, fontweight="bold", color="#174ea6")
    ax.set_xlabel("Operational efficiency")
    ax.set_ylabel("Layout adaptability")
    ax.set_title("Efficiency-Adaptability Trade-off Across Elite Layouts", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.25)
    fig.colorbar(sc, ax=ax, label="Safety compliance")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    path = output_dir / "figure_objective_tradeoff.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths["objective_tradeoff"] = path

    category_counts = {
        "Crane collision": 0,
        "Entrance clearance": 0,
        "Crane danger": 0,
        "Boundary": 0,
        "Facility overlap": 0,
        "Other": 0,
    }
    for record in records:
        for violation in record.layout.get("feasibility", {}).get("violations", []) or []:
            if violation.startswith("crane_collision"):
                category_counts["Crane collision"] += 1
            elif violation.startswith("entrance_clearance"):
                category_counts["Entrance clearance"] += 1
            elif violation.startswith("crane_danger"):
                category_counts["Crane danger"] += 1
            elif violation.startswith("boundary"):
                category_counts["Boundary"] += 1
            elif violation.startswith("overlap"):
                category_counts["Facility overlap"] += 1
            else:
                category_counts["Other"] += 1
    category_counts = {key: value for key, value in category_counts.items() if value > 0}
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    labels = list(category_counts.keys())
    values = list(category_counts.values())
    ax.barh(labels, values, color="#4b82c8", alpha=0.85)
    ax.set_xlabel("Violation instances across exported layouts")
    ax.set_title("Constraint Violation Pattern in Non-Strict Elite Layouts", fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.25)
    for idx, value in enumerate(values):
        ax.text(value + max(values) * 0.01, idx, str(value), va="center", fontsize=8)
    fig.tight_layout()
    path = output_dir / "figure_violation_categories.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths["violation_categories"] = path

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2))
    for index, (ax, (label, record, _)) in enumerate(zip(axes.flatten(), selected), start=1):
        title = (
            f"{index}. {label}\n"
            f"S {record.safety:.2f} | E {record.efficiency:.2f} | A {record.adaptability:.2f} | C {record.combined:.2f}"
        )
        draw_layout(ax, record, title)
    legend_elements = [
        Line2D([0], [0], marker="s", color="w", label="Core", markerfacecolor=FACILITY_COLORS["core"], markersize=8),
        Line2D([0], [0], marker="s", color="w", label="Crane", markerfacecolor=FACILITY_COLORS["crane"], markersize=8),
        Line2D([0], [0], marker="s", color="w", label="Storage", markerfacecolor=FACILITY_COLORS["storage"], markersize=8),
        Line2D([0], [0], marker="s", color="w", label="Office", markerfacecolor=FACILITY_COLORS["office"], markersize=8),
        Line2D([0], [0], marker="s", color="w", label="Rest", markerfacecolor=FACILITY_COLORS["rest_area"], markersize=8),
        Line2D([0], [0], marker="*", color="w", label="Entrance", markerfacecolor="#f4c430", markeredgecolor="#8c5a00", markersize=9),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=6, frameon=False, fontsize=9)
    fig.suptitle("Representative CEXO Layouts for the Bulleen Case Study", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    path = output_dir / "figure_showcase_layouts.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths["showcase_layouts"] = path

    return paths


def write_narrative(records: list[LayoutRecord], selected: list[tuple[str, LayoutRecord, str]], output_dir: Path) -> Path:
    safe_count = sum(1 for record in records if record.strict_safe)
    best = max(records, key=lambda record: record.combined)
    safety = percentile_stats([record.safety for record in records])
    efficiency = percentile_stats([record.efficiency for record in records])
    adaptability = percentile_stats([record.adaptability for record in records])

    lines = [
        "# Suggested Case Study Result Text",
        "",
        (
            f"The Bulleen practical case generated {len(records)} archived elite layouts across "
            f"382 occupied cells of a 20 x 20 behavioural archive, corresponding to 95.5% archive coverage. "
            f"Of these layouts, {safe_count} ({100 * safe_count / len(records):.1f}%) were strictly feasible under the "
            "implemented boundary, overlap, entrance and road-exclusion constraints."
        ),
        "",
        (
            f"The best overall layout achieved a combined score of {best.combined:.3f}, with safety "
            f"{best.safety:.3f}, operational efficiency {best.efficiency:.3f}, and adaptability {best.adaptability:.3f}. "
            f"Across the full elite set, median scores were safety {safety['median']:.3f}, efficiency "
            f"{efficiency['median']:.3f}, and adaptability {adaptability['median']:.3f}; the 25th-75th percentile ranges "
            f"were {safety['p25']:.3f}-{safety['p75']:.3f}, {efficiency['p25']:.3f}-{efficiency['p75']:.3f}, and "
            f"{adaptability['p25']:.3f}-{adaptability['p75']:.3f}, respectively."
        ),
        "",
        "Recommended layout showcase:",
    ]
    for index, (label, record, reason) in enumerate(selected, start=1):
        lines.append(
            f"{index}. {label}: {record.layout_id}, combined {record.combined:.3f}; "
            f"S/E/A = {record.safety:.3f}/{record.efficiency:.3f}/{record.adaptability:.3f}. {reason}."
        )
    lines.append("")
    lines.append(
        "For the paper, use the overview table for run settings and headline performance, "
        "the range table to report the spread of generated solutions, the behavioural-space figure to demonstrate "
        "quality-diversity coverage, and the six-layout showcase figure as the visual case-study result."
    )
    path = output_dir / "suggested_case_study_text.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


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
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(results_dir)
    table_paths = write_tables(records, results_dir, output_dir)
    selected = unique_select(records)
    showcase_path = write_showcase_table(selected, output_dir)
    figure_paths = save_figures(records, selected, results_dir, output_dir)
    narrative_path = write_narrative(records, selected, output_dir)

    print(f"Analysed {len(records)} layouts")
    print(f"Strictly feasible: {sum(1 for record in records if record.strict_safe)}")
    print(f"Output directory: {output_dir}")
    for path in list(table_paths.values()) + [showcase_path] + list(figure_paths.values()) + [narrative_path]:
        print(path)


if __name__ == "__main__":
    main()
