#!/usr/bin/env python3
"""
Run diversity-only construction site layout search with Pure MAP-Elites.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.cli_utils import make_run_output_dir, resolve_facility_types
from core.config import MapElitesConfig, SiteConfig
from core.mapelites_algorithm import PureMapElitesOptimizer
from core.visualization import create_pure_mapelites_visualizations, export_mapelites_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Pure MAP-Elites as the diversity-only baseline."
    )
    parser.add_argument("--facilities", type=int, default=5, help="Number of facilities to generate when --facility-mix is not used.")
    parser.add_argument("--facility-mix", type=str, default=None, help="Explicit mix, e.g. core=1,crane=1,storage=2,office=1.")
    parser.add_argument("--iterations", type=int, default=15000, help="Number of MAP-Elites iterations.")
    parser.add_argument("--initial-pop", type=int, default=500, help="Initial random/targeted population size.")
    parser.add_argument("--grid-size", type=int, default=20, help="Square MAP-Elites archive grid size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible facility mix and search.")
    parser.add_argument("--output", type=str, default="results", help="Base output directory; each run creates a timestamped subfolder.")
    parser.add_argument("--visualize", action="store_true", help="Export layout files and PNG visualizations.")
    parser.add_argument("--export-count", type=int, default=30, help="Number of top layouts to export when --visualize is used.")
    parser.add_argument("--export-all", action="store_true", help="Export every occupied archive layout when --visualize is used.")
    parser.add_argument("--no-export-pngs", action="store_true", help="Skip individual layout PNG export when --visualize is used.")
    return parser


def individual_to_summary(individual, archive):
    if individual is None:
        return None

    return {
        "objectives": {
            "safety": float(individual.objectives[0]),
            "efficiency": float(individual.objectives[1]),
            "adaptability": float(individual.objectives[2]),
            "scalar_fitness": float(getattr(individual, "scalar_fitness", archive.calculate_scalar_fitness(individual))),
        },
        "behaviors": {
            "bd1": float(individual.behaviors[0]),
            "bd2": float(individual.behaviors[1]),
        },
        "feasible": bool(individual.feasible),
        "violations": list(individual.violations),
    }


def main() -> None:
    args = build_parser().parse_args()

    facility_types = resolve_facility_types(args.facilities, seed=args.seed, facility_mix=args.facility_mix)
    site_config = SiteConfig(facility_count=len(facility_types), seed=args.seed)
    mapelites_config = MapElitesConfig(
        grid_size=(args.grid_size, args.grid_size),
        iterations=args.iterations,
        initial_population=args.initial_pop,
    )
    output_dir = make_run_output_dir(args.output, "mapelites")

    optimizer = PureMapElitesOptimizer(site_config, facility_types, mapelites_config)
    results = optimizer.run(iterations=args.iterations, initial_population=args.initial_pop)
    archive = results["archive"]
    stats = archive.get_stats()
    best = archive.get_best_individual()

    exported_layouts = 0
    if args.visualize:
        max_layouts = None if args.export_all else args.export_count
        exported_layouts = export_mapelites_results(
            archive,
            site_config,
            output_dir,
            max_layouts=max_layouts,
            export_pngs=not args.no_export_pngs,
        )
        create_pure_mapelites_visualizations(archive, site_config, output_dir)

    results_json = {
        "algorithm": "Pure MAP-Elites",
        "configuration": {
            "facility_count": len(facility_types),
            "facility_types": facility_types,
            "iterations": args.iterations,
            "initial_population": args.initial_pop,
            "grid_size": [args.grid_size, args.grid_size],
            "seed": args.seed,
            "facility_mix_overrode_facilities": bool(args.facility_mix),
        },
        "statistics": {
            "coverage": int(stats["coverage"]),
            "coverage_percentage": float(stats["coverage_pct"]),
            "total_individuals": int(stats["total_individuals"]),
            "safety_feasible_count": int(stats["safety_feasible_count"]),
            "avg_safety": float(stats["avg_safety"]),
            "avg_efficiency": float(stats["avg_efficiency"]),
            "avg_adaptability": float(stats["avg_adaptability"]),
            "avg_scalar_fitness": float(stats["avg_scalar_fitness"]),
            "best_scalar_fitness": float(stats["best_scalar_fitness"]),
            "runtime_seconds": float(results["runtime"]),
        },
        "best_solution": individual_to_summary(best, archive),
        "exported_layouts": int(exported_layouts),
    }

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)

    print(f"\nMAP-Elites results saved to: {output_dir}")


if __name__ == "__main__":
    main()
