#!/usr/bin/env python3
"""
Run optimisation-only construction site layout search with Pure NSGA-II.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.cli_utils import make_run_output_dir, resolve_facility_types
from core.config import NSGA2Config, SiteConfig
from core.nsga2_algorithm import PureNSGA2Optimizer
from core.visualization import create_nsga2_visualizations, export_nsga2_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Pure NSGA-II as the optimisation-only baseline."
    )
    parser.add_argument("--facilities", type=int, default=5, help="Number of facilities to generate when --facility-mix is not used.")
    parser.add_argument("--facility-mix", type=str, default=None, help="Explicit mix, e.g. core=1,crane=1,storage=2,office=1.")
    parser.add_argument("--population", type=int, default=200, help="NSGA-II population size.")
    parser.add_argument("--generations", type=int, default=300, help="Number of NSGA-II generations.")
    parser.add_argument("--tournament-size", type=int, default=3, help="Tournament size for parent selection.")
    parser.add_argument("--crossover-rate", type=float, default=0.8, help="Probability of crossover.")
    parser.add_argument("--mutation-rate", type=float, default=0.4, help="Mutation rate used by the NSGA-II config.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible facility mix and search.")
    parser.add_argument("--output", type=str, default="results", help="Base output directory; each run creates a timestamped subfolder.")
    parser.add_argument("--no-repair", action="store_true", help="Disable the constraint-repair step before NSGA-II evaluates layouts.")
    parser.add_argument("--visualize", action="store_true", help="Export layout files and PNG visualizations.")
    parser.add_argument("--export-count", type=int, default=30, help="Number of Pareto layouts to export when --visualize is used.")
    parser.add_argument("--export-all", action="store_true", help="Export every Pareto-front layout when --visualize is used.")
    parser.add_argument("--no-export-pngs", action="store_true", help="Skip individual layout PNG export when --visualize is used.")
    parser.add_argument("--allow-infeasible-export", action="store_true", help="Allow infeasible Pareto layouts in exported JSON/PNG previews.")
    return parser


def population_objective_summary(individuals):
    if not individuals:
        return {"safety": 0.0, "efficiency": 0.0, "adaptability": 0.0}

    objectives = np.array([ind.objectives for ind in individuals])
    return {
        "safety": float(np.mean(objectives[:, 0])),
        "efficiency": float(np.mean(objectives[:, 1])),
        "adaptability": float(np.mean(objectives[:, 2])),
    }


def individual_to_summary(individual):
    return {
        "objectives": {
            "safety": float(individual.objectives[0]),
            "efficiency": float(individual.objectives[1]),
            "adaptability": float(individual.objectives[2]),
            "combined_score": float(sum(individual.objectives) / 3.0),
        },
        "feasible": bool(individual.feasible),
        "violations": list(individual.violations),
    }


def main() -> None:
    args = build_parser().parse_args()

    facility_types = resolve_facility_types(args.facilities, seed=args.seed, facility_mix=args.facility_mix)
    site_config = SiteConfig(facility_count=len(facility_types), seed=args.seed)
    nsga2_config = NSGA2Config(
        population_size=args.population,
        generations=args.generations,
        tournament_size=args.tournament_size,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
    )
    output_dir = make_run_output_dir(args.output, "nsga2")

    optimizer = PureNSGA2Optimizer(
        site_config,
        facility_types,
        nsga2_config,
        repair_layouts=not args.no_repair,
    )
    results = optimizer.run(population_size=args.population, generations=args.generations)

    exported_layouts = 0
    if args.visualize:
        max_layouts = None if args.export_all else args.export_count
        exported_layouts = export_nsga2_results(
            results,
            site_config,
            output_dir,
            max_layouts=max_layouts,
            export_pngs=not args.no_export_pngs,
            feasible_only=not args.allow_infeasible_export,
        )
        create_nsga2_visualizations(results, site_config, output_dir)

    population = results["population"]
    pareto_front = results["pareto_front"]
    best = max(pareto_front or population, key=lambda ind: sum(ind.objectives), default=None)
    results_json = {
        "algorithm": "Pure NSGA-II",
        "configuration": {
            "facility_count": len(facility_types),
            "facility_types": facility_types,
            "population": args.population,
            "generations": args.generations,
            "tournament_size": args.tournament_size,
            "crossover_rate": args.crossover_rate,
            "mutation_rate": args.mutation_rate,
            "seed": args.seed,
            "repair_layouts": not args.no_repair,
            "facility_mix_overrode_facilities": bool(args.facility_mix),
        },
        "statistics": {
            "population_size": len(population),
            "pareto_front_size": len(pareto_front),
            "total_fronts": len(results["fronts"]),
            "feasible_count": sum(1 for ind in population if ind.feasible),
            "population_objective_averages": population_objective_summary(population),
            "pareto_objective_averages": population_objective_summary(pareto_front),
            "evaluations": int(results["evaluations"]),
            "runtime_seconds": float(results["runtime"]),
        },
        "best_solution": individual_to_summary(best) if best else None,
        "exported_layouts": int(exported_layouts),
    }

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)

    print(f"\nNSGA-II results saved to: {output_dir}")


if __name__ == "__main__":
    main()
