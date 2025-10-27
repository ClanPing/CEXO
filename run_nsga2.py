#!/usr/bin/env python3
"""
Main Pure NSGA-II Construction Site Layout Optimizer
====================================================

Main execution script for the pure NSGA-II algorithm focusing on 
multi-objective optimization without behavioral descriptors.

Usage:
    python run_pure_nsga2.py --facilities 6 --population 200 --generations 300 --visualize
"""

import sys
import os
import argparse
import time
from pathlib import Path

from core.config import SiteConfig, NSGA2Config, generate_facility_mix
from core.nsga2_algorithm import PureNSGA2Optimizer, calculate_nsga2_metrics
from core.visualization import create_nsga2_visualizations, export_nsga2_results

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Pure NSGA-II Construction Site Layout Optimizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Core parameters
    parser.add_argument("--facilities", type=int, default=6, choices=range(3, 9),
                       help="Number of facilities (3-8)")
    parser.add_argument("--population", type=int, default=200,
                       help="Population size")
    parser.add_argument("--generations", type=int, default=300,
                       help="Number of generations")
    
    # NSGA-II specific parameters
    parser.add_argument("--tournament-size", type=int, default=3,
                       help="Tournament size for selection")
    parser.add_argument("--crossover-rate", type=float, default=0.8,
                       help="Crossover rate")
    parser.add_argument("--mutation-rate", type=float, default=0.4,
                       help="Mutation rate")
    
    # Site configuration
    parser.add_argument("--min-entrances", type=int, default=1, choices=range(1, 7),
                       help="Minimum number of entrances (1-6)")
    parser.add_argument("--max-entrances", type=int, default=3, choices=range(1, 7),
                       help="Maximum number of entrances (1-6)")
    parser.add_argument("--margin", type=float, default=0.08,
                       help="Boundary margin")
    parser.add_argument("--entrance-clearance", type=float, default=0.15,
                       help="Entrance clearance distance")
    parser.add_argument("--crane-safety", type=float, default=0.30,
                       help="Crane safety distance")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="nsga2_output",
                       help="Output directory")
    parser.add_argument("--export-count", type=int, default=25,
                       help="Number of layouts to export")
    parser.add_argument("--visualize", action="store_true",
                       help="Create visualizations")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--test", action="store_true",
                       help="Run quick test")
    
    args = parser.parse_args()
    
    # Test mode adjustments
    if args.test:
        print("RUNNING IN TEST MODE")
        args.population = 100
        args.generations = 150
        args.export_count = 15
    
    # Create configurations
    site_config = SiteConfig(
        facility_count=args.facilities,
        boundary_margin=args.margin,
        seed=args.seed,
        min_entrances=args.min_entrances,
        max_entrances=args.max_entrances,
        entrance_clearance=args.entrance_clearance,
        crane_safety_distance=args.crane_safety
    )
    
    nsga2_config = NSGA2Config(
        population_size=args.population,
        generations=args.generations,
        tournament_size=args.tournament_size,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate
    )
    
    facility_types = generate_facility_mix(args.facilities, args.seed)
    
    print("=" * 80)
    print("PURE NSGA-II OPTIMIZATION")
    print("Construction Site Layout Optimization")
    print("=" * 80)
    print(f"Objectives: Safety & Compliance, Operational Efficiency, Layout Adaptability")
    print(f"Population: {args.population}")
    print(f"Generations: {args.generations}")
    print(f"Facilities: {args.facilities} ({', '.join(facility_types)})")
    print(f"Algorithm: Multi-objective optimization without behavioral descriptors")
    
    # Run optimization
    start_time = time.time()
    
    optimizer = PureNSGA2Optimizer(site_config, facility_types, nsga2_config)
    results = optimizer.run()
    
    runtime = time.time() - start_time
    population = results["population"]
    pareto_front = results["pareto_front"]
    
    # Results
    print(f"\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Population size: {len(population)}")
    print(f"Pareto front size: {len(pareto_front)}")
    print(f"Total fronts: {len(results['fronts'])}")
    
    feasible_count = sum(1 for ind in population if ind.feasible)
    print(f"Feasible solutions: {feasible_count}/{len(population)} ({100*feasible_count/len(population):.1f}%)")
    
    if population:
        import numpy as np
        population_objectives = np.array([ind.objectives for ind in population])
        print(f"Population averages:")
        print(f"  Safety: {np.mean(population_objectives[:, 0]):.3f}")
        print(f"  Efficiency: {np.mean(population_objectives[:, 1]):.3f}")
        print(f"  Adaptability: {np.mean(population_objectives[:, 2]):.3f}")
    
    if pareto_front:
        import numpy as np
        pareto_objectives = np.array([ind.objectives for ind in pareto_front])
        print(f"Pareto front averages:")
        print(f"  Safety: {np.mean(pareto_objectives[:, 0]):.3f}")
        print(f"  Efficiency: {np.mean(pareto_objectives[:, 1]):.3f}")
        print(f"  Adaptability: {np.mean(pareto_objectives[:, 2]):.3f}")
    
    print(f"Total evaluations: {results['evaluations']:,}")
    print(f"Runtime: {results['runtime']:.2f} seconds")
    
    # NSGA-II Performance Evaluation
    print(f"\n" + "=" * 60)
    print("NSGA-II PERFORMANCE EVALUATION")
    print("=" * 60)
    
    metrics = calculate_nsga2_metrics(results, site_config)
    
    print(f"Quality Metrics:")
    print(f"  Hypervolume: {metrics['hypervolume']:.4f}")
    print(f"  Spread: {metrics['spread']:.4f}")
    print(f"  Spacing: {metrics['spacing_metric']:.4f}")
    print(f"  Pareto Coverage: {metrics['pareto_coverage']:.4f}")
    
    print(f"Solution Quality:")
    print(f"  Feasible Pareto Ratio: {metrics['feasible_pareto_ratio']:.3f}")
    print(f"  Avg Feasible Quality: {metrics['avg_feasible_quality']:.3f}")
    
    # Export and visualize
    if pareto_front:
        print(f"\nExporting results to {args.output_dir}...")
        
        output_path = Path(args.output_dir)
        output_path.mkdir(exist_ok=True)
        
        exported = export_nsga2_results(
            results, site_config, args.output_dir,
            min(args.export_count, len(pareto_front))
        )
        
        if args.visualize:
            create_nsga2_visualizations(results, site_config, args.output_dir)
            print("✓ NSGA-II visualizations created")
        
        # Export detailed metrics
        import json
        def convert_numpy(obj):
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        detailed_metrics = {
            "nsga2_performance": {
                "algorithm": "Pure NSGA-II",
                "population_size": len(population),
                "pareto_front_size": len(pareto_front),
                "total_fronts": len(results["fronts"]),
                "evaluations": results["evaluations"],
                "runtime_seconds": results["runtime"],
                "evaluations_per_second": results["evaluations"] / results["runtime"]
            },
            "quality_metrics": metrics,
            "configuration": {
                "population_size": args.population,
                "generations": args.generations,
                "tournament_size": args.tournament_size,
                "crossover_rate": args.crossover_rate,
                "mutation_rate": args.mutation_rate
            }
        }
        
        metrics_json = json.loads(json.dumps(detailed_metrics, default=convert_numpy))
        with open(os.path.join(args.output_dir, "nsga2_detailed_metrics.json"), 'w') as f:
            json.dump(metrics_json, f, indent=2)
        
        print(f"✓ SUCCESS: Exported {exported} Pareto-optimal layouts to {args.output_dir}/")
        print(f"✓ Detailed performance metrics saved")
        
        # Compare with MAP-Elites approach
        print(f"\nNSGA-II vs MAP-Elites Comparison:")
        print(f"  Focus: Pure multi-objective optimization vs Behavioral diversity")
        print(f"  Output: {len(pareto_front)} Pareto solutions vs Archive grid coverage")
        print(f"  Strength: Direct Pareto optimization vs Diverse solution space")
        print(f"  Use case: Best trade-off solutions vs Comprehensive layout variety")
        
    else:
        print("\n✗ No Pareto-optimal solutions found!")
        print("Try: --margin 0.1 --generations 500 --population 300")
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        print(f"\nPure NSGA-II finished with exit code: {exit_code}")
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)