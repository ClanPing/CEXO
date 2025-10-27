#!/usr/bin/env python3
"""
Main Pure MAP-Elites Construction Site Layout Optimizer
=======================================================

Main execution script for the pure MAP-Elites algorithm focusing on 
behavioral diversity exploration with scalar fitness function.

Usage:
    python run_pure_mapelites.py --facilities 6 --iterations 15000 --visualize
"""

import sys
import os
import argparse
import time
from pathlib import Path

from core.config import SiteConfig, MapElitesConfig, generate_facility_mix
from core.mapelites_algorithm import (
    PureMapElitesOptimizer,
    evaluate_pure_mapelites_performance,
)
from core.visualization import (
    create_pure_mapelites_visualizations,
    export_mapelites_results,
)
from core.behavioral_descriptors import analyze_behavioral_regions

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Pure MAP-Elites Construction Site Layout Optimizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Core parameters
    parser.add_argument("--facilities", type=int, default=6, choices=range(3, 9),
                       help="Number of facilities (3-8)")
    parser.add_argument("--iterations", type=int, default=15000,
                       help="Evolution iterations")
    parser.add_argument("--init-pop", type=int, default=500,
                       help="Initial population size")
    
    # Archive configuration
    parser.add_argument("--grid-size", type=int, default=20, choices=range(10, 31),
                       help="2D grid size per dimension (10-30)")
    
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
    
    # Scalar fitness weighting
    parser.add_argument("--safety-weight", type=float, default=0.5,
                       help="Weight for safety objective in scalar fitness")
    parser.add_argument("--efficiency-weight", type=float, default=0.3,
                       help="Weight for efficiency objective in scalar fitness")
    parser.add_argument("--adaptability-weight", type=float, default=0.2,
                       help="Weight for adaptability objective in scalar fitness")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="output/mapelites",
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
    
    # Validate weights
    total_weight = args.safety_weight + args.efficiency_weight + args.adaptability_weight
    if abs(total_weight - 1.0) > 0.01:
        print(f"Warning: Objective weights sum to {total_weight:.3f}, normalizing to 1.0")
        args.safety_weight /= total_weight
        args.efficiency_weight /= total_weight
        args.adaptability_weight /= total_weight
    
    # Test mode adjustments
    if args.test:
        print("RUNNING IN TEST MODE")
        args.iterations = 3000
        args.init_pop = 300
        args.grid_size = 15
        args.export_count = 12
    
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
    
    mapelites_config = MapElitesConfig(
        grid_size=(args.grid_size, args.grid_size),
        iterations=args.iterations,
        initial_population=args.init_pop
    )
    
    facility_types = generate_facility_mix(args.facilities, args.seed)
    
    print("=" * 80)
    print("PURE MAP-ELITES OPTIMIZATION")
    print("Construction Site Layout Optimization")
    print("=" * 80)
    print(f"Objectives: Safety & Compliance, Operational Efficiency, Layout Adaptability")
    print(f"Scalar Fitness: Safety({args.safety_weight:.1f}) + Efficiency({args.efficiency_weight:.1f}) + Adaptability({args.adaptability_weight:.1f})")
    print(f"Behavioral Space: Compactness vs Spread × Worker-Operational Separation")
    print(f"Archive: {args.grid_size}×{args.grid_size} = {args.grid_size**2:,} cells (one solution per cell)")
    print(f"Facilities: {args.facilities} ({', '.join(facility_types)})")
    print(f"Evolution: {args.iterations:,} iterations")
    
    # Run optimization
    start_time = time.time()
    
    optimizer = PureMapElitesOptimizer(site_config, facility_types, mapelites_config)
    
    # Override scalar fitness weights if specified
    if args.safety_weight != 0.5 or args.efficiency_weight != 0.3 or args.adaptability_weight != 0.2:
        def custom_scalar_fitness(individual):
            safety, efficiency, adaptability = individual.objectives
            weights = [args.safety_weight, args.efficiency_weight, args.adaptability_weight]
            scalar_fitness = sum(obj * weight for obj, weight in zip(individual.objectives, weights))
            
            # Apply feasibility bonus/penalty
            if individual.feasible:
                scalar_fitness += 0.1
            else:
                scalar_fitness *= 0.7
            
            return float(max(0.0, min(1.0, scalar_fitness)))
        
        optimizer.archive.calculate_scalar_fitness = custom_scalar_fitness
    
    results = optimizer.run()
    
    runtime = time.time() - start_time
    archive = results["archive"]
    stats = results["stats"]
    best_individual = results["best_individual"]
    
    # Results
    print(f"\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Archive coverage: {stats['coverage']:,}/{archive.total_cells:,} cells ({stats['coverage_pct']:.3f}%)")
    print(f"Total solutions: {stats['total_individuals']:,}")
    print(f"Safety feasible (≥0.7): {stats['safety_feasible_count']:,}")
    print(f"Average scalar fitness: {stats['avg_scalar_fitness']:.3f}")
    print(f"Best scalar fitness: {stats['best_scalar_fitness']:.3f}")
    print(f"Average objectives:")
    print(f"  Safety: {stats['avg_safety']:.3f}")
    print(f"  Efficiency: {stats['avg_efficiency']:.3f}")
    print(f"  Adaptability: {stats['avg_adaptability']:.3f}")
    print(f"Total evaluations: {archive.evaluations:,}")
    print(f"Runtime: {runtime:.2f} seconds")
    
    if best_individual:
        print(f"\nBest Individual:")
        print(f"  Scalar Fitness: {best_individual.scalar_fitness:.3f}")
        print(f"  Safety: {best_individual.objectives[0]:.3f}")
        print(f"  Efficiency: {best_individual.objectives[1]:.3f}")
        print(f"  Adaptability: {best_individual.objectives[2]:.3f}")
        print(f"  Behaviors: Spatial={best_individual.behaviors[0]:.3f}, Functional={best_individual.behaviors[1]:.3f}")
        print(f"  Feasible: {best_individual.feasible}")
    
    # Enhanced Pure MAP-Elites evaluation
    print(f"\n" + "=" * 60)
    print("PURE MAP-ELITES EVALUATION")
    print("=" * 60)
    
    evaluation_results = evaluate_pure_mapelites_performance(archive, site_config)
    
    print(f"Coverage Analysis:")
    print(f"  Cells filled: {evaluation_results['coverage_metrics']['cells_filled']:,}/{evaluation_results['coverage_metrics']['total_cells']:,}")
    print(f"  Coverage: {evaluation_results['coverage_metrics']['coverage_percentage']:.2f}%")
    
    print(f"\nBehavioral Diversity:")
    print(f"  BD1 range: {evaluation_results['diversity_metrics']['behavioral_range_bd1']:.3f}")
    print(f"  BD2 range: {evaluation_results['diversity_metrics']['behavioral_range_bd2']:.3f}")
    print(f"  Diversity score: {evaluation_results['summary']['behavioral_diversity_score']:.3f}")
    
    print(f"\nScalar Fitness Quality:")
    print(f"  Average fitness: {evaluation_results['quality_metrics']['average_scalar_fitness']:.3f}")
    print(f"  Best fitness: {evaluation_results['quality_metrics']['best_scalar_fitness']:.3f}")
    print(f"  Fitness range: {evaluation_results['quality_metrics']['best_scalar_fitness'] - evaluation_results['quality_metrics']['average_scalar_fitness']:.3f}")
    
    print(f"\nPure MAP-Elites Effectiveness: {evaluation_results['summary']['pure_mapelites_effectiveness']:.3f}")
    
    # Behavioral region analysis
    all_individuals = archive.get_all_individuals()
    behavioral_analysis = analyze_behavioral_regions(all_individuals)
    
    print(f"\nBehavioral Region Analysis:")
    for region_name, region_data in behavioral_analysis.items():
        print(f"  {region_name.replace('_', ' ').title()}: {region_data['count']} solutions ({region_data['percentage']:.1f}%)")
    
    # Export and visualize
    if stats['safety_feasible_count'] > 0:
        print(f"\nExporting results to {args.output_dir}...")
        
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        exported = export_mapelites_results(
            archive, site_config, args.output_dir,
            min(args.export_count, stats['safety_feasible_count'])
        )
        
        if args.visualize:
            create_pure_mapelites_visualizations(archive, site_config, args.output_dir)
            print("✓ Visualizations created")
        
        # Export detailed evaluation metrics
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
        
        evaluation_json = json.loads(json.dumps(evaluation_results, default=convert_numpy))
        with open(os.path.join(args.output_dir, "pure_mapelites_evaluation.json"), 'w') as f:
            json.dump(evaluation_json, f, indent=2)
        
        with open(os.path.join(args.output_dir, "behavioral_analysis.json"), 'w') as f:
            json.dump(behavioral_analysis, f, indent=2)
        
        # Save algorithm configuration
        algorithm_config = {
            "algorithm": "Pure MAP-Elites",
            "scalar_fitness_weights": {
                "safety": args.safety_weight,
                "efficiency": args.efficiency_weight,
                "adaptability": args.adaptability_weight
            },
            "grid_size": f"{args.grid_size}×{args.grid_size}",
            "iterations": args.iterations,
            "initial_population": args.init_pop,
            "facilities": args.facilities,
            "facility_types": facility_types,
            "runtime_seconds": runtime,
            "evaluations": archive.evaluations,
            "evaluations_per_second": archive.evaluations / runtime
        }
        
        with open(os.path.join(args.output_dir, "algorithm_config.json"), 'w') as f:
            json.dump(algorithm_config, f, indent=2)
        
        print(f"✓ SUCCESS: Exported {exported} layouts to {args.output_dir}/")
        print(f"✓ Detailed evaluation metrics saved")
        
        # Compare with other approaches
        print(f"\nPure MAP-Elites vs Other Approaches:")
        print(f"  Focus: Behavioral diversity with scalar fitness optimization")
        print(f"  Output: {stats['coverage']} diverse solutions across behavioral space")
        print(f"  Strength: Systematic exploration of layout variety")
        print(f"  Use case: Discovering novel layout patterns and design insights")
        print(f"  Archive: One best solution per behavioral niche")
        
    else:
        print("\n✗ No safety-feasible layouts found!")
        print("Try: --margin 0.1 --iterations 20000 --entrance-clearance 0.12")
        print("Or adjust fitness weights: --safety-weight 0.6 --efficiency-weight 0.25 --adaptability-weight 0.15")
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        print(f"\nPure MAP-Elites finished with exit code: {exit_code}")
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)