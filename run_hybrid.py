#!/usr/bin/env python3
"""
Main MAP-Elites + NSGA-II Construction Site Layout Optimizer
===========================================================

Main execution script for the combined MAP-Elites + NSGA-II algorithm.
Combines behavioral diversity exploration with multi-objective optimization.

Usage:
    python run_mapelites_nsga2.py --facilities 6 --iterations 15000 --visualize
"""

import sys
import os
import argparse
import time
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import SiteConfig, MapElitesConfig, generate_facility_mix
from mapelites_algorithm import MapElitesNSGA2Optimizer, evaluate_mapelites_performance
from visualization import create_mapelites_visualizations, export_mapelites_results
from behavioral_descriptors import analyze_behavioral_regions

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="MAP-Elites + NSGA-II Construction Site Layout Optimizer",
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
    parser.add_argument("--pareto-size", type=int, default=12, choices=range(5, 26),
                       help="Pareto front size per cell (5-25)")
    
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
    parser.add_argument("--output-dir", type=str, default="mapelites_output",
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
        args.iterations = 3000
        args.init_pop = 300
        args.grid_size = 15
        args.pareto_size = 8
        args.export_count = 12
    
    # Create configurations
    site_config = SiteConfig(
        facility_count=args.facilities,
        boundary_margin=args.margin,
        seed=args.seed,
        min_entrances=args.min_entrances,
        max_entrances=args.max_entrances,
        pareto_size=args.pareto_size,
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
    print("MAP-ELITES + 3-OBJECTIVE NSGA-II")
    print("Construction Site Layout Optimization")
    print("=" * 80)
    print(f"Objectives: Safety & Compliance, Operational Efficiency, Layout Adaptability")
    print(f"Behavioral Space: Compactness vs Spread × Worker-Operational Separation")
    print(f"Archive: {args.grid_size}×{args.grid_size} = {args.grid_size**2:,} cells")
    print(f"Facilities: {args.facilities} ({', '.join(facility_types)})")
    print(f"Evolution: {args.iterations:,} iterations")
    
    # Run optimization
    start_time = time.time()
    
    optimizer = MapElitesNSGA2Optimizer(site_config, facility_types, mapelites_config)
    results = optimizer.run()
    
    runtime = time.time() - start_time
    archive = results["archive"]
    stats = results["stats"]
    
    # Results
    print(f"\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Archive coverage: {stats['coverage']:,}/{archive.total_cells:,} ({stats['coverage_pct']:.3f}%)")
    print(f"Total individuals: {stats['total_individuals']:,}")
    print(f"Safety feasible (≥0.7): {stats['safety_feasible_count']:,}")
    print(f"Average objectives:")
    print(f"  Safety: {stats['avg_safety']:.3f}")
    print(f"  Efficiency: {stats['avg_efficiency']:.3f}")
    print(f"  Adaptability: {stats['avg_adaptability']:.3f}")
    print(f"Total evaluations: {archive.evaluations:,}")
    print(f"Runtime: {runtime:.2f} seconds")
    
    # Enhanced MAP-Elites evaluation
    print(f"\n" + "=" * 60)
    print("MAP-ELITES EVALUATION")
    print("=" * 60)
    
    evaluation_results = evaluate_mapelites_performance(archive, site_config)
    
    print(f"Coverage Analysis:")
    print(f"  Cells filled: {evaluation_results['coverage_metrics']['cells_filled']:,}/{evaluation_results['coverage_metrics']['total_cells']:,}")
    print(f"  Coverage: {evaluation_results['coverage_metrics']['coverage_percentage']:.2f}%")
    print(f"  Avg solutions per cell: {evaluation_results['coverage_metrics']['average_individuals_per_cell']:.2f}")
    
    print(f"\nBehavioral Diversity:")
    print(f"  BD1 range: {evaluation_results['diversity_metrics']['behavioral_range_bd1']:.3f}")
    print(f"  BD2 range: {evaluation_results['diversity_metrics']['behavioral_range_bd2']:.3f}")
    print(f"  Diversity score: {evaluation_results['summary']['behavioral_diversity_score']:.3f}")
    
    print(f"\nMAP-Elites Effectiveness: {evaluation_results['summary']['mapelites_effectiveness']:.3f}")
    
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
        output_path.mkdir(exist_ok=True)
        
        exported = export_mapelites_results(
            archive, site_config, args.output_dir,
            min(args.export_count, stats['safety_feasible_count'])
        )
        
        if args.visualize:
            create_mapelites_visualizations(archive, site_config, args.output_dir)
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
        with open(os.path.join(args.output_dir, "mapelites_evaluation.json"), 'w') as f:
            json.dump(evaluation_json, f, indent=2)
        
        with open(os.path.join(args.output_dir, "behavioral_analysis.json"), 'w') as f:
            json.dump(behavioral_analysis, f, indent=2)
        
        print(f"✓ SUCCESS: Exported {exported} layouts to {args.output_dir}/")
        print(f"✓ Detailed evaluation metrics saved")
        
    else:
        print("\n✗ No safety-feasible layouts found!")
        print("Try: --margin 0.1 --iterations 20000 --entrance-clearance 0.12")
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        print(f"\nMAP-Elites + NSGA-II finished with exit code: {exit_code}")
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)