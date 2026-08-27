#!/usr/bin/env python3
"""
Scalability Analysis for CEXO Algorithm
========================================

Tests algorithm scalability across different problem sizes:
- Number of facilities (5, 10, 15, 20, 25)
- Measures: runtime, memory, convergence quality
- Analyzes scaling behavior and computational complexity
"""

import os
import sys
import json
import time
import psutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Set environment variables before torch import
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONHASHSEED'] = '0'

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.config import (
    SiteConfig,
    MapElitesConfig,
    AutoencoderConfig,
    generate_facility_mix
)
from core.mapelites_with_autoencoder import MapElitesWithAutoencoder
from core.layout_autoencoder import set_random_seeds


def get_memory_usage():
    """Get current process memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def run_scalability_test(facilities, iterations, seed):
    """Run algorithm and measure performance metrics"""
    print(f"\n{'='*80}")
    print(f"Testing: {facilities} facilities")
    print(f"{'='*80}")
    
    # Set seeds
    set_random_seeds(seed)
    
    # Setup
    site_config = SiteConfig(facility_count=facilities, seed=seed)
    facility_types = generate_facility_mix(facilities, seed=seed)
    
    mapelites_config = MapElitesConfig(
        grid_size=(20, 20),
        iterations=iterations,
        initial_population=500
    )
    
    autoencoder_config = AutoencoderConfig(
        use_learned_descriptors=True,
        latent_dim=2,
        pretrain_iterations=min(2000, iterations // 5),
        training_frequency=min(1000, iterations // 10),
        seed=seed
    )
    
    # Measure initial memory
    initial_memory = get_memory_usage()
    
    # Run
    start_time = time.time()
    algorithm = MapElitesWithAutoencoder(
        facility_types=facility_types,
        site_config=site_config,
        mapelites_config=mapelites_config,
        autoencoder_config=autoencoder_config
    )
    
    result = algorithm.run()
    runtime = time.time() - start_time
    
    # Measure final memory
    final_memory = get_memory_usage()
    memory_used = final_memory - initial_memory
    
    # Calculate metrics
    archive = result['archive']
    coverage = len(archive.archive) / archive.total_cells * 100
    all_inds = archive.get_all_individuals()
    
    if all_inds:
        avg_fitness = np.mean([archive.calculate_scalar_fitness(ind) for ind in all_inds])
        best_fitness = max([archive.calculate_scalar_fitness(ind) for ind in all_inds])
        
        # Calculate objective averages
        avg_safety = np.mean([ind.objectives[0] for ind in all_inds])
        avg_efficiency = np.mean([ind.objectives[1] for ind in all_inds])
        avg_adaptability = np.mean([ind.objectives[2] for ind in all_inds])
    else:
        avg_fitness = best_fitness = 0
        avg_safety = avg_efficiency = avg_adaptability = 0
    
    metrics = {
        'facilities': facilities,
        'runtime': runtime,
        'memory_mb': memory_used,
        'coverage': coverage,
        'num_solutions': len(all_inds),
        'avg_fitness': avg_fitness,
        'best_fitness': best_fitness,
        'avg_safety': avg_safety,
        'avg_efficiency': avg_efficiency,
        'avg_adaptability': avg_adaptability,
        'iterations': iterations
    }
    
    print(f"✓ Runtime: {runtime:.1f}s, Memory: {memory_used:.1f}MB")
    print(f"  Coverage: {coverage:.2f}%, Solutions: {len(all_inds)}")
    print(f"  Best Fitness: {best_fitness:.3f}")
    
    return metrics


def plot_scalability_results(results, output_dir):
    """Create comprehensive scalability visualization"""
    os.makedirs(output_dir, exist_ok=True)
    
    facilities = [r['facilities'] for r in results]
    runtimes = [r['runtime'] for r in results]
    memory = [r['memory_mb'] for r in results]
    coverage = [r['coverage'] for r in results]
    best_fitness = [r['best_fitness'] for r in results]
    num_solutions = [r['num_solutions'] for r in results]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('CEXO Scalability Analysis', fontsize=16, fontweight='bold')
    
    # Runtime vs Facilities
    ax = axes[0, 0]
    ax.plot(facilities, runtimes, 'b-o', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Facilities', fontsize=11)
    ax.set_ylabel('Runtime (seconds)', fontsize=11)
    ax.set_title('Computational Time', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(facilities, runtimes, 2)
    p = np.poly1d(z)
    ax.plot(facilities, p(facilities), 'r--', alpha=0.5, label='Polynomial fit')
    ax.legend()
    
    # Memory vs Facilities
    ax = axes[0, 1]
    ax.plot(facilities, memory, 'g-s', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Facilities', fontsize=11)
    ax.set_ylabel('Memory Usage (MB)', fontsize=11)
    ax.set_title('Memory Consumption', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Coverage vs Facilities
    ax = axes[0, 2]
    ax.plot(facilities, coverage, 'm-^', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Facilities', fontsize=11)
    ax.set_ylabel('Coverage (%)', fontsize=11)
    ax.set_title('Archive Coverage', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Best Fitness vs Facilities
    ax = axes[1, 0]
    ax.plot(facilities, best_fitness, 'r-o', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Facilities', fontsize=11)
    ax.set_ylabel('Best Fitness', fontsize=11)
    ax.set_title('Solution Quality', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Number of Solutions vs Facilities
    ax = axes[1, 1]
    ax.plot(facilities, num_solutions, 'c-d', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Facilities', fontsize=11)
    ax.set_ylabel('Number of Solutions', fontsize=11)
    ax.set_title('Archive Size', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Efficiency: Solutions per Second
    ax = axes[1, 2]
    efficiency = [n / t for n, t in zip(num_solutions, runtimes)]
    ax.plot(facilities, efficiency, 'orange', marker='p', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Facilities', fontsize=11)
    ax.set_ylabel('Solutions per Second', fontsize=11)
    ax.set_title('Computational Efficiency', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'scalability_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved scalability plot: {plot_path}")
    plt.close()


def analyze_complexity(results, output_dir):
    """Analyze computational complexity"""
    facilities = np.array([r['facilities'] for r in results])
    runtimes = np.array([r['runtime'] for r in results])
    
    # Fit different complexity models
    # Linear: O(n)
    linear_coef = np.polyfit(facilities, runtimes, 1)
    linear_fit = np.poly1d(linear_coef)
    linear_r2 = 1 - (np.sum((runtimes - linear_fit(facilities))**2) / 
                     np.sum((runtimes - np.mean(runtimes))**2))
    
    # Quadratic: O(n²)
    quad_coef = np.polyfit(facilities, runtimes, 2)
    quad_fit = np.poly1d(quad_coef)
    quad_r2 = 1 - (np.sum((runtimes - quad_fit(facilities))**2) / 
                   np.sum((runtimes - np.mean(runtimes))**2))
    
    complexity_analysis = {
        'linear_r2': float(linear_r2),
        'quadratic_r2': float(quad_r2),
        'best_fit': 'quadratic' if quad_r2 > linear_r2 else 'linear',
        'linear_coefficients': linear_coef.tolist(),
        'quadratic_coefficients': quad_coef.tolist()
    }
    
    print(f"\n{'='*80}")
    print("COMPLEXITY ANALYSIS")
    print(f"{'='*80}")
    print(f"Linear fit R²: {linear_r2:.4f}")
    print(f"Quadratic fit R²: {quad_r2:.4f}")
    print(f"Best fit model: {complexity_analysis['best_fit']}")
    print("="*80)
    
    return complexity_analysis


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='CEXO Scalability Analysis')
    parser.add_argument('--min-facilities', type=int, default=3, help='Minimum facilities')
    parser.add_argument('--max-facilities', type=int, default=8, help='Maximum facilities (practical limit ~8-10)')
    parser.add_argument('--step', type=int, default=1, help='Step size')
    parser.add_argument('--iterations', type=int, default=5000, help='Iterations per run')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='results/scalability', help='Output directory')
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    facility_counts = range(args.min_facilities, args.max_facilities + 1, args.step)
    
    print("="*80)
    print("CEXO SCALABILITY ANALYSIS")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Facility range: {args.min_facilities} to {args.max_facilities} (step: {args.step})")
    print(f"  Iterations per run: {args.iterations}")
    print(f"  Seed: {args.seed}")
    print(f"  Tests to run: {len(list(facility_counts))}")
    print("="*80)
    
    results = []
    start_time = time.time()
    
    for facilities in facility_counts:
        metrics = run_scalability_test(facilities, args.iterations, args.seed)
        results.append(metrics)
    
    total_time = time.time() - start_time
    
    # Analyze complexity
    complexity = analyze_complexity(results, args.output)
    
    # Save results
    json_path = os.path.join(args.output, 'scalability_results.json')
    with open(json_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'min_facilities': args.min_facilities,
                'max_facilities': args.max_facilities,
                'step': args.step,
                'iterations': args.iterations,
                'seed': args.seed
            },
            'results': results,
            'complexity_analysis': complexity,
            'total_runtime': total_time
        }, f, indent=2)
    print(f"\n✓ Saved results: {json_path}")
    
    # Create visualizations
    plot_scalability_results(results, args.output)
    
    print(f"\n{'='*80}")
    print("SCALABILITY ANALYSIS COMPLETE")
    print(f"Total runtime: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Results saved to: {args.output}/")
    print("="*80)


if __name__ == "__main__":
    main()
