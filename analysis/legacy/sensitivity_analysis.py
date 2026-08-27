#!/usr/bin/env python3
"""
Sensitivity Analysis for CEXO Algorithm
========================================

Tests algorithm sensitivity to key parameters:
- Population size
- Training frequency
- Latent dimensions
- Pretrain iterations

Runs multiple configurations and analyzes impact on performance.
"""

import os
import sys
import json
import time
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


def run_single_config(config_name, facilities, iterations, initial_pop, 
                      pretrain_iter, train_freq, latent_dim, seed):
    """Run algorithm with specific configuration"""
    print(f"\n{'='*80}")
    print(f"Running: {config_name}")
    print(f"{'='*80}")
    
    # Set seeds
    set_random_seeds(seed)
    
    # Setup
    site_config = SiteConfig(facility_count=facilities, seed=seed)
    facility_types = generate_facility_mix(facilities, seed=seed)
    
    mapelites_config = MapElitesConfig(
        grid_size=(20, 20),
        iterations=iterations,
        initial_population=initial_pop
    )
    
    autoencoder_config = AutoencoderConfig(
        use_learned_descriptors=True,
        latent_dim=latent_dim,
        pretrain_iterations=pretrain_iter,
        training_frequency=train_freq,
        seed=seed
    )
    
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
    
    # Calculate metrics
    archive = result['archive']
    coverage = len(archive.archive) / archive.total_cells * 100
    all_inds = archive.get_all_individuals()
    
    if all_inds:
        avg_fitness = np.mean([archive.calculate_scalar_fitness(ind) for ind in all_inds])
        best_fitness = max([archive.calculate_scalar_fitness(ind) for ind in all_inds])
    else:
        avg_fitness = 0
        best_fitness = 0
    
    metrics = {
        'config_name': config_name,
        'coverage': coverage,
        'avg_fitness': avg_fitness,
        'best_fitness': best_fitness,
        'runtime': runtime,
        'num_solutions': len(all_inds)
    }
    
    print(f"✓ Coverage: {coverage:.2f}%, Best Fitness: {best_fitness:.3f}, Time: {runtime:.1f}s")
    
    return metrics


def analyze_population_size(facilities=5, iterations=5000, seed=42, output_dir="results/sensitivity"):
    """Test sensitivity to initial population size"""
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS: Population Size")
    print("="*80)
    
    population_sizes = [100, 200, 500, 1000, 2000]
    results = []
    
    for pop_size in population_sizes:
        metrics = run_single_config(
            config_name=f"PopSize_{pop_size}",
            facilities=facilities,
            iterations=iterations,
            initial_pop=pop_size,
            pretrain_iter=min(1000, iterations // 5),
            train_freq=min(500, iterations // 10),
            latent_dim=2,
            seed=seed
        )
        metrics['population_size'] = pop_size
        results.append(metrics)
    
    return results


def analyze_training_frequency(facilities=5, iterations=5000, seed=42):
    """Test sensitivity to autoencoder training frequency"""
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS: Training Frequency")
    print("="*80)
    
    training_frequencies = [250, 500, 1000, 2000]
    results = []
    
    for train_freq in training_frequencies:
        metrics = run_single_config(
            config_name=f"TrainFreq_{train_freq}",
            facilities=facilities,
            iterations=iterations,
            initial_pop=500,
            pretrain_iter=1000,
            train_freq=train_freq,
            latent_dim=2,
            seed=seed
        )
        metrics['training_frequency'] = train_freq
        results.append(metrics)
    
    return results


def analyze_latent_dimensions(facilities=5, iterations=5000, seed=42):
    """Test sensitivity to latent dimension size"""
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS: Latent Dimensions")
    print("="*80)
    
    latent_dims = [2, 4, 8, 16]
    results = []
    
    for latent_dim in latent_dims:
        metrics = run_single_config(
            config_name=f"LatentDim_{latent_dim}",
            facilities=facilities,
            iterations=iterations,
            initial_pop=500,
            pretrain_iter=1000,
            train_freq=500,
            latent_dim=latent_dim,
            seed=seed
        )
        metrics['latent_dim'] = latent_dim
        results.append(metrics)
    
    return results


def analyze_pretrain_iterations(facilities=5, iterations=5000, seed=42):
    """Test sensitivity to pretrain period"""
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS: Pretrain Iterations")
    print("="*80)
    
    pretrain_iters = [500, 1000, 2000, 3000]
    results = []
    
    for pretrain_iter in pretrain_iters:
        metrics = run_single_config(
            config_name=f"Pretrain_{pretrain_iter}",
            facilities=facilities,
            iterations=iterations,
            initial_pop=500,
            pretrain_iter=pretrain_iter,
            train_freq=500,
            latent_dim=2,
            seed=seed
        )
        metrics['pretrain_iterations'] = pretrain_iter
        results.append(metrics)
    
    return results


def plot_sensitivity_results(all_results, output_dir):
    """Create visualization of sensitivity analysis"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subplots for each parameter
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('CEXO Sensitivity Analysis', fontsize=16, fontweight='bold')
    
    # Population Size
    if 'population_size' in all_results:
        ax = axes[0, 0]
        data = all_results['population_size']
        x = [d['population_size'] for d in data]
        coverage = [d['coverage'] for d in data]
        best_fitness = [d['best_fitness'] for d in data]
        
        ax2 = ax.twinx()
        ax.plot(x, coverage, 'b-o', linewidth=2, markersize=8, label='Coverage')
        ax2.plot(x, best_fitness, 'r-s', linewidth=2, markersize=8, label='Best Fitness')
        
        ax.set_xlabel('Population Size', fontsize=11)
        ax.set_ylabel('Coverage (%)', color='b', fontsize=11)
        ax2.set_ylabel('Best Fitness', color='r', fontsize=11)
        ax.set_title('Population Size Sensitivity', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
    
    # Training Frequency
    if 'training_frequency' in all_results:
        ax = axes[0, 1]
        data = all_results['training_frequency']
        x = [d['training_frequency'] for d in data]
        coverage = [d['coverage'] for d in data]
        best_fitness = [d['best_fitness'] for d in data]
        
        ax2 = ax.twinx()
        ax.plot(x, coverage, 'b-o', linewidth=2, markersize=8, label='Coverage')
        ax2.plot(x, best_fitness, 'r-s', linewidth=2, markersize=8, label='Best Fitness')
        
        ax.set_xlabel('Training Frequency (iterations)', fontsize=11)
        ax.set_ylabel('Coverage (%)', color='b', fontsize=11)
        ax2.set_ylabel('Best Fitness', color='r', fontsize=11)
        ax.set_title('Training Frequency Sensitivity', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
    
    # Latent Dimensions
    if 'latent_dimensions' in all_results:
        ax = axes[1, 0]
        data = all_results['latent_dimensions']
        x = [d['latent_dim'] for d in data]
        coverage = [d['coverage'] for d in data]
        best_fitness = [d['best_fitness'] for d in data]
        
        ax2 = ax.twinx()
        ax.plot(x, coverage, 'b-o', linewidth=2, markersize=8, label='Coverage')
        ax2.plot(x, best_fitness, 'r-s', linewidth=2, markersize=8, label='Best Fitness')
        
        ax.set_xlabel('Latent Dimensions', fontsize=11)
        ax.set_ylabel('Coverage (%)', color='b', fontsize=11)
        ax2.set_ylabel('Best Fitness', color='r', fontsize=11)
        ax.set_title('Latent Dimension Sensitivity', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
    
    # Pretrain Iterations
    if 'pretrain_iterations' in all_results:
        ax = axes[1, 1]
        data = all_results['pretrain_iterations']
        x = [d['pretrain_iterations'] for d in data]
        coverage = [d['coverage'] for d in data]
        best_fitness = [d['best_fitness'] for d in data]
        
        ax2 = ax.twinx()
        ax.plot(x, coverage, 'b-o', linewidth=2, markersize=8, label='Coverage')
        ax2.plot(x, best_fitness, 'r-s', linewidth=2, markersize=8, label='Best Fitness')
        
        ax.set_xlabel('Pretrain Iterations', fontsize=11)
        ax.set_ylabel('Coverage (%)', color='b', fontsize=11)
        ax2.set_ylabel('Best Fitness', color='r', fontsize=11)
        ax.set_title('Pretrain Period Sensitivity', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'sensitivity_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved sensitivity plot: {plot_path}")
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='CEXO Sensitivity Analysis')
    parser.add_argument('--facilities', type=int, default=5, help='Number of facilities')
    parser.add_argument('--iterations', type=int, default=5000, help='Iterations per run')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='results/sensitivity', help='Output directory')
    parser.add_argument('--test', type=str, choices=['all', 'population', 'frequency', 'latent', 'pretrain'],
                       default='all', help='Which sensitivity test to run')
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    print("="*80)
    print("CEXO SENSITIVITY ANALYSIS")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Facilities: {args.facilities}")
    print(f"  Iterations per run: {args.iterations}")
    print(f"  Seed: {args.seed}")
    print(f"  Test: {args.test}")
    print("="*80)
    
    all_results = {}
    start_time = time.time()
    
    if args.test in ['all', 'population']:
        results = analyze_population_size(args.facilities, args.iterations, args.seed)
        all_results['population_size'] = results
    
    if args.test in ['all', 'frequency']:
        results = analyze_training_frequency(args.facilities, args.iterations, args.seed)
        all_results['training_frequency'] = results
    
    if args.test in ['all', 'latent']:
        results = analyze_latent_dimensions(args.facilities, args.iterations, args.seed)
        all_results['latent_dimensions'] = results
    
    if args.test in ['all', 'pretrain']:
        results = analyze_pretrain_iterations(args.facilities, args.iterations, args.seed)
        all_results['pretrain_iterations'] = results
    
    total_time = time.time() - start_time
    
    # Save results to JSON
    json_path = os.path.join(args.output, 'sensitivity_results.json')
    with open(json_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'facilities': args.facilities,
                'iterations': args.iterations,
                'seed': args.seed
            },
            'results': all_results,
            'total_runtime': total_time
        }, f, indent=2)
    print(f"\n✓ Saved results: {json_path}")
    
    # Create visualizations
    plot_sensitivity_results(all_results, args.output)
    
    print(f"\n{'='*80}")
    print("SENSITIVITY ANALYSIS COMPLETE")
    print(f"Total runtime: {total_time:.1f}s")
    print(f"Results saved to: {args.output}/")
    print("="*80)


if __name__ == "__main__":
    main()
