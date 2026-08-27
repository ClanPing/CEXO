#!/usr/bin/env python3
"""
Statistical Analysis for CEXO Algorithm
========================================

Performs statistical validation across multiple random seeds:
- Runs CEXO, MAP-Elites baseline, and NSGA-II baseline
- Multiple seeds for statistical significance
- Calculates mean, std, confidence intervals
- Statistical tests (t-test, Mann-Whitney U)
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats

# Set environment variables before torch import
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONHASHSEED'] = '0'

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.config import (
    SiteConfig,
    MapElitesConfig,
    AutoencoderConfig,
    NSGA2Config,
    generate_facility_mix
)
from core.mapelites_with_autoencoder import MapElitesWithAutoencoder
from core.mapelites_algorithm import PureMapElitesOptimizer
from core.nsga2_algorithm import PureNSGA2Optimizer
from core.layout_autoencoder import set_random_seeds


def run_cexo(facilities, iterations, initial_pop, seed):
    """Run CEXO with specific seed"""
    set_random_seeds(seed)
    
    site_config = SiteConfig(facility_count=facilities, seed=seed)
    facility_types = generate_facility_mix(facilities, seed=seed)
    
    mapelites_config = MapElitesConfig(
        iterations=iterations,
        initial_population=initial_pop
    )
    
    autoencoder_config = AutoencoderConfig(
        use_learned_descriptors=True,
        pretrain_iterations=min(2000, iterations // 5),
        training_frequency=min(1000, iterations // 10),
        latent_dim=2,
        seed=seed
    )
    
    start_time = time.time()
    algorithm = MapElitesWithAutoencoder(
        facility_types=facility_types,
        site_config=site_config,
        mapelites_config=mapelites_config,
        autoencoder_config=autoencoder_config
    )
    
    result = algorithm.run()
    runtime = time.time() - start_time
    
    archive = result['archive']
    coverage = len(archive.archive) / archive.total_cells * 100
    all_inds = archive.get_all_individuals()
    
    if all_inds:
        avg_fitness = np.mean([archive.calculate_scalar_fitness(ind) for ind in all_inds])
        best_fitness = max([archive.calculate_scalar_fitness(ind) for ind in all_inds])
    else:
        avg_fitness = best_fitness = 0
    
    return {
        'coverage': coverage,
        'avg_fitness': avg_fitness,
        'best_fitness': best_fitness,
        'runtime': runtime,
        'num_solutions': len(all_inds)
    }


def run_mapelites_baseline(facilities, iterations, initial_pop, seed):
    """Run MAP-Elites baseline with specific seed"""
    set_random_seeds(seed)
    
    site_config = SiteConfig(facility_count=facilities, seed=seed)
    facility_types = generate_facility_mix(facilities, seed=seed)
    
    mapelites_config = MapElitesConfig(
        iterations=iterations,
        initial_population=initial_pop
    )
    
    start_time = time.time()
    optimizer = PureMapElitesOptimizer(
        site_config=site_config,
        facility_types=facility_types,
        mapelites_config=mapelites_config
    )
    
    result = optimizer.run()
    runtime = time.time() - start_time
    
    archive = result['archive']
    coverage = len(archive.archive) / archive.total_cells * 100
    all_inds = list(archive.archive.values())
    
    if all_inds:
        avg_fitness = np.mean([archive.calculate_scalar_fitness(ind) for ind in all_inds])
        best_fitness = max([archive.calculate_scalar_fitness(ind) for ind in all_inds])
    else:
        avg_fitness = best_fitness = 0
    
    return {
        'coverage': coverage,
        'avg_fitness': avg_fitness,
        'best_fitness': best_fitness,
        'runtime': runtime,
        'num_solutions': len(all_inds)
    }


def run_nsga2_baseline(facilities, iterations, initial_pop, seed):
    """Run NSGA-II baseline with specific seed"""
    set_random_seeds(seed)
    
    site_config = SiteConfig(facility_count=facilities, seed=seed)
    facility_types = generate_facility_mix(facilities, seed=seed)
    
    nsga2_config = NSGA2Config(
        population_size=initial_pop,
        generations=iterations // 10
    )
    
    start_time = time.time()
    optimizer = PureNSGA2Optimizer(
        site_config=site_config,
        facility_types=facility_types,
        nsga2_config=nsga2_config
    )
    
    result = optimizer.run()
    runtime = time.time() - start_time
    
    final_population = result['population']
    
    if final_population:
        avg_fitness = np.mean([
            0.5 * ind.objectives[0] + 0.3 * ind.objectives[1] + 0.2 * ind.objectives[2]
            for ind in final_population
        ])
        best_fitness = max([
            0.5 * ind.objectives[0] + 0.3 * ind.objectives[1] + 0.2 * ind.objectives[2]
            for ind in final_population
        ])
    else:
        avg_fitness = best_fitness = 0
    
    return {
        'coverage': 0.0,  # N/A for NSGA-II
        'avg_fitness': avg_fitness,
        'best_fitness': best_fitness,
        'runtime': runtime,
        'num_solutions': len(final_population)
    }


def calculate_statistics(data):
    """Calculate statistical metrics"""
    data = np.array(data)
    return {
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'median': float(np.median(data)),
        'ci_95': (float(np.percentile(data, 2.5)), float(np.percentile(data, 97.5)))
    }


def statistical_comparison(method1_data, method2_data, method1_name, method2_name):
    """Perform statistical tests between two methods"""
    # T-test
    t_stat, t_pvalue = stats.ttest_ind(method1_data, method2_data)
    
    # Mann-Whitney U test (non-parametric)
    u_stat, u_pvalue = stats.mannwhitneyu(method1_data, method2_data, alternative='two-sided')
    
    # Effect size (Cohen's d)
    mean_diff = np.mean(method1_data) - np.mean(method2_data)
    pooled_std = np.sqrt((np.std(method1_data)**2 + np.std(method2_data)**2) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
    
    return {
        't_test': {
            'statistic': float(t_stat),
            'p_value': float(t_pvalue),
            'significant': t_pvalue < 0.05
        },
        'mann_whitney_u': {
            'statistic': float(u_stat),
            'p_value': float(u_pvalue),
            'significant': u_pvalue < 0.05
        },
        'effect_size': {
            'cohens_d': float(cohens_d),
            'interpretation': 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'
        },
        'winner': method1_name if mean_diff > 0 else method2_name
    }


def plot_statistical_results(all_results, output_dir):
    """Create statistical visualization with error bars"""
    os.makedirs(output_dir, exist_ok=True)
    
    methods = list(all_results.keys())
    metrics = ['coverage', 'best_fitness', 'avg_fitness', 'runtime']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('CEXO Statistical Analysis (Multiple Seeds)', fontsize=16, fontweight='bold')
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        means = []
        stds = []
        method_names = []
        
        for method in methods:
            if metric in all_results[method]:
                stats_data = all_results[method][metric]
                means.append(stats_data['mean'])
                stds.append(stats_data['std'])
                method_names.append(method)
        
        x = np.arange(len(method_names))
        bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(method_names)])
        
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.set_title(f'{metric.replace("_", " ").title()} Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(method_names, rotation=15, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, mean + std, f'{mean:.2f}±{std:.2f}', 
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'statistical_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved statistical plot: {plot_path}")
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='CEXO Statistical Analysis')
    parser.add_argument('--runs', type=int, default=10, help='Number of runs per method')
    parser.add_argument('--facilities', type=int, default=5, help='Number of facilities')
    parser.add_argument('--iterations', type=int, default=5000, help='Iterations per run')
    parser.add_argument('--initial-pop', type=int, default=500, help='Initial population')
    parser.add_argument('--start-seed', type=int, default=42, help='Starting seed')
    parser.add_argument('--output', type=str, default='results/statistical', help='Output directory')
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    seeds = [args.start_seed + i for i in range(args.runs)]
    
    print("="*80)
    print("CEXO STATISTICAL ANALYSIS")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Number of runs: {args.runs}")
    print(f"  Facilities: {args.facilities}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Seeds: {seeds}")
    print("="*80)
    
    all_results = {
        'CEXO (Proposed)': {'coverage': [], 'best_fitness': [], 'avg_fitness': [], 'runtime': []},
        'MAP-Elites Baseline': {'coverage': [], 'best_fitness': [], 'avg_fitness': [], 'runtime': []},
        'NSGA-II Baseline': {'coverage': [], 'best_fitness': [], 'avg_fitness': [], 'runtime': []}
    }
    
    start_time = time.time()
    
    for i, seed in enumerate(seeds):
        print(f"\n{'='*80}")
        print(f"Run {i+1}/{args.runs} (Seed: {seed})")
        print(f"{'='*80}")
        
        # CEXO
        print("\n  Running: CEXO (Proposed)...")
        result = run_cexo(args.facilities, args.iterations, args.initial_pop, seed)
        for key in ['coverage', 'best_fitness', 'avg_fitness', 'runtime']:
            all_results['CEXO (Proposed)'][key].append(result[key])
        print(f"    ✓ Coverage: {result['coverage']:.2f}%, Best: {result['best_fitness']:.3f}")
        
        # MAP-Elites
        print("  Running: MAP-Elites Baseline...")
        result = run_mapelites_baseline(args.facilities, args.iterations, args.initial_pop, seed)
        for key in ['coverage', 'best_fitness', 'avg_fitness', 'runtime']:
            all_results['MAP-Elites Baseline'][key].append(result[key])
        print(f"    ✓ Coverage: {result['coverage']:.2f}%, Best: {result['best_fitness']:.3f}")
        
        # NSGA-II
        print("  Running: NSGA-II Baseline...")
        result = run_nsga2_baseline(args.facilities, args.iterations, args.initial_pop, seed)
        for key in ['best_fitness', 'avg_fitness', 'runtime']:
            all_results['NSGA-II Baseline'][key].append(result[key])
        all_results['NSGA-II Baseline']['coverage'].append(0.0)  # N/A
        print(f"    ✓ Best: {result['best_fitness']:.3f}")
    
    total_time = time.time() - start_time
    
    # Calculate statistics
    statistics = {}
    for method in all_results:
        statistics[method] = {}
        for metric in all_results[method]:
            if all_results[method][metric]:  # Check if not empty
                statistics[method][metric] = calculate_statistics(all_results[method][metric])
    
    # Statistical comparisons
    comparisons = {}
    
    # CEXO vs MAP-Elites
    comparisons['CEXO_vs_MapElites'] = {
        'best_fitness': statistical_comparison(
            all_results['CEXO (Proposed)']['best_fitness'],
            all_results['MAP-Elites Baseline']['best_fitness'],
            'CEXO', 'MAP-Elites'
        ),
        'coverage': statistical_comparison(
            all_results['CEXO (Proposed)']['coverage'],
            all_results['MAP-Elites Baseline']['coverage'],
            'CEXO', 'MAP-Elites'
        )
    }
    
    # CEXO vs NSGA-II
    comparisons['CEXO_vs_NSGA2'] = {
        'best_fitness': statistical_comparison(
            all_results['CEXO (Proposed)']['best_fitness'],
            all_results['NSGA-II Baseline']['best_fitness'],
            'CEXO', 'NSGA-II'
        )
    }
    
    # Print summary
    print(f"\n{'='*80}")
    print("STATISTICAL SUMMARY")
    print(f"{'='*80}")
    
    for method in statistics:
        print(f"\n{method}:")
        for metric in ['coverage', 'best_fitness', 'avg_fitness']:
            if metric in statistics[method]:
                s = statistics[method][metric]
                print(f"  {metric}: {s['mean']:.3f} ± {s['std']:.3f} "
                      f"(95% CI: [{s['ci_95'][0]:.3f}, {s['ci_95'][1]:.3f}])")
    
    print(f"\n{'='*80}")
    print("STATISTICAL TESTS")
    print(f"{'='*80}")
    
    for comparison_name, comparison_data in comparisons.items():
        print(f"\n{comparison_name.replace('_', ' ')}:")
        for metric, test_results in comparison_data.items():
            print(f"  {metric}:")
            print(f"    Winner: {test_results['winner']}")
            print(f"    T-test p-value: {test_results['t_test']['p_value']:.4f} "
                  f"({'significant' if test_results['t_test']['significant'] else 'not significant'})")
            print(f"    Effect size (Cohen's d): {test_results['effect_size']['cohens_d']:.3f} "
                  f"({test_results['effect_size']['interpretation']})")
    
    # Save results
    json_path = os.path.join(args.output, 'statistical_results.json')
    with open(json_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'runs': args.runs,
                'facilities': args.facilities,
                'iterations': args.iterations,
                'seeds': seeds
            },
            'raw_results': {k: {m: [float(x) for x in v] for m, v in data.items()} 
                           for k, data in all_results.items()},
            'statistics': statistics,
            'comparisons': comparisons,
            'total_runtime': total_time
        }, f, indent=2)
    print(f"\n✓ Saved results: {json_path}")
    
    # Create visualizations
    plot_statistical_results(statistics, args.output)
    
    print(f"\n{'='*80}")
    print("STATISTICAL ANALYSIS COMPLETE")
    print(f"Total runtime: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Results saved to: {args.output}/")
    print("="*80)


if __name__ == "__main__":
    main()
