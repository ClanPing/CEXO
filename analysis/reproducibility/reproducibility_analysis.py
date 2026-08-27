#!/usr/bin/env python3
"""
CEXO Reproducibility Analysis
==============================

Runs 30 independent CEXO experiments with different random seeds to assess
statistical stability and reproducibility of the algorithm with autoencoder-learned
behavioral descriptors.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

# Set CUDA environment variables for reproducibility
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


def run_single_trial(trial_num, seed, facility_types, output_dir):
    """Run a single CEXO trial"""
    
    print(f"\n{'='*80}")
    print(f"Trial {trial_num}/30 (Seed: {seed})")
    print(f"{'='*80}")
    
    # Configuration
    site_config = SiteConfig(
        seed=seed,
        boundary_margin=0.05,
        pareto_size=12,
        facility_count=len(facility_types)
    )
    
    mapelites_config = MapElitesConfig(
        grid_size=(20, 20),
        iterations=15000,
        initial_population=500
    )
    
    autoencoder_config = AutoencoderConfig(
        use_learned_descriptors=True,
        pretrain_iterations=5000,
        latent_dim=2,
        encoder_hidden=[128, 64, 32],
        decoder_hidden=[32, 64, 128],
        learning_rate=0.001,
        training_epochs=50,
        batch_size=32,
        training_frequency=2500,
        min_samples_for_training=200,
        seed=seed
    )
    
    # Run CEXO
    start_time = time.time()
    
    algorithm = MapElitesWithAutoencoder(
        facility_types=facility_types,
        site_config=site_config,
        mapelites_config=mapelites_config,
        autoencoder_config=autoencoder_config
    )
    
    results = algorithm.run(
        iterations=mapelites_config.iterations,
        initial_population=mapelites_config.initial_population
    )
    
    runtime = time.time() - start_time
    
    # Collect metrics
    all_individuals = algorithm.archive.get_all_individuals()
    feasible_individuals = [ind for ind in all_individuals if ind.feasible]
    safe_individuals = [ind for ind in all_individuals if ind.objectives[0] >= 0.7]
    
    stats = algorithm.archive.get_stats()
    
    objectives = np.array([ind.objectives for ind in all_individuals])
    
    # Calculate BD distribution (average of both BDs)
    behaviors = np.array([ind.behaviors for ind in all_individuals])
    avg_bd = np.mean(behaviors)
    
    metrics = {
        'trial': trial_num,
        'seed': seed,
        'coverage_pct': stats['coverage_pct'],
        'coverage_cells': stats['coverage'],
        'total_individuals': len(all_individuals),
        'safety_feasible_count': len(safe_individuals),
        'feasible_count': len(feasible_individuals),
        'avg_safety': float(np.mean(objectives[:, 0])),
        'avg_efficiency': float(np.mean(objectives[:, 1])),
        'avg_adaptability': float(np.mean(objectives[:, 2])),
        'bd_avg': float(avg_bd),
        'runtime': runtime,
        'bd_mode': algorithm.bd_manager.get_mode(),
        'autoencoder_trained': algorithm.autoencoder_trained
    }
    
    print(f"\nTrial {trial_num} Results:")
    print(f"  Coverage: {metrics['coverage_cells']}/{algorithm.archive.total_cells} ({metrics['coverage_pct']:.1f}%)")
    print(f"  Total Individuals: {metrics['total_individuals']}")
    print(f"  Safety Feasible: {metrics['safety_feasible_count']}")
    print(f"  Avg Safety: {metrics['avg_safety']:.4f}")
    print(f"  Avg Efficiency: {metrics['avg_efficiency']:.4f}")
    print(f"  Avg Adaptability: {metrics['avg_adaptability']:.4f}")
    print(f"  Runtime: {runtime:.2f}s")
    print(f"  BD Mode: {metrics['bd_mode']}")
    
    return metrics


def calculate_statistics(data):
    """Calculate comprehensive statistics"""
    
    metrics = [
        'coverage_pct', 'coverage_cells', 'total_individuals', 
        'safety_feasible_count', 'avg_safety', 'avg_efficiency', 
        'avg_adaptability', 'bd_avg', 'runtime'
    ]
    
    stats_summary = {}
    
    for metric in metrics:
        values = data[metric].values
        
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        cv = (std / mean * 100) if mean != 0 else 0
        
        # 95% confidence interval
        confidence = 0.95
        n = len(values)
        se = stats.sem(values)
        margin = se * stats.t.ppf((1 + confidence) / 2, n - 1)
        ci_lower = mean - margin
        ci_upper = mean + margin
        
        # IQR
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        
        # Range
        value_range = np.max(values) - np.min(values)
        
        stats_summary[metric] = {
            'mean': float(mean),
            'std': float(std),
            'cv': float(cv),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'iqr': float(iqr),
            'range': float(value_range),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values))
        }
    
    return stats_summary


def format_table_row(metric_name, stats):
    """Format a row for the markdown table"""
    mean = stats['mean']
    std = stats['std']
    cv = stats['cv']
    ci_lower = stats['ci_lower']
    ci_upper = stats['ci_upper']
    iqr = stats['iqr']
    value_range = stats['range']
    
    metric_key = metric_name.lower()

    # Format based on display metric type.
    if 'coverage (%)' in metric_key:
        mean_std = f"{mean:.2f} ± {std:.2f}"
        ci = f"[{ci_lower:.2f}, {ci_upper:.2f}]"
        iqr_str = f"{iqr:.2f}"
        range_str = f"{value_range:.2f}"
    elif metric_key.startswith('average') or metric_key == 'bd':
        mean_std = f"{mean:.3f} ± {std:.3f}"
        ci = f"[{ci_lower:.3f}, {ci_upper:.3f}]"
        iqr_str = f"{iqr:.3f}"
        range_str = f"{value_range:.3f}"
    elif 'runtime' in metric_key:
        mean_std = f"{mean:.2f} ± {std:.2f}"
        ci = f"[{ci_lower:.2f}, {ci_upper:.2f}]"
        iqr_str = f"{iqr:.2f}"
        range_str = f"{value_range:.2f}"
    else:
        mean_std = f"{mean:.2f} ± {std:.2f}"
        ci = f"[{ci_lower:.2f}, {ci_upper:.2f}]"
        iqr_str = f"{iqr:.0f}"
        range_str = f"{value_range:.0f}"
    
    cv_str = f"{cv:.2f}"
    
    return f"| {metric_name:<25} | {mean_std:<15} | {cv_str:<10} | {ci:<20} | {iqr_str:<7} | {range_str:<9} |"


def main():
    """Main reproducibility analysis"""
    parser = argparse.ArgumentParser(description="Run CEXO reproducibility analysis.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed; trials use seed+1 through seed+num_trials.")
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("CEXO REPRODUCIBILITY ANALYSIS")
    print("30 Independent Runs with Autoencoder-Learned BDs")
    print("="*80)
    
    # Configuration
    num_trials = 30
    base_seed = args.seed
    num_facilities = 6
    output_dir = "results/reproducibility_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate facility mix (same for all trials)
    facility_types = generate_facility_mix(num_facilities, seed=base_seed)
    print(f"\nFacility mix ({num_facilities} facilities): {facility_types}")
    
    print(f"\nConfiguration:")
    print(f"  Trials: {num_trials}")
    print(f"  Base seed: {base_seed}")
    print(f"  Grid: 20×20 (400 cells)")
    print(f"  Iterations: 15,000")
    print(f"  Initial population: 500")
    print(f"  Pareto size per cell: 12")
    print(f"  Autoencoder: Learned BDs (2D latent)")
    print(f"  Pretrain iterations: 5,000")
    print(f"  Training frequency: 2,500")
    
    # Run all trials
    all_results = []
    
    for trial_num in range(1, num_trials + 1):
        seed = base_seed + trial_num
        
        try:
            metrics = run_single_trial(trial_num, seed, facility_types, output_dir)
            all_results.append(metrics)
            
            # Save intermediate results
            df = pd.DataFrame(all_results)
            df.to_csv(os.path.join(output_dir, 'intermediate_results.csv'), index=False)
            
        except Exception as e:
            print(f"\n✗ Trial {trial_num} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(all_results) == 0:
        print("\n✗ No trials completed successfully")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Calculate statistics
    print("\n" + "="*80)
    print("CALCULATING STATISTICS")
    print("="*80)
    
    stats_summary = calculate_statistics(df)
    
    # Save results
    df.to_csv(os.path.join(output_dir, 'all_trials.csv'), index=False)
    
    with open(os.path.join(output_dir, 'statistics.json'), 'w') as f:
        json.dump(stats_summary, f, indent=2)
    
    # Generate markdown table
    print("\n" + "="*80)
    print("REPRODUCIBILITY STATISTICS SUMMARY")
    print("="*80)
    
    print("\n**Table 6.** Reproducibility analysis statistical summary.\n")
    print("| **Metric**                | **Mean ± Std** | **CV (%)** | **CI<sub>95</sub>** | **IQR** | **Range** |")
    print("|---------------------------|----------------|------------|---------------------|---------|-----------|")
    
    metric_labels = {
        'coverage_pct': 'Coverage (%)',
        'coverage_cells': 'Coverage (cells)',
        'total_individuals': 'Total individuals',
        'safety_feasible_count': 'Safety feasible count',
        'avg_safety': 'Average safety',
        'avg_efficiency': 'Average efficiency',
        'avg_adaptability': 'Average adaptability',
        'bd_avg': 'BD',
        'runtime': 'Runtime (seconds)'
    }
    
    for metric_key, metric_label in metric_labels.items():
        print(format_table_row(metric_label, stats_summary[metric_key]))
    
    # Save markdown table
    with open(os.path.join(output_dir, 'table_6.md'), 'w') as f:
        f.write("**Table 6.** Reproducibility analysis statistical summary.\n\n")
        f.write("| **Metric**                | **Mean ± Std** | **CV (%)** | **CI<sub>95</sub>** | **IQR** | **Range** |\n")
        f.write("|---------------------------|----------------|------------|---------------------|---------|-----------|")
        f.write("\n")
        for metric_key, metric_label in metric_labels.items():
            f.write(format_table_row(metric_label, stats_summary[metric_key]) + "\n")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nCompleted {len(all_results)}/{num_trials} trials successfully")
    print(f"\nKey Findings:")
    print(f"  Mean Coverage: {stats_summary['coverage_pct']['mean']:.2f}% ± {stats_summary['coverage_pct']['std']:.2f}% (CV: {stats_summary['coverage_pct']['cv']:.2f}%)")
    print(f"  Mean Safety: {stats_summary['avg_safety']['mean']:.4f} ± {stats_summary['avg_safety']['std']:.4f} (CV: {stats_summary['avg_safety']['cv']:.2f}%)")
    print(f"  Mean Efficiency: {stats_summary['avg_efficiency']['mean']:.4f} ± {stats_summary['avg_efficiency']['std']:.4f} (CV: {stats_summary['avg_efficiency']['cv']:.2f}%)")
    print(f"  Mean Adaptability: {stats_summary['avg_adaptability']['mean']:.4f} ± {stats_summary['avg_adaptability']['std']:.4f} (CV: {stats_summary['avg_adaptability']['cv']:.2f}%)")
    print(f"  Mean Runtime: {stats_summary['runtime']['mean']:.2f}s ± {stats_summary['runtime']['std']:.2f}s (CV: {stats_summary['runtime']['cv']:.2f}%)")
    
    print("\n" + "="*80)
    print(f"✓ All results saved to {output_dir}/")
    print("="*80)


if __name__ == "__main__":
    main()
