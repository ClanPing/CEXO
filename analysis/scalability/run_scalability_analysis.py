"""
Scalability Analysis for CEXO with Autoencoder-Learned Behavioral Descriptors
Tests algorithm performance across varying facility counts (3-8 facilities)
"""

import sys
import time
import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

# Import CEXO components
from core.mapelites_with_autoencoder import MapElitesWithAutoencoder
from core.config import SiteConfig, MapElitesConfig, AutoencoderConfig

def run_single_configuration(num_facilities, seed, site_config, mapelites_config, autoencoder_config):
    """
    Execute one CEXO run for a specific facility count and seed.
    
    Args:
        num_facilities: Number of facilities to place (3-8)
        seed: Random seed for reproducibility
        site_config: SiteConfig instance
        mapelites_config: MapElitesConfig instance
        autoencoder_config: AutoencoderConfig with learned descriptors enabled
        
    Returns:
        Dictionary with metrics: coverage %, cells, safety, efficiency, adaptability, runtime
    """
    print(f"  Running {num_facilities} facilities, seed {seed}...", end=" ", flush=True)
    
    # Define facility types based on count
    # Standard mix: crane, office, rest_area, storage, crane, core + additional storage
    facility_types = ['crane', 'office', 'rest_area', 'storage', 'crane', 'core']
    if num_facilities > 6:
        facility_types.extend(['storage'] * (num_facilities - 6))
    elif num_facilities < 6:
        facility_types = facility_types[:num_facilities]
    
    # Update site config for this run
    site_config.seed = seed
    site_config.facility_count = len(facility_types)
    
    # Initialize algorithm with autoencoder config
    algorithm = MapElitesWithAutoencoder(
        facility_types=facility_types,
        site_config=site_config,
        mapelites_config=mapelites_config,
        autoencoder_config=autoencoder_config
    )
    
    # Run optimization
    start_time = time.time()
    algorithm.run(
        iterations=mapelites_config.iterations,
        initial_population=mapelites_config.initial_population
    )
    runtime = time.time() - start_time
    
    # Extract final archive
    archive = algorithm.archive.archive
    
    # Calculate metrics
    total_cells = 400  # 20x20 grid
    occupied_cells = len(archive)
    coverage_pct = (occupied_cells / total_cells) * 100
    
    # Count safety-feasible solutions and collect all individuals
    safety_threshold = 0.7
    safety_feasible = 0
    all_solutions = []
    
    for cell_coords, individual in archive.items():
        all_solutions.append(individual)
        if individual.objectives[0] >= safety_threshold:
            safety_feasible += 1
    
    # Calculate average objectives across all archived solutions
    avg_safety = np.mean([ind.objectives[0] for ind in all_solutions]) if all_solutions else 0
    avg_efficiency = np.mean([ind.objectives[1] for ind in all_solutions]) if all_solutions else 0
    avg_adaptability = np.mean([ind.objectives[2] for ind in all_solutions]) if all_solutions else 0
    
    print(f"Coverage: {coverage_pct:.2f}%, Runtime: {runtime:.2f}s")
    
    return {
        'num_facilities': num_facilities,
        'seed': seed,
        'coverage_pct': float(coverage_pct),
        'coverage_cells': int(occupied_cells),
        'total_individuals': len(all_solutions),
        'safety_feasible_count': int(safety_feasible),
        'avg_safety': float(avg_safety),
        'avg_efficiency': float(avg_efficiency),
        'avg_adaptability': float(avg_adaptability),
        'runtime_seconds': float(runtime)
    }

def calculate_statistics(results_df):
    """
    Calculate summary statistics for each facility configuration.
    
    Args:
        results_df: DataFrame with all trial results
        
    Returns:
        DataFrame with mean ± std for each configuration
    """
    summary = []
    
    for n_fac in sorted(results_df['num_facilities'].unique()):
        config_data = results_df[results_df['num_facilities'] == n_fac]
        
        # Calculate mean and std for each metric
        coverage_mean = config_data['coverage_pct'].mean()
        coverage_std = config_data['coverage_pct'].std()
        
        safety_mean = config_data['avg_safety'].mean()
        safety_std = config_data['avg_safety'].std()
        
        efficiency_mean = config_data['avg_efficiency'].mean()
        efficiency_std = config_data['avg_efficiency'].std()
        
        adaptability_mean = config_data['avg_adaptability'].mean()
        adaptability_std = config_data['avg_adaptability'].std()
        
        runtime_mean = config_data['runtime_seconds'].mean()
        
        # Calculate scaling factor relative to 3 facilities
        base_runtime = results_df[results_df['num_facilities'] == 3]['runtime_seconds'].mean()
        scaling_factor = runtime_mean / base_runtime if base_runtime > 0 else 1.0
        
        summary.append({
            'Facilities': int(n_fac),
            'Coverage (%)': f"{coverage_mean:.2f} ± {coverage_std:.2f}",
            'Average safety': f"{safety_mean:.3f} ± {safety_std:.3f}",
            'Average efficiency': f"{efficiency_mean:.3f} ± {efficiency_std:.3f}",
            'Average adaptability': f"{adaptability_mean:.3f} ± {adaptability_std:.3f}",
            'Runtime (s)': f"{runtime_mean:.2f}",
            'Scaling factor': f"{scaling_factor:.2f}×"
        })
    
    return pd.DataFrame(summary)

def format_markdown_table(summary_df):
    """
    Format summary statistics as markdown table for experiment.md.
    
    Args:
        summary_df: DataFrame with summary statistics
        
    Returns:
        String containing markdown-formatted table
    """
    lines = [
        "**Table 7.** Scalability analysis statistical summary.",
        "",
        "| **Facilities** | **Coverage (%)** | **Average safety** | **Average efficiency** | **Average adaptability** | **Runtime (s)** | **Scaling factor** |",
        "|----------------|------------------|--------------------|------------------------|--------------------------|-----------------|--------------------|"
    ]
    
    for _, row in summary_df.iterrows():
        line = f"| {row['Facilities']} | {row['Coverage (%)']} | {row['Average safety']} | {row['Average efficiency']} | {row['Average adaptability']} | {row['Runtime (s)']} | {row['Scaling factor']} |"
        lines.append(line)
    
    return "\n".join(lines)

def main():
    """Run scalability analysis across 3-8 facilities with autoencoder."""
    
    print("="*80)
    print("SCALABILITY ANALYSIS - CEXO with Autoencoder-Learned Behavioral Descriptors")
    print("="*80)
    print()
    
    # Configure autoencoder with learned descriptors
    autoencoder_config = AutoencoderConfig(
        use_learned_descriptors=True,
        latent_dim=2,
        encoder_hidden=[128, 64, 32],
        decoder_hidden=[32, 64, 128],
        learning_rate=0.001,
        batch_size=32,
        training_epochs=50,
        pretrain_iterations=5000,
        training_frequency=2500
    )
    
    print("Configuration:")
    print(f"  Facility range: 3-8")
    print(f"  Seeds per config: 3 (seeds 42, 43, 44)")
    print(f"  Grid size: 20x20 (400 cells)")
    print(f"  Iterations: 15,000")
    print(f"  Population: 500")
    print(f"  Autoencoder: Learned BDs (2D latent space)")
    print(f"  Pretrain: {autoencoder_config.pretrain_iterations} iterations")
    print(f"  Retrain frequency: {autoencoder_config.training_frequency} iterations")
    print()
    
    # Configure site and algorithm settings
    site_config = SiteConfig(
        seed=42,
        boundary_margin=0.05,
        pareto_size=12,
        facility_count=6  # Will be updated per run
    )
    
    mapelites_config = MapElitesConfig(
        grid_size=(20, 20),
        iterations=15000,
        initial_population=500
    )
    
    # Create output directory
    output_dir = Path("results/scalability_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run all configurations
    all_results = []
    facility_range = range(3, 9)  # 3 to 8 inclusive
    seeds = [42, 43, 44]  # 3 seeds per configuration
    
    total_runs = len(facility_range) * len(seeds)
    current_run = 0
    
    for num_facilities in facility_range:
        print(f"\nTesting {num_facilities} facilities:")
        print("-" * 60)
        
        for seed in seeds:
            current_run += 1
            print(f"[Run {current_run}/{total_runs}]", end=" ")
            
            try:
                result = run_single_configuration(
                    num_facilities, seed, site_config, mapelites_config, autoencoder_config
                )
                all_results.append(result)
                
                # Save intermediate results after each trial
                df = pd.DataFrame(all_results)
                df.to_csv(output_dir / "intermediate_results.csv", index=False)
                
            except Exception as e:
                print(f"ERROR: {e}")
                continue
    
    # Calculate summary statistics
    print("\n" + "="*80)
    print("CALCULATING SUMMARY STATISTICS")
    print("="*80)
    
    results_df = pd.DataFrame(all_results)
    summary_df = calculate_statistics(results_df)
    
    # Save all results
    results_df.to_csv(output_dir / "all_trials.csv", index=False)
    summary_df.to_csv(output_dir / "summary_statistics.csv", index=False)
    
    # Generate markdown table
    markdown_table = format_markdown_table(summary_df)
    with open(output_dir / "table_7.md", 'w') as f:
        f.write(markdown_table)
    
    # Display results
    print("\nSummary Statistics:")
    print(summary_df.to_string(index=False))
    print()
    print(f"Results saved to: {output_dir}")
    print(f"  - all_trials.csv: Raw data from all {len(all_results)} runs")
    print(f"  - summary_statistics.csv: Aggregated statistics")
    print(f"  - table_7.md: Formatted markdown table for experiment.md")
    print()
    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
