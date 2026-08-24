"""
Comprehensive sensitivity analysis for CEXO with autoencoder-learned BDs.
Tests both site parameters and autoencoder parameters.
Baseline: 6 facilities, 98.83% coverage (from Table 7)
"""

import os
import json
import numpy as np
from core.config import SiteConfig, MapElitesConfig, AutoencoderConfig
from core.mapelites_with_autoencoder import MapElitesWithAutoencoder

# Create results directory
os.makedirs("results/sensitivity_analysis", exist_ok=True)

# Baseline configuration (6 facilities, from Table 7)
def get_baseline_config():
    site_config = SiteConfig()
    site_config.facility_count = 6
    site_config.boundary_margin = 0.08
    site_config.pareto_size = 12
    site_config.seed = 42
    site_config.crane_safety_distance = 0.30
    site_config.entrance_clearance = 0.15
    site_config.min_entrances = 1
    site_config.max_entrances = 3
    
    mapelites_config = MapElitesConfig()
    mapelites_config.grid_size = (20, 20)
    mapelites_config.iterations = 15000
    mapelites_config.initial_population = 500
    
    autoencoder_config = AutoencoderConfig()
    autoencoder_config.use_learned_descriptors = True
    autoencoder_config.latent_dim = 2
    autoencoder_config.encoder_hidden = [128, 64, 32]
    autoencoder_config.decoder_hidden = [32, 64, 128]
    autoencoder_config.pretrain_iterations = 5000
    autoencoder_config.training_frequency = 2500
    autoencoder_config.training_epochs = 50
    autoencoder_config.learning_rate = 0.001
    
    return site_config, mapelites_config, autoencoder_config

def run_single_experiment(site_config, mapelites_config, autoencoder_config, param_name, param_value):
    """Run single experiment and extract key metrics"""
    print(f"\n{'='*60}")
    print(f"Testing {param_name} = {param_value}")
    print(f"{'='*60}")
    
    # Define facility types for 6 facilities
    facility_types = ['crane', 'office', 'rest_area', 'storage', 'crane', 'core']
    
    algorithm = MapElitesWithAutoencoder(facility_types, site_config, mapelites_config, autoencoder_config)
    algorithm.run()
    
    # Extract metrics from archive
    archive = algorithm.archive.archive
    total_cells = mapelites_config.grid_size[0] * mapelites_config.grid_size[1]
    filled_cells = len(archive)
    coverage = (filled_cells / total_cells) * 100
    
    # Calculate objective averages
    safety_scores = []
    efficiency_scores = []
    adaptability_scores = []
    feasible_count = 0
    
    for individual in archive.values():
        if individual.objectives[0] >= 0.7:  # Safety threshold
            feasible_count += 1
            safety_scores.append(individual.objectives[0])
            efficiency_scores.append(individual.objectives[1])
            adaptability_scores.append(individual.objectives[2])
    
    result = {
        'param_name': param_name,
        'param_value': param_value,
        'coverage_pct': coverage,
        'filled_cells': filled_cells,
        'total_individuals': filled_cells,
        'feasible_count': feasible_count,
        'avg_safety': np.mean(safety_scores) if safety_scores else 0.0,
        'avg_efficiency': np.mean(efficiency_scores) if efficiency_scores else 0.0,
        'avg_adaptability': np.mean(adaptability_scores) if adaptability_scores else 0.0,
    }
    
    print(f"\nResults:")
    print(f"  Coverage: {coverage:.2f}%")
    print(f"  Feasible: {feasible_count}/{filled_cells}")
    print(f"  Avg Safety: {result['avg_safety']:.3f}")
    print(f"  Avg Efficiency: {result['avg_efficiency']:.3f}")
    print(f"  Avg Adaptability: {result['avg_adaptability']:.3f}")
    
    return result

# Parameter sweep configurations
parameter_sweeps = {
    # Site parameters
    'boundary_margin': [0.04, 0.06, 0.08, 0.10, 0.12],
    'crane_safety_distance': [0.20, 0.25, 0.30, 0.35, 0.40],
    'entrance_clearance': [0.10, 0.125, 0.15, 0.175, 0.20],
    
    # Autoencoder parameters
    'pretrain_iterations': [1000, 3000, 5000, 7000, 10000],
    'training_frequency': [1000, 2000, 2500, 5000, 7500],
    'latent_dim': [2, 3, 4, 5, 6],
    
    # Algorithm parameters
    'initial_population': [100, 250, 500, 750, 1000],
    'pareto_size': [4, 8, 12, 16, 20],
}

# Baseline values for each parameter
baseline_values = {
    'boundary_margin': 0.08,
    'crane_safety_distance': 0.30,
    'entrance_clearance': 0.15,
    'pretrain_iterations': 5000,
    'training_frequency': 2500,
    'latent_dim': 2,
    'initial_population': 500,
    'pareto_size': 12,
}

def main():
    all_results = []
    
    # First, run baseline
    print("\n" + "="*70)
    print("RUNNING BASELINE CONFIGURATION")
    print("="*70)
    site_config, mapelites_config, autoencoder_config = get_baseline_config()
    baseline_result = run_single_experiment(site_config, mapelites_config, autoencoder_config, 'baseline', 'baseline')
    all_results.append(baseline_result)
    
    baseline_coverage = baseline_result['coverage_pct']
    print(f"\n*** BASELINE COVERAGE: {baseline_coverage:.2f}% ***\n")
    
    # Now test each parameter
    for param_name, param_values in parameter_sweeps.items():
        print(f"\n\n{'#'*70}")
        print(f"# TESTING PARAMETER: {param_name}")
        print(f"{'#'*70}")
        
        for param_value in param_values:
            # Skip baseline value (already ran)
            if param_value == baseline_values[param_name]:
                print(f"\nSkipping {param_name} = {param_value} (baseline already run)")
                continue
            
            # Create config with modified parameter
            site_config, mapelites_config, autoencoder_config = get_baseline_config()
            
            # Apply parameter change
            if param_name == 'boundary_margin':
                site_config.boundary_margin = param_value
            elif param_name == 'crane_safety_distance':
                site_config.crane_safety_distance = param_value
            elif param_name == 'entrance_clearance':
                site_config.entrance_clearance = param_value
            elif param_name == 'pretrain_iterations':
                autoencoder_config.pretrain_iterations = param_value
            elif param_name == 'training_frequency':
                autoencoder_config.training_frequency = param_value
            elif param_name == 'latent_dim':
                autoencoder_config.latent_dim = param_value
            elif param_name == 'initial_population':
                mapelites_config.initial_population = param_value
            elif param_name == 'pareto_size':
                site_config.pareto_size = param_value
            
            result = run_single_experiment(site_config, mapelites_config, autoencoder_config, param_name, param_value)
            all_results.append(result)
    
    # Save all results
    results_file = "results/sensitivity_analysis/all_sensitivity_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n\n{'='*70}")
    print(f"All results saved to: {results_file}")
    print(f"Total experiments run: {len(all_results)}")
    print(f"{'='*70}")
    
    # Print summary table
    print("\n\nSUMMARY TABLE:")
    print(f"{'Parameter':<25} {'Value':<15} {'Coverage %':<12} {'Change %':<10}")
    print("-" * 70)
    
    for result in all_results:
        param = result['param_name']
        value = result['param_value']
        coverage = result['coverage_pct']
        change = coverage - baseline_coverage
        print(f"{param:<25} {str(value):<15} {coverage:>10.2f}% {change:>+9.2f}%")

if __name__ == "__main__":
    main()
