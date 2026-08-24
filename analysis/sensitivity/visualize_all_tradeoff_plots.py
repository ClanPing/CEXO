"""
Generate trade-off plots for ALL parameters (7 plots total)
Each parameter gets its own figure showing how all 4 objectives change
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load results
results_file = "results/sensitivity_analysis/all_sensitivity_results.json"
with open(results_file, 'r') as f:
    all_results = json.load(f)

# Extract baseline
baseline = [r for r in all_results if r['param_name'] == 'baseline'][0]
baseline_coverage = baseline['coverage_pct']
baseline_safety = baseline['avg_safety']
baseline_efficiency = baseline['avg_efficiency']
baseline_adaptability = baseline['avg_adaptability']

print(f"Baseline: Coverage={baseline_coverage:.2f}%, Safety={baseline_safety:.3f}, "
      f"Efficiency={baseline_efficiency:.3f}, Adaptability={baseline_adaptability:.3f}\n")

# Organize results by parameter
params_data = {}
for result in all_results:
    param_name = result['param_name']
    if param_name == 'baseline':
        continue
    
    if param_name not in params_data:
        params_data[param_name] = {
            'values': [],
            'coverage': [],
            'safety': [],
            'efficiency': [],
            'adaptability': []
        }
    
    params_data[param_name]['values'].append(result['param_value'])
    params_data[param_name]['coverage'].append(result['coverage_pct'])
    params_data[param_name]['safety'].append(result['avg_safety'])
    params_data[param_name]['efficiency'].append(result['avg_efficiency'])
    params_data[param_name]['adaptability'].append(result['avg_adaptability'])

# Pretty names and baseline values
param_display_names = {
    'boundary_margin': 'Boundary Margin',
    'crane_safety_distance': 'Crane Safety Distance',
    'entrance_clearance': 'Entrance Clearance',
    'pretrain_iterations': 'Pretrain Iterations',
    'training_frequency': 'Training Frequency',
    'latent_dim': 'Latent Dimensions',
    'initial_population': 'Initial Population Size',
    'pareto_size': 'Pareto Front Size'
}

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

# Create output directory
output_dir = Path("results/sensitivity_analysis/individual_tradeoff_plots")
output_dir.mkdir(parents=True, exist_ok=True)

# Generate a trade-off plot for each parameter
for param_name, param_data in sorted(params_data.items()):
    print(f"Generating trade-off plot for: {param_display_names[param_name]}")
    
    # Extract and sort data
    param_values = np.array(param_data['values'])
    coverage_data = np.array(param_data['coverage'])
    safety_data = np.array(param_data['safety'])
    efficiency_data = np.array(param_data['efficiency'])
    adaptability_data = np.array(param_data['adaptability'])
    
    # Sort by parameter value
    sort_idx = np.argsort(param_values)
    param_values = param_values[sort_idx]
    coverage_data = coverage_data[sort_idx]
    safety_data = safety_data[sort_idx]
    efficiency_data = efficiency_data[sort_idx]
    adaptability_data = adaptability_data[sort_idx]
    
    # Calculate relative changes from baseline (%)
    coverage_change = ((coverage_data - baseline_coverage) / baseline_coverage) * 100
    safety_change = ((safety_data - baseline_safety) / baseline_safety) * 100
    efficiency_change = ((efficiency_data - baseline_efficiency) / baseline_efficiency) * 100
    adaptability_change = ((adaptability_data - baseline_adaptability) / baseline_adaptability) * 100
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot lines for each objective
    ax.plot(param_values, coverage_change, marker='o', linewidth=2.5, markersize=8, 
            color='#9b59b6', label='Coverage', zorder=3)
    ax.plot(param_values, safety_change, marker='s', linewidth=2.5, markersize=8, 
            color='#e67e22', label='Safety', zorder=3)
    ax.plot(param_values, efficiency_change, marker='^', linewidth=2.5, markersize=8, 
            color='#27ae60', label='Efficiency', zorder=3)
    ax.plot(param_values, adaptability_change, marker='D', linewidth=2.5, markersize=8, 
            color='#e74c3c', label='Adaptability', zorder=3)
    
    # Add horizontal reference lines
    ax.axhline(y=10, color='#e67e22', linestyle='--', linewidth=1.5, alpha=0.7, 
               label='Medium sensitivity (±10%)', zorder=1)
    ax.axhline(y=-10, color='#e67e22', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)
    ax.axhline(y=20, color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.7, 
               label='High sensitivity (±20%)', zorder=1)
    ax.axhline(y=-20, color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5, zorder=1)
    
    # Customize plot
    baseline_val = baseline_values[param_name]
    ax.set_xlabel(f'{param_display_names[param_name]} (Baseline: {baseline_val})', 
                  fontsize=12, fontweight='bold')
    ax.set_ylabel('Relative Change from Baseline (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Relative change based on varying {param_display_names[param_name]}', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--', zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=10)
    
    # Set reasonable y-axis limits
    all_changes = np.concatenate([coverage_change, safety_change, efficiency_change, adaptability_change])
    y_min = all_changes.min()
    y_max = all_changes.max()
    y_range = y_max - y_min
    ax.set_ylim(y_min - y_range*0.15, y_max + y_range*0.15)
    
    # Save figure
    safe_filename = param_name.replace('_', '-')
    output_file = output_dir / f"tradeoff_{safe_filename}.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_file}")
    print(f"  Max coverage change: {max(abs(coverage_change)):.2f}%")
    print(f"  Max adaptability change: {max(abs(adaptability_change)):.2f}%\n")

print(f"\nAll {len(params_data)} trade-off plots generated successfully!")
print(f"Files saved in: {output_dir}")
