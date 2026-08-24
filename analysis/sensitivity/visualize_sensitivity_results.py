"""
Generate sensitivity analysis visualizations (Figures 7 & 8)
Matches the style from the reference images provided
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load results
results_file = "results/sensitivity_analysis/all_sensitivity_results.json"
with open(results_file, 'r') as f:
    all_results = json.load(f)

# Extract baseline
baseline = [r for r in all_results if r['param_name'] == 'baseline'][0]
baseline_coverage = baseline['coverage_pct']

print(f"Baseline coverage: {baseline_coverage:.2f}%")

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

# Calculate max absolute change for each parameter
param_sensitivities = {}
for param_name, data in params_data.items():
    coverage_values = data['coverage']
    changes = [abs(c - baseline_coverage) for c in coverage_values]
    max_change = max(changes)
    param_sensitivities[param_name] = max_change

# Sort parameters by sensitivity
sorted_params = sorted(param_sensitivities.items(), key=lambda x: x[1], reverse=True)

# Pretty names for display
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

# =============================================================================
# FIGURE 7: TORNADO PLOT (Parameter Sensitivity Ranking)
# =============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

# Prepare data for plotting
param_names = [param_display_names[p[0]] for p in sorted_params]
sensitivities = [p[1] for p in sorted_params]

# Color by sensitivity level
colors = []
for sens in sensitivities:
    if sens >= 20:
        colors.append('#e74c3c')  # Red: High sensitivity (>20%)
    elif sens >= 10:
        colors.append('#e67e22')  # Orange: Medium sensitivity (10-20%)
    else:
        colors.append('#27ae60')  # Green: Low sensitivity (<10%)

# Create horizontal bar chart
bars = ax.barh(param_names, sensitivities, color=colors, edgecolor='black', linewidth=0.5)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, sensitivities)):
    ax.text(val + 0.3, i, f'{val:.2f}', va='center', fontsize=10, fontweight='bold')

# Customize plot
ax.set_xlabel('Max coverage change from baseline (%)', fontsize=12, fontweight='bold')
ax.set_title('Parameter Sensitivity Ranking', fontsize=14, fontweight='bold', pad=20)
ax.set_xlim(0, max(sensitivities) * 1.15)
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#e74c3c', edgecolor='black', label='High sensitivity (>20%)'),
    Patch(facecolor='#e67e22', edgecolor='black', label='Medium sensitivity (10-20%)'),
    Patch(facecolor='#27ae60', edgecolor='black', label='Low sensitivity (<10%)')
]
ax.legend(handles=legend_elements, loc='lower right', frameon=True, fancybox=True, shadow=True)

plt.tight_layout()
plt.savefig('results/sensitivity_analysis/figure_7_tornado_plot.png', dpi=300, bbox_inches='tight')
plt.savefig('results/sensitivity_analysis/figure_7_tornado_plot.pdf', bbox_inches='tight')
print("Saved Figure 7: Tornado plot")

# =============================================================================
# FIGURE 8: TRADE-OFF PLOT (Boundary Margin parameter effects)
# =============================================================================

# Find the parameter with highest sensitivity for detailed plot
most_sensitive_param = sorted_params[0][0]
print(f"\nMost sensitive parameter: {most_sensitive_param} ({sorted_params[0][1]:.2f}% max change)")

# Extract data for most sensitive parameter
param_data = params_data[most_sensitive_param]
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

# Calculate relative changes from baseline
coverage_change = ((coverage_data - baseline_coverage) / baseline_coverage) * 100
safety_change = ((safety_data - baseline['avg_safety']) / baseline['avg_safety']) * 100
efficiency_change = ((efficiency_data - baseline['avg_efficiency']) / baseline['avg_efficiency']) * 100
adaptability_change = ((adaptability_data - baseline['avg_adaptability']) / baseline['avg_adaptability']) * 100

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

# Add horizontal reference lines for sensitivity thresholds
ax.axhline(y=10, color='#e67e22', linestyle='--', linewidth=1.5, alpha=0.7, label='Medium sensitivity (±10%)', zorder=1)
ax.axhline(y=-10, color='#e67e22', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)
ax.axhline(y=20, color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.7, label='High sensitivity (±20%)', zorder=1)
ax.axhline(y=-20, color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5, zorder=1)

# Get baseline value for this parameter
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
baseline_val = baseline_values[most_sensitive_param]

# Customize plot
ax.set_xlabel(f'{param_display_names[most_sensitive_param]} (Baseline: {baseline_val})', 
              fontsize=12, fontweight='bold')
ax.set_ylabel('Relative Change from Baseline (%)', fontsize=12, fontweight='bold')
ax.set_title(f'Relative change based on varying {param_display_names[most_sensitive_param]}', 
             fontsize=14, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--', zorder=0)
ax.set_axisbelow(True)
ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=10)

# Set reasonable y-axis limits
y_min = min(coverage_change.min(), safety_change.min(), efficiency_change.min(), adaptability_change.min())
y_max = max(coverage_change.max(), safety_change.max(), efficiency_change.max(), adaptability_change.max())
y_range = y_max - y_min
ax.set_ylim(y_min - y_range*0.15, y_max + y_range*0.15)

plt.tight_layout()
plt.savefig('results/sensitivity_analysis/figure_8_tradeoff_plot.png', dpi=300, bbox_inches='tight')
plt.savefig('results/sensitivity_analysis/figure_8_tradeoff_plot.pdf', bbox_inches='tight')
print(f"Saved Figure 8: Trade-off plot for {param_display_names[most_sensitive_param]}")

print("\nAll visualizations generated successfully!")
print("Files saved in results/sensitivity_analysis/")
