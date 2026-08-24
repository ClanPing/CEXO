"""
Visualize Scalability Analysis Results
Creates a 2x2 grid showing coverage, safety, efficiency, and adaptability vs facilities
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load the scalability data
data_file = Path("results/scalability_analysis/all_trials.csv")
df = pd.read_csv(data_file)

# Calculate mean and std for each metric by facility count
summary_stats = []
for n_fac in sorted(df['num_facilities'].unique()):
    fac_data = df[df['num_facilities'] == n_fac]
    
    summary_stats.append({
        'facilities': n_fac,
        'coverage_mean': fac_data['coverage_pct'].mean(),
        'coverage_std': fac_data['coverage_pct'].std(),
        'runtime_mean': fac_data['runtime_seconds'].mean(),
        'runtime_std': fac_data['runtime_seconds'].std()
    })

summary_df = pd.DataFrame(summary_stats)

# Objective estimates (from extract_scalability_results.py)
objectives = {
    3: {'safety': 0.990, 'safety_std': 0.004, 'efficiency': 0.540, 'efficiency_std': 0.040, 'adaptability': 0.500, 'adaptability_std': 0.002},
    4: {'safety': 0.990, 'safety_std': 0.003, 'efficiency': 0.840, 'efficiency_std': 0.030, 'adaptability': 0.520, 'adaptability_std': 0.005},
    5: {'safety': 0.990, 'safety_std': 0.000, 'efficiency': 0.830, 'efficiency_std': 0.026, 'adaptability': 0.510, 'adaptability_std': 0.001},
    6: {'safety': 0.985, 'safety_std': 0.003, 'efficiency': 0.820, 'efficiency_std': 0.009, 'adaptability': 0.500, 'adaptability_std': 0.004},
    7: {'safety': 0.985, 'safety_std': 0.002, 'efficiency': 0.830, 'efficiency_std': 0.031, 'adaptability': 0.480, 'adaptability_std': 0.054},
    8: {'safety': 0.985, 'safety_std': 0.001, 'efficiency': 0.820, 'efficiency_std': 0.036, 'adaptability': 0.460, 'adaptability_std': 0.065}
}

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('Scalability Analysis: Performance vs Number of Facilities', fontsize=14, fontweight='bold')

# Plot 1: Archive Coverage
ax = axes[0, 0]
ax.errorbar(summary_df['facilities'], summary_df['coverage_mean'], 
            yerr=summary_df['coverage_std'], marker='o', markersize=8, 
            linewidth=2, capsize=5, color='#8B4789', label='Coverage')
ax.set_xlabel('Facilities', fontsize=11)
ax.set_ylabel('Coverage (%)', fontsize=11)
ax.set_title('Archive Coverage', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 105)
ax.set_xticks(range(3, 9))

# Plot 2: Average Safety
ax = axes[0, 1]
safety_means = [objectives[n]['safety'] for n in range(3, 9)]
safety_stds = [objectives[n]['safety_std'] for n in range(3, 9)]
ax.errorbar(range(3, 9), safety_means, yerr=safety_stds, 
            marker='o', markersize=8, linewidth=2, capsize=5, 
            color='#FF8C00', label='Safety')
ax.set_xlabel('Facilities', fontsize=11)
ax.set_ylabel('Safety Score', fontsize=11)
ax.set_title('Average Safety', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_ylim(0.982, 0.992)
ax.set_xticks(range(3, 9))

# Plot 3: Average Efficiency
ax = axes[1, 0]
efficiency_means = [objectives[n]['efficiency'] for n in range(3, 9)]
efficiency_stds = [objectives[n]['efficiency_std'] for n in range(3, 9)]
ax.errorbar(range(3, 9), efficiency_means, yerr=efficiency_stds,
            marker='o', markersize=8, linewidth=2, capsize=5,
            color='#6B8E23', label='Efficiency')
ax.set_xlabel('Facilities', fontsize=11)
ax.set_ylabel('Efficiency Score', fontsize=11)
ax.set_title('Average Efficiency', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_ylim(0.50, 0.90)
ax.set_xticks(range(3, 9))

# Plot 4: Average Adaptability
ax = axes[1, 1]
adaptability_means = [objectives[n]['adaptability'] for n in range(3, 9)]
adaptability_stds = [objectives[n]['adaptability_std'] for n in range(3, 9)]
ax.errorbar(range(3, 9), adaptability_means, yerr=adaptability_stds,
            marker='o', markersize=8, linewidth=2, capsize=5,
            color='#DC143C', label='Adaptability')
ax.set_xlabel('Facilities', fontsize=11)
ax.set_ylabel('Adaptability Score', fontsize=11)
ax.set_title('Average Adaptability', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_ylim(0.40, 0.57)
ax.set_xticks(range(3, 9))

# Adjust layout
plt.tight_layout()

# Save figure
output_dir = Path("results/scalability_analysis")
output_file = output_dir / "scalability_performance.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Saved: {output_file}")

# Also save as PDF
output_file_pdf = output_dir / "scalability_performance.pdf"
plt.savefig(output_file_pdf, bbox_inches='tight')
print(f"Saved: {output_file_pdf}")

plt.show()

print("\n" + "="*80)
print("SCALABILITY VISUALIZATION COMPLETE")
print("="*80)
print(f"\nGenerated 2x2 grid showing:")
print("  - Archive Coverage: 21% (3 fac) → 98-99% (4-8 fac)")
print("  - Average Safety: Consistently high (0.985-0.990)")
print("  - Average Efficiency: 0.54 (3 fac) → 0.82-0.84 (4-8 fac)")
print("  - Average Adaptability: Decreases from 0.52 to 0.46")
print("\nThis figure can replace Figure 6 in experiment.md")
