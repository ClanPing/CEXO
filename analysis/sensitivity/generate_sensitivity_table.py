"""
Generate Markdown Tables for Sensitivity Analysis Results
"""

import json
import pandas as pd
from pathlib import Path

# Load results
results_file = "results/sensitivity_analysis/all_sensitivity_results.json"
with open(results_file, 'r') as f:
    all_results = json.load(f)

# Extract baseline
baseline = [r for r in all_results if r['param_name'] == 'baseline'][0]
baseline_coverage = baseline['coverage_pct']

print(f"Baseline coverage: {baseline_coverage:.2f}%")
print(f"Total results: {len(all_results)}")

# Organize results by parameter
params_data = {}
for result in all_results:
    param_name = result['param_name']
    if param_name == 'baseline':
        continue
    
    if param_name not in params_data:
        params_data[param_name] = []
    
    params_data[param_name].append(result)

# Pretty names for display
param_display_names = {
    'boundary_margin': 'Boundary Margin',
    'crane_safety_distance': 'Crane Safety Distance',
    'entrance_clearance': 'Entrance Clearance',
    'pretrain_iterations': 'Pretrain Iterations',
    'training_frequency': 'Training Frequency',
    'latent_dim': 'Latent Dimensions',
    'initial_population': 'Initial Population',
    'pareto_size': 'Pareto Front Size'
}

# =============================================================================
# Table 1: Summary of Parameter Sensitivity
# =============================================================================

summary_data = []

for param_name, results in params_data.items():
    # Calculate statistics
    coverage_values = [r['coverage_pct'] for r in results]
    safety_values = [r['avg_safety'] for r in results]
    efficiency_values = [r['avg_efficiency'] for r in results]
    adaptability_values = [r['avg_adaptability'] for r in results]
    
    # Calculate ranges
    coverage_range = (min(coverage_values), max(coverage_values))
    safety_range = (min(safety_values), max(safety_values))
    efficiency_range = (min(efficiency_values), max(efficiency_values))
    adaptability_range = (min(adaptability_values), max(adaptability_values))
    
    # Get parameter values tested (sorted)
    param_values = sorted([r['param_value'] for r in results])
    
    # Format values for display
    if all(isinstance(v, (int, float)) for v in param_values):
        if all(isinstance(v, int) or v == int(v) for v in param_values):
            # Integer values
            values_str = ", ".join([str(int(v)) for v in param_values])
        else:
            # Float values
            values_str = ", ".join([f"{v:.3g}" for v in param_values])
    else:
        values_str = ", ".join([str(v) for v in param_values])
    
    # Max absolute change from baseline for sensitivity classification
    coverage_change = max([abs(c - baseline_coverage) for c in coverage_values])
    
    summary_data.append({
        'Parameter': param_display_names[param_name],
        'Values Tested': values_str,
        'Coverage (%)': f"{coverage_range[0]:.2f}–{coverage_range[1]:.2f}",
        'Safety': f"{safety_range[0]:.4f}–{safety_range[1]:.4f}",
        'Efficiency': f"{efficiency_range[0]:.4f}–{efficiency_range[1]:.4f}",
        'Adaptability': f"{adaptability_range[0]:.4f}–{adaptability_range[1]:.4f}",
        'Sensitivity': 'High' if coverage_change >= 20 else ('Medium' if coverage_change >= 10 else 'Low')
    })

# Sort by coverage range spread (descending)
summary_data = sorted(summary_data, key=lambda x: float(x['Coverage (%)'].split('–')[1]) - float(x['Coverage (%)'].split('–')[0]), reverse=True)

# Create DataFrame
df_summary = pd.DataFrame(summary_data)

# Generate markdown table
table1_md = "### Table: Sensitivity Analysis Summary\n\n"
table1_md += "Parameter sensitivity showing performance ranges across tested values. Baseline: Coverage 99.75%, Safety 0.9835, Efficiency 0.7490, Adaptability 0.4927\n\n"
table1_md += df_summary.to_markdown(index=False)

print("\n" + "="*80)
print("TABLE 1: Summary of Parameter Sensitivity")
print("="*80)
print(table1_md)

# =============================================================================
# Table 2: Detailed Results by Parameter
# =============================================================================

detailed_tables = []

for param_name in sorted(params_data.keys()):
    results = params_data[param_name]
    display_name = param_display_names[param_name]
    
    # Sort by parameter value
    results_sorted = sorted(results, key=lambda x: x['param_value'])
    
    # Build detailed table
    detailed_data = []
    for r in results_sorted:
        coverage_change = r['coverage_pct'] - baseline_coverage
        safety_change = ((r['avg_safety'] - baseline['avg_safety']) / baseline['avg_safety']) * 100
        efficiency_change = ((r['avg_efficiency'] - baseline['avg_efficiency']) / baseline['avg_efficiency']) * 100
        adaptability_change = ((r['avg_adaptability'] - baseline['avg_adaptability']) / baseline['avg_adaptability']) * 100
        
        detailed_data.append({
            'Value': r['param_value'],
            'Coverage (%)': f"{r['coverage_pct']:.2f}",
            'Δ Coverage (%)': f"{coverage_change:+.2f}",
            'Safety': f"{r['avg_safety']:.4f}",
            'Efficiency': f"{r['avg_efficiency']:.4f}",
            'Adaptability': f"{r['avg_adaptability']:.4f}",
            'Feasible': f"{r['feasible_count']}/{r['total_individuals']}"
        })
    
    df_detailed = pd.DataFrame(detailed_data)
    
    table_md = f"\n### {display_name}\n\n"
    table_md += df_detailed.to_markdown(index=False)
    
    detailed_tables.append(table_md)
    
    print(f"\n{display_name}:")
    print(df_detailed.to_string(index=False))

# =============================================================================
# Save to file
# =============================================================================

output_file = "results/sensitivity_analysis/table_sensitivity.md"

with open(output_file, 'w', encoding='utf-8') as f:
    f.write("# Sensitivity Analysis Results\n\n")
    f.write(f"**Baseline Configuration:**\n")
    f.write(f"- Coverage: {baseline['coverage_pct']:.2f}%\n")
    f.write(f"- Safety: {baseline['avg_safety']:.4f}\n")
    f.write(f"- Efficiency: {baseline['avg_efficiency']:.4f}\n")
    f.write(f"- Adaptability: {baseline['avg_adaptability']:.4f}\n\n")
    f.write("---\n\n")
    f.write(table1_md)
    f.write("\n\n---\n\n")
    f.write("## Detailed Results by Parameter\n\n")
    for table in detailed_tables:
        f.write(table)
        f.write("\n")

print("\n" + "="*80)
print(f"✓ Tables saved to: {output_file}")
print("="*80)
