#!/usr/bin/env python3
"""
Process existing reproducibility results and generate statistics
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats


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
    
    return f"| {metric_name:<25} | {mean_std:<18} | {cv_str:<10} | {ci:<23} | {iqr_str:<9} | {range_str:<9} |"


def main():
    output_dir = "results/reproducibility_analysis"
    
    # Load existing CSV
    csv_file = os.path.join(output_dir, 'intermediate_results.csv')
    
    if not os.path.exists(csv_file):
        print(f"✗ CSV file not found: {csv_file}")
        return
    
    print("\n" + "="*80)
    print("PROCESSING EXISTING REPRODUCIBILITY RESULTS")
    print("="*80)
    
    df = pd.read_csv(csv_file)
    
    print(f"\nLoaded {len(df)} trials from CSV")
    print(f"Columns: {list(df.columns)}")
    
    # Calculate statistics
    stats_summary = calculate_statistics(df)
    
    # Save to JSON
    with open(os.path.join(output_dir, 'statistics.json'), 'w') as f:
        json.dump(stats_summary, f, indent=2)
    
    print("\n✓ Statistics saved to statistics.json")
    
    # Generate markdown table
    print("\n" + "="*80)
    print("REPRODUCIBILITY STATISTICS SUMMARY")
    print("="*80)
    
    print("\n**Table 6.** Reproducibility analysis statistical summary.\n")
    print("| **Metric**                | **Mean ± Std**     | **CV (%)** | **CI<sub>95</sub>**    | **IQR**   | **Range** |")
    print("|---------------------------|--------------------|-----------|-----------------------|-----------|-----------|")
    
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
        f.write("| **Metric**                | **Mean ± Std**     | **CV (%)** | **CI<sub>95</sub>**    | **IQR**   | **Range** |\n")
        f.write("|---------------------------|--------------------|-----------|-----------------------|-----------|-----------|")
        f.write("\n")
        for metric_key, metric_label in metric_labels.items():
            f.write(format_table_row(metric_label, stats_summary[metric_key]) + "\n")
    
    print("\n✓ Table saved to table_6.md")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nProcessed {len(df)} trials successfully")
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
