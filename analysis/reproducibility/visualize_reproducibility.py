#!/usr/bin/env python3
"""
Generate box plots for reproducibility analysis results
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_reproducibility_boxplots():
    """Create box plots for reproducibility analysis"""
    
    # Load data
    csv_file = "results/reproducibility_analysis/intermediate_results.csv"
    
    if not os.path.exists(csv_file):
        print(f"✗ CSV file not found: {csv_file}")
        return
    
    df = pd.read_csv(csv_file)
    
    print(f"\nLoaded {len(df)} trials")
    print(f"Creating box plots...")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10
    
    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle('Reproducibility Analysis of CEXO (n=30)', fontsize=14, fontweight='bold', y=0.995)
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    # Define metrics to plot
    metrics = [
        ('coverage_pct', 'Coverage (%)', '.2f'),
        ('avg_safety', 'Average Safety', '.4f'),
        ('avg_efficiency', 'Average Efficiency', '.3f'),
        ('avg_adaptability', 'Average Adaptability', '.3f'),
        ('bd_avg', 'Behavioural Diversity', '.3f'),
        ('safety_feasible_count', 'Safety Feasible Count', '.0f')
    ]
    
    # Create box plots
    for idx, (metric, title, fmt) in enumerate(metrics):
        ax = axes[idx]
        
        # Create box plot
        bp = ax.boxplot([df[metric]], widths=0.5, patch_artist=True,
                        boxprops=dict(facecolor='lightblue', edgecolor='navy', linewidth=1.5),
                        medianprops=dict(color='red', linewidth=2),
                        whiskerprops=dict(color='navy', linewidth=1.5),
                        capprops=dict(color='navy', linewidth=1.5),
                        flierprops=dict(marker='o', markerfacecolor='red', markersize=6, 
                                       markeredgecolor='darkred', alpha=0.7))
        
        # Add statistics text box
        mean = df[metric].mean()
        median = df[metric].median()
        std = df[metric].std()
        
        stats_text = f"Mean: {mean:{fmt}}\nMedian: {median:{fmt}}\nStd: {std:{fmt}}"
        
        # Position text box in upper left
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=0.5))
        
        # Set title and labels
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.set_ylabel('Value', fontsize=10)
        ax.set_xticks([])
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save figure
    output_dir = "results/reproducibility_analysis"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "reproducibility_boxplots.png")
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Box plots saved to: {output_file}")
    
    # Also save as high-res PDF
    pdf_file = os.path.join(output_dir, "reproducibility_boxplots.pdf")
    plt.savefig(pdf_file, bbox_inches='tight')
    print(f"✓ PDF version saved to: {pdf_file}")
    
    plt.close()
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    create_reproducibility_boxplots()
