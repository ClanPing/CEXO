#!/usr/bin/env python3
"""
Algorithm Performance Comparison
=================================

Comprehensive comparison of CEXO, MAP-Elites, and NSGA-II algorithms
for construction site layout optimization.

Metrics compared:
- Feasible solutions count
- Coverage (for MAP-Elites variants)
- Distinct layouts
- Average Safety
- Average Efficiency
- Average Adaptability
- Computation time
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List
import seaborn as sns

# Set CUDA environment variables for reproducibility
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


def count_distinct_layouts(individuals, tolerance=1e-4):
    """
    Count distinct layouts by comparing facility positions.
    
    Args:
        individuals: List of Individual objects
        tolerance: Position tolerance for considering layouts as identical
    
    Returns:
        Number of distinct layouts
    """
    if not individuals:
        return 0
    
    distinct_count = 0
    compared = []
    
    for ind in individuals:
        is_distinct = True
        for comp_ind in compared:
            # Compare facility positions
            if len(ind.solution) != len(comp_ind.solution):
                continue
            
            # Sort both by facility type for fair comparison
            sorted_ind = sorted(ind.solution, key=lambda f: f['type'])
            sorted_comp = sorted(comp_ind.solution, key=lambda f: f['type'])
            
            # Check if positions are similar
            all_similar = True
            for f1, f2 in zip(sorted_ind, sorted_comp):
                if f1['type'] != f2['type']:
                    all_similar = False
                    break
                    
                center_diff = np.linalg.norm(
                    np.array(f1['center']) - np.array(f2['center'])
                )
                if center_diff > tolerance:
                    all_similar = False
                    break
            
            if all_similar:
                is_distinct = False
                break
        
        if is_distinct:
            distinct_count += 1
            compared.append(ind)
    
    return distinct_count


def get_best_cell_representatives(archive):
    """Return one best representative from each occupied behavioural cell."""
    representatives = []
    for coords in archive.archive:
        if hasattr(archive, "get_best_for_cell"):
            individual = archive.get_best_for_cell(coords)
        else:
            individual = archive.archive[coords]
        if individual is not None:
            representatives.append(individual)
    return representatives


def run_cexo_experiment(site_config, facility_types, mapelites_config, 
                        autoencoder_config, experiment_id):
    """Run CEXO (MAP-Elites with Autoencoder) experiment"""
    print("\n" + "="*80)
    print("RUNNING CEXO (MAP-Elites with Autoencoder)")
    print("="*80)
    
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
    
    computation_time = time.time() - start_time
    
    # Report one best representative per occupied behavioural cell. The full
    # CEXO archive still stores bounded Pareto fronts inside each occupied cell.
    all_archive_individuals = algorithm.archive.get_all_individuals()
    reported_individuals = get_best_cell_representatives(algorithm.archive)
    feasible_individuals = [ind for ind in reported_individuals if ind.feasible]
    safe_individuals = [ind for ind in reported_individuals if ind.objectives[0] >= 0.7]
    
    # Calculate metrics
    objectives = np.array([ind.objectives for ind in reported_individuals])
    
    stats = results.get('stats', algorithm.archive.get_stats())
    
    metrics = {
        'algorithm': 'CEXO',
        'reported_solution_set': 'best_per_occupied_cell',
        'total_solutions': len(reported_individuals),
        'stored_pareto_solutions': len(all_archive_individuals),
        'feasible_solutions': len(feasible_individuals),
        'safe_solutions': len(safe_individuals),
        'coverage': stats.get('coverage', len(algorithm.archive.archive)),
        'coverage_pct': stats.get('coverage_pct', 100.0 * len(algorithm.archive.archive) / 400),
        'distinct_layouts': count_distinct_layouts(reported_individuals),
        'avg_safety': float(np.mean(objectives[:, 0])),
        'avg_efficiency': float(np.mean(objectives[:, 1])),
        'avg_adaptability': float(np.mean(objectives[:, 2])),
        'max_safety': float(np.max(objectives[:, 0])),
        'max_efficiency': float(np.max(objectives[:, 1])),
        'max_adaptability': float(np.max(objectives[:, 2])),
        'computation_time': computation_time,
        'evaluations': algorithm.archive.evaluations,
        'autoencoder_trained': algorithm.autoencoder_trained,
        'bd_mode': algorithm.bd_manager.get_mode()
    }
    
    print("\n" + "-"*80)
    print("CEXO RESULTS:")
    print(f"  Reported Solution Set: best representative per occupied cell")
    print(f"  Stored Pareto Solutions: {metrics['stored_pareto_solutions']}")
    print(f"  Total Solutions: {metrics['total_solutions']}")
    print(f"  Feasible Solutions: {metrics['feasible_solutions']}")
    print(f"  Safe Solutions (>=0.7): {metrics['safe_solutions']}")
    print(f"  Coverage: {metrics['coverage']} cells ({metrics['coverage_pct']:.1f}%)")
    print(f"  Distinct Layouts: {metrics['distinct_layouts']}")
    print(f"  Avg Safety: {metrics['avg_safety']:.4f}")
    print(f"  Avg Efficiency: {metrics['avg_efficiency']:.4f}")
    print(f"  Avg Adaptability: {metrics['avg_adaptability']:.4f}")
    print(f"  Computation Time: {computation_time:.2f}s")
    print(f"  BD Mode: {metrics['bd_mode']}")
    print("-"*80)
    
    return metrics, algorithm, reported_individuals


def run_mapelites_experiment(site_config, facility_types, mapelites_config, 
                              experiment_id):
    """Run Pure MAP-Elites experiment"""
    print("\n" + "="*80)
    print("RUNNING MAP-ELITES (Manual BDs)")
    print("="*80)
    
    start_time = time.time()
    
    algorithm = PureMapElitesOptimizer(
        site_config=site_config,
        facility_types=facility_types,
        mapelites_config=mapelites_config
    )
    
    results = algorithm.run(
        iterations=mapelites_config.iterations,
        initial_population=mapelites_config.initial_population
    )
    
    computation_time = time.time() - start_time
    
    # Get all individuals
    all_individuals = algorithm.archive.get_all_individuals()
    feasible_individuals = [ind for ind in all_individuals if ind.feasible]
    safe_individuals = [ind for ind in all_individuals if ind.objectives[0] >= 0.7]
    
    # Calculate metrics
    objectives = np.array([ind.objectives for ind in all_individuals])
    stats = results.get('stats', algorithm.archive.get_stats())
    
    metrics = {
        'algorithm': 'MAP-Elites',
        'total_solutions': len(all_individuals),
        'feasible_solutions': len(feasible_individuals),
        'safe_solutions': len(safe_individuals),
        'coverage': stats.get('coverage', len(algorithm.archive.archive)),
        'coverage_pct': stats.get('coverage_pct', 100.0 * len(algorithm.archive.archive) / 400),
        'distinct_layouts': count_distinct_layouts(all_individuals),
        'avg_safety': float(np.mean(objectives[:, 0])),
        'avg_efficiency': float(np.mean(objectives[:, 1])),
        'avg_adaptability': float(np.mean(objectives[:, 2])),
        'max_safety': float(np.max(objectives[:, 0])),
        'max_efficiency': float(np.max(objectives[:, 1])),
        'max_adaptability': float(np.max(objectives[:, 2])),
        'computation_time': computation_time,
        'evaluations': algorithm.archive.evaluations,
        'bd_type': 'manual'
    }
    
    print("\n" + "-"*80)
    print("MAP-ELITES RESULTS:")
    print(f"  Total Solutions: {metrics['total_solutions']}")
    print(f"  Feasible Solutions: {metrics['feasible_solutions']}")
    print(f"  Safe Solutions (>=0.7): {metrics['safe_solutions']}")
    print(f"  Coverage: {metrics['coverage']} cells ({metrics['coverage_pct']:.1f}%)")
    print(f"  Distinct Layouts: {metrics['distinct_layouts']}")
    print(f"  Avg Safety: {metrics['avg_safety']:.4f}")
    print(f"  Avg Efficiency: {metrics['avg_efficiency']:.4f}")
    print(f"  Avg Adaptability: {metrics['avg_adaptability']:.4f}")
    print(f"  Computation Time: {computation_time:.2f}s")
    print(f"  BD Type: {metrics['bd_type']}")
    print("-"*80)
    
    return metrics, algorithm, all_individuals


def run_nsga2_experiment(site_config, facility_types, nsga2_config, experiment_id):
    """Run NSGA-II experiment"""
    print("\n" + "="*80)
    print("RUNNING NSGA-II (Pure Multi-Objective)")
    print("="*80)
    
    start_time = time.time()
    
    algorithm = PureNSGA2Optimizer(
        site_config=site_config,
        facility_types=facility_types,
        nsga2_config=nsga2_config
    )
    
    results = algorithm.run(
        generations=nsga2_config.generations,
        population_size=nsga2_config.population_size
    )
    
    computation_time = time.time() - start_time
    
    # Get Pareto front and all evaluated individuals
    pareto_front = results['pareto_front']
    
    # For NSGA-II, we only keep the final population
    # Approximate total distinct solutions as population_size * generations
    # But for fair comparison, use the Pareto front
    all_individuals = pareto_front
    feasible_individuals = [ind for ind in all_individuals if ind.feasible]
    safe_individuals = [ind for ind in all_individuals if ind.objectives[0] >= 0.7]
    
    # Calculate metrics
    objectives = np.array([ind.objectives for ind in all_individuals])
    
    metrics = {
        'algorithm': 'NSGA-II',
        'total_solutions': len(all_individuals),  # Pareto front size
        'feasible_solutions': len(feasible_individuals),
        'safe_solutions': len(safe_individuals),
        'coverage': 'N/A',  # No grid-based coverage
        'coverage_pct': 'N/A',
        'distinct_layouts': count_distinct_layouts(all_individuals),
        'avg_safety': float(np.mean(objectives[:, 0])),
        'avg_efficiency': float(np.mean(objectives[:, 1])),
        'avg_adaptability': float(np.mean(objectives[:, 2])),
        'max_safety': float(np.max(objectives[:, 0])),
        'max_efficiency': float(np.max(objectives[:, 1])),
        'max_adaptability': float(np.max(objectives[:, 2])),
        'computation_time': computation_time,
        'evaluations': results['evaluations'],
        'pareto_front_size': len(pareto_front)
    }
    
    print("\n" + "-"*80)
    print("NSGA-II RESULTS:")
    print(f"  Pareto Front Size: {metrics['total_solutions']}")
    print(f"  Feasible Solutions: {metrics['feasible_solutions']}")
    print(f"  Safe Solutions (>=0.7): {metrics['safe_solutions']}")
    print(f"  Distinct Layouts: {metrics['distinct_layouts']}")
    print(f"  Avg Safety: {metrics['avg_safety']:.4f}")
    print(f"  Avg Efficiency: {metrics['avg_efficiency']:.4f}")
    print(f"  Avg Adaptability: {metrics['avg_adaptability']:.4f}")
    print(f"  Computation Time: {computation_time:.2f}s")
    print("-"*80)
    
    return metrics, algorithm, all_individuals


def create_comparison_visualizations(all_results, output_dir):
    """Create comprehensive comparison visualizations"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for plotting
    algorithms = [r['algorithm'] for r in all_results]
    
    # Set style
    sns.set_style("whitegrid")
    colors = {'CEXO': '#2E86AB', 'MAP-Elites': '#A23B72', 'NSGA-II': '#F18F01'}
    
    # 1. Main Metrics Comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Algorithm Performance Comparison', fontsize=16, fontweight='bold')
    
    # Feasible Solutions
    ax = axes[0, 0]
    feasible = [r['feasible_solutions'] for r in all_results]
    bars = ax.bar(algorithms, feasible, color=[colors[a] for a in algorithms], alpha=0.8, edgecolor='black')
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Feasible Solutions', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(feasible):
        ax.text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
    
    # Distinct Layouts
    ax = axes[0, 1]
    distinct = [r['distinct_layouts'] for r in all_results]
    bars = ax.bar(algorithms, distinct, color=[colors[a] for a in algorithms], alpha=0.8, edgecolor='black')
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distinct Layouts', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(distinct):
        ax.text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
    
    # Coverage (only for MAP-Elites variants)
    ax = axes[0, 2]
    coverage_data = []
    coverage_labels = []
    coverage_colors = []
    for r in all_results:
        if r['coverage'] != 'N/A':
            coverage_data.append(r['coverage'])
            coverage_labels.append(r['algorithm'])
            coverage_colors.append(colors[r['algorithm']])
    if coverage_data:
        bars = ax.bar(coverage_labels, coverage_data, color=coverage_colors, alpha=0.8, edgecolor='black')
        ax.set_ylabel('Cells Occupied', fontsize=12)
        ax.set_title('Archive Coverage', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        for i, v in enumerate(coverage_data):
            pct = (v / 400) * 100  # Assuming 20x20 grid
            ax.text(i, v, f'{v}\n({pct:.1f}%)', ha='center', va='bottom', fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'N/A for NSGA-II', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Archive Coverage', fontsize=14, fontweight='bold')
    
    # Average Safety
    ax = axes[1, 0]
    safety = [r['avg_safety'] for r in all_results]
    bars = ax.bar(algorithms, safety, color=[colors[a] for a in algorithms], alpha=0.8, edgecolor='black')
    ax.set_ylabel('Score (0-1)', fontsize=12)
    ax.set_title('Average Safety', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(safety):
        ax.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Average Efficiency
    ax = axes[1, 1]
    efficiency = [r['avg_efficiency'] for r in all_results]
    bars = ax.bar(algorithms, efficiency, color=[colors[a] for a in algorithms], alpha=0.8, edgecolor='black')
    ax.set_ylabel('Score (0-1)', fontsize=12)
    ax.set_title('Average Efficiency', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(efficiency):
        ax.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Average Adaptability
    ax = axes[1, 2]
    adaptability = [r['avg_adaptability'] for r in all_results]
    bars = ax.bar(algorithms, adaptability, color=[colors[a] for a in algorithms], alpha=0.8, edgecolor='black')
    ax.set_ylabel('Score (0-1)', fontsize=12)
    ax.set_title('Average Adaptability', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(adaptability):
        ax.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'algorithm_comparison_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Computation Time Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    times = [r['computation_time'] for r in all_results]
    bars = ax.bar(algorithms, times, color=[colors[a] for a in algorithms], alpha=0.8, edgecolor='black')
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Computation Time Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(times):
        ax.text(i, v, f'{v:.1f}s', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'computation_time.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Multi-objective radar chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    objectives_labels = ['Safety', 'Efficiency', 'Adaptability']
    angles = np.linspace(0, 2 * np.pi, len(objectives_labels), endpoint=False).tolist()
    angles += angles[:1]
    
    for r in all_results:
        values = [r['avg_safety'], r['avg_efficiency'], r['avg_adaptability']]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=r['algorithm'], 
                color=colors[r['algorithm']], markersize=8)
        ax.fill(angles, values, alpha=0.15, color=colors[r['algorithm']])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(objectives_labels, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.set_title('Average Objective Scores', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'objectives_radar.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Summary table
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    headers = ['Metric'] + [r['algorithm'] for r in all_results]
    
    metrics_to_show = [
        ('Feasible Solutions', 'feasible_solutions'),
        ('Distinct Layouts', 'distinct_layouts'),
        ('Coverage', 'coverage'),
        ('Avg Safety', 'avg_safety'),
        ('Avg Efficiency', 'avg_efficiency'),
        ('Avg Adaptability', 'avg_adaptability'),
        ('Computation Time (s)', 'computation_time')
    ]
    
    for label, key in metrics_to_show:
        row = [label]
        for r in all_results:
            value = r[key]
            if key in ['avg_safety', 'avg_efficiency', 'avg_adaptability']:
                row.append(f'{value:.4f}')
            elif key == 'computation_time':
                row.append(f'{value:.2f}')
            else:
                row.append(str(value))
        table_data.append(row)
    
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4A4A4A')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color cells by algorithm - adapt to number of results
    cell_colors = ['#E3F2FD', '#FCE4EC', '#FFF3E0']  # CEXO, MAP-Elites, NSGA-II
    for i in range(1, len(table_data) + 1):
        table[(i, 0)].set_facecolor('#E8E8E8')
        table[(i, 0)].set_text_props(weight='bold')
        for j in range(1, len(headers)):
            table[(i, j)].set_facecolor(cell_colors[(j-1) % len(cell_colors)])
    
    plt.title('Algorithm Performance Summary', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(os.path.join(output_dir, 'comparison_table.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualizations saved to {output_dir}/")


def save_results(all_results, output_dir):
    """Save comparison results to JSON, CSV, and README-ready Markdown."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON
    json_path = os.path.join(output_dir, 'comparison_results.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {json_path}")
    
    # Save CSV
    csv_path = os.path.join(output_dir, 'comparison_results.csv')
    df = pd.DataFrame(all_results)
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    md_path = os.path.join(output_dir, 'comparison_results.md')
    with open(md_path, 'w') as f:
        f.write(render_markdown_comparison(all_results))
    print(f"Results saved to {md_path}")


def _format_feasible_ratio(result):
    total = result['total_solutions']
    feasible = result['feasible_solutions']
    pct = 100.0 * feasible / total if total else 0.0
    return f"{pct:.1f} ({feasible}/{total})"


def _format_safe_ratio(result):
    total = result['total_solutions']
    safe = result.get('safe_solutions', 0)
    pct = 100.0 * safe / total if total else 0.0
    return f"{pct:.1f} ({safe}/{total})"


def _format_coverage(result):
    coverage_pct = result['coverage_pct']
    if coverage_pct == 'N/A':
        return 'N/A'
    return f"{float(coverage_pct):.1f}"


def render_markdown_comparison(all_results):
    """Create a compact quantitative comparison table for README use."""
    lines = [
        "### Quantitative Comparison",
        "",
        "| Algorithm | Strict Feasible (%) | Safety >= 0.7 (%) | Behavioural Coverage (%) | Distinct Layouts | Avg Safety | Avg Efficiency | Avg Adaptability |",
        "|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|",
    ]

    for result in all_results:
        lines.append(
            "| {algorithm} | {feasible} | {safe} | {coverage} | {distinct} | {safety:.3f} | {efficiency:.3f} | {adaptability:.3f} |".format(
                algorithm=result['algorithm'],
                feasible=_format_feasible_ratio(result),
                safe=_format_safe_ratio(result),
                coverage=_format_coverage(result),
                distinct=result['distinct_layouts'],
                safety=result['avg_safety'],
                efficiency=result['avg_efficiency'],
                adaptability=result['avg_adaptability'],
            )
        )

    lines.extend([
        "",
        "Notes:",
        "- CEXO reports one best representative layout per occupied behavioural cell; the full archive stores additional Pareto alternatives inside each cell.",
        "- Strict feasibility means no recorded boundary, overlap, crane-safety, or entrance-clearance violations.",
        "- Safety >= 0.7 reports the looser safety-threshold metric used in several earlier result summaries.",
        "- Behavioural coverage is reported for archive-based methods only.",
        "- NSGA-II reports the final Pareto front and does not use a behavioural archive.",
        "- Values are generated from the current run configuration and random seed.",
        "",
    ])

    return "\n".join(lines)


def main():
    """Main comparison experiment"""
    parser = argparse.ArgumentParser(description="Compare CEXO, MAP-Elites, and NSGA-II.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the shared comparison run.")
    parser.add_argument("--facilities", type=int, default=6, help="Number of facilities in the shared comparison.")
    parser.add_argument("--iterations", type=int, default=15000, help="CEXO/MAP-Elites optimisation iterations.")
    parser.add_argument("--initial-pop", type=int, default=500, help="CEXO/MAP-Elites initial population.")
    parser.add_argument("--nsga-population", type=int, default=100, help="NSGA-II population size.")
    parser.add_argument("--nsga-generations", type=int, default=150, help="NSGA-II generations.")
    parser.add_argument("--nsga-crossover", type=float, default=0.8, help="NSGA-II crossover probability.")
    parser.add_argument("--nsga-mutation", type=float, default=0.1, help="NSGA-II mutation probability.")
    parser.add_argument("--pretrain", type=int, default=5000, help="CEXO autoencoder pretraining iterations.")
    parser.add_argument("--train-freq", type=int, default=2500, help="CEXO autoencoder retraining frequency.")
    parser.add_argument("--output", type=str, default="results/algorithm_comparison", help="Output directory for comparison files.")
    args = parser.parse_args()
    
    # Configuration
    experiment_id = "comparison_v1"
    output_dir = args.output
    
    print("\n" + "="*80)
    print("ALGORITHM PERFORMANCE COMPARISON")
    print("CEXO vs MAP-Elites vs NSGA-II")
    print("="*80)
    
    # Common configuration
    seed = args.seed
    num_facilities = args.facilities
    
    # Site configuration (shared)
    site_config = SiteConfig(
        seed=seed,
        boundary_margin=0.05,
        pareto_size=12,
        facility_count=num_facilities
    )
    
    # Generate facility mix
    facility_types = generate_facility_mix(num_facilities, seed=seed)
    print(f"\nFacility mix ({num_facilities} facilities): {facility_types}")
    
    # Algorithm-specific configurations
    # Make them comparable in terms of computational budget
    total_evaluations = args.iterations  # Target similar number of evaluations
    
    # MAP-Elites configuration
    mapelites_config = MapElitesConfig(
        grid_size=(20, 20),  # 400 cells
        iterations=args.iterations,
        initial_population=args.initial_pop
    )
    
    # Autoencoder configuration for CEXO
    autoencoder_config = AutoencoderConfig(
        use_learned_descriptors=True,
        pretrain_iterations=args.pretrain,
        latent_dim=2,
        encoder_hidden=[128, 64, 32],
        decoder_hidden=[32, 64, 128],
        learning_rate=0.001,
        training_epochs=50,
        batch_size=32,
        training_frequency=args.train_freq,
        min_samples_for_training=min(200, args.initial_pop),
        seed=seed
    )
    
    # NSGA-II configuration
    nsga2_config = NSGA2Config(
        population_size=args.nsga_population,
        generations=args.nsga_generations,
        crossover_rate=args.nsga_crossover,
        mutation_rate=args.nsga_mutation
    )
    
    print("\nConfiguration:")
    print(f"  Seed: {seed}")
    print(f"  Facilities: {num_facilities}")
    print(f"  Approximate Evaluation Budget: ~{total_evaluations}")
    print(f"  MAP-Elites: {mapelites_config.grid_size[0]}x{mapelites_config.grid_size[1]} grid, "
          f"{mapelites_config.iterations} iterations, {mapelites_config.initial_population} initial pop")
    print(f"  CEXO: Same as MAP-Elites + Autoencoder (switch at iteration {autoencoder_config.pretrain_iterations})")
    print(
        f"  NSGA-II: {nsga2_config.population_size} population, "
        f"{nsga2_config.generations} generations, crossover {nsga2_config.crossover_rate}, "
        f"mutation {nsga2_config.mutation_rate}"
    )
    
    # Run experiments
    all_results = []
    
    # 1. CEXO
    try:
        cexo_metrics, cexo_algo, cexo_individuals = run_cexo_experiment(
            site_config, facility_types, mapelites_config, 
            autoencoder_config, experiment_id
        )
        all_results.append(cexo_metrics)
    except Exception as e:
        print(f"\nCEXO failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. MAP-Elites
    try:
        mapelites_metrics, mapelites_algo, mapelites_individuals = run_mapelites_experiment(
            site_config, facility_types, mapelites_config, experiment_id
        )
        all_results.append(mapelites_metrics)
    except Exception as e:
        print(f"\nMAP-Elites failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. NSGA-II
    try:
        nsga2_metrics, nsga2_algo, nsga2_individuals = run_nsga2_experiment(
            site_config, facility_types, nsga2_config, experiment_id
        )
        all_results.append(nsga2_metrics)
    except Exception as e:
        print(f"\nNSGA-II failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Generate comparison visualizations
    if len(all_results) > 0:
        print("\n" + "="*80)
        print("GENERATING COMPARISON VISUALIZATIONS")
        print("="*80)
        
        create_comparison_visualizations(all_results, output_dir)
        save_results(all_results, output_dir)
        
        # Print summary
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        
        for r in all_results:
            print(f"\n{r['algorithm']}:")
            print(f"  Feasible Solutions: {r['feasible_solutions']}")
            print(f"  Distinct Layouts: {r['distinct_layouts']}")
            print(f"  Coverage: {r['coverage']}")
            print(f"  Avg Safety: {r['avg_safety']:.4f}")
            print(f"  Avg Efficiency: {r['avg_efficiency']:.4f}")
            print(f"  Avg Adaptability: {r['avg_adaptability']:.4f}")
            print(f"  Time: {r['computation_time']:.2f}s")
        
        print("\n" + "="*80)
        print(f"Comparison complete! Results saved to {output_dir}/")
        print("="*80)
    else:
        print("\nNo results to compare")


if __name__ == "__main__":
    main()
