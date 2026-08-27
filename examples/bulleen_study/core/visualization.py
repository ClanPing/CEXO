#!/usr/bin/env python3
"""
Visualization Module
====================

Comprehensive visualization functions for both MAP-Elites and NSGA-II results.
Includes layout visualizations, performance analysis, and comparative studies.
"""

import os
import json
from typing import List, Dict, Optional
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
from mpl_toolkits.mplot3d import Axes3D

from .config import (
    Individual,
    SiteConfig,
    FACILITY_SPECS,
    FACILITY_COLORS,
    get_facility_dimensions,
)


def draw_site_boundary(ax, config: SiteConfig, **kwargs):
    """Draw rectangular or polygonal site boundary."""
    if config.boundary_polygon:
        xs = [p[0] for p in config.boundary_polygon] + [config.boundary_polygon[0][0]]
        ys = [p[1] for p in config.boundary_polygon] + [config.boundary_polygon[0][1]]
        ax.plot(xs, ys, **kwargs)

        for zone in config.exclusion_zones or []:
            polygon = zone["polygon"]
            zx = [p[0] for p in polygon] + [polygon[0][0]]
            zy = [p[1] for p in polygon] + [polygon[0][1]]
            ax.fill(zx, zy, facecolor='lightgray', edgecolor='dimgray',
                    linewidth=1.5, alpha=0.65, hatch='//')
    else:
        margin = config.boundary_margin
        boundary_x = [margin, 1-margin, 1-margin, margin, margin]
        boundary_y = [margin, margin, 1-margin, 1-margin, margin]
        ax.plot(boundary_x, boundary_y, **kwargs)

# =============================================================================
# LAYOUT VISUALIZATION
# =============================================================================

def visualize_layout(ax, individual: Individual, config: SiteConfig, title: str = "Layout"):
    """Visualize layout with 3-objective information"""
    facilities = individual.solution
    entrances = individual.entrances
    objectives = individual.objectives
    behaviors = individual.behaviors
    
    # Draw boundary
    draw_site_boundary(ax, config, color='black', linestyle='-', linewidth=2)
    
    # Draw entrance clearance zones
    for entrance in entrances:
        clearance_circle = Circle(entrance, config.entrance_clearance, 
                                fill=False, linestyle='-', edgecolor='red', alpha=0.5, linewidth=1)
        ax.add_patch(clearance_circle)
    
    # Draw entrances
    entrance_colors = ['gold', 'orange', 'yellow', 'lightcoral']
    for i, entrance in enumerate(entrances):
        color = entrance_colors[i % len(entrance_colors)]
        edge_color = 'darkorange' if i == 0 else 'darkred'
        
        ax.plot(entrance[0], entrance[1], marker='*', markersize=16, 
                color=color, markeredgecolor=edge_color, markeredgewidth=2, zorder=10)
        
        label = f'E{i+1}' if len(entrances) > 1 else 'ENTRANCE'
        offset_y = -0.06 if entrance[1] < 0.5 else 0.06
        ax.text(entrance[0], entrance[1] + offset_y, label, ha='center', 
                va='bottom' if entrance[1] < 0.5 else 'top',
                fontsize=7, fontweight='bold', color=edge_color)
    
    # Draw facilities
    for facility in facilities:
        ftype = facility["type"]
        x, y = facility["center"]
        spec = FACILITY_SPECS[ftype]
        w, h = get_facility_dimensions(facility)
        
        category = spec["category"]
        if category == "worker":
            rect = Rectangle((x - w/2, y - h/2), w, h, 
                            facecolor=FACILITY_COLORS[ftype], 
                            edgecolor='darkblue', linewidth=2, alpha=0.85)
        else:
            rect = Rectangle((x - w/2, y - h/2), w, h, 
                            facecolor=FACILITY_COLORS[ftype], 
                            edgecolor='black', linewidth=1.5, alpha=0.8)
        ax.add_patch(rect)
        
        # Add facility labels
        label_map = {"core": "C", "crane": "CR", "storage": "S", "office": "OF", "rest_area": "RA"}
        label = label_map.get(ftype, ftype[0].upper())
        ax.text(x, y, label, ha='center', va='center', 
                fontsize=9, fontweight='bold', color='white')
        
        # Crane danger zones
        if ftype == "crane":
            operating_circle = Circle((x, y), spec["operating_radius"],
                                    fill=False, linestyle='--', edgecolor='darkorange',
                                    alpha=0.7, linewidth=2)
            ax.add_patch(operating_circle)
            danger_circle = Circle((x, y), spec["danger_radius"], 
                                 fill=False, linestyle='-', edgecolor='red', alpha=0.8, linewidth=2)
            ax.add_patch(danger_circle)
    
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Enhanced title
    safety, efficiency, adaptability = objectives
    
    status = "SAFE" if individual.feasible else "UNSAFE"
    obj_text = f"S: {safety:.3f}, E: {efficiency:.3f}, A: {adaptability:.3f}"
    
    if behaviors is not None:
        bd1, bd2 = behaviors
        behavior_text = f"Spatial: {bd1:.2f}, Functional: {bd2:.2f}"
        full_title = f"{title} | {status}\n{obj_text}\n{behavior_text}"
    else:
        combined_score = (safety + efficiency + adaptability) / 3
        full_title = f"{title} | {status}\n{obj_text}\nCombined: {combined_score:.3f}"
    
    ax.set_title(full_title, fontweight='bold', fontsize=8)

# =============================================================================
# MAP-ELITES + NSGA-II VISUALIZATIONS
# =============================================================================

def create_mapelites_visualizations(archive, config: SiteConfig, output_dir: str):
    """Create comprehensive MAP-Elites visualizations"""
    print("Creating MAP-Elites visualizations...")
    
    all_individuals = archive.get_all_individuals()
    if not all_individuals:
        return
    
    # 3D objective space + 2D behavioral space visualization
    fig = plt.figure(figsize=(16, 12))
    
    # 3D objective space
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    
    objectives = np.array([ind.objectives for ind in all_individuals])
    safety_feasible = [ind for ind in all_individuals if ind.objectives[0] >= 0.7]
    safety_infeasible = [ind for ind in all_individuals if ind.objectives[0] < 0.7]
    
    if safety_feasible:
        feas_obj = np.array([ind.objectives for ind in safety_feasible])
        ax1.scatter(feas_obj[:, 0], feas_obj[:, 1], feas_obj[:, 2], 
                   c='green', alpha=0.7, s=30, label=f'Safe ({len(safety_feasible)})')
    
    if safety_infeasible:
        infeas_obj = np.array([ind.objectives for ind in safety_infeasible])
        ax1.scatter(infeas_obj[:, 0], infeas_obj[:, 1], infeas_obj[:, 2], 
                   c='red', alpha=0.5, s=20, label=f'Unsafe ({len(safety_infeasible)})')
    
    ax1.set_xlabel('Safety & Compliance')
    ax1.set_ylabel('Operational Efficiency')
    ax1.set_zlabel('Layout Adaptability')
    ax1.set_title('3D Objective Space')
    ax1.legend()
    
    # 2D Behavioral space colored by safety
    ax2 = fig.add_subplot(2, 3, 2)
    behaviors = np.array([ind.behaviors for ind in all_individuals])
    safety_scores = [ind.objectives[0] for ind in all_individuals]
    
    scatter2 = ax2.scatter(behaviors[:, 0], behaviors[:, 1], 
                          c=safety_scores, cmap='RdYlGn', alpha=0.7, s=30)
    ax2.set_xlabel('BD1: Same-Type Module Dispersion\n(Clustered -> Dispersed)')
    ax2.set_ylabel('BD2: Worker-Operational Separation\n(Embedded → Segregated)')
    ax2.set_title('2D Behavioral Space\n(Color = Safety Score)')
    plt.colorbar(scatter2, ax=ax2, label='Safety Score')
    
    # Efficiency in behavioral space  
    ax3 = fig.add_subplot(2, 3, 3)
    efficiency_scores = [ind.objectives[1] for ind in all_individuals]
    scatter3 = ax3.scatter(behaviors[:, 0], behaviors[:, 1], 
                          c=efficiency_scores, cmap='plasma', alpha=0.7, s=30)
    ax3.set_xlabel('BD1: Same-Type Module Dispersion')
    ax3.set_ylabel('BD2: Worker-Operational Separation')
    ax3.set_title('2D Behavioral Space\n(Color = Efficiency Score)')
    plt.colorbar(scatter3, ax=ax3, label='Efficiency Score')
    
    # Archive grid visualization
    ax4 = fig.add_subplot(2, 3, 4)
    create_archive_grid_plot(ax4, archive)
    
    # Objective correlations
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.scatter(objectives[:, 0], objectives[:, 1], alpha=0.6, s=20)
    ax5.set_xlabel('Safety & Compliance')
    ax5.set_ylabel('Operational Efficiency')
    ax5.set_title('Safety vs Efficiency')
    ax5.grid(True, alpha=0.3)
    
    # Archive statistics
    stats = archive.get_stats()
    stats_text = [
        f"MAP-Elites + 3D NSGA-II Results:",
        f"",
        f"Archive Coverage:",
        f"  Cells: {stats['coverage']:,} / {archive.total_cells:,}",
        f"  Coverage: {stats['coverage_pct']:.3f}%",
        f"",
        f"Population Quality:",
        f"  Total: {stats['total_individuals']:,}",
        f"  Safe (≥0.7): {stats['safety_feasible_count']:,}",
        f"",
        f"Objective Averages:",
        f"  Safety: {stats['avg_safety']:.3f}",
        f"  Efficiency: {stats['avg_efficiency']:.3f}",
        f"  Adaptability: {stats['avg_adaptability']:.3f}",
        f"",
        f"Total Evaluations: {archive.evaluations:,}"
    ]
    
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.text(0.1, 0.9, '\n'.join(stats_text), transform=ax6.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.axis('off')
    ax6.set_title('Results Summary', fontweight='bold')
    
    plt.tight_layout()
    analysis_path = os.path.join(output_dir, "cslpelites_analysis.png")
    plt.savefig(analysis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved CSLP-Elites analysis: {analysis_path}")
    
    # Show diverse high-quality layouts
    show_quality_layouts(all_individuals, config, output_dir, "cslpelites_layouts.png")

def create_archive_grid_plot(ax, archive):
    """Create MAP-Elites archive grid visualization"""
    grid_h, grid_w = archive.grid_size
    pareto_size_grid = np.zeros((grid_h, grid_w))
    
    max_pareto_size = 0
    for (i, j), pareto_front in archive.archive.items():
        pareto_size_grid[i, j] = pareto_front.size()
        max_pareto_size = max(max_pareto_size, pareto_front.size())
    
    im = ax.imshow(pareto_size_grid, cmap='Blues', origin='lower', vmin=0, vmax=max_pareto_size)
    ax.set_title('Archive Grid\n(Color = Solutions per cell)')
    ax.set_xlabel('BD2: Worker-Operational Separation')
    ax.set_ylabel('BD1: Spatial Organization')
    
    # Add numbers to non-empty cells
    for i in range(grid_h):
        for j in range(grid_w):
            if pareto_size_grid[i, j] > 0:
                text_color = 'white' if pareto_size_grid[i, j] > max_pareto_size * 0.6 else 'black'
                ax.text(j, i, f'{int(pareto_size_grid[i, j])}', 
                        ha='center', va='center', color=text_color, fontweight='bold', fontsize=8)

# =============================================================================
# NSGA-II VISUALIZATIONS
# =============================================================================

def create_nsga2_visualizations(results: Dict, config: SiteConfig, output_dir: str):
    """Create comprehensive NSGA-II visualizations"""
    print("Creating NSGA-II visualizations...")
    
    population = results["population"]
    pareto_front = results["pareto_front"]
    fronts = results["fronts"]
    
    if not population:
        return
    
    # 3D objective space + Pareto analysis
    fig = plt.figure(figsize=(16, 12))
    
    # 3D objective space with Pareto fronts
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    
    # Color different fronts
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
    
    for i, front in enumerate(fronts[:6]):  # Show first 6 fronts
        if front:
            objectives = np.array([ind.objectives for ind in front])
            color = colors[i % len(colors)]
            label = f'Front {i+1}' if i > 0 else 'Pareto Front'
            size = 50 if i == 0 else 30
            ax1.scatter(objectives[:, 0], objectives[:, 1], objectives[:, 2], 
                       c=color, alpha=0.8, s=size, label=label)
    
    ax1.set_xlabel('Safety & Compliance')
    ax1.set_ylabel('Operational Efficiency')
    ax1.set_zlabel('Layout Adaptability')
    ax1.set_title('NSGA-II 3D Objective Space')
    ax1.legend()
    
    # 2D projections of objective space
    ax2 = fig.add_subplot(2, 3, 2)
    objectives = np.array([ind.objectives for ind in population])
    pareto_objectives = np.array([ind.objectives for ind in pareto_front]) if pareto_front else np.array([])
    
    ax2.scatter(objectives[:, 0], objectives[:, 1], alpha=0.6, s=20, c='lightblue', label='Population')
    if len(pareto_objectives) > 0:
        ax2.scatter(pareto_objectives[:, 0], pareto_objectives[:, 1], 
                   c='red', s=50, alpha=0.8, label='Pareto Front')
    ax2.set_xlabel('Safety & Compliance')
    ax2.set_ylabel('Operational Efficiency')
    ax2.set_title('Safety vs Efficiency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Convergence analysis if available
    ax3 = fig.add_subplot(2, 3, 3)
    if "convergence_history" in results:
        convergence_data = results["convergence_history"]
        generations = list(range(len(convergence_data)))
        
        avg_safety = [gen["avg_safety"] for gen in convergence_data]
        avg_efficiency = [gen["avg_efficiency"] for gen in convergence_data]
        avg_adaptability = [gen["avg_adaptability"] for gen in convergence_data]
        
        ax3.plot(generations, avg_safety, 'r-', label='Safety', linewidth=2)
        ax3.plot(generations, avg_efficiency, 'g-', label='Efficiency', linewidth=2)
        ax3.plot(generations, avg_adaptability, 'b-', label='Adaptability', linewidth=2)
        
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Average Objective Value')
        ax3.set_title('NSGA-II Convergence')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Convergence history\nnot available', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Convergence Analysis')
    
    # Pareto front analysis
    ax4 = fig.add_subplot(2, 3, 4)
    if pareto_front:
        pareto_safety = [ind.objectives[0] for ind in pareto_front]
        pareto_efficiency = [ind.objectives[1] for ind in pareto_front]
        pareto_adaptability = [ind.objectives[2] for ind in pareto_front]
        
        x_pos = range(len(pareto_front))
        ax4.bar(x_pos, pareto_safety, alpha=0.7, label='Safety', color='red')
        ax4.bar(x_pos, pareto_efficiency, alpha=0.7, label='Efficiency', 
               bottom=pareto_safety, color='blue')
        ax4.bar(x_pos, pareto_adaptability, alpha=0.7, label='Adaptability',
               bottom=np.array(pareto_safety) + np.array(pareto_efficiency), color='green')
        
        ax4.set_xlabel('Pareto Solutions')
        ax4.set_ylabel('Objective Values')
        ax4.set_title('Pareto Front Trade-offs')
        ax4.legend()
    
    # Additional projections
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.scatter(objectives[:, 1], objectives[:, 2], alpha=0.6, s=20, c='lightblue', label='Population')
    if len(pareto_objectives) > 0:
        ax5.scatter(pareto_objectives[:, 1], pareto_objectives[:, 2], 
                   c='red', s=50, alpha=0.8, label='Pareto Front')
    ax5.set_xlabel('Operational Efficiency')
    ax5.set_ylabel('Layout Adaptability')
    ax5.set_title('Efficiency vs Adaptability')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # NSGA-II statistics
    feasible_count = sum(1 for ind in population if ind.feasible)
    avg_objectives = np.mean(objectives, axis=0)
    
    stats_text = [
        f"Pure NSGA-II Results:",
        f"",
        f"Population: {len(population)}",
        f"Pareto Front: {len(pareto_front)}",
        f"Total Fronts: {len(fronts)}",
        f"Feasible Solutions: {feasible_count}/{len(population)}",
        f"",
        f"Average Objectives:",
        f"  Safety: {avg_objectives[0]:.3f}",
        f"  Efficiency: {avg_objectives[1]:.3f}",
        f"  Adaptability: {avg_objectives[2]:.3f}",
        f"",
        f"Total Evaluations: {results['evaluations']:,}",
        f"Runtime: {results['runtime']:.2f}s"
    ]
    
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.text(0.1, 0.9, '\n'.join(stats_text), transform=ax6.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.axis('off')
    ax6.set_title('NSGA-II Statistics', fontweight='bold')
    
    plt.tight_layout()
    analysis_path = os.path.join(output_dir, "nsga2_analysis.png")
    plt.savefig(analysis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved NSGA-II analysis: {analysis_path}")
    
    # Show best feasible Pareto solutions
    if pareto_front:
        feasible_pareto = [ind for ind in pareto_front if ind.feasible]
        if feasible_pareto:
            show_quality_layouts(feasible_pareto, config, output_dir, "nsga2_pareto_layouts.png")
            print(f"  Showing {len(feasible_pareto)} feasible solutions out of {len(pareto_front)} total Pareto solutions")
        else:
            print("  Warning: No feasible solutions in Pareto front!")

# =============================================================================
# SHARED VISUALIZATION FUNCTIONS
# =============================================================================

def show_quality_layouts(individuals: List[Individual], config: SiteConfig, 
                        output_dir: str, filename: str):
    """Show diverse high-quality layouts"""
    if not individuals:
        return
    
    # Filter to high quality solutions and sort
    high_quality = [ind for ind in individuals if ind.objectives[0] >= 0.7]
    if not high_quality:
        high_quality = sorted(individuals, key=lambda x: x.objectives[0], reverse=True)[:9]
    
    high_quality.sort(
        key=lambda x: (0.4 * x.objectives[0] + 0.3 * x.objectives[1] + 0.3 * x.objectives[2]), 
        reverse=True
    )
    
    n_samples = min(9, len(high_quality))
    step = max(1, len(high_quality) // n_samples)
    samples = high_quality[::step][:n_samples]
    
    cols = 3
    rows = (len(samples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    
    if rows == 1:
        axes = [axes] if cols == 1 else list(axes)
    else:
        axes = axes.flatten()
    
    for i, individual in enumerate(samples):
        ax = axes[i]
        title = f"Layout {i+1}"
        visualize_layout(ax, individual, config, title)
    
    for i in range(len(samples), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    layouts_path = os.path.join(output_dir, filename)
    plt.savefig(layouts_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved quality layouts: {layouts_path}")

# =============================================================================
# PURE MAP-ELITES VISUALIZATIONS
# =============================================================================

def create_pure_mapelites_visualizations(archive, config: SiteConfig, output_dir: str):
    """Create comprehensive Pure MAP-Elites visualizations"""
    print("Creating Pure MAP-Elites visualizations...")
    
    all_individuals = archive.get_all_individuals()
    if not all_individuals:
        return
    
    # 3D objective space + 2D behavioral space + scalar fitness visualization
    fig = plt.figure(figsize=(16, 12))
    
    # 3D objective space
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    
    objectives = np.array([ind.objectives for ind in all_individuals])
    scalar_fitnesses = [getattr(ind, 'scalar_fitness', archive.calculate_scalar_fitness(ind)) 
                       for ind in all_individuals]
    
    scatter1 = ax1.scatter(objectives[:, 0], objectives[:, 1], objectives[:, 2], 
                          c=scalar_fitnesses, cmap='viridis', alpha=0.7, s=40)
    ax1.set_xlabel('Safety & Compliance')
    ax1.set_ylabel('Operational Efficiency')
    ax1.set_zlabel('Layout Adaptability')
    ax1.set_title('3D Objective Space\n(Color = Scalar Fitness)')
    plt.colorbar(scatter1, ax=ax1, label='Scalar Fitness', shrink=0.6)
    
    # 2D Behavioral space colored by scalar fitness
    ax2 = fig.add_subplot(2, 3, 2)
    behaviors = np.array([ind.behaviors for ind in all_individuals])
    
    scatter2 = ax2.scatter(behaviors[:, 0], behaviors[:, 1], 
                          c=scalar_fitnesses, cmap='viridis', alpha=0.7, s=40)
    ax2.set_xlabel('BD1: Same-Type Module Dispersion\n(Clustered -> Dispersed)')
    ax2.set_ylabel('BD2: Worker-Operational Separation\n(Embedded → Segregated)')
    ax2.set_title('2D Behavioral Space\n(Color = Scalar Fitness)')
    plt.colorbar(scatter2, ax=ax2, label='Scalar Fitness')
    
    # Safety in behavioral space  
    ax3 = fig.add_subplot(2, 3, 3)
    safety_scores = [ind.objectives[0] for ind in all_individuals]
    scatter3 = ax3.scatter(behaviors[:, 0], behaviors[:, 1], 
                          c=safety_scores, cmap='RdYlGn', alpha=0.7, s=40)
    ax3.set_xlabel('BD1: Same-Type Module Dispersion')
    ax3.set_ylabel('BD2: Worker-Operational Separation')
    ax3.set_title('2D Behavioral Space\n(Color = Safety Score)')
    plt.colorbar(scatter3, ax=ax3, label='Safety Score')
    
    # Archive grid visualization
    ax4 = fig.add_subplot(2, 3, 4)
    create_pure_mapelites_grid_plot(ax4, archive)
    
    # Scalar fitness distribution
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.hist(scalar_fitnesses, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax5.axvline(np.mean(scalar_fitnesses), color='red', linestyle='--', 
                label=f'Mean: {np.mean(scalar_fitnesses):.3f}')
    ax5.axvline(np.max(scalar_fitnesses), color='green', linestyle='--', 
                label=f'Best: {np.max(scalar_fitnesses):.3f}')
    ax5.set_xlabel('Scalar Fitness')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Scalar Fitness Distribution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Archive statistics
    stats = archive.get_stats()
    best_individual = archive.get_best_individual()
    best_obj = best_individual.objectives if best_individual else (0, 0, 0)
    
    stats_text = [
        f"Pure MAP-Elites Results:",
        f"",
        f"Archive Coverage:",
        f"  Cells: {stats['coverage']:,} / {archive.total_cells:,}",
        f"  Coverage: {stats['coverage_pct']:.3f}%",
        f"",
        f"Scalar Fitness Quality:",
        f"  Average: {stats['avg_scalar_fitness']:.3f}",
        f"  Best: {stats['best_scalar_fitness']:.3f}",
        f"",
        f"Best Solution Objectives:",
        f"  Safety: {best_obj[0]:.3f}",
        f"  Efficiency: {best_obj[1]:.3f}",
        f"  Adaptability: {best_obj[2]:.3f}",
        f"",
        f"Safety Feasible: {stats['safety_feasible_count']:,}",
        f"Total Evaluations: {archive.evaluations:,}"
    ]
    
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.text(0.1, 0.9, '\n'.join(stats_text), transform=ax6.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.axis('off')
    ax6.set_title('Results Summary', fontweight='bold')
    
    plt.tight_layout()
    analysis_path = os.path.join(output_dir, "mapelites_analysis.png")
    plt.savefig(analysis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved MAP-Elites analysis: {analysis_path}")
    
    # Show diverse high-quality layouts
    show_quality_layouts(all_individuals, config, output_dir, "mapelites_layouts.png")

def create_pure_mapelites_grid_plot(ax, archive):
    """Create Pure MAP-Elites archive grid visualization"""
    grid_h, grid_w = archive.grid_size
    fitness_grid = np.zeros((grid_h, grid_w))
    
    max_fitness = 0.0
    for (i, j), individual in archive.archive.items():
        fitness = getattr(individual, 'scalar_fitness', archive.calculate_scalar_fitness(individual))
        fitness_grid[i, j] = fitness
        max_fitness = max(max_fitness, fitness)
    
    # Use a mask for empty cells
    fitness_grid_masked = np.ma.masked_where(fitness_grid == 0, fitness_grid)
    
    im = ax.imshow(fitness_grid_masked, cmap='viridis', origin='lower', vmin=0, vmax=max_fitness)
    ax.set_title('Archive Grid\n(Color = Scalar Fitness)')
    ax.set_xlabel('BD2: Worker-Operational Separation')
    ax.set_ylabel('BD1: Spatial Organization')
    
    # Add fitness values to occupied cells
    for i in range(grid_h):
        for j in range(grid_w):
            if fitness_grid[i, j] > 0:
                text_color = 'white' if fitness_grid[i, j] > max_fitness * 0.6 else 'black'
                ax.text(j, i, f'{fitness_grid[i, j]:.2f}', 
                        ha='center', va='center', color=text_color, fontweight='bold', fontsize=7)

# =============================================================================
# RESULTS EXPORT FUNCTIONS
# =============================================================================

def export_cslpelite_results(
    archive,
    config: SiteConfig,
    output_dir: str,
    max_layouts: Optional[int] = 30,
    include_unsafe: bool = False,
    site_width_m: float = 100.0,
    site_length_m: float = 100.0,
    coordinate_space: str = "normalized",
    export_pngs: bool = True,
) -> int:
    """Export CSLP Elite (MAP-Elites + NSGA-II) results to JSON and optional PNG."""
    print(f"Exporting CSLP Elite (MAP-Elites + NSGA-II) results to {output_dir}/...")
    os.makedirs(output_dir, exist_ok=True)
    
    all_individuals = archive.get_all_individuals()
    safety_feasible = [ind for ind in all_individuals if ind.objectives[0] >= 0.7]
    candidates = all_individuals if include_unsafe else safety_feasible
    
    # Sort by weighted combination
    candidates.sort(
        key=lambda x: (0.4 * x.objectives[0] + 0.3 * x.objectives[1] + 0.3 * x.objectives[2]), 
        reverse=True
    )

    export_limit = len(candidates) if not max_layouts or max_layouts <= 0 else max_layouts
    
    exported = 0
    boundary_polygon = None
    if config.boundary_polygon:
        boundary_polygon = [[float(x), float(y)] for x, y in config.boundary_polygon]

    exclusion_zones = None
    if config.exclusion_zones:
        exclusion_zones = []
        for zone in config.exclusion_zones:
            polygon = zone.get("polygon") or []
            exclusion_zones.append({
                "name": zone.get("name", "zone"),
                "polygon": [[float(x), float(y)] for x, y in polygon]
            })

    for i, individual in enumerate(candidates[:export_limit]):
        facilities_payload = []
        for facility in individual.solution:
            width, length = get_facility_dimensions(facility)
            spec = FACILITY_SPECS[facility["type"]]
            facilities_payload.append({
                "type": facility["type"],
                "x": float(facility["center"][0]),
                "y": float(facility["center"][1]),
                "width": float(width),
                "length": float(length),
                "rotation": int(facility.get("rotation", 0)),
                "category": spec["category"],
                "height_m": float(spec.get("height_m", 3.0)),
                **({"jib_length_m": float(spec.get("jib_length_m", 6.0))} if facility["type"].startswith("crane") else {})
            })

        layout_data = {
            "id": f"cslpelite_layout_{i:03d}",
            "coordinate_space": coordinate_space,
            "site_width_m": float(site_width_m),
            "site_length_m": float(site_length_m),
            "boundary_polygon": boundary_polygon,
            "exclusion_zones": exclusion_zones,
            "objectives": {
                "safety_compliance": float(individual.objectives[0]),
                "operational_efficiency": float(individual.objectives[1]),
                "layout_adaptability": float(individual.objectives[2]),
                "combined_score": float(0.4 * individual.objectives[0] + 0.3 * individual.objectives[1] + 0.3 * individual.objectives[2])
            },
            "behaviors": {
                "module_dispersion": float(individual.behaviors[0]),
                "worker_operational_separation": float(individual.behaviors[1])
            },
            "feasibility": {
                "safe": individual.feasible,
                "violations": individual.violations
            },
            "facilities": facilities_payload,
            "entrances": [
                {"x": float(e[0]), "y": float(e[1])} 
                for e in individual.entrances
            ]
        }
        
        filepath = os.path.join(output_dir, f"cslpelite_layout_{i:03d}.json")
        with open(filepath, 'w') as f:
            json.dump(layout_data, f, indent=2)

        if export_pngs:
            png_filepath = os.path.join(output_dir, f"cslpelite_layout_{i:03d}.png")
            try:
                from scripts.generate_2d_previews import draw_layout as draw_bulleen_layout

                draw_bulleen_layout(layout_data, Path(png_filepath), dpi=150)
            except ImportError:
                fig = visualize_layout(
                    individual.solution,
                    individual.entrances,
                    title=(
                        f"Bulleen Layout {i + 1} | Safety: {individual.objectives[0]:.3f} | "
                        f"Eff: {individual.objectives[1]:.3f} | Adapt: {individual.objectives[2]:.3f}"
                    ),
                    config=config,
                )
                fig.savefig(png_filepath, dpi=150, bbox_inches='tight')
                plt.close(fig)

        exported += 1
    
    # Create summary
    stats = archive.get_stats()
    summary = {
        "algorithm": "CSLP Elite (MAP-Elites + 3-Objective NSGA-II)",
        "objectives": {
            "safety_compliance": "Boundary + overlap + critical safety constraints",
            "operational_efficiency": "Material flows + equipment access + workflow",
            "layout_adaptability": "Expansion + redundancy + reconfiguration potential"
        },
        "behavioral_space": {
            "module_dispersion": "Clustered vs dispersed same-type modules",
            "worker_operational_separation": "Worker modules embedded with or separated from operations"
        },
        "total_exported": exported,
        "export_includes_unsafe": include_unsafe,
        "export_pngs": export_pngs,
        "archive_performance": {
            "grid_size": f"{archive.grid_size[0]}×{archive.grid_size[1]}",
            "coverage": stats['coverage'],
            "coverage_percentage": stats['coverage_pct'],
            "total_individuals": stats['total_individuals'],
            "safety_feasible": stats['safety_feasible_count']
        },
        "objective_averages": {
            "safety": stats['avg_safety'],
            "efficiency": stats['avg_efficiency'],
            "adaptability": stats['avg_adaptability']
        }
    }
    
    with open(os.path.join(output_dir, "cslpelite_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"  Exported {exported} layouts and summary")
    return exported

def export_nsga2_results(results: Dict, config: SiteConfig, output_dir: str, max_layouts: int = 30) -> int:
    """Export NSGA-II results to JSON"""
    print(f"Exporting NSGA-II results to {output_dir}/...")
    os.makedirs(output_dir, exist_ok=True)
    
    pareto_front = results["pareto_front"]
    population = results["population"]
    
    # Sort Pareto front by combined score
    pareto_front.sort(key=lambda x: sum(x.objectives), reverse=True)
    
    exported = 0
    for i, individual in enumerate(pareto_front[:max_layouts]):
        layout_data = {
            "id": f"nsga2_layout_{i:03d}",
            "rank": "pareto_optimal",
            "objectives": {
                "safety_compliance": float(individual.objectives[0]),
                "operational_efficiency": float(individual.objectives[1]),
                "layout_adaptability": float(individual.objectives[2]),
                "combined_score": float(sum(individual.objectives) / 3)
            },
            "feasibility": {
                "safe": individual.feasible,
                "violations": individual.violations
            },
            "facilities": [
                {
                    "type": f["type"],
                    "x": float(f["center"][0]),
                    "y": float(f["center"][1]),
                    "rotation": int(f.get("rotation", 0)),
                    "category": FACILITY_SPECS[f["type"]]["category"]
                }
                for f in individual.solution
            ],
            "entrances": [
                {"x": float(e[0]), "y": float(e[1])} 
                for e in individual.entrances
            ]
        }
        
        filepath = os.path.join(output_dir, f"nsga2_layout_{i:03d}.json")
        with open(filepath, 'w') as f:
            json.dump(layout_data, f, indent=2)
        exported += 1
    
    # Create summary
    feasible_count = sum(1 for ind in population if ind.feasible)
    population_objectives = np.array([ind.objectives for ind in population])
    pareto_objectives = np.array([ind.objectives for ind in pareto_front]) if pareto_front else np.array([])
    
    summary = {
        "algorithm": "Pure NSGA-II",
        "objectives": {
            "safety_compliance": "Boundary + overlap + critical safety constraints",
            "operational_efficiency": "Material flows + enhanced crane efficiency + workflow",
            "layout_adaptability": "Expansion + redundancy + reconfiguration potential"
        },
        "total_exported": exported,
        "nsga2_performance": {
            "population_size": len(population),
            "pareto_front_size": len(pareto_front),
            "total_fronts": len(results["fronts"]),
            "feasible_solutions": feasible_count,
            "feasibility_rate": feasible_count / len(population) if population else 0.0
        },
        "population_quality": {
            "avg_safety": float(np.mean(population_objectives[:, 0])),
            "avg_efficiency": float(np.mean(population_objectives[:, 1])),
            "avg_adaptability": float(np.mean(population_objectives[:, 2]))
        },
        "pareto_quality": {
            "avg_safety": float(np.mean(pareto_objectives[:, 0])) if len(pareto_objectives) > 0 else 0.0,
            "avg_efficiency": float(np.mean(pareto_objectives[:, 1])) if len(pareto_objectives) > 0 else 0.0,
            "avg_adaptability": float(np.mean(pareto_objectives[:, 2])) if len(pareto_objectives) > 0 else 0.0
        },
        "performance_metrics": {
            "total_evaluations": results["evaluations"],
            "runtime_seconds": results["runtime"],
            "evaluations_per_second": results["evaluations"] / results["runtime"]
        }
    }
    
    with open(os.path.join(output_dir, "nsga2_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"  Exported {exported} Pareto-optimal layouts and summary")
    return exported

def export_mapelites_results(archive, config: SiteConfig, output_dir: str, max_layouts: int = 30) -> int:
    """Export Pure MAP-Elites results to JSON"""
    print(f"Exporting Pure MAP-Elites results to {output_dir}/...")
    os.makedirs(output_dir, exist_ok=True)
    
    all_individuals = archive.get_all_individuals()
    
    # Sort by scalar fitness
    all_individuals.sort(key=lambda x: getattr(x, 'scalar_fitness', archive.calculate_scalar_fitness(x)), reverse=True)
    
    # Filter to safe solutions first, then fall back to best overall if needed
    safety_feasible = [ind for ind in all_individuals if ind.objectives[0] >= 0.7]
    
    if safety_feasible:
        export_candidates = safety_feasible[:max_layouts]
    else:
        export_candidates = all_individuals[:max_layouts]
    
    exported = 0
    for i, individual in enumerate(export_candidates):
        scalar_fitness = getattr(individual, 'scalar_fitness', archive.calculate_scalar_fitness(individual))
        
        layout_data = {
            "id": f"mapelites_layout_{i:03d}",
            "objectives": {
                "safety_compliance": float(individual.objectives[0]),
                "operational_efficiency": float(individual.objectives[1]),
                "layout_adaptability": float(individual.objectives[2]),
                "scalar_fitness": float(scalar_fitness)
            },
            "behaviors": {
                "module_dispersion": float(individual.behaviors[0]),
                "worker_operational_separation": float(individual.behaviors[1])
            },
            "feasibility": {
                "safe": individual.feasible,
                "violations": individual.violations
            },
            "facilities": [
                {
                    "type": f["type"],
                    "x": float(f["center"][0]),
                    "y": float(f["center"][1]),
                    "rotation": int(f.get("rotation", 0)),
                    "category": FACILITY_SPECS[f["type"]]["category"]
                }
                for f in individual.solution
            ],
            "entrances": [
                {"x": float(e[0]), "y": float(e[1])} 
                for e in individual.entrances
            ]
        }
        
        filepath = os.path.join(output_dir, f"mapelites_layout_{i:03d}.json")
        with open(filepath, 'w') as f:
            json.dump(layout_data, f, indent=2)
        exported += 1
    
    # Create summary
    stats = archive.get_stats()
    best_individual = archive.get_best_individual()
    best_obj = best_individual.objectives if best_individual else (0, 0, 0)
    best_fitness = getattr(best_individual, 'scalar_fitness', 0.0) if best_individual else 0.0
    
    summary = {
        "algorithm": "Pure MAP-Elites",
        "objectives": {
            "safety_compliance": "Boundary + overlap + critical safety constraints",
            "operational_efficiency": "Material flows + equipment access + workflow",
            "layout_adaptability": "Expansion + redundancy + reconfiguration potential"
        },
        "behavioral_space": {
            "module_dispersion": "Clustered vs dispersed same-type modules", 
            "worker_operational_separation": "Worker modules embedded with or separated from operations"
        },
        "fitness_function": "Weighted scalar combination of three objectives",
        "total_exported": exported,
        "archive_performance": {
            "grid_size": f"{archive.grid_size[0]}×{archive.grid_size[1]}",
            "coverage": stats['coverage'],
            "coverage_percentage": stats['coverage_pct'],
            "total_individuals": stats['total_individuals'],
            "safety_feasible": stats['safety_feasible_count']
        },
        "quality_metrics": {
            "best_scalar_fitness": best_fitness,
            "avg_scalar_fitness": stats['avg_scalar_fitness'],
            "best_solution_objectives": {
                "safety": best_obj[0],
                "efficiency": best_obj[1], 
                "adaptability": best_obj[2]
            }
        },
        "objective_averages": {
            "safety": stats['avg_safety'],
            "efficiency": stats['avg_efficiency'],
            "adaptability": stats['avg_adaptability']
        }
    }
    
    with open(os.path.join(output_dir, "mapelites_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"  Exported {exported} layouts and summary")
    return exported


# =============================================================================
# AUTOENCODER-BASED MAPELITES VISUALIZATIONS
# =============================================================================

def visualize_mapelites_archive_heatmap(archive, title: str = "MAP-Elites Archive"):
    """
    Create heatmap visualization of MAP-Elites archive.
    
    Args:
        archive: PureMapElitesArchive instance
        title: Plot title
    
    Returns:
        Matplotlib figure
    """
    grid_size = archive.grid_size
    fitness_grid = np.zeros(grid_size)
    fitness_grid[:] = np.nan  # Empty cells are NaN
    
    # Fill grid with fitness values. Archives may store either one individual
    # per cell or a Pareto front per cell.
    for coords, cell_value in archive.archive.items():
        i, j = coords
        if hasattr(archive, "get_best_for_cell"):
            individual = archive.get_best_for_cell(coords)
        else:
            individual = cell_value
        if individual is None:
            continue
        fitness = archive.calculate_scalar_fitness(individual)
        fitness_grid[j, i] = fitness  # Transpose for correct orientation
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot heatmap
    im = ax.imshow(fitness_grid, cmap='viridis', origin='lower', aspect='auto',
                   vmin=0.0, vmax=1.0, interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Scalar Fitness', rotation=270, labelpad=20, fontsize=12)
    
    # Labels
    ax.set_xlabel('Behavioral Descriptor 1', fontsize=12)
    ax.set_ylabel('Behavioral Descriptor 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Grid
    ax.set_xticks(np.arange(0, grid_size[0], 2))
    ax.set_yticks(np.arange(0, grid_size[1], 2))
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    # Add statistics
    stats = archive.get_stats()
    stats_text = f"Coverage: {stats['coverage']}/{archive.total_cells} ({stats['coverage_pct']:.1f}%)\n"
    stats_text += f"Avg Fitness: {stats['avg_scalar_fitness']:.3f}\n"
    stats_text += f"Best Fitness: {stats['best_scalar_fitness']:.3f}"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=10, fontfamily='monospace')
    
    plt.tight_layout()
    return fig


def visualize_layout(facilities: List[Dict], entrances: List,
                     title: str = "Construction Site Layout",
                     config: SiteConfig = None):
    """
    Standalone layout visualization (simplified version for autoencoder script).
    
    Args:
        facilities: List of facility dictionaries
        entrances: List of entrance positions
        title: Plot title
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    site_config = config or SiteConfig()

    if site_config.boundary_polygon:
        ax.add_patch(
            Polygon(
                site_config.boundary_polygon,
                closed=True,
                facecolor="#f7eadc",
                edgecolor="#d9272e",
                linewidth=2.4,
                alpha=0.55,
                zorder=1,
            )
        )
    else:
        draw_site_boundary(ax, site_config, color="#d9272e", linestyle="-", linewidth=2.4)

    for zone in site_config.exclusion_zones or []:
        polygon = zone.get("polygon") or []
        if len(polygon) >= 3:
            ax.add_patch(
                Polygon(
                    polygon,
                    closed=True,
                    facecolor="#ffe45c",
                    edgecolor="#d3bd00",
                    linewidth=1.0,
                    alpha=0.85,
                    zorder=2,
                )
            )
    
    # Draw entrances
    for i, entrance in enumerate(entrances, start=1):
        ax.plot(
            entrance[0],
            entrance[1],
            marker="*",
            markersize=13,
            color="#f4c430",
            markeredgecolor="#a36b00",
            zorder=6,
        )
        ax.text(entrance[0], entrance[1] - 0.025, f"E{i}", ha="center",
                va="top", fontsize=7, color="#8a5a00")
    
    # Draw facilities
    for facility in facilities:
        ftype = facility["type"]
        x, y = facility["center"]
        spec = FACILITY_SPECS[ftype]
        w, h = get_facility_dimensions(facility)
        
        rect = Rectangle((x - w/2, y - h/2), w, h, 
                        facecolor=FACILITY_COLORS[ftype], 
                        edgecolor="#1f2933", linewidth=1.0, alpha=0.86, zorder=4)
        ax.add_patch(rect)
        
        # Add labels
        label_map = {"core": "CORE", "crane": "CRANE", "storage": "STORAGE", 
                     "office": "OFFICE", "rest_area": "REST"}
        label = label_map.get(ftype, ftype.upper())
        ax.text(x, y, label, ha='center', va='center', 
                fontsize=5.5, fontweight='bold', color='white', zorder=5)
        
        # Crane danger zones
        if ftype == "crane":
            radius = min(float(spec.get("danger_radius", 0.10)), float(spec.get("jib_length_m", 6.0)) / 300.0)
            danger_circle = Circle((x, y), radius,
                                 fill=False, linestyle='--', edgecolor="#e63232", 
                                 alpha=0.7, linewidth=1.0, zorder=3)
            ax.add_patch(danger_circle)
    
    ax.set_xlim(0.08, 0.92)
    ax.set_ylim(0.14, 0.86)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.5, alpha=0.28)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Normalized X")
    ax.set_ylabel("Normalized Y")
    
    plt.tight_layout()
    return fig
