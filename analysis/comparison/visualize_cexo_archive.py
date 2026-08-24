#!/usr/bin/env python3
"""
CEXO Archive Visualization
===========================

Visualizes the behavioral archive from CEXO to show coverage and diversity.
Creates heatmaps showing fitness distribution across the learned behavioral space.
"""

import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set CUDA environment variables for reproducibility
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONHASHSEED'] = '0'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import (
    SiteConfig,
    MapElitesConfig,
    AutoencoderConfig,
    generate_facility_mix
)
from core.mapelites_with_autoencoder import MapElitesWithAutoencoder
from core.mapelites_algorithm import PureMapElitesOptimizer


def create_archive_heatmap(archive, title, bd_labels=None, show_values=False):
    """
    Create detailed heatmap visualization of archive.
    
    Args:
        archive: Archive object (PureMapElitesArchive)
        title: Plot title
        bd_labels: Tuple of (x_label, y_label) for behavioral descriptors
        show_values: Whether to show fitness values in cells
    
    Returns:
        Matplotlib figure
    """
    if bd_labels is None:
        bd_labels = ("Behavioral Descriptor 1", "Behavioral Descriptor 2")
    
    grid_size = archive.grid_size
    
    # Create grids for different metrics
    fitness_grid = np.full(grid_size, np.nan)
    safety_grid = np.full(grid_size, np.nan)
    efficiency_grid = np.full(grid_size, np.nan)
    adaptability_grid = np.full(grid_size, np.nan)
    feasible_grid = np.zeros(grid_size)
    
    # Fill grids
    for coords, individual in archive.archive.items():
        i, j = coords
        
        # Scalar fitness
        if hasattr(individual, 'scalar_fitness'):
            fitness_grid[j, i] = individual.scalar_fitness
        else:
            fitness_grid[j, i] = archive.calculate_scalar_fitness(individual)
        
        # Individual objectives
        safety_grid[j, i] = individual.objectives[0]
        efficiency_grid[j, i] = individual.objectives[1]
        adaptability_grid[j, i] = individual.objectives[2]
        
        # Feasibility
        feasible_grid[j, i] = 1 if individual.feasible else 0.5
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 10))
    
    # Main heatmap (larger)
    ax1 = plt.subplot(2, 3, (1, 4))
    im1 = ax1.imshow(fitness_grid, cmap='viridis', origin='lower', aspect='auto',
                     vmin=0.0, vmax=1.0, interpolation='nearest')
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Scalar Fitness', rotation=270, labelpad=20, fontsize=11)
    ax1.set_xlabel(bd_labels[0], fontsize=12, fontweight='bold')
    ax1.set_ylabel(bd_labels[1], fontsize=12, fontweight='bold')
    ax1.set_title(f'{title}\nFitness Distribution', fontsize=14, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.2, linewidth=0.5)
    
    # Add statistics box
    stats = archive.get_stats()
    stats_text = f"Coverage: {stats['coverage']}/{archive.total_cells} ({stats['coverage_pct']:.1f}%)\n"
    stats_text += f"Total Solutions: {stats['total_individuals']}\n"
    stats_text += f"Feasible: {stats.get('safety_feasible_count', 'N/A')}\n"
    stats_text += f"Avg Fitness: {stats.get('avg_scalar_fitness', 0):.3f}\n"
    stats_text += f"Best Fitness: {stats.get('best_scalar_fitness', 0):.3f}"
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
            fontsize=11, fontfamily='monospace', fontweight='bold')
    
    # Safety heatmap
    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.imshow(safety_grid, cmap='RdYlGn', origin='lower', aspect='auto',
                     vmin=0.0, vmax=1.0, interpolation='nearest')
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Safety', rotation=270, labelpad=15, fontsize=10)
    ax2.set_title('Safety Distribution', fontsize=11, fontweight='bold')
    ax2.set_xlabel(bd_labels[0], fontsize=10)
    ax2.set_ylabel(bd_labels[1], fontsize=10)
    ax2.grid(True, alpha=0.2, linewidth=0.5)
    
    # Efficiency heatmap
    ax3 = plt.subplot(2, 3, 3)
    im3 = ax3.imshow(efficiency_grid, cmap='Blues', origin='lower', aspect='auto',
                     vmin=0.0, vmax=1.0, interpolation='nearest')
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.set_label('Efficiency', rotation=270, labelpad=15, fontsize=10)
    ax3.set_title('Efficiency Distribution', fontsize=11, fontweight='bold')
    ax3.set_xlabel(bd_labels[0], fontsize=10)
    ax3.set_ylabel(bd_labels[1], fontsize=10)
    ax3.grid(True, alpha=0.2, linewidth=0.5)
    
    # Adaptability heatmap
    ax4 = plt.subplot(2, 3, 5)
    im4 = ax4.imshow(adaptability_grid, cmap='Oranges', origin='lower', aspect='auto',
                     vmin=0.0, vmax=1.0, interpolation='nearest')
    cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    cbar4.set_label('Adaptability', rotation=270, labelpad=15, fontsize=10)
    ax4.set_title('Adaptability Distribution', fontsize=11, fontweight='bold')
    ax4.set_xlabel(bd_labels[0], fontsize=10)
    ax4.set_ylabel(bd_labels[1], fontsize=10)
    ax4.grid(True, alpha=0.2, linewidth=0.5)
    
    # Feasibility map
    ax5 = plt.subplot(2, 3, 6)
    # Create binary map: occupied cells
    occupied_grid = ~np.isnan(fitness_grid)
    im5 = ax5.imshow(occupied_grid, cmap='Greys', origin='lower', aspect='auto',
                     interpolation='nearest', alpha=0.8)
    ax5.set_title('Coverage Map', fontsize=11, fontweight='bold')
    ax5.set_xlabel(bd_labels[0], fontsize=10)
    ax5.set_ylabel(bd_labels[1], fontsize=10)
    ax5.grid(True, alpha=0.2, linewidth=0.5)
    
    # Add text showing cell count
    occupied_count = np.sum(occupied_grid)
    ax5.text(0.5, 0.5, f'{int(occupied_count)}\ncells', 
             transform=ax5.transAxes, ha='center', va='center',
             fontsize=24, fontweight='bold', color='red', alpha=0.7)
    
    plt.tight_layout()
    return fig


def create_comparison_visualization(cexo_archive, mapelites_archive, output_dir):
    """Create side-by-side comparison of CEXO vs MAP-Elites archives"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # CEXO
    ax1 = axes[0]
    grid_size = cexo_archive.grid_size
    fitness_grid_cexo = np.full(grid_size, np.nan)
    
    for coords, individual in cexo_archive.archive.items():
        i, j = coords
        if hasattr(individual, 'scalar_fitness'):
            fitness_grid_cexo[j, i] = individual.scalar_fitness
        else:
            fitness_grid_cexo[j, i] = cexo_archive.calculate_scalar_fitness(individual)
    
    im1 = ax1.imshow(fitness_grid_cexo, cmap='viridis', origin='lower', aspect='auto',
                     vmin=0.0, vmax=1.0, interpolation='nearest')
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Fitness', rotation=270, labelpad=15, fontsize=11)
    
    stats_cexo = cexo_archive.get_stats()
    ax1.set_title(f'CEXO (Learned BDs)\nCoverage: {stats_cexo["coverage"]}/{cexo_archive.total_cells} ({stats_cexo["coverage_pct"]:.1f}%)',
                  fontsize=13, fontweight='bold')
    ax1.set_xlabel('Learned BD 1', fontsize=11)
    ax1.set_ylabel('Learned BD 2', fontsize=11)
    ax1.grid(True, alpha=0.3, linewidth=0.5)
    
    # MAP-Elites
    ax2 = axes[1]
    fitness_grid_me = np.full(grid_size, np.nan)
    
    for coords, individual in mapelites_archive.archive.items():
        i, j = coords
        fitness_grid_me[j, i] = mapelites_archive.calculate_scalar_fitness(individual)
    
    im2 = ax2.imshow(fitness_grid_me, cmap='viridis', origin='lower', aspect='auto',
                     vmin=0.0, vmax=1.0, interpolation='nearest')
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Fitness', rotation=270, labelpad=15, fontsize=11)
    
    stats_me = mapelites_archive.get_stats()
    ax2.set_title(f'MAP-Elites (Manual BDs)\nCoverage: {stats_me["coverage"]}/{mapelites_archive.total_cells} ({stats_me["coverage_pct"]:.1f}%)',
                  fontsize=13, fontweight='bold')
    ax2.set_xlabel('Manual BD 1 (Spatial)', fontsize=11)
    ax2.set_ylabel('Manual BD 2 (Functional)', fontsize=11)
    ax2.grid(True, alpha=0.3, linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'archive_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Comparison saved to {output_dir}/archive_comparison.png")
    
    return fig


def main():
    """Run CEXO and visualize the archive"""
    
    print("\n" + "="*80)
    print("CEXO ARCHIVE VISUALIZATION")
    print("="*80)
    
    # Configuration
    seed = 42
    num_facilities = 6
    output_dir = "results/archive_visualization"
    os.makedirs(output_dir, exist_ok=True)
    
    # Site configuration
    site_config = SiteConfig(
        seed=seed,
        boundary_margin=0.05,
        pareto_size=12,
        facility_count=num_facilities
    )
    
    # Generate facility mix
    facility_types = generate_facility_mix(num_facilities, seed=seed)
    print(f"\nFacility mix ({num_facilities} facilities): {facility_types}")
    
    # MAP-Elites configuration
    mapelites_config = MapElitesConfig(
        grid_size=(20, 20),
        iterations=15000,
        initial_population=500
    )
    
    # Autoencoder configuration
    autoencoder_config = AutoencoderConfig(
        use_learned_descriptors=True,
        pretrain_iterations=5000,
        latent_dim=2,
        encoder_hidden=[128, 64, 32],
        decoder_hidden=[32, 64, 128],
        learning_rate=0.001,
        training_epochs=50,
        batch_size=32,
        training_frequency=2500,
        min_samples_for_training=200,
        seed=seed
    )
    
    print(f"\nConfiguration:")
    print(f"  Grid: {mapelites_config.grid_size[0]}×{mapelites_config.grid_size[1]} = {mapelites_config.grid_size[0] * mapelites_config.grid_size[1]} cells")
    print(f"  Iterations: {mapelites_config.iterations}")
    print(f"  Initial population: {mapelites_config.initial_population}")
    
    # Run CEXO
    print("\n" + "="*80)
    print("Running CEXO...")
    print("="*80)
    
    cexo_algorithm = MapElitesWithAutoencoder(
        facility_types=facility_types,
        site_config=site_config,
        mapelites_config=mapelites_config,
        autoencoder_config=autoencoder_config
    )
    
    cexo_results = cexo_algorithm.run(
        iterations=mapelites_config.iterations,
        initial_population=mapelites_config.initial_population
    )
    
    print("\n✓ CEXO completed!")
    
    # Run MAP-Elites for comparison
    print("\n" + "="*80)
    print("Running MAP-Elites...")
    print("="*80)
    
    mapelites_algorithm = PureMapElitesOptimizer(
        site_config=site_config,
        facility_types=facility_types,
        mapelites_config=mapelites_config
    )
    
    mapelites_results = mapelites_algorithm.run(
        iterations=mapelites_config.iterations,
        initial_population=mapelites_config.initial_population
    )
    
    print("\n✓ MAP-Elites completed!")
    
    # Create visualizations
    print("\n" + "="*80)
    print("Creating Visualizations...")
    print("="*80)
    
    # CEXO detailed heatmap
    bd_mode = cexo_algorithm.bd_manager.get_mode()
    bd_labels = ("Learned BD 1", "Learned BD 2") if bd_mode == "learned" else ("Manual BD 1", "Manual BD 2")
    
    fig_cexo = create_archive_heatmap(
        cexo_algorithm.archive,
        title=f"CEXO Archive (BD Mode: {bd_mode})",
        bd_labels=bd_labels
    )
    cexo_path = os.path.join(output_dir, 'cexo_archive_detailed.png')
    fig_cexo.savefig(cexo_path, dpi=300, bbox_inches='tight')
    print(f"✓ CEXO archive saved to {cexo_path}")
    plt.close(fig_cexo)
    
    # MAP-Elites detailed heatmap
    fig_me = create_archive_heatmap(
        mapelites_algorithm.archive,
        title="MAP-Elites Archive (Manual BDs)",
        bd_labels=("Manual BD 1 (Spatial)", "Manual BD 2 (Functional)")
    )
    me_path = os.path.join(output_dir, 'mapelites_archive_detailed.png')
    fig_me.savefig(me_path, dpi=300, bbox_inches='tight')
    print(f"✓ MAP-Elites archive saved to {me_path}")
    plt.close(fig_me)
    
    # Comparison visualization
    fig_comp = create_comparison_visualization(
        cexo_algorithm.archive,
        mapelites_algorithm.archive,
        output_dir
    )
    plt.close(fig_comp)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    cexo_stats = cexo_algorithm.archive.get_stats()
    me_stats = mapelites_algorithm.archive.get_stats()
    
    print(f"\nCEXO ({bd_mode} BDs):")
    print(f"  Coverage: {cexo_stats['coverage']}/{cexo_algorithm.archive.total_cells} ({cexo_stats['coverage_pct']:.1f}%)")
    print(f"  Total Solutions: {cexo_stats['total_individuals']}")
    print(f"  Avg Safety: {cexo_stats['avg_safety']:.4f}")
    print(f"  Avg Efficiency: {cexo_stats['avg_efficiency']:.4f}")
    print(f"  Avg Adaptability: {cexo_stats['avg_adaptability']:.4f}")
    
    print(f"\nMAP-Elites (manual BDs):")
    print(f"  Coverage: {me_stats['coverage']}/{mapelites_algorithm.archive.total_cells} ({me_stats['coverage_pct']:.1f}%)")
    print(f"  Total Solutions: {me_stats['total_individuals']}")
    print(f"  Avg Safety: {me_stats['avg_safety']:.4f}")
    print(f"  Avg Efficiency: {me_stats['avg_efficiency']:.4f}")
    print(f"  Avg Adaptability: {me_stats['avg_adaptability']:.4f}")
    
    coverage_improvement = cexo_stats['coverage_pct'] - me_stats['coverage_pct']
    print(f"\n✨ Coverage Improvement: +{coverage_improvement:.1f} percentage points")
    print(f"   ({cexo_stats['coverage'] - me_stats['coverage']} more cells occupied)")
    
    print("\n" + "="*80)
    print(f"✓ All visualizations saved to {output_dir}/")
    print("="*80)


if __name__ == "__main__":
    main()
