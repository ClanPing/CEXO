#!/usr/bin/env python3
"""
Latent Space Visualization for CEXO
====================================

Visualizes the autoencoder-learned behavioral descriptors in 2D latent space,
showing how the autoencoder organizes construction site layouts.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import (
    SiteConfig,
    MapElitesConfig,
    AutoencoderConfig,
    generate_facility_mix
)
from core.mapelites_with_autoencoder import MapElitesWithAutoencoder


def create_latent_space_visualization():
    """Create comprehensive latent space visualizations"""
    
    print("\n" + "="*80)
    print("LATENT SPACE VISUALIZATION")
    print("="*80)
    
    # Configuration aligned with experiment.md
    seed = 42
    num_facilities = 6
    facility_types = generate_facility_mix(num_facilities, seed=seed)
    
    print(f"\nConfiguration (aligned with experiment.md):")
    print(f"  Facilities: {num_facilities} - {facility_types}")
    print(f"  Grid: 20×20 (400 cells)")
    print(f"  Iterations: 15,000")
    print(f"  Initial population: 500")
    print(f"  Pareto size per cell: 12")
    print(f"  Autoencoder: 2D latent space")
    print(f"  Pretrain: 5,000 iterations")
    print(f"  Training frequency: 2,500 iterations")
    
    site_config = SiteConfig(
        seed=seed,
        boundary_margin=0.05,
        pareto_size=12,
        facility_count=len(facility_types)
    )
    
    mapelites_config = MapElitesConfig(
        grid_size=(20, 20),
        iterations=15000,
        initial_population=500
    )
    
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
    
    # Run CEXO
    print("\n" + "="*80)
    print("RUNNING CEXO")
    print("="*80)
    
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
    
    # Extract data
    all_individuals = algorithm.archive.get_all_individuals()
    
    print(f"\nExtracted {len(all_individuals)} individuals from archive")
    
    # Extract latent coordinates and objectives
    latent_coords = np.array([ind.behaviors for ind in all_individuals])
    objectives = np.array([ind.objectives for ind in all_individuals])
    feasible = np.array([ind.feasible for ind in all_individuals])
    
    safety = objectives[:, 0]
    efficiency = objectives[:, 1]
    adaptability = objectives[:, 2]
    
    # Calculate aggregate fitness
    fitness = np.mean(objectives, axis=1)
    
    print(f"Latent space range:")
    print(f"  BD1: [{latent_coords[:, 0].min():.3f}, {latent_coords[:, 0].max():.3f}]")
    print(f"  BD2: [{latent_coords[:, 1].min():.3f}, {latent_coords[:, 1].max():.3f}]")
    
    # Create visualization
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Latent space colored by aggregate fitness
    ax1 = fig.add_subplot(gs[0, 0])
    scatter1 = ax1.scatter(latent_coords[:, 0], latent_coords[:, 1], 
                           c=fitness, s=50, alpha=0.6, cmap='viridis', 
                           edgecolors='black', linewidths=0.5)
    ax1.set_xlabel('Learned BD 1', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Learned BD 2', fontsize=11, fontweight='bold')
    ax1.set_title('Latent Space: Aggregate Fitness', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Fitness', fontsize=10)
    
    # 2. Latent space colored by safety
    ax2 = fig.add_subplot(gs[0, 1])
    scatter2 = ax2.scatter(latent_coords[:, 0], latent_coords[:, 1], 
                           c=safety, s=50, alpha=0.6, cmap='RdYlGn', 
                           edgecolors='black', linewidths=0.5, vmin=0, vmax=1)
    ax2.set_xlabel('Learned BD 1', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Learned BD 2', fontsize=11, fontweight='bold')
    ax2.set_title('Latent Space: Safety', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Safety', fontsize=10)
    
    # 3. Latent space colored by efficiency
    ax3 = fig.add_subplot(gs[0, 2])
    scatter3 = ax3.scatter(latent_coords[:, 0], latent_coords[:, 1], 
                           c=efficiency, s=50, alpha=0.6, cmap='plasma', 
                           edgecolors='black', linewidths=0.5, vmin=0, vmax=1)
    ax3.set_xlabel('Learned BD 1', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Learned BD 2', fontsize=11, fontweight='bold')
    ax3.set_title('Latent Space: Efficiency', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    cbar3 = plt.colorbar(scatter3, ax=ax3)
    cbar3.set_label('Efficiency', fontsize=10)
    
    # 4. Latent space colored by adaptability
    ax4 = fig.add_subplot(gs[1, 0])
    scatter4 = ax4.scatter(latent_coords[:, 0], latent_coords[:, 1], 
                           c=adaptability, s=50, alpha=0.6, cmap='coolwarm', 
                           edgecolors='black', linewidths=0.5, vmin=0, vmax=1)
    ax4.set_xlabel('Learned BD 1', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Learned BD 2', fontsize=11, fontweight='bold')
    ax4.set_title('Latent Space: Adaptability', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--')
    cbar4 = plt.colorbar(scatter4, ax=ax4)
    cbar4.set_label('Adaptability', fontsize=10)
    
    # 5. Latent space colored by feasibility
    ax5 = fig.add_subplot(gs[1, 1])
    colors = ['red' if not f else 'green' for f in feasible]
    ax5.scatter(latent_coords[:, 0], latent_coords[:, 1], 
                c=colors, s=50, alpha=0.6, edgecolors='black', linewidths=0.5)
    ax5.set_xlabel('Learned BD 1', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Learned BD 2', fontsize=11, fontweight='bold')
    ax5.set_title('Latent Space: Feasibility', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, linestyle='--')
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', alpha=0.6, label='Feasible'),
                      Patch(facecolor='red', alpha=0.6, label='Infeasible')]
    ax5.legend(handles=legend_elements, loc='best', fontsize=9)
    
    # 6. Density heatmap of latent space
    ax6 = fig.add_subplot(gs[1, 2])
    # Create 2D histogram
    hist, xedges, yedges = np.histogram2d(latent_coords[:, 0], latent_coords[:, 1], 
                                          bins=50)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax6.imshow(hist.T, origin='lower', extent=extent, aspect='auto', 
                    cmap='hot', interpolation='gaussian')
    ax6.set_xlabel('Learned BD 1', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Learned BD 2', fontsize=11, fontweight='bold')
    ax6.set_title('Latent Space: Density Heatmap', fontsize=12, fontweight='bold')
    cbar6 = plt.colorbar(im, ax=ax6)
    cbar6.set_label('Count', fontsize=10)
    
    plt.suptitle('CEXO Autoencoder-Learned Latent Space Analysis', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    # Save figure
    output_dir = "results/latent_space_visualization"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "latent_space_analysis.png")
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Latent space visualization saved to: {output_file}")
    
    # Save PDF version
    pdf_file = os.path.join(output_dir, "latent_space_analysis.pdf")
    plt.savefig(pdf_file, bbox_inches='tight')
    print(f"✓ PDF version saved to: {pdf_file}")
    
    plt.close()
    
    # Create summary statistics
    print("\n" + "="*80)
    print("LATENT SPACE STATISTICS")
    print("="*80)
    
    print(f"\nTotal solutions: {len(all_individuals)}")
    print(f"Feasible solutions: {np.sum(feasible)} ({np.sum(feasible)/len(feasible)*100:.1f}%)")
    print(f"\nLatent Space Coverage:")
    print(f"  BD1 range: [{latent_coords[:, 0].min():.3f}, {latent_coords[:, 0].max():.3f}]")
    print(f"  BD2 range: [{latent_coords[:, 1].min():.3f}, {latent_coords[:, 1].max():.3f}]")
    print(f"  BD1 std: {latent_coords[:, 0].std():.3f}")
    print(f"  BD2 std: {latent_coords[:, 1].std():.3f}")
    
    print(f"\nObjective Statistics:")
    print(f"  Safety: {safety.mean():.3f} ± {safety.std():.3f}")
    print(f"  Efficiency: {efficiency.mean():.3f} ± {efficiency.std():.3f}")
    print(f"  Adaptability: {adaptability.mean():.3f} ± {adaptability.std():.3f}")
    print(f"  Aggregate Fitness: {fitness.mean():.3f} ± {fitness.std():.3f}")
    
    # Calculate coverage in archive grid
    stats = algorithm.archive.get_stats()
    print(f"\nArchive Grid Coverage:")
    print(f"  Occupied cells: {stats['coverage']}/400 ({stats['coverage_pct']:.1f}%)")
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    create_latent_space_visualization()
