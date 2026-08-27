#!/usr/bin/env python3
"""
CEXO: Construction Site eXploration & Optimisation
===================================================

CEXO combines quality-diversity optimization (MAP-Elites) with learned behavioral 
descriptors for construction site layout planning.

Process:
1. Start with hand-crafted behavioral descriptors
2. Build diverse archive through MAP-Elites
3. Train autoencoder on archive population
4. Switch to learned latent behavioral descriptors
5. Continue optimization with periodic autoencoder retraining
"""

import os
import sys
from datetime import datetime

# Set CUDA environment variables for reproducibility BEFORE importing torch
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONHASHSEED'] = '0'

import json
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional

# Ensure imports resolve when the script is launched from outside the repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import (
    SiteConfig,
    MapElitesConfig,
    AutoencoderConfig,
    NSGA2Config,
    FACILITY_SPECS,
    generate_facility_mix
)
from core.mapelites_with_autoencoder import MapElitesWithAutoencoder
from core.mapelites_algorithm import PureMapElitesOptimizer
from core.nsga2_algorithm import PureNSGA2Optimizer
from core.visualization import (
    visualize_mapelites_archive_heatmap,
    visualize_layout_preview,
    export_cslpelite_results
)


def parse_facility_mix(mix_spec: str) -> List[str]:
    """Parse a CLI facility mix such as core=2,crane=1,storage=2."""
    if not mix_spec or not mix_spec.strip():
        raise ValueError("Facility mix cannot be empty.")

    facilities: List[str] = []
    valid_types = set(FACILITY_SPECS)
    parts = [part.strip() for part in mix_spec.replace(";", ",").split(",") if part.strip()]

    for part in parts:
        if "=" in part:
            name, count_text = part.split("=", 1)
        elif ":" in part:
            name, count_text = part.split(":", 1)
        else:
            name, count_text = part, "1"

        facility_type = name.strip()
        if facility_type not in valid_types:
            valid = ", ".join(sorted(valid_types))
            raise ValueError(f"Unknown facility type '{facility_type}'. Valid types: {valid}.")

        try:
            count = int(count_text.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid count for '{facility_type}': {count_text!r}.") from exc

        if count < 0:
            raise ValueError(f"Facility count for '{facility_type}' must be zero or greater.")

        facilities.extend([facility_type] * count)

    if not facilities:
        raise ValueError("Facility mix must include at least one facility.")

    return facilities


def make_run_output_dir(base_dir: str, run_label: str) -> str:
    """Create a timestamped output folder for one run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(base_dir, f"{run_label}_{timestamp}")


def visualize_diverse_layouts(archive, algorithm, num_layouts=9):
    """
    Visualize diverse layouts from different behavioral regions.
    
    Selects representative layouts from different parts of the behavioral space
    to showcase the diversity achieved.
    
    Args:
        archive: MAP-Elites archive
        algorithm: MapElitesWithAutoencoder instance
        num_layouts: Number of layouts to display (default: 9 for 3x3 grid)
    
    Returns:
        Matplotlib figure
    """
    # Get all individuals
    all_individuals = archive.get_all_individuals()
    
    if len(all_individuals) < num_layouts:
        print(f"    Warning: Only {len(all_individuals)} layouts available, showing all")
        num_layouts = len(all_individuals)
    
    if num_layouts == 0:
        return None
    
    # Strategy: Select layouts from different behavioral regions
    # Divide behavioral space into grid and pick best from each region
    grid_dim = int(np.ceil(np.sqrt(num_layouts)))
    
    # Create behavioral regions
    bd1_bins = np.linspace(0, 1, grid_dim + 1)
    bd2_bins = np.linspace(0, 1, grid_dim + 1)
    
    selected_layouts = []
    region_labels = []
    
    for i in range(grid_dim):
        for j in range(grid_dim):
            if len(selected_layouts) >= num_layouts:
                break
            
            # Find individuals in this region
            bd1_min, bd1_max = bd1_bins[i], bd1_bins[i + 1]
            bd2_min, bd2_max = bd2_bins[j], bd2_bins[j + 1]
            
            region_individuals = [
                ind for ind in all_individuals
                if bd1_min <= ind.behaviors[0] < bd1_max and 
                   bd2_min <= ind.behaviors[1] < bd2_max
            ]
            
            if region_individuals:
                # Pick best individual from this region
                best_in_region = max(
                    region_individuals,
                    key=lambda ind: archive.calculate_scalar_fitness(ind)
                )
                selected_layouts.append(best_in_region)
                region_labels.append(f"BD1:[{bd1_min:.2f}-{bd1_max:.2f}], BD2:[{bd2_min:.2f}-{bd2_max:.2f}]")
    
    # If we didn't get enough layouts, fill with high-fitness ones
    if len(selected_layouts) < num_layouts:
        remaining = [ind for ind in all_individuals if ind not in selected_layouts]
        remaining_sorted = sorted(
            remaining,
            key=lambda ind: archive.calculate_scalar_fitness(ind),
            reverse=True
        )
        selected_layouts.extend(remaining_sorted[:num_layouts - len(selected_layouts)])
        for _ in range(len(selected_layouts) - len(region_labels)):
            region_labels.append("High Fitness")
    
    # Create visualization
    rows = int(np.ceil(np.sqrt(len(selected_layouts))))
    cols = int(np.ceil(len(selected_layouts) / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    
    # Flatten axes for easy iteration
    if rows * cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (ind, ax) in enumerate(zip(selected_layouts, axes)):
        # Draw boundary
        margin = 0.08
        boundary_x = [margin, 1-margin, 1-margin, margin, margin]
        boundary_y = [margin, margin, 1-margin, 1-margin, margin]
        ax.plot(boundary_x, boundary_y, 'k-', linewidth=2)
        
        # Draw entrances
        for i, entrance in enumerate(ind.entrances):
            ax.plot(entrance[0], entrance[1], marker='*', markersize=15, 
                    color='gold', markeredgecolor='darkorange', markeredgewidth=2, zorder=10)
        
        # Draw facilities
        from core.config import FACILITY_SPECS, FACILITY_COLORS
        from matplotlib.patches import Rectangle, Circle
        
        for facility in ind.solution:
            ftype = facility["type"]
            x, y = facility["center"]
            spec = FACILITY_SPECS[ftype]
            w, h = spec["w"], spec["d"]
            
            rect = Rectangle((x - w/2, y - h/2), w, h, 
                            facecolor=FACILITY_COLORS[ftype], 
                            edgecolor='black', linewidth=1.5, alpha=0.8)
            ax.add_patch(rect)
            
            # Add labels
            label_map = {"core": "C", "crane": "CR", "storage": "S", 
                        "office": "OF", "rest_area": "RA"}
            label = label_map.get(ftype, ftype[0].upper())
            ax.text(x, y, label, ha='center', va='center', 
                    fontsize=8, fontweight='bold', color='white')
            
            # Crane danger zones
            if ftype == "crane":
                danger_circle = Circle((x, y), spec["danger_radius"], 
                                     fill=False, linestyle='--', edgecolor='red', 
                                     alpha=0.5, linewidth=1.5)
                ax.add_patch(danger_circle)
        
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        
        # Title with metrics
        fitness = archive.calculate_scalar_fitness(ind)
        safety, efficiency, adaptability = ind.objectives
        bd1, bd2 = ind.behaviors
        
        title = f"Layout {idx + 1} - Fitness: {fitness:.3f}\n"
        title += f"S:{safety:.2f} E:{efficiency:.2f} A:{adaptability:.2f}\n"
        title += f"BD1:{bd1:.2f} BD2:{bd2:.2f}"
        ax.set_title(title, fontsize=9, fontweight='bold')
    
    # Hide unused subplots
    for idx in range(len(selected_layouts), len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(f"Diverse Layouts from MAP-Elites Archive (Showing {len(selected_layouts)} layouts)",
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    return fig


def run_mapelites_with_learned_bds(
    facility_count: int = 5,
    iterations: int = 10000,
    initial_population: int = 500,
    use_learned_descriptors: bool = True,
    pretrain_iterations: int = 2000,
    training_frequency: int = 1000,
    latent_dim: int = 2,
    save_model: bool = True,
    output_dir: str = "results",
    seed: int = 42,
    visualize: bool = False,
    export_count: int = 30,
    export_all: bool = False,
    export_pngs: bool = True,
    export_safe_only: bool = True,
    facility_mix: Optional[List[str]] = None
):
    """
    Run MAP-Elites with autoencoder-based behavioral descriptor learning.
    
    Args:
        facility_count: Number of facilities in layout
        iterations: Total iterations
        initial_population: Size of initial population
        use_learned_descriptors: Whether to use learned vs hand-crafted BDs
        pretrain_iterations: Iterations before first autoencoder training
        training_frequency: Retrain autoencoder every N iterations
        latent_dim: Latent dimension (typically 2 for MAP-Elites)
        save_model: Whether to save trained models
        output_dir: Directory for results
        seed: Random seed for reproducibility
        visualize: Whether to export individual layout JSON files and PNG previews
        export_count: Number of layouts to export when export_all is False
        export_all: Export every final archive layout instead of the top subset
        export_pngs: Save a PNG preview beside each exported layout JSON
        export_safe_only: Export only strictly feasible layouts with no violations
        facility_mix: Optional explicit facility type list overriding auto-generation
    """
    from core.layout_autoencoder import set_random_seeds
    
    # Set all random seeds at the very beginning
    set_random_seeds(seed)
    
    print("="*80)
    print("CEXO: Construction Site eXploration & Optimisation")
    print("Quality-Diversity with Learned Behavioral Descriptors")
    print("="*80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate or apply facility mix before creating the site config.
    if facility_mix is not None:
        facility_types = list(facility_mix)
        facility_count = len(facility_types)
        mix_source = "custom"
    else:
        facility_types = generate_facility_mix(facility_count, seed=seed)
        mix_source = "auto"

    # Configuration
    site_config = SiteConfig(
        facility_count=facility_count,
        boundary_margin=0.08,
        seed=seed
    )
    
    mapelites_config = MapElitesConfig(
        grid_size=(20, 20),
        iterations=iterations,
        initial_population=initial_population
    )
    
    autoencoder_config = AutoencoderConfig(
        use_learned_descriptors=use_learned_descriptors,
        latent_dim=latent_dim,
        encoder_hidden=[128, 64, 32],
        decoder_hidden=[32, 64, 128],
        learning_rate=0.001,
        batch_size=32,
        training_epochs=50,
        training_frequency=training_frequency,
        min_samples_for_training=100,
        pretrain_iterations=pretrain_iterations,
        save_model_path=os.path.join(output_dir, "autoencoder_model") if save_model else None,
        seed=seed
    )
    
    print(f"\nConfiguration:")
    print(f"  Facilities: {facility_count} - {', '.join(facility_types)}")
    print(f"  Facility mix source: {mix_source}")
    print(f"  Archive: {mapelites_config.grid_size[0]}x{mapelites_config.grid_size[1]} = {mapelites_config.grid_size[0] * mapelites_config.grid_size[1]} cells")
    print(f"  Iterations: {iterations}")
    print(f"  Initial population: {initial_population}")
    print(f"\nAutoencoder Configuration:")
    print(f"  Use learned BDs: {use_learned_descriptors}")
    if use_learned_descriptors:
        print(f"  Pretrain period: {pretrain_iterations} iterations")
        print(f"  Training frequency: every {training_frequency} iterations")
        print(f"  Latent dimensions: {latent_dim}")
        print(f"  Training epochs: {autoencoder_config.training_epochs}")
        print(f"  Batch size: {autoencoder_config.batch_size}")
    
    # Run MAP-Elites
    algorithm = MapElitesWithAutoencoder(
        facility_types=facility_types,
        site_config=site_config,
        mapelites_config=mapelites_config,
        autoencoder_config=autoencoder_config
    )
    
    results = algorithm.run()
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    stats = results['stats']
    print(f"\nArchive Coverage:")
    print(f"  Cells filled: {stats['coverage']}/{algorithm.archive.total_cells} ({stats['coverage_pct']:.2f}%)")
    
    print(f"\nQuality Metrics:")
    print(f"  Average scalar fitness: {stats['avg_scalar_fitness']:.4f}")
    print(f"  Best scalar fitness: {stats['best_scalar_fitness']:.4f}")
    print(f"  Safety-threshold solutions (>=0.7): {stats['safety_feasible_count']}")
    print(f"  Strict feasible solutions (no violations): {stats.get('strict_feasible_count', 'N/A')}")
    
    print(f"\nBehavioral Descriptors:")
    print(f"  Final mode: {results['bd_mode']}")
    if results['autoencoder_trained']:
        print(f"  Training sessions: {len(results['training_history'])}")
        if results['training_history']:
            final_loss = results['training_history'][-1]['loss']
            print(f"  Final reconstruction loss: {final_loss:.6f}")
    
    best = results['best_individual']
    if best:
        print(f"\nBest Solution:")
        print(f"  Safety: {best.objectives[0]:.4f}")
        print(f"  Efficiency: {best.objectives[1]:.4f}")
        print(f"  Adaptability: {best.objectives[2]:.4f}")
        print(f"  Behaviors: BD1={best.behaviors[0]:.3f}, BD2={best.behaviors[1]:.3f}")
        print(f"  Feasible: {best.feasible}")
    
    print(f"\nRuntime: {results['runtime']:.2f} seconds")
    
    # Visualizations
    print(f"\nGenerating visualizations...")
    
    # 1. Archive heatmap
    fig = visualize_mapelites_archive_heatmap(
        results['archive'],
        title=f"MAP-Elites Archive - {results['bd_mode']} BDs"
    )
    heatmap_path = os.path.join(output_dir, "archive_heatmap.png")
    fig.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved archive heatmap: {heatmap_path}")
    
    # 2. Best solution layout
    if best:
        fig = visualize_layout_preview(
            best.solution,
            best.entrances,
            title=f"Best Layout - Fitness: {algorithm.archive.calculate_scalar_fitness(best):.3f}"
        )
        layout_path = os.path.join(output_dir, "best_layout.png")
        fig.savefig(layout_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved best layout: {layout_path}")
    
    # 3. Training history (if autoencoder was trained)
    if results['autoencoder_trained'] and results['training_history']:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        iterations_list = [h['iteration'] for h in results['training_history']]
        losses = [h['loss'] for h in results['training_history']]
        samples = [h['num_samples'] for h in results['training_history']]
        
        ax1.plot(iterations_list, losses, 'b-o', linewidth=2, markersize=6)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Reconstruction Loss')
        ax1.set_title('Autoencoder Training Progress')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(iterations_list, samples, 'g-s', linewidth=2, markersize=6)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Training Samples')
        ax2.set_title('Archive Growth')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        training_path = os.path.join(output_dir, "training_history.png")
        fig.savefig(training_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved training history: {training_path}")
    
    # 4. Diverse layouts showcase (NEW!)
    print(f"\n  Generating diverse layouts showcase...")
    diverse_layouts_fig = visualize_diverse_layouts(results['archive'], algorithm)
    if diverse_layouts_fig:
        diverse_path = os.path.join(output_dir, "diverse_layouts.png")
        diverse_layouts_fig.savefig(diverse_path, dpi=300, bbox_inches='tight')
        plt.close(diverse_layouts_fig)
        print(f"  Saved diverse layouts: {diverse_path}")
    
    # Save results to JSON
    results_json = {
        "configuration": {
            "facility_count": facility_count,
            "facility_types": facility_types,
            "iterations": iterations,
            "initial_population": initial_population,
            "grid_size": list(mapelites_config.grid_size),
            "use_learned_descriptors": use_learned_descriptors,
            "latent_dim": latent_dim if use_learned_descriptors else None
        },
        "statistics": {
            "coverage": int(stats['coverage']),
            "coverage_percentage": float(stats['coverage_pct']),
            "avg_scalar_fitness": float(stats['avg_scalar_fitness']),
            "best_scalar_fitness": float(stats['best_scalar_fitness']),
            "safety_feasible_count": int(stats['safety_feasible_count']),
            "strict_feasible_count": int(stats.get('strict_feasible_count', 0)),
            "runtime_seconds": float(results['runtime'])
        },
        "bd_mode": results['bd_mode'],
        "autoencoder_trained": results['autoencoder_trained'],
        "training_sessions": len(results['training_history']) if results['autoencoder_trained'] else 0
    }
    
    if best:
        results_json["best_solution"] = {
            "objectives": {
                "safety": float(best.objectives[0]),
                "efficiency": float(best.objectives[1]),
                "adaptability": float(best.objectives[2])
            },
            "behaviors": {
                "bd1": float(best.behaviors[0]),
                "bd2": float(best.behaviors[1])
            },
            "feasible": bool(best.feasible)
        }

    if visualize:
        max_layouts = None if export_all else export_count
        exported_layouts = export_cslpelite_results(
            results['archive'],
            site_config,
            output_dir,
            max_layouts=max_layouts,
            export_pngs=export_pngs,
            safe_only=export_safe_only
        )
        results_json["exported_layouts"] = {
            "count": int(exported_layouts),
            "pngs": bool(export_pngs),
            "strict_feasible_only": bool(export_safe_only),
            "all_archive_layouts": bool(export_all)
        }
    
    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"  Saved results JSON: {json_path}")
    
    print(f"\nAll results saved to: {output_dir}/")
    
    return results


def compare_hand_crafted_vs_learned(
    seed=42,
    facility_mix: Optional[List[str]] = None,
    output_dir: str = "results/compare"
):
    """
    Run comparative experiment: hand-crafted vs learned behavioral descriptors.
    
    Args:
        seed: Random seed for reproducibility
        facility_mix: Optional explicit facility type list shared by both runs
        output_dir: Parent directory for comparison outputs
    """
    from core.layout_autoencoder import set_random_seeds
    
    # Set all random seeds at the very beginning
    set_random_seeds(seed)
    
    print("\n" + "="*80)
    print("COMPARATIVE EXPERIMENT: Hand-Crafted vs Learned BDs")
    print("="*80)
    
    configs = [
        {
            "name": "Hand-Crafted BDs",
            "use_learned": False,
            "output_dir": os.path.join(output_dir, "handcrafted")
        },
        {
            "name": "Learned BDs (Autoencoder)",
            "use_learned": True,
            "output_dir": os.path.join(output_dir, "learned")
        }
    ]
    
    all_results = {}
    
    for config in configs:
        print(f"\n\n{'='*80}")
        print(f"Running: {config['name']}")
        print(f"{'='*80}")
        
        results = run_mapelites_with_learned_bds(
            facility_count=len(facility_mix) if facility_mix is not None else 5,
            iterations=8000,
            initial_population=400,
            use_learned_descriptors=config['use_learned'],
            pretrain_iterations=2000,
            training_frequency=1000,
            output_dir=config['output_dir'],
            seed=seed,
            facility_mix=facility_mix
        )
        
        all_results[config['name']] = results
    
    # Comparison summary
    print("\n\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    for name, results in all_results.items():
        stats = results['stats']
        print(f"\n{name}:")
        print(f"  Coverage: {stats['coverage_pct']:.2f}%")
        print(f"  Avg Fitness: {stats['avg_scalar_fitness']:.4f}")
        print(f"  Best Fitness: {stats['best_scalar_fitness']:.4f}")
        print(f"  Runtime: {results['runtime']:.2f}s")


def run_ablation_study(
    facilities=5,
    iterations=10000,
    initial_population=500,
    seed=42,
    output_dir="results/ablation",
    facility_mix: Optional[List[str]] = None
):
    """
    Run ablation study comparing:
    1. CEXO (Proposed) - Full method with learned exploration
    2. Exploration Baseline - Quality diversity without learning
    3. Optimization Baseline - Pure multi-objective optimization
    
    Args:
        facilities: Number of facilities
        iterations: Number of iterations/generations
        initial_population: Initial population size
        seed: Random seed for reproducibility
        output_dir: Directory for ablation results
        facility_mix: Optional explicit facility type list shared by all methods
    """
    import time
    from core.layout_autoencoder import set_random_seeds
    
    # Set all random seeds at the very beginning for reproducibility
    set_random_seeds(seed)
    
    print("="*80)
    print("ABLATION STUDY: CEXO Components Analysis")
    print("="*80)
    if facility_mix is not None:
        facility_types = list(facility_mix)
        facilities = len(facility_types)
        mix_source = "custom"
    else:
        facility_types = generate_facility_mix(facilities, seed=seed)
        mix_source = "auto"

    print(f"\nConfiguration:")
    print(f"  Facilities: {facilities} - {', '.join(facility_types)}")
    print(f"  Facility mix source: {mix_source}")
    print(f"  Iterations/Generations: {iterations}")
    print(f"  Initial Population: {initial_population}")
    print(f"  Seed: {seed}")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # =========================================================================
    # 1. CEXO (Proposed) - Full method with learned exploration
    # =========================================================================
    print("\n\n" + "="*80)
    print("1/3: CEXO (Proposed)")
    print("     Quality-Diversity with Learned Behavioral Descriptors")
    print("="*80)
    
    start_time = time.time()
    
    site_config = SiteConfig(facility_count=facilities, seed=seed)
    mapelites_config = MapElitesConfig(
        iterations=iterations,
        initial_population=initial_population
    )
    autoencoder_config = AutoencoderConfig(
        use_learned_descriptors=True,
        pretrain_iterations=min(2000, iterations // 5),
        training_frequency=min(1000, iterations // 10),
        latent_dim=2,
        seed=seed
    )
    
    algorithm = MapElitesWithAutoencoder(
        facility_types=facility_types,
        site_config=site_config,
        mapelites_config=mapelites_config,
        autoencoder_config=autoencoder_config
    )
    
    result = algorithm.run()
    runtime = time.time() - start_time
    
    archive = result['archive']
    training_history = result['training_history']
    
    # Calculate metrics
    coverage = len(archive.archive) / archive.total_cells * 100
    all_inds = archive.get_all_individuals()
    avg_fitness = np.mean([archive.calculate_scalar_fitness(ind) for ind in all_inds])
    best_fitness = max([archive.calculate_scalar_fitness(ind) for ind in all_inds])
    
    results['CEXO (Proposed)'] = {
        'coverage': coverage,
        'avg_fitness': avg_fitness,
        'best_fitness': best_fitness,
        'runtime': runtime,
        'archive': archive,
        'num_solutions': len(all_inds)
    }
    
    print(f"\nCompleted: Coverage={coverage:.2f}%, Best Fitness={best_fitness:.3f}, Time={runtime:.1f}s")
    
    # =========================================================================
    # 2. Exploration Baseline - Quality diversity without learning
    # =========================================================================
    print("\n\n" + "="*80)
    print("2/3: Exploration Baseline")
    print("     Quality-Diversity with Hand-Crafted Behavioral Descriptors")
    print("="*80)
    
    start_time = time.time()
    
    site_config = SiteConfig(facility_count=facilities, seed=seed)
    mapelites_config = MapElitesConfig(
        iterations=iterations,
        initial_population=initial_population
    )
    
    optimizer = PureMapElitesOptimizer(
        site_config=site_config,
        facility_types=facility_types,
        mapelites_config=mapelites_config
    )
    
    result_baseline = optimizer.run()
    runtime = time.time() - start_time
    
    archive_baseline = result_baseline['archive']
    
    # Calculate metrics
    coverage = len(archive_baseline.archive) / archive_baseline.total_cells * 100
    all_inds = list(archive_baseline.archive.values())
    avg_fitness = np.mean([archive_baseline.calculate_scalar_fitness(ind) for ind in all_inds])
    best_fitness = max([archive_baseline.calculate_scalar_fitness(ind) for ind in all_inds])
    
    results['Exploration Baseline'] = {
        'coverage': coverage,
        'avg_fitness': avg_fitness,
        'best_fitness': best_fitness,
        'runtime': runtime,
        'archive': archive_baseline,
        'num_solutions': len(all_inds)
    }
    
    print(f"\nCompleted: Coverage={coverage:.2f}%, Best Fitness={best_fitness:.3f}, Time={runtime:.1f}s")
    
    # =========================================================================
    # 3. Optimization Baseline - Pure multi-objective optimization
    # =========================================================================
    print("\n\n" + "="*80)
    print("3/3: Optimization Baseline")
    print("     Pure Multi-Objective Optimization (NSGA-II)")
    print("="*80)
    
    start_time = time.time()
    
    site_config = SiteConfig(facility_count=facilities, seed=seed)
    nsga2_config = NSGA2Config(
        population_size=initial_population,
        generations=iterations // 10  # Adjust for comparable evaluation budget
    )
    
    optimizer_nsga2 = PureNSGA2Optimizer(
        site_config=site_config,
        facility_types=facility_types,
        nsga2_config=nsga2_config
    )
    
    result_nsga2 = optimizer_nsga2.run()
    runtime = time.time() - start_time
    
    final_population = result_nsga2['population']
    
    # Calculate metrics (no coverage for NSGA-II)
    avg_fitness = np.mean([
        0.5 * ind.objectives[0] + 0.3 * ind.objectives[1] + 0.2 * ind.objectives[2]
        for ind in final_population
    ])
    best_fitness = max([
        0.5 * ind.objectives[0] + 0.3 * ind.objectives[1] + 0.2 * ind.objectives[2]
        for ind in final_population
    ])
    
    results['Optimization Baseline'] = {
        'coverage': 0.0,  # N/A for NSGA-II
        'avg_fitness': avg_fitness,
        'best_fitness': best_fitness,
        'runtime': runtime,
        'population': final_population,
        'num_solutions': len(final_population)
    }
    
    print(f"\nCompleted: Best Fitness={best_fitness:.3f}, Time={runtime:.1f}s")
    
    # =========================================================================
    # Generate Comparison Report
    # =========================================================================
    print("\n\n" + "="*80)
    print("ABLATION STUDY RESULTS")
    print("="*80)
    print(f"\n{'Method':<28} {'Coverage':<12} {'Avg Fitness':<14} {'Best Fitness':<14} {'Runtime':<10}")
    print("-" * 80)
    
    for method_name, result in results.items():
        coverage_str = f"{result['coverage']:.2f}%" if result['coverage'] > 0 else "N/A"
        print(f"{method_name:<28} {coverage_str:<12} {result['avg_fitness']:<14.4f} {result['best_fitness']:<14.4f} {result['runtime']:<10.1f}s")
    
    # Key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    print(f"- CEXO (Proposed) achieves {results['CEXO (Proposed)']['coverage']:.1f}% behavioral coverage")
    print(f"- Exploration Baseline achieves {results['Exploration Baseline']['coverage']:.1f}% coverage")
    print(f"- Optimization Baseline focuses on quality without diversity")
    print(f"- CEXO combines high coverage ({results['CEXO (Proposed)']['coverage']:.1f}%) with high fitness ({results['CEXO (Proposed)']['best_fitness']:.3f})")
    
    # Save results
    results_json = {
        method: {
            'coverage': result['coverage'],
            'avg_fitness': float(result['avg_fitness']),
            'best_fitness': float(result['best_fitness']),
            'runtime': result['runtime'],
            'num_solutions': result['num_solutions']
        }
        for method, result in results.items()
    }
    
    json_path = os.path.join(output_dir, "ablation_results.json")
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\nResults saved to: {json_path}")
    
    # Generate comparative visualizations
    print(f"\n  Generating comparative visualizations...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Coverage comparison (bar chart)
    ax = axes[0]
    methods = list(results.keys())
    coverages = [results[m]['coverage'] for m in methods]
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    bars = ax.bar(range(len(methods)), coverages, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Behavioral Space Coverage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Behavioral Diversity Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width()/2., 5,
                   'N/A', ha='center', va='bottom', fontweight='bold')
    
    # Fitness comparison (bar chart)
    ax = axes[1]
    best_fitnesses = [results[m]['best_fitness'] for m in methods]
    avg_fitnesses = [results[m]['avg_fitness'] for m in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, best_fitnesses, width, label='Best Fitness', 
                   color=colors, alpha=0.9, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, avg_fitnesses, width, label='Avg Fitness',
                   color=colors, alpha=0.5, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Scalar Fitness', fontsize=12, fontweight='bold')
    ax.set_title('Solution Quality Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, "ablation_comparison.png")
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {comparison_path}")
    
    print("\n" + "="*80)
    print(f"Ablation study complete! Results saved to: {output_dir}")
    print("="*80)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CEXO: Construction Site eXploration & Optimisation")
    parser.add_argument('--facilities', type=int, default=5, help='Number of facilities')
    parser.add_argument('--iterations', type=int, default=10000, help='Number of iterations')
    parser.add_argument('--initial-pop', type=int, default=500, help='Initial population size')
    parser.add_argument('--no-learned', action='store_true', help='Disable learned BDs (use hand-crafted)')
    parser.add_argument('--pretrain', type=int, default=2000, help='Pretrain iterations')
    parser.add_argument('--train-freq', type=int, default=1000, help='Training frequency')
    parser.add_argument('--latent-dim', type=int, default=2, help='Latent dimension')
    parser.add_argument('--output', type=str, default='results', help='Base output directory; each run creates a timestamped subfolder')
    parser.add_argument('--compare', action='store_true', help='Run comparison experiment')
    parser.add_argument('--ablation', action='store_true', help='Run ablation study')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument(
        '--facility-mix',
        type=str,
        default=None,
        help="Override auto facility selection, e.g. core=2,crane=1,storage=2,office=1,rest_area=1"
    )
    parser.add_argument('--visualize', action='store_true', help='Export final archive layouts as JSON files and PNG previews')
    parser.add_argument('--export-count', type=int, default=30, help='Number of final archive layouts to export')
    parser.add_argument('--export-all', action='store_true', help='Export every final archive layout instead of only the top export-count layouts')
    parser.add_argument('--no-export-pngs', action='store_true', help='Only export layout JSON files, without PNG previews')
    parser.add_argument('--export-unsafe', action='store_true', help='Include layouts with recorded feasibility violations in exports')
    
    args = parser.parse_args()

    try:
        facility_mix = parse_facility_mix(args.facility_mix) if args.facility_mix else None
    except ValueError as exc:
        parser.error(str(exc))
    
    if args.ablation:
        output_dir = make_run_output_dir(args.output, "ablation")
        run_ablation_study(
            facilities=args.facilities,
            iterations=args.iterations,
            initial_population=args.initial_pop,
            seed=args.seed,
            output_dir=output_dir,
            facility_mix=facility_mix
        )
    elif args.compare:
        output_dir = make_run_output_dir(args.output, "compare")
        compare_hand_crafted_vs_learned(
            seed=args.seed,
            facility_mix=facility_mix,
            output_dir=output_dir
        )
    else:
        output_dir = make_run_output_dir(args.output, "cexo")
        run_mapelites_with_learned_bds(
            facility_count=args.facilities,
            iterations=args.iterations,
            initial_population=args.initial_pop,
            use_learned_descriptors=not args.no_learned,
            pretrain_iterations=args.pretrain,
            training_frequency=args.train_freq,
            latent_dim=args.latent_dim,
            output_dir=output_dir,
            seed=args.seed,
            visualize=args.visualize,
            export_count=args.export_count,
            export_all=args.export_all,
            export_pngs=not args.no_export_pngs,
            export_safe_only=not args.export_unsafe,
            facility_mix=facility_mix
        )
