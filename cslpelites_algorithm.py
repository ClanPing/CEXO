#!/usr/bin/env python3
"""
CSLP Elite Algorithm Module
===========================

CSLP Elite (MAP-Elites + NSGA-II) implementation with 2D behavioral space and 
3-objective NSGA-II optimization for construction site layout optimization 
with enhanced variation and quality.
"""

import random
import time
from typing import List, Dict, Tuple, Optional
import numpy as np

from config import Individual, SiteConfig, MapElitesConfig
from objectives import evaluate_individual
from behavioral_descriptors import calculate_spatial_organization, calculate_functional_integration
from layout_generation import (generate_random_entrances, create_random_layout, create_targeted_layout,
                              mutate_layout, mutate_toward_behavioral_diversity, crossover_layouts, 
                              repair_layout_constraints)
from nsga2_algorithm import dominates, calculate_crowding_distance

# =============================================================================
# PARETO FRONT MANAGEMENT
# =============================================================================

class ParetoFront:
    """Manages a 3-objective Pareto front"""
    
    def __init__(self, max_size: int = 12):
        self.max_size = max_size
        self.individuals: List[Individual] = []
    
    def add_individual(self, individual: Individual) -> bool:
        """Add individual to Pareto front"""
        if individual.objectives[0] < 0.95:  # Strict safety threshold - only accept very safe layouts
            return False

        self.individuals = [ind for ind in self.individuals if not dominates(individual, ind)]
        
        if any(dominates(existing, individual) for existing in self.individuals):
            return False
        
        self.individuals.append(individual)
        
        if len(self.individuals) > self.max_size:
            self._trim_front()
        
        return True
    
    def _trim_front(self):
        """Trim front to max_size using crowding distance"""
        if len(self.individuals) <= self.max_size:
            return
        
        calculate_crowding_distance(self.individuals)
        
        self.individuals.sort(key=lambda x: -x.crowding_distance if x.crowding_distance != float('inf') else -1e6)
        self.individuals = self.individuals[:self.max_size]
    
    def get_random_individual(self) -> Optional[Individual]:
        """Get random individual from front"""
        return random.choice(self.individuals) if self.individuals else None
    
    def size(self) -> int:
        return len(self.individuals)

# =============================================================================
# MAP-ELITES ARCHIVE
# =============================================================================

class MapElitesArchive:
    """MAP-Elites archive with 2D behavioral space and 3-objective optimization"""
    
    def __init__(self, grid_size: Tuple[int, int] = (20, 20), pareto_size: int = 12):
        self.grid_size = grid_size
        self.pareto_size = pareto_size
        self.total_cells = grid_size[0] * grid_size[1]
        self.archive: Dict[Tuple[int, int], ParetoFront] = {}
        self.evaluations = 0
    
    def get_cell_coords(self, behaviors: Tuple[float, float]) -> Tuple[int, int]:
        """Convert 2D behaviors to grid coordinates"""
        b1, b2 = behaviors
        i = int(np.clip(b1 * self.grid_size[0], 0, self.grid_size[0] - 1))
        j = int(np.clip(b2 * self.grid_size[1], 0, self.grid_size[1] - 1))
        return (i, j)
    
    def add_individual(self, individual: Individual) -> bool:
        """Add individual to appropriate cell's Pareto front"""
        coords = self.get_cell_coords(individual.behaviors)
        
        if coords not in self.archive:
            self.archive[coords] = ParetoFront(self.pareto_size)
        
        return self.archive[coords].add_individual(individual)
    
    def get_random_individual(self) -> Optional[Individual]:
        """Get random individual from random occupied cell"""
        if not self.archive:
            return None
        
        coords = random.choice(list(self.archive.keys()))
        return self.archive[coords].get_random_individual()
    
    def get_all_individuals(self) -> List[Individual]:
        """Get all individuals from all cells"""
        individuals = []
        for pareto_front in self.archive.values():
            individuals.extend(pareto_front.individuals)
        return individuals
    
    def get_stats(self) -> Dict:
        """Get comprehensive archive statistics"""
        if not self.archive:
            return {
                "coverage": 0, "coverage_pct": 0.0, "total_individuals": 0,
                "safety_feasible_count": 0, "avg_safety": 0.0, 
                "avg_efficiency": 0.0, "avg_adaptability": 0.0
            }
        
        all_individuals = self.get_all_individuals()
        
        coverage = len(self.archive)
        total_individuals = len(all_individuals)
        safety_feasible = [ind for ind in all_individuals if ind.objectives[0] >= 0.95]
        
        objectives_array = np.array([ind.objectives for ind in all_individuals])
        avg_safety = np.mean(objectives_array[:, 0])
        avg_efficiency = np.mean(objectives_array[:, 1])
        avg_adaptability = np.mean(objectives_array[:, 2])
        
        return {
            "coverage": coverage,
            "coverage_pct": 100.0 * coverage / self.total_cells,
            "total_individuals": total_individuals,
            "safety_feasible_count": len(safety_feasible),
            "avg_safety": avg_safety,
            "avg_efficiency": avg_efficiency,
            "avg_adaptability": avg_adaptability
        }

# =============================================================================
# MAP-ELITES OPTIMIZER
# =============================================================================

class MapElitesNSGA2Optimizer:
    """MAP-Elites with 3-objective NSGA-II optimization"""
    
    def __init__(self, site_config: SiteConfig, facility_types: List[str], 
                 mapelites_config: MapElitesConfig = None):
        self.site_config = site_config
        self.facility_types = facility_types
        self.mapelites_config = mapelites_config or MapElitesConfig()
        self.archive = MapElitesArchive(self.mapelites_config.grid_size, site_config.pareto_size)
        
        random.seed(site_config.seed)
        np.random.seed(site_config.seed)
    
    def evaluate_solution(self, solution: List[Dict], 
                         entrances: List[Tuple[float, float]]) -> Individual:
        """Evaluate solution with 3-objective fitness and behavioral descriptors"""
        self.archive.evaluations += 1
        
        result = evaluate_individual(solution, entrances, self.site_config, calculate_behaviors=True)
        
        return Individual(
            solution=solution,
            entrances=entrances,
            objectives=result['objectives'],
            behaviors=result['behaviors'],
            feasible=result['feasible'],
            violations=result['violations']
        )
    
    def run(self, iterations: int = None, initial_population: int = None) -> Dict:
        """Run MAP-Elites with 3-objective NSGA-II"""
        if iterations is None:
            iterations = self.mapelites_config.iterations
        if initial_population is None:
            initial_population = self.mapelites_config.initial_population
            
        print(f"\nRunning MAP-Elites + 3-Objective NSGA-II:")
        print(f"Facilities: {len(self.facility_types)} ({', '.join(self.facility_types)})")
        print(f"Archive: {self.archive.total_cells:,} cells × {self.site_config.pareto_size} Pareto size")
        print(f"Evolution: {iterations} iterations, {initial_population} initial population")
        
        start_time = time.time()
        
        # Initialize with diverse population targeting different behavioral regions
        print("Creating initial population...")
        behavioral_targets = [
            (0.1, 0.1), (0.1, 0.9), (0.9, 0.1), (0.9, 0.9),  # Corners
            (0.1, 0.5), (0.9, 0.5), (0.5, 0.1), (0.5, 0.9),  # Edges
            (0.3, 0.3), (0.3, 0.7), (0.7, 0.3), (0.7, 0.7)   # Intermediate
        ]
        
        for i in range(initial_population):
            entrances = generate_random_entrances(self.site_config, seed=self.site_config.seed + i)
            
            if i < len(behavioral_targets) * 20:  # First portion: targeted layouts
                target_idx = i // 20
                target_spatial, target_functional = behavioral_targets[target_idx % len(behavioral_targets)]
                solution = create_targeted_layout(
                    self.facility_types, self.site_config.boundary_margin,
                    target_spatial, target_functional)
            else:  # Remaining: random layouts
                solution = create_random_layout(self.facility_types, self.site_config.boundary_margin)
            
            # Apply constraint repair to initial solutions with reduced safety threshold
            if random.random() > 0.3:  # 70% chance to repair in initial population
                solution = repair_layout_constraints(solution, self.site_config.boundary_margin, 
                                                   entrances, self.site_config)
            individual = self.evaluate_solution(solution, entrances)
            
            # Much more accepting threshold for initial population to encourage safety diversity
            if individual.objectives[0] >= 0.4:  # Very low threshold for initial diversity
                self.archive.add_individual(individual)
            
            if (i + 1) % (initial_population // 10) == 0:
                print(f"  Generated {i+1}/{initial_population} initial solutions")
        
        initial_stats = self.archive.get_stats()
        print(f"Initial archive: {initial_stats['coverage']} cells, "
              f"{initial_stats['total_individuals']} individuals")
        
        # Main evolution loop
        print("Starting evolution...")
        report_interval = max(1, iterations // 10)
        
        for iteration in range(iterations):
            # Progressive safety threshold: start at 0.8, gradually increase to 0.95
            progress = iteration / iterations
            safety_threshold = 0.8 + 0.15 * progress
            
            # Determine variation strategy based on iteration and archive coverage
            current_coverage = self.archive.get_stats()['coverage_pct']
            
            # Increase diversity-focused operations when coverage is low
            if current_coverage < 15.0:  # Low coverage - emphasize diversity
                diversity_rate = 0.3
                mutation_rate = 0.4
                targeted_rate = 0.3
            else:
                diversity_rate = 0.15
                mutation_rate = 0.65
                targeted_rate = 0.2
            
            variation_type = random.random()
            
            if variation_type < diversity_rate:  # Behavioral diversity mutation
                parent = self.archive.get_random_individual()
                if parent is not None:
                    # Use behavioral diversity mutation
                    current_spatial = parent.behaviors[0]
                    current_functional = parent.behaviors[1]
                    offspring_solution = mutate_toward_behavioral_diversity(
                        parent.solution, self.site_config.boundary_margin, 
                        current_spatial, current_functional)
                    offspring_entrances = parent.entrances[:]
                else:
                    # Create targeted layout in under-explored region
                    target_spatial = random.choice([0.1, 0.3, 0.7, 0.9])
                    target_functional = random.choice([0.1, 0.3, 0.7, 0.9])
                    offspring_solution = create_targeted_layout(
                        self.facility_types, self.site_config.boundary_margin,
                        target_spatial, target_functional)
                    offspring_entrances = generate_random_entrances(
                        self.site_config, seed=self.site_config.seed + iteration)
                        
            elif variation_type < diversity_rate + targeted_rate:  # Targeted generation
                # Generate layout targeting specific behavioral regions
                target_spatial = random.choice([0.1, 0.9])
                target_functional = random.choice([0.1, 0.9])
                offspring_solution = create_targeted_layout(
                    self.facility_types, self.site_config.boundary_margin,
                    target_spatial, target_functional)
                offspring_entrances = generate_random_entrances(
                    self.site_config, seed=self.site_config.seed + iteration)
                    
            elif variation_type < diversity_rate + targeted_rate + mutation_rate:  # Standard mutation
                parent = self.archive.get_random_individual()
                if parent is not None:
                    # Increase mutation strength for better exploration
                    enhanced_sigma = 0.06 if current_coverage < 10.0 else 0.04
                    offspring_solution = mutate_layout(
                        parent.solution, self.site_config.boundary_margin, 
                        p_mut=0.5, sigma=enhanced_sigma)
                    if random.random() < 0.2:
                        offspring_entrances = generate_random_entrances(
                            self.site_config, seed=self.site_config.seed + iteration)
                    else:
                        offspring_entrances = parent.entrances[:]
                else:
                    offspring_entrances = generate_random_entrances(
                        self.site_config, seed=self.site_config.seed + iteration)
                    offspring_solution = create_random_layout(
                        self.facility_types, self.site_config.boundary_margin)
                
            else:  # Crossover
                parent1 = self.archive.get_random_individual()
                parent2 = self.archive.get_random_individual()
                
                if parent1 is not None and parent2 is not None:
                    child1, child2 = crossover_layouts(parent1.solution, parent2.solution)
                    offspring_solution = random.choice([child1, child2])
                    offspring_entrances = random.choice([parent1.entrances, parent2.entrances])
                else:
                    offspring_entrances = generate_random_entrances(
                        self.site_config, seed=self.site_config.seed + iteration)
                    offspring_solution = create_random_layout(
                        self.facility_types, self.site_config.boundary_margin)
            
            # Apply constraint repair with reduced frequency for more safety diversity
            if progress > 0.4 or random.random() > 0.4:  # 60% chance to repair after 40% progress
                offspring_solution = repair_layout_constraints(
                    offspring_solution, self.site_config.boundary_margin, 
                    offspring_entrances, self.site_config)
            
            individual = self.evaluate_solution(offspring_solution, offspring_entrances)
            
            # More accepting criteria with lower and more graduated threshold
            min_safety_threshold = 0.5 + 0.3 * progress  # Start at 0.5, end at 0.8
            actual_threshold = min(safety_threshold, min_safety_threshold)
            
            if individual.objectives[0] >= actual_threshold:
                self.archive.add_individual(individual)
            
            if (iteration + 1) % report_interval == 0:
                stats = self.archive.get_stats()
                print(f"  Iter {iteration+1:>6}: Cells={stats['coverage']:>4} "
                      f"({stats['coverage_pct']:>6.3f}%), Indiv={stats['total_individuals']:>5}, "
                      f"Safety={stats['avg_safety']:.3f}, Eff={stats['avg_efficiency']:.3f}, "
                      f"Adapt={stats['avg_adaptability']:.3f}, Threshold={safety_threshold:.3f}")
        
        runtime = time.time() - start_time
        final_stats = self.archive.get_stats()
        
        print(f"\nEvolution completed in {runtime:.2f} seconds")
        print(f"Final archive: {final_stats['coverage']:,}/{self.archive.total_cells:,} cells")
        print(f"Total individuals: {final_stats['total_individuals']:,}")
        print(f"Safety feasible (≥0.7): {final_stats['safety_feasible_count']}")
        
        return {
            "archive": self.archive,
            "stats": final_stats,
            "runtime": runtime
        }

# =============================================================================
# MAP-ELITES EVALUATION FUNCTIONS
# =============================================================================

def evaluate_mapelites_performance(archive: MapElitesArchive, site_config: SiteConfig) -> Dict:
    """Comprehensive MAP-Elites evaluation metrics"""
    all_individuals = archive.get_all_individuals()
    
    if not all_individuals:
        return {"error": "Empty archive"}
    
    # Basic coverage metrics
    coverage = len(archive.archive)
    coverage_percentage = 100.0 * coverage / archive.total_cells
    total_individuals = len(all_individuals)
    
    # Behavioral space analysis
    behaviors = np.array([ind.behaviors for ind in all_individuals])
    objectives = np.array([ind.objectives for ind in all_individuals])
    
    # Coverage metrics
    coverage_metrics = {
        "cells_filled": coverage,
        "total_cells": archive.total_cells,
        "coverage_percentage": coverage_percentage,
        "cells_per_dimension": archive.grid_size,
        "average_individuals_per_cell": total_individuals / max(1, coverage)
    }
    
    # Behavioral diversity metrics
    bd1_range = behaviors[:, 0].max() - behaviors[:, 0].min() if len(behaviors) > 0 else 0
    bd2_range = behaviors[:, 1].max() - behaviors[:, 1].min() if len(behaviors) > 0 else 0
    bd1_variance = np.var(behaviors[:, 0]) if len(behaviors) > 0 else 0
    bd2_variance = np.var(behaviors[:, 1]) if len(behaviors) > 0 else 0
    
    diversity_metrics = {
        "behavioral_range_bd1": bd1_range,
        "behavioral_range_bd2": bd2_range,
        "behavioral_coverage_bd1": bd1_range,
        "behavioral_coverage_bd2": bd2_range,
        "behavioral_variance_bd1": bd1_variance,
        "behavioral_variance_bd2": bd2_variance,
        "behavioral_uniformity": min(bd1_variance, bd2_variance)
    }
    
    # Quality metrics
    safety_feasible = [ind for ind in all_individuals if ind.objectives[0] >= 0.95]
    
    quality_metrics = {
        "average_safety": np.mean(objectives[:, 0]),
        "average_efficiency": np.mean(objectives[:, 1]),
        "average_adaptability": np.mean(objectives[:, 2]),
        "max_safety": np.max(objectives[:, 0]),
        "max_efficiency": np.max(objectives[:, 1]),
        "max_adaptability": np.max(objectives[:, 2]),
        "safety_feasible_count": len(safety_feasible),
        "safety_feasible_percentage": 100.0 * len(safety_feasible) / len(all_individuals),
        "pareto_front_sizes": [pf.size() for pf in archive.archive.values()]
    }
    
    return {
        "coverage_metrics": coverage_metrics,
        "diversity_metrics": diversity_metrics,
        "quality_metrics": quality_metrics,
        "summary": {
            "total_cells_filled": coverage,
            "coverage_percentage": coverage_percentage,
            "total_solutions": total_individuals,
            "high_quality_solutions": len(safety_feasible),
            "behavioral_diversity_score": (bd1_range + bd2_range) / 2.0,
            "average_objective_quality": np.mean(objectives),
            "mapelites_effectiveness": coverage_percentage * np.mean(objectives) / 100.0
        }
    }
