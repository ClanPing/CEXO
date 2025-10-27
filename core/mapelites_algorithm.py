#!/usr/bin/env python3
"""
Pure MAP-Elites Algorithm Module
================================

Pure MAP-Elites implementation with scalar fitness function combining the three objectives.
Focuses on behavioral diversity exploration across the 2D behavioral space without
multi-objective optimization complexity.
"""

import random
import time
from typing import List, Dict, Tuple, Optional
import numpy as np

from .config import Individual, SiteConfig, MapElitesConfig
from .objectives import evaluate_individual
from .behavioral_descriptors import (
    calculate_spatial_organization,
    calculate_functional_integration,
)
from .layout_generation import (
    generate_random_entrances,
    create_random_layout,
    create_targeted_layout,
    mutate_layout,
    mutate_toward_behavioral_diversity,
    crossover_layouts,
    repair_layout_constraints,
)

# =============================================================================
# PURE MAP-ELITES ARCHIVE
# =============================================================================

class PureMapElitesArchive:
    """Pure MAP-Elites archive with single best individual per cell"""
    
    def __init__(self, grid_size: Tuple[int, int] = (20, 20)):
        self.grid_size = grid_size
        self.total_cells = grid_size[0] * grid_size[1]
        self.archive: Dict[Tuple[int, int], Individual] = {}
        self.evaluations = 0
    
    def get_cell_coords(self, behaviors: Tuple[float, float]) -> Tuple[int, int]:
        """Convert 2D behaviors to grid coordinates"""
        b1, b2 = behaviors
        i = int(np.clip(b1 * self.grid_size[0], 0, self.grid_size[0] - 1))
        j = int(np.clip(b2 * self.grid_size[1], 0, self.grid_size[1] - 1))
        return (i, j)
    
    def add_individual(self, individual: Individual) -> bool:
        """Add individual to cell if it's better than current occupant"""
        coords = self.get_cell_coords(individual.behaviors)
        
        # For pure MAP-Elites, we use scalar fitness for comparison
        scalar_fitness = self.calculate_scalar_fitness(individual)
        individual.scalar_fitness = scalar_fitness
        
        if coords not in self.archive:
            # Empty cell - always add
            self.archive[coords] = individual
            return True
        else:
            # Cell occupied - compare scalar fitness
            current_fitness = getattr(self.archive[coords], 'scalar_fitness', 
                                    self.calculate_scalar_fitness(self.archive[coords]))
            
            if scalar_fitness > current_fitness:
                self.archive[coords] = individual
                return True
        
        return False
    
    def calculate_scalar_fitness(self, individual: Individual) -> float:
        """
        Convert 3-objective vector to scalar fitness
        Uses weighted combination with safety priority
        """
        safety, efficiency, adaptability = individual.objectives
        
        # Weighted combination emphasizing safety
        weights = np.array([0.5, 0.3, 0.2])  # Safety gets highest weight
        scalar_fitness = np.dot([safety, efficiency, adaptability], weights)
        
        # Apply feasibility bonus/penalty
        if individual.feasible:
            scalar_fitness += 0.1  # Feasibility bonus
        else:
            scalar_fitness *= 0.7  # Feasibility penalty
        
        return float(np.clip(scalar_fitness, 0.0, 1.0))
    
    def get_random_individual(self) -> Optional[Individual]:
        """Get random individual from random occupied cell"""
        if not self.archive:
            return None
        
        coords = random.choice(list(self.archive.keys()))
        return self.archive[coords]
    
    def get_best_individual(self) -> Optional[Individual]:
        """Get individual with highest scalar fitness"""
        if not self.archive:
            return None
        
        best_individual = None
        best_fitness = -1.0
        
        for individual in self.archive.values():
            fitness = getattr(individual, 'scalar_fitness', 
                            self.calculate_scalar_fitness(individual))
            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = individual
        
        return best_individual
    
    def get_all_individuals(self) -> List[Individual]:
        """Get all individuals from all cells"""
        return list(self.archive.values())
    
    def get_stats(self) -> Dict:
        """Get comprehensive archive statistics"""
        if not self.archive:
            return {
                "coverage": 0, "coverage_pct": 0.0, "total_individuals": 0,
                "safety_feasible_count": 0, "avg_safety": 0.0, 
                "avg_efficiency": 0.0, "avg_adaptability": 0.0,
                "avg_scalar_fitness": 0.0, "best_scalar_fitness": 0.0
            }
        
        all_individuals = list(self.archive.values())
        
        coverage = len(self.archive)
        total_individuals = len(all_individuals)
        safety_feasible = [ind for ind in all_individuals if ind.objectives[0] >= 0.7]
        
        objectives_array = np.array([ind.objectives for ind in all_individuals])
        avg_safety = np.mean(objectives_array[:, 0])
        avg_efficiency = np.mean(objectives_array[:, 1])
        avg_adaptability = np.mean(objectives_array[:, 2])
        
        # Scalar fitness statistics
        scalar_fitnesses = [getattr(ind, 'scalar_fitness', self.calculate_scalar_fitness(ind)) 
                          for ind in all_individuals]
        avg_scalar_fitness = np.mean(scalar_fitnesses)
        best_scalar_fitness = np.max(scalar_fitnesses)
        
        return {
            "coverage": coverage,
            "coverage_pct": 100.0 * coverage / self.total_cells,
            "total_individuals": total_individuals,
            "safety_feasible_count": len(safety_feasible),
            "avg_safety": avg_safety,
            "avg_efficiency": avg_efficiency,
            "avg_adaptability": avg_adaptability,
            "avg_scalar_fitness": avg_scalar_fitness,
            "best_scalar_fitness": best_scalar_fitness
        }

# =============================================================================
# PURE MAP-ELITES OPTIMIZER
# =============================================================================

class PureMapElitesOptimizer:
    """Pure MAP-Elites optimizer with scalar fitness function"""
    
    def __init__(self, site_config: SiteConfig, facility_types: List[str], 
                 mapelites_config: MapElitesConfig = None):
        self.site_config = site_config
        self.facility_types = facility_types
        self.mapelites_config = mapelites_config or MapElitesConfig()
        self.archive = PureMapElitesArchive(self.mapelites_config.grid_size)
        
        random.seed(site_config.seed)
        np.random.seed(site_config.seed)
    
    def evaluate_solution(self, solution: List[Dict], 
                         entrances: List[Tuple[float, float]]) -> Individual:
        """Evaluate solution with 3-objective fitness and behavioral descriptors"""
        self.archive.evaluations += 1
        
        result = evaluate_individual(solution, entrances, self.site_config, calculate_behaviors=True)
        
        individual = Individual(
            solution=solution,
            entrances=entrances,
            objectives=result['objectives'],
            behaviors=result['behaviors'],
            feasible=result['feasible'],
            violations=result['violations']
        )
        
        # Add scalar fitness for pure MAP-Elites
        individual.scalar_fitness = self.archive.calculate_scalar_fitness(individual)
        
        return individual
    
    def run(self, iterations: int = None, initial_population: int = None) -> Dict:
        """Run Pure MAP-Elites algorithm"""
        if iterations is None:
            iterations = self.mapelites_config.iterations
        if initial_population is None:
            initial_population = self.mapelites_config.initial_population
            
        print(f"\nRunning Pure MAP-Elites:")
        print(f"Facilities: {len(self.facility_types)} ({', '.join(self.facility_types)})")
        print(f"Archive: {self.archive.total_cells:,} cells (one solution per cell)")
        print(f"Evolution: {iterations} iterations, {initial_population} initial population")
        print(f"Fitness: Scalar combination of Safety(50%), Efficiency(30%), Adaptability(20%)")
        
        start_time = time.time()
        
        # Initialize with diverse population targeting different behavioral regions
        print("Creating initial population...")
        behavioral_targets = [
            (0.1, 0.1), (0.1, 0.9), (0.9, 0.1), (0.9, 0.9),  # Corners
            (0.1, 0.5), (0.9, 0.5), (0.5, 0.1), (0.5, 0.9),  # Edges
            (0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75),  # Intermediate
            (0.3, 0.3), (0.3, 0.7), (0.7, 0.3), (0.7, 0.7)   # More intermediate
        ]
        
        for i in range(initial_population):
            entrances = generate_random_entrances(self.site_config, seed=self.site_config.seed + i)
            
            if i < len(behavioral_targets) * 15:  # First portion: targeted layouts
                target_idx = i // 15
                target_spatial, target_functional = behavioral_targets[target_idx % len(behavioral_targets)]
                solution = create_targeted_layout(
                    self.facility_types, self.site_config.boundary_margin,
                    target_spatial, target_functional)
            else:  # Remaining: random layouts
                solution = create_random_layout(self.facility_types, self.site_config.boundary_margin)
            
            # Apply constraint repair less frequently to maintain diversity
            if random.random() > 0.5:  # 50% chance to repair in initial population
                solution = repair_layout_constraints(solution, self.site_config.boundary_margin, 
                                                   entrances, self.site_config)
            
            individual = self.evaluate_solution(solution, entrances)
            
            # Accept all solutions in initial population for diversity
            self.archive.add_individual(individual)
            
            if (i + 1) % (initial_population // 10) == 0:
                print(f"  Generated {i+1}/{initial_population} initial solutions")
        
        initial_stats = self.archive.get_stats()
        print(f"Initial archive: {initial_stats['coverage']} cells, "
              f"Avg fitness: {initial_stats['avg_scalar_fitness']:.3f}")
        
        # Main evolution loop
        print("Starting evolution...")
        report_interval = max(1, iterations // 10)
        
        for iteration in range(iterations):
            # Determine variation strategy based on iteration and archive coverage
            current_coverage = self.archive.get_stats()['coverage_pct']
            
            # Increase diversity-focused operations when coverage is low
            if current_coverage < 20.0:  # Low coverage - emphasize diversity
                diversity_rate = 0.4
                mutation_rate = 0.35
                targeted_rate = 0.25
            elif current_coverage < 50.0:  # Medium coverage
                diversity_rate = 0.25
                mutation_rate = 0.5
                targeted_rate = 0.25
            else:  # High coverage - focus on quality improvement
                diversity_rate = 0.15
                mutation_rate = 0.7
                targeted_rate = 0.15
            
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
                    # Adaptive mutation strength based on coverage
                    mutation_sigma = 0.08 if current_coverage < 25.0 else 0.05
                    offspring_solution = mutate_layout(
                        parent.solution, self.site_config.boundary_margin, 
                        p_mut=0.4, sigma=mutation_sigma)
                    if random.random() < 0.15:
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
            
            # Apply constraint repair with moderate frequency
            if random.random() > 0.3:  # 70% chance to repair
                offspring_solution = repair_layout_constraints(
                    offspring_solution, self.site_config.boundary_margin, 
                    offspring_entrances, self.site_config)
            
            individual = self.evaluate_solution(offspring_solution, offspring_entrances)
            
            # Always try to add - archive will decide based on scalar fitness
            self.archive.add_individual(individual)
            
            if (iteration + 1) % report_interval == 0:
                stats = self.archive.get_stats()
                best_individual = self.archive.get_best_individual()
                best_objectives = best_individual.objectives if best_individual else (0, 0, 0)
                
                print(f"  Iter {iteration+1:>6}: Cells={stats['coverage']:>4} "
                      f"({stats['coverage_pct']:>6.2f}%), "
                      f"AvgFit={stats['avg_scalar_fitness']:.3f}, "
                      f"BestFit={stats['best_scalar_fitness']:.3f}, "
                      f"Best: S={best_objectives[0]:.3f}, E={best_objectives[1]:.3f}, A={best_objectives[2]:.3f}")
        
        runtime = time.time() - start_time
        final_stats = self.archive.get_stats()
        best_individual = self.archive.get_best_individual()
        
        print(f"\nEvolution completed in {runtime:.2f} seconds")
        print(f"Final archive: {final_stats['coverage']:,}/{self.archive.total_cells:,} cells")
        print(f"Average scalar fitness: {final_stats['avg_scalar_fitness']:.3f}")
        print(f"Best scalar fitness: {final_stats['best_scalar_fitness']:.3f}")
        print(f"Safety feasible (≥0.7): {final_stats['safety_feasible_count']}")
        
        return {
            "archive": self.archive,
            "stats": final_stats,
            "best_individual": best_individual,
            "runtime": runtime
        }

# =============================================================================
# PURE MAP-ELITES EVALUATION FUNCTIONS
# =============================================================================

def evaluate_pure_mapelites_performance(archive: PureMapElitesArchive, site_config: SiteConfig) -> Dict:
    """Comprehensive Pure MAP-Elites evaluation metrics"""
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
    
    # Scalar fitness analysis
    scalar_fitnesses = [getattr(ind, 'scalar_fitness', archive.calculate_scalar_fitness(ind)) 
                       for ind in all_individuals]
    
    # Coverage metrics
    coverage_metrics = {
        "cells_filled": coverage,
        "total_cells": archive.total_cells,
        "coverage_percentage": coverage_percentage,
        "cells_per_dimension": archive.grid_size
    }
    
    # Behavioral diversity metrics
    bd1_range = behaviors[:, 0].max() - behaviors[:, 0].min() if len(behaviors) > 0 else 0
    bd2_range = behaviors[:, 1].max() - behaviors[:, 1].min() if len(behaviors) > 0 else 0
    
    diversity_metrics = {
        "behavioral_range_bd1": bd1_range,
        "behavioral_range_bd2": bd2_range,
        "behavioral_coverage": (bd1_range + bd2_range) / 2.0,
        "behavioral_uniformity": coverage_percentage / 100.0  # How well we fill the space
    }
    
    # Quality metrics
    safety_feasible = [ind for ind in all_individuals if ind.objectives[0] >= 0.7]
    
    quality_metrics = {
        "average_safety": np.mean(objectives[:, 0]),
        "average_efficiency": np.mean(objectives[:, 1]),
        "average_adaptability": np.mean(objectives[:, 2]),
        "average_scalar_fitness": np.mean(scalar_fitnesses),
        "best_scalar_fitness": np.max(scalar_fitnesses),
        "max_safety": np.max(objectives[:, 0]),
        "max_efficiency": np.max(objectives[:, 1]),
        "max_adaptability": np.max(objectives[:, 2]),
        "safety_feasible_count": len(safety_feasible),
        "safety_feasible_percentage": 100.0 * len(safety_feasible) / len(all_individuals)
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
            "average_scalar_fitness": np.mean(scalar_fitnesses),
            "best_scalar_fitness": np.max(scalar_fitnesses),
            "pure_mapelites_effectiveness": coverage_percentage * np.mean(scalar_fitnesses) / 100.0
        }
    }