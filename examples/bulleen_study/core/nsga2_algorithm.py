#!/usr/bin/env python3
"""
NSGA-II Algorithm Module
========================

Pure NSGA-II implementation for multi-objective optimization without behavioral descriptors.
Focuses on the three construction site layout objectives: Safety, Efficiency, Adaptability.
"""

import random
from typing import List, Dict, Tuple
import numpy as np

from .config import Individual, SiteConfig, NSGA2Config
from .objectives import evaluate_individual
from .layout_generation import (
    generate_random_entrances,
    create_random_layout,
    mutate_layout,
    crossover_layouts,
)

# =============================================================================
# NSGA-II CORE FUNCTIONS
# =============================================================================

def dominates(ind1: Individual, ind2: Individual) -> bool:
    """Check if ind1 dominates ind2 (higher values are better for all 3 objectives)"""
    better_in_all = all(ind1.objectives[i] >= ind2.objectives[i] for i in range(3))
    strictly_better_in_one = any(ind1.objectives[i] > ind2.objectives[i] for i in range(3))
    return better_in_all and strictly_better_in_one

def non_dominated_sort(population: List[Individual]) -> List[List[Individual]]:
    """Perform non-dominated sorting for 3-objective population"""
    fronts = []
    
    for i, ind1 in enumerate(population):
        ind1.dominance_rank = 0
        dominated_by_count = 0
        dominates_list = []
        
        for j, ind2 in enumerate(population):
            if i != j:
                if dominates(ind1, ind2):
                    dominates_list.append(j)
                elif dominates(ind2, ind1):
                    dominated_by_count += 1
        
        ind1.dominates_list = dominates_list
        ind1.dominated_by_count = dominated_by_count
        
        if dominated_by_count == 0:
            if len(fronts) == 0:
                fronts.append([])
            fronts[0].append(ind1)
    
    current_front = 0
    while current_front < len(fronts) and len(fronts[current_front]) > 0:
        next_front = []
        
        for ind1 in fronts[current_front]:
            for j in ind1.dominates_list:
                ind2 = population[j]
                ind2.dominated_by_count -= 1
                
                if ind2.dominated_by_count == 0:
                    ind2.dominance_rank = current_front + 1
                    next_front.append(ind2)
        
        if len(next_front) > 0:
            fronts.append(next_front)
        current_front += 1
    
    return fronts

def calculate_crowding_distance(front: List[Individual]):
    """Calculate crowding distance for 3-objective front"""
    if len(front) <= 2:
        for ind in front:
            ind.crowding_distance = float('inf')
        return
    
    for ind in front:
        ind.crowding_distance = 0.0
    
    for obj_idx in range(3):
        front.sort(key=lambda x: x.objectives[obj_idx])
        
        front[0].crowding_distance = float('inf')
        front[-1].crowding_distance = float('inf')
        
        obj_range = front[-1].objectives[obj_idx] - front[0].objectives[obj_idx]
        
        if obj_range == 0:
            continue
        
        for i in range(1, len(front) - 1):
            if front[i].crowding_distance != float('inf'):
                distance = (front[i+1].objectives[obj_idx] - front[i-1].objectives[obj_idx]) / obj_range
                front[i].crowding_distance += distance

def select_parents(population: List[Individual], tournament_size: int = 3) -> Individual:
    """Tournament selection for parent selection"""
    tournament = random.sample(population, min(tournament_size, len(population)))
    
    # Select based on dominance rank first, then crowding distance
    best = min(tournament, key=lambda x: (x.dominance_rank, -x.crowding_distance))
    return best

# =============================================================================
# NSGA-II OPTIMIZER CLASS
# =============================================================================

class PureNSGA2Optimizer:
    """Pure NSGA-II optimizer for construction site layout optimization"""
    
    def __init__(self, site_config: SiteConfig, facility_types: List[str], 
                 nsga2_config: NSGA2Config = None):
        self.site_config = site_config
        self.facility_types = facility_types
        self.nsga2_config = nsga2_config or NSGA2Config()
        self.evaluations = 0
        
        random.seed(site_config.seed)
        np.random.seed(site_config.seed)
    
    def evaluate_solution(self, solution: List[Dict], 
                         entrances: List[Tuple[float, float]]) -> Individual:
        """Evaluate solution with 3-objective fitness"""
        self.evaluations += 1
        
        result = evaluate_individual(solution, entrances, self.site_config, calculate_behaviors=False)
        
        return Individual(
            solution=solution,
            entrances=entrances,
            objectives=result['objectives'],
            behaviors=None,  # No behaviors for pure NSGA-II
            feasible=result['feasible'],
            violations=result['violations']
        )
    
    def initialize_population(self, population_size: int) -> List[Individual]:
        """Initialize population with diverse layouts"""
        population = []
        
        print("Initializing diverse population...")
        for i in range(population_size):
            entrances = generate_random_entrances(self.site_config, seed=self.site_config.seed + i)
            solution = create_random_layout(self.facility_types, self.site_config.boundary_margin, self.site_config)
            individual = self.evaluate_solution(solution, entrances)
            population.append(individual)
            
            if (i + 1) % (population_size // 10) == 0:
                print(f"  Generated {i+1}/{population_size} initial solutions")
        
        return population
    
    def create_offspring(self, population: List[Individual], offspring_size: int) -> List[Individual]:
        """Create offspring through selection, crossover, and mutation"""
        offspring = []
        
        while len(offspring) < offspring_size:
            if random.random() < self.nsga2_config.crossover_rate:  # Crossover
                parent1 = select_parents(population, self.nsga2_config.tournament_size)
                parent2 = select_parents(population, self.nsga2_config.tournament_size)
                
                child1_solution, child2_solution = crossover_layouts(
                    parent1.solution, parent2.solution
                )
                
                # Choose one child and mutate it
                child_solution = random.choice([child1_solution, child2_solution])
                child_solution = mutate_layout(child_solution, self.site_config.boundary_margin, config=self.site_config)
                
                # Use parent's entrances or generate new ones
                if random.random() < 0.8:
                    child_entrances = random.choice([parent1.entrances, parent2.entrances])
                else:
                    child_entrances = generate_random_entrances(
                        self.site_config, seed=self.site_config.seed + len(offspring))
                
            else:  # Mutation only
                parent = select_parents(population, self.nsga2_config.tournament_size)
                child_solution = mutate_layout(parent.solution, self.site_config.boundary_margin, 
                                             p_mut=0.6, config=self.site_config)
                
                if random.random() < 0.2:
                    child_entrances = generate_random_entrances(
                        self.site_config, seed=self.site_config.seed + len(offspring))
                else:
                    child_entrances = parent.entrances[:]
            
            # Evaluate offspring
            child = self.evaluate_solution(child_solution, child_entrances)
            offspring.append(child)
        
        return offspring[:offspring_size]
    
    def environmental_selection(self, combined_population: List[Individual], 
                              population_size: int) -> List[Individual]:
        """Select next generation using NSGA-II environmental selection"""
        
        # Non-dominated sorting
        fronts = non_dominated_sort(combined_population)
        
        # Calculate crowding distance for each front
        for front in fronts:
            calculate_crowding_distance(front)
        
        # Select individuals for next generation
        next_generation = []
        front_idx = 0
        
        while front_idx < len(fronts) and len(next_generation) + len(fronts[front_idx]) <= population_size:
            next_generation.extend(fronts[front_idx])
            front_idx += 1
        
        # If we need to partially include the next front
        if front_idx < len(fronts) and len(next_generation) < population_size:
            remaining_slots = population_size - len(next_generation)
            last_front = fronts[front_idx]
            
            # Sort by crowding distance (descending)
            last_front.sort(key=lambda x: -x.crowding_distance if x.crowding_distance != float('inf') else -1e6)
            next_generation.extend(last_front[:remaining_slots])
        
        return next_generation
    
    def run(self, population_size: int = None, generations: int = None) -> Dict:
        """Run pure NSGA-II optimization with convergence tracking"""
        if population_size is None:
            population_size = self.nsga2_config.population_size
        if generations is None:
            generations = self.nsga2_config.generations
            
        print(f"\nRunning Pure NSGA-II Optimization:")
        print(f"Facilities: {len(self.facility_types)} ({', '.join(self.facility_types)})")
        print(f"Population: {population_size}")
        print(f"Generations: {generations}")
        print(f"Objectives: Safety, Efficiency, Adaptability")
        
        import time
        start_time = time.time()
        
        # Initialize population
        population = self.initialize_population(population_size)
        
        # Track convergence history
        convergence_history = []
        
        # Evolution loop
        print("Starting evolution...")
        report_interval = max(1, generations // 10)
        
        for generation in range(generations):
            # Create offspring
            offspring = self.create_offspring(population, population_size)
            
            # Combine parent and offspring populations
            combined_population = population + offspring
            
            # Environmental selection
            population = self.environmental_selection(combined_population, population_size)
            
            # Track convergence metrics
            population_objectives = np.array([ind.objectives for ind in population])
            generation_metrics = {
                "generation": generation,
                "avg_safety": float(np.mean(population_objectives[:, 0])),
                "avg_efficiency": float(np.mean(population_objectives[:, 1])),
                "avg_adaptability": float(np.mean(population_objectives[:, 2])),
                "feasible_count": sum(1 for ind in population if ind.feasible)
            }
            convergence_history.append(generation_metrics)
            
            # Progress reporting
            if (generation + 1) % report_interval == 0:
                fronts = non_dominated_sort(population)
                pareto_front = fronts[0] if fronts else []
                
                if pareto_front:
                    avg_objectives = np.mean([ind.objectives for ind in pareto_front], axis=0)
                    feasible_count = sum(1 for ind in population if ind.feasible)
                    
                    print(f"  Gen {generation+1:>4}: Pareto={len(pareto_front):>3}, "
                          f"Feasible={feasible_count:>3}/{population_size}, "
                          f"Avg: S={avg_objectives[0]:.3f}, E={avg_objectives[1]:.3f}, A={avg_objectives[2]:.3f}")
        
        runtime = time.time() - start_time
        
        # Final analysis
        final_fronts = non_dominated_sort(population)
        pareto_front = final_fronts[0] if final_fronts else []
        
        print(f"\nNSGA-II Evolution completed in {runtime:.2f} seconds")
        print(f"Final Pareto front: {len(pareto_front)} solutions")
        print(f"Total evaluations: {self.evaluations:,}")
        
        return {
            "population": population,
            "pareto_front": pareto_front,
            "fronts": final_fronts,
            "evaluations": self.evaluations,
            "runtime": runtime,
            "convergence_history": convergence_history
        }

# =============================================================================
# NSGA-II UTILITY FUNCTIONS
# =============================================================================

def calculate_nsga2_metrics(results: Dict, site_config: SiteConfig) -> Dict:
    """Calculate comprehensive NSGA-II performance metrics"""
    population = results["population"]
    pareto_front = results["pareto_front"]
    
    metrics = {}
    
    if not pareto_front:
        return {
            'hypervolume': 0.0,
            'spread': 0.0,
            'spacing_metric': 0.0,
            'pareto_coverage': 0.0,
            'convergence_rate': 0.0
        }
    
    pareto_objectives = np.array([ind.objectives for ind in pareto_front])
    
    # 1. Hypervolume (approximation using reference point)
    reference_point = np.array([0.0, 0.0, 0.0])  # Worst case for all objectives
    hypervolume = calculate_hypervolume_approximation(pareto_objectives, reference_point)
    metrics['hypervolume'] = hypervolume
    
    # 2. Spread metric (extent of Pareto front)
    if len(pareto_objectives) > 1:
        spread = calculate_spread_metric(pareto_objectives)
    else:
        spread = 0.0
    metrics['spread'] = spread
    
    # 3. Spacing metric (distribution uniformity)
    if len(pareto_objectives) > 2:
        spacing = calculate_spacing_metric(pareto_objectives)
    else:
        spacing = 1.0
    metrics['spacing_metric'] = spacing
    
    # 4. Pareto coverage (percentage of population in first front)
    metrics['pareto_coverage'] = len(pareto_front) / len(population)
    
    # 5. Solution quality metrics
    feasible_pareto = [ind for ind in pareto_front if ind.feasible]
    metrics['feasible_pareto_ratio'] = len(feasible_pareto) / len(pareto_front)
    
    if feasible_pareto:
        feasible_obj = np.array([ind.objectives for ind in feasible_pareto])
        metrics['avg_feasible_quality'] = np.mean(np.sum(feasible_obj, axis=1))
    else:
        metrics['avg_feasible_quality'] = 0.0
    
    return metrics

def calculate_hypervolume_approximation(pareto_objectives: np.ndarray, reference_point: np.ndarray) -> float:
    """Calculate approximate hypervolume using Monte Carlo sampling"""
    if len(pareto_objectives) == 0:
        return 0.0
    
    # Define bounds for sampling
    max_bounds = np.max(pareto_objectives, axis=0)
    min_bounds = reference_point
    
    # Monte Carlo sampling
    n_samples = 10000
    count_dominated = 0
    
    for _ in range(n_samples):
        # Random point in objective space
        random_point = np.random.uniform(min_bounds, max_bounds)
        
        # Check if any Pareto solution dominates this point
        dominated = False
        for pareto_point in pareto_objectives:
            if np.all(pareto_point >= random_point):
                dominated = True
                break
        
        if dominated:
            count_dominated += 1
    
    # Calculate hypervolume as proportion of dominated space
    total_volume = np.prod(max_bounds - min_bounds)
    hypervolume = (count_dominated / n_samples) * total_volume
    
    # Normalize by maximum possible volume
    max_possible = np.prod(np.ones(3) - reference_point)
    return hypervolume / max_possible

def calculate_spread_metric(pareto_objectives: np.ndarray) -> float:
    """Calculate spread metric (extent of Pareto front coverage)"""
    if len(pareto_objectives) <= 1:
        return 0.0
    
    # Calculate range in each objective
    ranges = np.max(pareto_objectives, axis=0) - np.min(pareto_objectives, axis=0)
    
    # Normalize by maximum possible range (0 to 1 for each objective)
    max_range = 1.0
    normalized_ranges = ranges / max_range
    
    # Spread is the average normalized range
    spread = np.mean(normalized_ranges)
    return min(spread, 1.0)

def calculate_spacing_metric(pareto_objectives: np.ndarray) -> float:
    """Calculate spacing metric (uniformity of distribution)"""
    if len(pareto_objectives) <= 2:
        return 1.0
    
    # Calculate distances between consecutive points (sorted)
    distances = []
    for i in range(len(pareto_objectives)):
        min_dist = float('inf')
        for j in range(len(pareto_objectives)):
            if i != j:
                dist = np.linalg.norm(pareto_objectives[i] - pareto_objectives[j])
                min_dist = min(min_dist, dist)
        distances.append(min_dist)
    
    # Calculate coefficient of variation
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    
    if mean_dist == 0:
        return 1.0
    
    # Spacing metric (lower variation = better spacing)
    spacing_metric = 1.0 - (std_dist / mean_dist)
    return max(0.0, min(1.0, spacing_metric))
