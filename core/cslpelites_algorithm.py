#!/usr/bin/env python3
"""
CEXO Algorithm Module
=====================

CEXO combines three components for construction site layout optimisation:

1. NSGA-II style multi-objective selection through Pareto fronts and crowding
   distance.
2. MAP-Elites quality-diversity archiving across a 2D behavioural space.
3. Optional autoencoder-learned behavioural descriptors. In learned mode,
   CEXO first builds an unarchived training pool, trains the autoencoder, and
   only then creates the MAP-Elites archive from learned latent descriptors.
"""

import os
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .behavioral_descriptors import create_descriptor_manager
from .config import AutoencoderConfig, Individual, MapElitesConfig, SiteConfig
from .layout_autoencoder import (
    AutoencoderTrainer,
    LearnedBehavioralDescriptors,
    create_autoencoder,
    set_random_seeds,
)
from .layout_generation import (
    create_random_layout,
    create_targeted_layout,
    crossover_layouts,
    generate_random_entrances,
    mutate_layout,
    mutate_toward_behavioral_diversity,
    repair_layout_constraints,
)
from .nsga2_algorithm import calculate_crowding_distance, dominates
from .objectives import evaluate_individual


class ParetoFront:
    """Bounded per-cell Pareto set for the three CSLP objectives."""

    def __init__(self, max_size: int = 12):
        self.max_size = max_size
        self.individuals: List[Individual] = []

    def add_individual(self, individual: Individual) -> bool:
        """Add an individual if it is not dominated in this cell."""
        self.individuals = [
            existing
            for existing in self.individuals
            if not dominates(individual, existing)
        ]

        if any(dominates(existing, individual) for existing in self.individuals):
            return False

        self.individuals.append(individual)
        if len(self.individuals) > self.max_size:
            self._trim_front()
        return True

    def _trim_front(self):
        """Trim front with NSGA-II crowding distance."""
        calculate_crowding_distance(self.individuals)
        self.individuals.sort(
            key=lambda ind: (
                -ind.crowding_distance
                if ind.crowding_distance != float("inf")
                else -1e6
            )
        )
        self.individuals = self.individuals[: self.max_size]

    def get_random_individual(self) -> Optional[Individual]:
        return random.choice(self.individuals) if self.individuals else None

    def get_best_individual(self, archive) -> Optional[Individual]:
        if not self.individuals:
            return None
        return max(self.individuals, key=archive.calculate_scalar_fitness)

    def size(self) -> int:
        return len(self.individuals)


class CEXOArchive:
    """MAP-Elites grid whose cells each contain a bounded Pareto front."""

    def __init__(self, grid_size: Tuple[int, int] = (20, 20), pareto_size: int = 12):
        self.grid_size = grid_size
        self.pareto_size = pareto_size
        self.total_cells = grid_size[0] * grid_size[1]
        self.archive: Dict[Tuple[int, int], ParetoFront] = {}
        self.evaluations = 0

    def get_cell_coords(self, behaviors: Tuple[float, float]) -> Tuple[int, int]:
        b1, b2 = behaviors
        i = int(np.clip(b1 * self.grid_size[0], 0, self.grid_size[0] - 1))
        j = int(np.clip(b2 * self.grid_size[1], 0, self.grid_size[1] - 1))
        return (i, j)

    def add_individual(self, individual: Individual) -> bool:
        coords = self.get_cell_coords(individual.behaviors)
        if coords not in self.archive:
            self.archive[coords] = ParetoFront(self.pareto_size)
        return self.archive[coords].add_individual(individual)

    def calculate_scalar_fitness(self, individual: Individual) -> float:
        """Scalar quality proxy used for reports and visualisation only."""
        safety, efficiency, adaptability = individual.objectives
        score = 0.5 * safety + 0.3 * efficiency + 0.2 * adaptability
        if individual.feasible:
            score += 0.1
        else:
            score *= 0.7
        return float(np.clip(score, 0.0, 1.0))

    def get_random_individual(self) -> Optional[Individual]:
        if not self.archive:
            return None
        coords = random.choice(list(self.archive.keys()))
        return self.archive[coords].get_random_individual()

    def get_best_individual(self) -> Optional[Individual]:
        all_individuals = self.get_all_individuals()
        if not all_individuals:
            return None
        return max(all_individuals, key=self.calculate_scalar_fitness)

    def get_best_for_cell(self, coords: Tuple[int, int]) -> Optional[Individual]:
        front = self.archive.get(coords)
        if front is None:
            return None
        return front.get_best_individual(self)

    def get_all_individuals(self) -> List[Individual]:
        individuals = []
        for pareto_front in self.archive.values():
            individuals.extend(pareto_front.individuals)
        return individuals

    def get_stats(self) -> Dict:
        if not self.archive:
            return {
                "coverage": 0,
                "coverage_pct": 0.0,
                "total_individuals": 0,
                "safety_feasible_count": 0,
                "strict_feasible_count": 0,
                "avg_safety": 0.0,
                "avg_efficiency": 0.0,
                "avg_adaptability": 0.0,
                "avg_scalar_fitness": 0.0,
                "best_scalar_fitness": 0.0,
                "avg_pareto_size": 0.0,
            }

        all_individuals = self.get_all_individuals()
        objectives_array = np.array([ind.objectives for ind in all_individuals])
        scalar_fitnesses = [self.calculate_scalar_fitness(ind) for ind in all_individuals]
        safety_feasible = [ind for ind in all_individuals if ind.objectives[0] >= 0.7]
        strict_feasible = [ind for ind in all_individuals if ind.feasible]
        pareto_sizes = [front.size() for front in self.archive.values()]

        return {
            "coverage": len(self.archive),
            "coverage_pct": 100.0 * len(self.archive) / self.total_cells,
            "total_individuals": len(all_individuals),
            "safety_feasible_count": len(safety_feasible),
            "strict_feasible_count": len(strict_feasible),
            "avg_safety": float(np.mean(objectives_array[:, 0])),
            "avg_efficiency": float(np.mean(objectives_array[:, 1])),
            "avg_adaptability": float(np.mean(objectives_array[:, 2])),
            "avg_scalar_fitness": float(np.mean(scalar_fitnesses)),
            "best_scalar_fitness": float(np.max(scalar_fitnesses)),
            "avg_pareto_size": float(np.mean(pareto_sizes)),
        }


class CEXOOptimizer:
    """NSGA-II + MAP-Elites + optional autoencoder-learned descriptors."""

    def __init__(
        self,
        facility_types: List[str],
        site_config: SiteConfig,
        mapelites_config: MapElitesConfig,
        autoencoder_config: AutoencoderConfig = None,
    ):
        self.facility_types = facility_types
        self.site_config = site_config
        self.mapelites_config = mapelites_config
        self.autoencoder_config = autoencoder_config or AutoencoderConfig(
            use_learned_descriptors=False
        )
        self.seed = (
            self.autoencoder_config.seed
            if self.autoencoder_config.seed is not None
            else site_config.seed
        )

        if self.seed is not None:
            set_random_seeds(self.seed)

        self.archive = CEXOArchive(
            self.mapelites_config.grid_size,
            self.site_config.pareto_size,
        )
        self.bd_manager = create_descriptor_manager(use_learned=False)
        self.autoencoder = None
        self.autoencoder_trainer = None
        self.learned_bd_extractor = None
        self.autoencoder_trained = False
        self.training_history = []
        self.last_training_iteration = 0

        if self.autoencoder_config.use_learned_descriptors:
            self._initialize_autoencoder()

    def _initialize_autoencoder(self):
        print("Initializing autoencoder...")
        self.autoencoder = create_autoencoder(
            latent_dim=self.autoencoder_config.latent_dim,
            encoder_hidden=self.autoencoder_config.encoder_hidden,
            decoder_hidden=self.autoencoder_config.decoder_hidden,
        )
        self.autoencoder_trainer = AutoencoderTrainer(
            self.autoencoder,
            learning_rate=self.autoencoder_config.learning_rate,
            seed=self.seed,
        )

        if self.autoencoder_config.load_model_path and os.path.exists(
            self.autoencoder_config.load_model_path
        ):
            print(f"Loading pretrained model from {self.autoencoder_config.load_model_path}")
            self.autoencoder_trainer.load_model(self.autoencoder_config.load_model_path)
            self.autoencoder_trained = True
            self.learned_bd_extractor = LearnedBehavioralDescriptors(self.autoencoder)
            self.bd_manager.switch_mode("learned", self.learned_bd_extractor)

    def _should_train_autoencoder(self, iteration: int) -> bool:
        if not self.autoencoder_config.use_learned_descriptors:
            return False
        archive_size = len(self.archive.get_all_individuals())
        if archive_size < self.autoencoder_config.min_samples_for_training:
            return False
        if not self.autoencoder_trained:
            return iteration >= self.autoencoder_config.pretrain_iterations
        return iteration - self.last_training_iteration >= self.autoencoder_config.training_frequency

    def _train_autoencoder_on_layouts(self, layouts, iteration: int):
        print(f"\n[Iteration {iteration}] Training autoencoder...")
        training_seed = self.seed + iteration if self.seed is not None else None
        training_stats = self.autoencoder_trainer.train(
            layouts,
            epochs=self.autoencoder_config.training_epochs,
            batch_size=self.autoencoder_config.batch_size,
            verbose=False,
            training_seed=training_seed,
        )
        print(f"  Training complete: Loss = {training_stats['final_loss']:.6f}")

        self.training_history.append(
            {
                "iteration": iteration,
                "loss": training_stats["final_loss"],
                "num_samples": len(layouts),
            }
        )
        self.last_training_iteration = iteration
        self.learned_bd_extractor = LearnedBehavioralDescriptors(self.autoencoder)

        if not self.autoencoder_trained:
            print("  Switching to learned behavioral descriptors")
            self.bd_manager.switch_mode("learned", self.learned_bd_extractor)
            self.autoencoder_trained = True
            self._reevaluate_archive_behaviors()

        if self.autoencoder_config.save_model_path:
            save_path = f"{self.autoencoder_config.save_model_path}_iter{iteration}.pt"
            self.autoencoder_trainer.save_model(save_path)
            print(f"  Model saved to {save_path}")

    def _train_autoencoder(self, iteration: int):
        all_individuals = self.archive.get_all_individuals()
        layouts = [(ind.solution, ind.entrances) for ind in all_individuals]
        self._train_autoencoder_on_layouts(layouts, iteration)

    def _reevaluate_archive_behaviors(self):
        old_individuals = self.archive.get_all_individuals()
        self.archive.archive.clear()
        for individual in old_individuals:
            individual.behaviors = self.bd_manager.get_descriptors(
                individual.solution,
                individual.entrances,
            )
            self.archive.add_individual(individual)
        print(f"  Archive re-indexed: {len(self.archive.archive)} occupied cells")

    def evaluate_solution(
        self,
        solution: List[Dict],
        entrances: List[Tuple[float, float]],
    ) -> Individual:
        self.archive.evaluations += 1
        eval_result = evaluate_individual(solution, entrances, self.site_config)
        behaviors = self.bd_manager.get_descriptors(solution, entrances)
        return Individual(
            solution=solution,
            entrances=entrances,
            objectives=eval_result["objectives"],
            behaviors=behaviors,
            feasible=eval_result["feasible"],
            violations=eval_result["violations"],
        )

    def evaluate_objectives_only(
        self,
        solution: List[Dict],
        entrances: List[Tuple[float, float]],
    ) -> Individual:
        """Evaluate objectives without assigning any behavioural descriptor."""
        self.archive.evaluations += 1
        eval_result = evaluate_individual(solution, entrances, self.site_config)
        return Individual(
            solution=solution,
            entrances=entrances,
            objectives=eval_result["objectives"],
            behaviors=None,
            feasible=eval_result["feasible"],
            violations=eval_result["violations"],
        )

    def assign_learned_behavior(self, individual: Individual) -> Individual:
        """Attach learned latent descriptors to an evaluated individual."""
        individual.behaviors = self.bd_manager.get_descriptors(
            individual.solution,
            individual.entrances,
        )
        return individual

    def _generate_unbiased_solution(self, index: int):
        """Generate a layout without hand-crafted descriptor targets."""
        entrances = generate_random_entrances(
            self.site_config,
            seed=self.site_config.seed + index,
        )
        solution = create_random_layout(
            self.facility_types,
            self.site_config.boundary_margin,
            self.site_config,
        )
        if random.random() > 0.4:
            solution = repair_layout_constraints(
                solution,
                self.site_config.boundary_margin,
                entrances,
                self.site_config,
            )
        return solution, entrances

    def _generate_initial_solution(self, index: int, behavioral_targets):
        entrances = generate_random_entrances(
            self.site_config,
            seed=self.site_config.seed + index,
        )
        if index < len(behavioral_targets) * 15:
            target_idx = index // 15
            target_spatial, target_functional = behavioral_targets[
                target_idx % len(behavioral_targets)
            ]
            solution = create_targeted_layout(
                self.facility_types,
                self.site_config.boundary_margin,
                target_spatial,
                target_functional,
                self.site_config,
            )
        else:
            solution = create_random_layout(
                self.facility_types,
                self.site_config.boundary_margin,
                self.site_config,
            )
        if random.random() > 0.5:
            solution = repair_layout_constraints(
                solution,
                self.site_config.boundary_margin,
                entrances,
                self.site_config,
            )
        return solution, entrances

    def _create_unbiased_offspring(self, iteration: int):
        """Create offspring without hand-crafted descriptor-directed operators."""
        variation_type = random.random()
        if variation_type < 0.65:
            parent = self.archive.get_random_individual()
            if parent is not None:
                solution = mutate_layout(
                    parent.solution,
                    self.site_config.boundary_margin,
                    p_mut=0.45,
                    sigma=0.06,
                    config=self.site_config,
                )
                entrances = parent.entrances[:]
                if random.random() < 0.15:
                    entrances = generate_random_entrances(
                        self.site_config,
                        seed=self.site_config.seed + iteration + 2000,
                    )
            else:
                solution, entrances = self._generate_unbiased_solution(iteration + 2000)
        elif variation_type < 0.85:
            parent1 = self.archive.get_random_individual()
            parent2 = self.archive.get_random_individual()
            if parent1 is not None and parent2 is not None:
                child1, child2 = crossover_layouts(parent1.solution, parent2.solution)
                solution = random.choice([child1, child2])
                entrances = random.choice([parent1.entrances, parent2.entrances])[:]
            else:
                solution, entrances = self._generate_unbiased_solution(iteration + 3000)
        else:
            solution, entrances = self._generate_unbiased_solution(iteration + 4000)

        if random.random() > 0.3:
            solution = repair_layout_constraints(
                solution,
                self.site_config.boundary_margin,
                entrances,
                self.site_config,
            )
        return solution, entrances

    def _bootstrap_learned_archive(self, initial_population: int) -> List[Individual]:
        """Train the autoencoder before any archive insertion in learned mode."""
        print("\nCreating unbiased autoencoder training pool...")
        pool = []
        for i in range(initial_population):
            solution, entrances = self._generate_unbiased_solution(i)
            pool.append(self.evaluate_objectives_only(solution, entrances))
            if (i + 1) % max(1, initial_population // 10) == 0:
                print(f"  Generated {i + 1}/{initial_population} training layouts")

        layouts = [(ind.solution, ind.entrances) for ind in pool]
        if len(layouts) < self.autoencoder_config.min_samples_for_training:
            raise ValueError(
                "Not enough layouts to train learned descriptors before archive creation: "
                f"{len(layouts)} < {self.autoencoder_config.min_samples_for_training}"
            )

        self._train_autoencoder_on_layouts(layouts, iteration=0)
        if not self.autoencoder_trained:
            print("  Switching to learned behavioral descriptors before archive creation")
            self.bd_manager.switch_mode("learned", self.learned_bd_extractor)
            self.autoencoder_trained = True

        print("Creating learned-descriptor archive from training pool...")
        for individual in pool:
            self.archive.add_individual(self.assign_learned_behavior(individual))

        return pool

    def _create_offspring(self, iteration: int, current_coverage: float):
        if current_coverage < 20.0:
            diversity_rate, mutation_rate, targeted_rate = 0.4, 0.35, 0.25
        elif current_coverage < 50.0:
            diversity_rate, mutation_rate, targeted_rate = 0.25, 0.5, 0.25
        else:
            diversity_rate, mutation_rate, targeted_rate = 0.15, 0.7, 0.15

        variation_type = random.random()
        if variation_type < diversity_rate:
            parent = self.archive.get_random_individual()
            if parent is not None:
                solution = mutate_toward_behavioral_diversity(
                    parent.solution,
                    self.site_config.boundary_margin,
                    parent.behaviors[0],
                    parent.behaviors[1],
                    self.site_config,
                )
                entrances = parent.entrances[:]
            else:
                solution = create_random_layout(
                    self.facility_types,
                    self.site_config.boundary_margin,
                    self.site_config,
                )
                entrances = generate_random_entrances(
                    self.site_config,
                    seed=self.site_config.seed + iteration,
                )
        elif variation_type < diversity_rate + targeted_rate:
            target_spatial = random.choice([0.1, 0.3, 0.7, 0.9])
            target_functional = random.choice([0.1, 0.3, 0.7, 0.9])
            solution = create_targeted_layout(
                self.facility_types,
                self.site_config.boundary_margin,
                target_spatial,
                target_functional,
                self.site_config,
            )
            entrances = generate_random_entrances(
                self.site_config,
                seed=self.site_config.seed + iteration + 1000,
            )
        elif variation_type < diversity_rate + targeted_rate + mutation_rate:
            parent = self.archive.get_random_individual()
            if parent is not None:
                mutation_sigma = 0.08 if current_coverage < 25.0 else 0.05
                solution = mutate_layout(
                    parent.solution,
                    self.site_config.boundary_margin,
                    p_mut=0.4,
                    sigma=mutation_sigma,
                    config=self.site_config,
                )
                entrances = parent.entrances[:]
            else:
                solution = create_random_layout(
                    self.facility_types,
                    self.site_config.boundary_margin,
                    self.site_config,
                )
                entrances = generate_random_entrances(
                    self.site_config,
                    seed=self.site_config.seed + iteration,
                )
        else:
            parent1 = self.archive.get_random_individual()
            parent2 = self.archive.get_random_individual()
            if parent1 is not None and parent2 is not None:
                child1, child2 = crossover_layouts(parent1.solution, parent2.solution)
                solution = random.choice([child1, child2])
                entrances = random.choice([parent1.entrances, parent2.entrances])[:]
            else:
                solution = create_random_layout(
                    self.facility_types,
                    self.site_config.boundary_margin,
                    self.site_config,
                )
                entrances = generate_random_entrances(
                    self.site_config,
                    seed=self.site_config.seed + iteration,
                )

        if random.random() > 0.3:
            solution = repair_layout_constraints(
                solution,
                self.site_config.boundary_margin,
                entrances,
                self.site_config,
            )
        return solution, entrances

    def run(self, iterations: int = None, initial_population: int = None) -> Dict:
        if iterations is None:
            iterations = self.mapelites_config.iterations
        if initial_population is None:
            initial_population = self.mapelites_config.initial_population

        mode = (
            "CEXO (NSGA-II + MAP-Elites + Autoencoder)"
            if self.autoencoder_config.use_learned_descriptors
            else "MAP-Elites + NSGA-II Pareto Archive"
        )
        print(f"\nRunning {mode}:")
        print(f"Facilities: {len(self.facility_types)} ({', '.join(self.facility_types)})")
        print(f"Archive: {self.archive.total_cells:,} cells x {self.site_config.pareto_size} Pareto size")
        print(f"Iterations: {iterations}, Initial: {initial_population}")
        if self.autoencoder_config.use_learned_descriptors:
            print("Descriptor bootstrap: learned-only, no hand-crafted archive phase")
            print(f"Training frequency: {self.autoencoder_config.training_frequency}")

        start_time = time.time()
        behavioral_targets = [
            (0.1, 0.1), (0.1, 0.9), (0.9, 0.1), (0.9, 0.9),
            (0.1, 0.5), (0.9, 0.5), (0.5, 0.1), (0.5, 0.9),
            (0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75),
            (0.3, 0.3), (0.3, 0.7), (0.7, 0.3), (0.7, 0.7),
        ]

        if self.autoencoder_config.use_learned_descriptors:
            self._bootstrap_learned_archive(initial_population)
        else:
            print("\nCreating initial population...")
            for i in range(initial_population):
                solution, entrances = self._generate_initial_solution(i, behavioral_targets)
                individual = self.evaluate_solution(solution, entrances)
                self.archive.add_individual(individual)
                if (i + 1) % max(1, initial_population // 10) == 0:
                    print(f"  Generated {i + 1}/{initial_population} initial solutions")

        stats = self.archive.get_stats()
        print(
            f"Initial archive: {stats['coverage']} cells, "
            f"{stats['total_individuals']} individuals"
        )

        print("\nStarting evolution...")
        report_interval = max(1, iterations // 20)
        for iteration in range(iterations):
            if self._should_train_autoencoder(iteration):
                self._train_autoencoder(iteration)

            current_coverage = self.archive.get_stats()["coverage_pct"]
            if self.autoencoder_config.use_learned_descriptors:
                solution, entrances = self._create_unbiased_offspring(iteration)
                individual = self.evaluate_objectives_only(solution, entrances)
                self.assign_learned_behavior(individual)
            else:
                solution, entrances = self._create_offspring(iteration, current_coverage)
                individual = self.evaluate_solution(solution, entrances)
            self.archive.add_individual(individual)

            if (iteration + 1) % report_interval == 0:
                stats = self.archive.get_stats()
                best = self.archive.get_best_individual()
                best_objectives = best.objectives if best else (0.0, 0.0, 0.0)
                print(
                    f"  Iter {iteration + 1:>6}: Cells={stats['coverage']:>4} "
                    f"({stats['coverage_pct']:>6.2f}%), "
                    f"Indiv={stats['total_individuals']:>5}, "
                    f"Fit={stats['avg_scalar_fitness']:.3f}/{stats['best_scalar_fitness']:.3f}, "
                    f"Best: S={best_objectives[0]:.3f}, E={best_objectives[1]:.3f}, A={best_objectives[2]:.3f}, "
                    f"BD={self.bd_manager.get_mode()}"
                )

        runtime = time.time() - start_time
        final_stats = self.archive.get_stats()
        best_individual = self.archive.get_best_individual()
        print(f"\nEvolution completed in {runtime:.2f} seconds")
        print(f"Final archive: {final_stats['coverage']:,}/{self.archive.total_cells:,} cells")
        print(f"Total individuals: {final_stats['total_individuals']:,}")
        print(f"Best scalar fitness: {final_stats['best_scalar_fitness']:.3f}")
        print(f"Final BD mode: {self.bd_manager.get_mode()}")

        return {
            "archive": self.archive,
            "stats": final_stats,
            "best_individual": best_individual,
            "runtime": runtime,
            "bd_mode": self.bd_manager.get_mode(),
            "autoencoder": self.autoencoder,
            "training_history": self.training_history,
            "autoencoder_trained": self.autoencoder_trained,
        }


# Backwards-compatible names for older scripts/notebooks.
MapElitesArchive = CEXOArchive
MapElitesNSGA2Optimizer = CEXOOptimizer


def evaluate_mapelites_performance(archive: CEXOArchive, site_config: SiteConfig) -> Dict:
    """Summarise CEXO/MAP-Elites archive performance."""
    all_individuals = archive.get_all_individuals()
    if not all_individuals:
        return {"error": "Empty archive"}

    behaviors = np.array([ind.behaviors for ind in all_individuals])
    objectives = np.array([ind.objectives for ind in all_individuals])
    stats = archive.get_stats()
    bd1_range = behaviors[:, 0].max() - behaviors[:, 0].min()
    bd2_range = behaviors[:, 1].max() - behaviors[:, 1].min()

    return {
        "coverage_metrics": {
            "cells_filled": stats["coverage"],
            "total_cells": archive.total_cells,
            "coverage_percentage": stats["coverage_pct"],
            "average_individuals_per_cell": stats["avg_pareto_size"],
        },
        "diversity_metrics": {
            "behavioral_range_bd1": float(bd1_range),
            "behavioral_range_bd2": float(bd2_range),
            "behavioral_diversity_score": float((bd1_range + bd2_range) / 2.0),
        },
        "quality_metrics": {
            "average_safety": float(np.mean(objectives[:, 0])),
            "average_efficiency": float(np.mean(objectives[:, 1])),
            "average_adaptability": float(np.mean(objectives[:, 2])),
            "best_scalar_fitness": stats["best_scalar_fitness"],
            "pareto_front_sizes": [front.size() for front in archive.archive.values()],
        },
        "summary": stats,
    }
