#!/usr/bin/env python3
"""
Configuration Module for Construction Site Layout Optimization
============================================================

Defines all facility specifications, configurations, and data structures
used by both MAP-Elites and NSGA-II algorithms.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple

# =============================================================================
# FACILITY SPECIFICATIONS
# =============================================================================

FACILITY_SPECS = {
    "core":      {"w": 0.20, "d": 0.20, "category": "operational", "noise_level": 0.8},
    "crane":     {"w": 0.10, "d": 0.10, "category": "equipment", "noise_level": 0.95, "danger_radius": 0.25},
    "storage":   {"w": 0.20, "d": 0.15, "category": "operational", "noise_level": 0.6},
    "office":    {"w": 0.15, "d": 0.12, "category": "worker", "noise_level": 0.1},
    "rest_area": {"w": 0.18, "d": 0.10, "category": "worker", "noise_level": 0.1}
}

FACILITY_COLORS = {
    "core":      "#4e79a7",  # Blue
    "crane":     "#e15759",  # Red
    "storage":   "#59a14f",  # Green
    "office":    "#af7aa1",  # Purple
    "rest_area": "#ff9d9a"   # Pink
}

FACILITY_TYPES = list(FACILITY_SPECS.keys())

# =============================================================================
# CONFIGURATION CLASSES
# =============================================================================

@dataclass
class SiteConfig:
    """Configuration for construction site layout generation"""
    facility_count: int = 5
    boundary_margin: float = 0.08
    seed: int = 42
    min_entrances: int = 1
    max_entrances: int = 3
    pareto_size: int = 12
    entrance_clearance: float = 0.15
    crane_safety_distance: float = 0.30

@dataclass 
class Individual:
    """Individual solution with 3-objective evaluation"""
    solution: List[Dict]
    entrances: List[Tuple[float, float]]
    objectives: Tuple[float, float, float]  # (safety, efficiency, adaptability)
    behaviors: Tuple[float, float] = None  # (spatial_org, functional_int) - None for pure NSGA-II
    feasible: bool = True
    violations: List[str] = None
    dominance_rank: int = 0
    crowding_distance: float = 0.0
    
    def __post_init__(self):
        if self.violations is None:
            self.violations = []

@dataclass
class MapElitesConfig:
    """Configuration specific to MAP-Elites algorithm"""
    grid_size: Tuple[int, int] = (20, 20)
    iterations: int = 15000
    initial_population: int = 500
    
@dataclass
class NSGA2Config:
    """Configuration specific to NSGA-II algorithm"""
    population_size: int = 200
    generations: int = 300
    tournament_size: int = 3
    crossover_rate: float = 0.8
    mutation_rate: float = 0.4

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def generate_facility_mix(count: int, seed: int = None) -> List[str]:
    """Generate smart mix of facilities"""
    import random
    
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random
    
    facilities = []
    
    if count >= 3:
        facilities.extend(["core", "crane", "storage"])
    
    if count >= 5:
        facilities.extend(["office", "rest_area"])
    
    remaining = count - len(facilities)
    operational_types = ["core", "storage", "crane"]
    
    for _ in range(remaining):
        facilities.append(rng.choice(operational_types))
    
    rng.shuffle(facilities)
    return facilities

def rectangles_overlap(f1: Dict, f2: Dict) -> bool:
    """Check if two facility rectangles overlap"""
    x1, y1 = f1["center"]
    x2, y2 = f2["center"]
    
    spec1 = FACILITY_SPECS[f1["type"]]
    spec2 = FACILITY_SPECS[f2["type"]]
    
    left1, right1 = x1 - spec1["w"]/2, x1 + spec1["w"]/2
    bottom1, top1 = y1 - spec1["d"]/2, y1 + spec1["d"]/2
    
    left2, right2 = x2 - spec2["w"]/2, x2 + spec2["w"]/2
    bottom2, top2 = y2 - spec2["d"]/2, y2 + spec2["d"]/2
    
    buffer = 0.02
    no_overlap = (right1 + buffer <= left2 or right2 + buffer <= left1 or 
                  top1 + buffer <= bottom2 or top2 + buffer <= bottom1)
    
    return not no_overlap

def calculate_overlap_area(f1: Dict, f2: Dict) -> float:
    """Calculate overlap area between two facilities"""
    if not rectangles_overlap(f1, f2):
        return 0.0
    
    x1, y1 = f1["center"]
    x2, y2 = f2["center"]
    
    spec1 = FACILITY_SPECS[f1["type"]]
    spec2 = FACILITY_SPECS[f2["type"]]
    
    left1, right1 = x1 - spec1["w"]/2, x1 + spec1["w"]/2
    bottom1, top1 = y1 - spec1["d"]/2, y1 + spec1["d"]/2
    
    left2, right2 = x2 - spec2["w"]/2, x2 + spec2["w"]/2
    bottom2, top2 = y2 - spec2["d"]/2, y2 + spec2["d"]/2
    
    overlap_left = max(left1, left2)
    overlap_right = min(right1, right2)
    overlap_bottom = max(bottom1, bottom2)
    overlap_top = min(top1, top2)
    
    if overlap_right > overlap_left and overlap_top > overlap_bottom:
        return (overlap_right - overlap_left) * (overlap_top - overlap_bottom)
    
    return 0.0