#!/usr/bin/env python3
"""
Behavioral Descriptors Module
=============================

Implements behavioral descriptors for MAP-Elites algorithm:
- BD1: Compactness vs Spread (spatial organization)
- BD2: Worker-Operational Separation (functional integration)

These descriptors define the 2D behavioral space for the MAP-Elites archive.
"""

from typing import List, Dict
import numpy as np

# =============================================================================
# BEHAVIORAL DESCRIPTOR 1: COMPACTNESS VS SPREAD
# =============================================================================

def calculate_compactness_vs_spread(facilities: List[Dict]) -> float:
    """BD1: Compactness vs. Spread (0 = very compact, 1 = very spread)"""
    if len(facilities) < 2:
        return 0.5
    
    # Calculate global centroid
    positions = np.array([f["center"] for f in facilities])
    centroid = np.mean(positions, axis=0)
    
    # Calculate mean distance to centroid
    distances = [np.linalg.norm(np.array(f["center"]) - centroid) for f in facilities]
    mean_distance = np.mean(distances)
    
    # Normalize with bound B ≈ 0.50 (maximum possible mean distance in unit square)
    B = 0.50
    compactness_spread = mean_distance / B
    
    return float(np.clip(compactness_spread, 0.0, 1.0))

# =============================================================================
# BEHAVIORAL DESCRIPTOR 2: WORKER-OPERATIONAL SEPARATION
# =============================================================================

def calculate_worker_operational_separation(facilities: List[Dict]) -> float:
    """BD2: Worker-Operational separation (0 = workers embedded in ops, 1 = strong segregation)"""
    # Define facility categories
    worker_types = {"office", "rest_area"}
    operational_types = {"core", "storage", "crane"}
    
    # Get worker and operational facilities
    worker_facilities = [f for f in facilities if f["type"] in worker_types]
    operational_facilities = [f for f in facilities if f["type"] in operational_types]
    
    if not worker_facilities or not operational_facilities:
        return 0.5
    
    # For each worker facility, find distance to nearest operational facility
    separation_distances = []
    for worker in worker_facilities:
        worker_pos = np.array(worker["center"])
        min_distance = min(np.linalg.norm(worker_pos - np.array(op["center"])) 
                          for op in operational_facilities)
        separation_distances.append(min_distance)
    
    # Average separation distance
    avg_separation = np.mean(separation_distances)
    
    # Normalize with bound S ≈ 0.30 (reasonable maximum separation in construction site)
    S = 0.30
    separation_ratio = avg_separation / S
    
    return float(np.clip(separation_ratio, 0.0, 1.0))

# =============================================================================
# LEGACY FUNCTION NAMES FOR COMPATIBILITY
# =============================================================================

def calculate_spatial_organization(facilities: List[Dict]) -> float:
    """BD1: Alias for compactness vs spread"""
    return calculate_compactness_vs_spread(facilities)

def calculate_functional_integration(facilities: List[Dict]) -> float:
    """BD2: Alias for worker-operational separation (inverted)"""
    # Note: Invert because integration is opposite of separation
    return 1.0 - calculate_worker_operational_separation(facilities)

# =============================================================================
# BEHAVIORAL DESCRIPTOR UTILITIES
# =============================================================================

def get_behavioral_description(bd1: float, bd2: float) -> str:
    """Get human-readable description of behavioral descriptors"""
    spatial = "Centralized" if bd1 < 0.5 else "Distributed"
    functional = "Segregated" if bd2 < 0.5 else "Integrated"
    
    descriptions = {
        ("Centralized", "Segregated"): "Facilities clustered near center with clear functional separation",
        ("Centralized", "Integrated"): "Facilities clustered near center with mixed functional zones",
        ("Distributed", "Segregated"): "Facilities spread across site with clear functional separation",
        ("Distributed", "Integrated"): "Facilities spread across site with mixed functional zones"
    }
    
    return descriptions.get((spatial, functional), f"{spatial} {functional}")

def get_behavioral_quadrant(bd1: float, bd2: float) -> str:
    """Get behavioral quadrant name"""
    if bd1 < 0.5 and bd2 < 0.5:
        return "centralized_segregated"
    elif bd1 < 0.5 and bd2 >= 0.5:
        return "centralized_integrated"
    elif bd1 >= 0.5 and bd2 < 0.5:
        return "distributed_segregated"
    else:
        return "distributed_integrated"

def analyze_behavioral_regions(individuals: List) -> Dict:
    """Analyze which regions of the 2D behavioral space are well-explored"""
    
    # Divide behavioral space into quadrants
    regions = {
        "centralized_segregated": {"bd1_range": (0.0, 0.5), "bd2_range": (0.0, 0.5)},
        "centralized_integrated": {"bd1_range": (0.0, 0.5), "bd2_range": (0.5, 1.0)},
        "distributed_segregated": {"bd1_range": (0.5, 1.0), "bd2_range": (0.0, 0.5)},
        "distributed_integrated": {"bd1_range": (0.5, 1.0), "bd2_range": (0.5, 1.0)}
    }
    
    region_analysis = {}
    
    for region_name, bounds in regions.items():
        # Count individuals in this region
        region_individuals = []
        for ind in individuals:
            if hasattr(ind, 'behaviors') and ind.behaviors:
                bd1, bd2 = ind.behaviors
                if (bounds["bd1_range"][0] <= bd1 < bounds["bd1_range"][1] and
                    bounds["bd2_range"][0] <= bd2 < bounds["bd2_range"][1]):
                    region_individuals.append(ind)
        
        if region_individuals:
            region_objectives = np.array([ind.objectives for ind in region_individuals])
            region_analysis[region_name] = {
                "count": len(region_individuals),
                "percentage": 100.0 * len(region_individuals) / len(individuals),
                "avg_safety": np.mean(region_objectives[:, 0]),
                "avg_efficiency": np.mean(region_objectives[:, 1]),
                "avg_adaptability": np.mean(region_objectives[:, 2]),
                "best_solution": {
                    "safety": np.max(region_objectives[:, 0]),
                    "efficiency": np.max(region_objectives[:, 1]),
                    "adaptability": np.max(region_objectives[:, 2])
                }
            }
        else:
            region_analysis[region_name] = {
                "count": 0,
                "percentage": 0.0,
                "avg_safety": 0.0,
                "avg_efficiency": 0.0,
                "avg_adaptability": 0.0,
                "best_solution": {"safety": 0.0, "efficiency": 0.0, "adaptability": 0.0}
            }
    
    return region_analysis