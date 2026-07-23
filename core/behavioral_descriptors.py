#!/usr/bin/env python3
"""
Behavioral Descriptors Module
=============================

Hand-crafted descriptors for MAP-Elites plus a manager for switching to learned
autoencoder descriptors.

For the practical Bulleen case, facilities are now small repeatable modules.
The descriptors therefore focus on module clustering and functional zoning,
instead of treating each facility as one large block.
"""

from typing import List, Dict, Tuple
import numpy as np


# =============================================================================
# BEHAVIORAL DESCRIPTOR 1: MODULE CLUSTERING VS DISPERSION
# =============================================================================

def calculate_compactness_vs_spread(facilities: List[Dict]) -> float:
    """BD1: same-type module clustering vs dispersion.

    0 = repeated modules of the same type are clustered.
    1 = repeated modules of the same type are dispersed.
    """
    if len(facilities) < 2:
        return 0.5

    nearest_same_type_distances = []
    for facility in facilities:
        same_type = [
            other for other in facilities
            if other is not facility and other["type"] == facility["type"]
        ]
        if not same_type:
            continue

        facility_pos = np.array(facility["center"])
        nearest_distance = min(
            np.linalg.norm(facility_pos - np.array(other["center"]))
            for other in same_type
        )
        nearest_same_type_distances.append(nearest_distance)

    if not nearest_same_type_distances:
        positions = np.array([f["center"] for f in facilities])
        centroid = np.mean(positions, axis=0)
        mean_distance = np.mean([
            np.linalg.norm(np.array(f["center"]) - centroid)
            for f in facilities
        ])
        return float(np.clip(mean_distance / 0.50, 0.0, 1.0))

    cluster_distance = np.mean(nearest_same_type_distances)
    return float(np.clip(cluster_distance / 0.16, 0.0, 1.0))


# =============================================================================
# BEHAVIORAL DESCRIPTOR 2: WORKER-OPERATIONAL SEPARATION
# =============================================================================

def calculate_worker_operational_separation(facilities: List[Dict]) -> float:
    """BD2: worker-operational separation.

    0 = office/rest modules are embedded near work operations.
    1 = office/rest modules are segregated from operational modules.
    """
    worker_types = {"office", "rest_area"}
    operational_types = {"core", "storage", "crane"}

    worker_facilities = [f for f in facilities if f["type"] in worker_types]
    operational_facilities = [f for f in facilities if f["type"] in operational_types]

    if not worker_facilities or not operational_facilities:
        return 0.5

    separation_distances = []
    for worker in worker_facilities:
        worker_pos = np.array(worker["center"])
        min_distance = min(
            np.linalg.norm(worker_pos - np.array(op["center"]))
            for op in operational_facilities
        )
        separation_distances.append(min_distance)

    worker_centroid = np.mean([np.array(f["center"]) for f in worker_facilities], axis=0)
    operational_centroid = np.mean([np.array(f["center"]) for f in operational_facilities], axis=0)
    centroid_separation = np.linalg.norm(worker_centroid - operational_centroid)

    avg_nearest_separation = np.mean(separation_distances)
    separation_ratio = (0.6 * avg_nearest_separation + 0.4 * centroid_separation) / 0.28
    return float(np.clip(separation_ratio, 0.0, 1.0))


# =============================================================================
# LEGACY FUNCTION NAMES FOR COMPATIBILITY
# =============================================================================

def calculate_spatial_organization(facilities: List[Dict]) -> float:
    """BD1 alias retained for existing algorithm code."""
    return calculate_compactness_vs_spread(facilities)


def calculate_functional_integration(facilities: List[Dict]) -> float:
    """BD2 alias retained for existing algorithm code.

    Despite the old name, this now returns worker-operational separation directly
    so the archive axis matches BehavioralDescriptorManager.
    """
    return calculate_worker_operational_separation(facilities)


# =============================================================================
# BEHAVIORAL DESCRIPTOR UTILITIES
# =============================================================================

def get_behavioral_description(bd1: float, bd2: float) -> str:
    """Get human-readable description of behavioral descriptors."""
    spatial = "Clustered" if bd1 < 0.5 else "Dispersed"
    functional = "Embedded" if bd2 < 0.5 else "Segregated"

    descriptions = {
        ("Clustered", "Embedded"): "Same-type modules form clusters with worker facilities embedded near operations",
        ("Clustered", "Segregated"): "Same-type modules form clusters with worker facilities separated from operations",
        ("Dispersed", "Embedded"): "Same-type modules are distributed while worker facilities remain near operations",
        ("Dispersed", "Segregated"): "Same-type modules are distributed with worker facilities separated from operations",
    }

    return descriptions.get((spatial, functional), f"{spatial} {functional}")


def get_behavioral_quadrant(bd1: float, bd2: float) -> str:
    """Get behavioral quadrant name."""
    if bd1 < 0.5 and bd2 < 0.5:
        return "clustered_embedded"
    if bd1 < 0.5 and bd2 >= 0.5:
        return "clustered_segregated"
    if bd1 >= 0.5 and bd2 < 0.5:
        return "dispersed_embedded"
    return "dispersed_segregated"


def analyze_behavioral_regions(individuals: List) -> Dict:
    """Analyze which regions of the 2D behavioral space are well-explored."""
    regions = {
        "clustered_embedded": {"bd1_range": (0.0, 0.5), "bd2_range": (0.0, 0.5)},
        "clustered_segregated": {"bd1_range": (0.0, 0.5), "bd2_range": (0.5, 1.0)},
        "dispersed_embedded": {"bd1_range": (0.5, 1.0), "bd2_range": (0.0, 0.5)},
        "dispersed_segregated": {"bd1_range": (0.5, 1.0), "bd2_range": (0.5, 1.0)},
    }

    region_analysis = {}

    for region_name, bounds in regions.items():
        region_individuals = []
        for ind in individuals:
            if hasattr(ind, "behaviors") and ind.behaviors:
                bd1, bd2 = ind.behaviors
                if (
                    bounds["bd1_range"][0] <= bd1 < bounds["bd1_range"][1]
                    and bounds["bd2_range"][0] <= bd2 < bounds["bd2_range"][1]
                ):
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
                    "adaptability": np.max(region_objectives[:, 2]),
                },
            }
        else:
            region_analysis[region_name] = {
                "count": 0,
                "percentage": 0.0,
                "avg_safety": 0.0,
                "avg_efficiency": 0.0,
                "avg_adaptability": 0.0,
                "best_solution": {"safety": 0.0, "efficiency": 0.0, "adaptability": 0.0},
            }

    return region_analysis


# =============================================================================
# BEHAVIORAL DESCRIPTOR MANAGER
# =============================================================================

class BehavioralDescriptorManager:
    """Manages hand-crafted and autoencoder-learned descriptors."""

    def __init__(self, mode: str = "hand-crafted", autoencoder_model=None):
        if mode not in ["hand-crafted", "learned"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'hand-crafted' or 'learned'")

        self.mode = mode
        self.autoencoder_model = autoencoder_model

        if self.mode == "learned" and self.autoencoder_model is None:
            raise ValueError("autoencoder_model must be provided when mode='learned'")

    def get_descriptors(
        self,
        facilities: List[Dict],
        entrances: List[Tuple[float, float]] = None,
    ) -> Tuple[float, float]:
        """Extract descriptors as a tuple in [0, 1]."""
        if self.mode == "hand-crafted":
            bd1 = calculate_compactness_vs_spread(facilities)
            bd2 = calculate_worker_operational_separation(facilities)
            return (bd1, bd2)

        return self.autoencoder_model.get_behavioral_descriptors(facilities, entrances)

    def switch_mode(self, new_mode: str, autoencoder_model=None):
        """Switch between hand-crafted and learned descriptor modes."""
        if new_mode not in ["hand-crafted", "learned"]:
            raise ValueError(f"Invalid mode: {new_mode}")

        if new_mode == "learned" and autoencoder_model is None:
            raise ValueError("autoencoder_model must be provided when switching to learned mode")

        self.mode = new_mode
        if autoencoder_model is not None:
            self.autoencoder_model = autoencoder_model

    def get_mode(self) -> str:
        """Get current descriptor mode."""
        return self.mode


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_descriptor_manager(
    use_learned: bool = False,
    autoencoder_model=None,
) -> BehavioralDescriptorManager:
    """Create a behavioral descriptor manager."""
    mode = "learned" if use_learned else "hand-crafted"
    return BehavioralDescriptorManager(mode=mode, autoencoder_model=autoencoder_model)
