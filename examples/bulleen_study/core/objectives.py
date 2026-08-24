#!/usr/bin/env python3
"""
Objective Functions Module
=========================

Implements the three main objectives for construction site layout optimization:
- O1: Safety & Constraint Compliance
- O2: Operational Efficiency 
- O3: Layout Adaptability

Used by both MAP-Elites and NSGA-II algorithms.
"""

import math
import random
from typing import List, Dict, Tuple
import numpy as np

from .config import (
    FACILITY_SPECS,
    SiteConfig,
    facility_within_site_boundary,
    get_facility_dimensions,
    rectangles_overlap,
    calculate_overlap_area,
)

# =============================================================================
# OBJECTIVE FUNCTION 1: SAFETY & CONSTRAINT COMPLIANCE
# =============================================================================

def calculate_safety_compliance(facilities: List[Dict], entrances: List[Tuple[float, float]], 
                              config: SiteConfig) -> Tuple[float, bool, List[str]]:
    """O1: Safety & Constraint Compliance with gradient scoring"""
    violations = []
    
    # Boundary violations (40%) - Graduated penalty based on severity
    boundary_violations = 0
    boundary_severity = 0.0
    margin = config.boundary_margin + 0.01
    
    for i, facility in enumerate(facilities):
        x, y = facility["center"]
        w, d = get_facility_dimensions(facility)
        half_w, half_h = w/2, d/2
        
        if config.boundary_polygon:
            total_violation = 0.0 if facility_within_site_boundary(facility, config, margin=0.01) else 0.1
        else:
            # Calculate how far outside boundaries (if any)
            left_violation = max(0, margin - (x - half_w))
            right_violation = max(0, (x + half_w) - (1.0 - margin))
            bottom_violation = max(0, margin - (y - half_h))
            top_violation = max(0, (y + half_h) - (1.0 - margin))
            total_violation = left_violation + right_violation + bottom_violation + top_violation

        if total_violation > 0:
            boundary_violations += 1
            boundary_severity += total_violation
            violations.append(f"boundary_violation_{i}")
    
    # Graduated boundary score: small violations get partial penalty
    boundary_penalty = min(1.0, (boundary_violations / len(facilities)) * 0.7 + boundary_severity * 2.0)
    boundary_score = max(0.0, 1.0 - boundary_penalty)
    
    # Overlap violations (30%) - Graduated by area and count
    overlap_count = 0
    total_overlap_area = 0.0
    
    for i in range(len(facilities)):
        for j in range(i+1, len(facilities)):
            if rectangles_overlap(facilities[i], facilities[j]):
                overlap_count += 1
                overlap_area = calculate_overlap_area(facilities[i], facilities[j])
                total_overlap_area += overlap_area
                violations.append(f"overlap_{i}_{j}")
    
    # Module layouts can include 20+ small facilities, so overlap count is
    # normalized by layout size and area is weighted more than raw pair count.
    overlap_pair_ratio = overlap_count / max(1, len(facilities))
    overlap_penalty = min(1.0, (overlap_pair_ratio * 0.8) + (total_overlap_area * 20.0))
    overlap_score = max(0.0, 1.0 - overlap_penalty)
    
    # Critical safety violations (30%) - Graduated by proximity to danger
    safety_violations = 0
    safety_severity = 0.0
    cranes = [f for f in facilities if f["type"] == "crane"]
    workers = [f for f in facilities if f["type"] in ("office", "rest_area")]
    
    # Crane-to-crane clearance: prevent jib collisions
    danger_radius = FACILITY_SPECS["crane"]["danger_radius"]
    min_crane_clearance = danger_radius * 2.0  # Cranes must stay 2x danger radius apart
    
    for i in range(len(cranes)):
        for j in range(i+1, len(cranes)):
            crane1_pos = np.array(cranes[i]["center"])
            crane2_pos = np.array(cranes[j]["center"])
            distance = np.linalg.norm(crane1_pos - crane2_pos)
            if distance < min_crane_clearance:
                safety_violations += 1
                # Graduated penalty: closer cranes = higher collision risk
                severity = (min_crane_clearance - distance) / min_crane_clearance
                safety_severity += severity
                violations.append(f"crane_collision_risk_{i}_{j}")
    
    # Crane danger zones vs worker facilities
    for crane in cranes:
        crane_pos = np.array(crane["center"])
        
        for worker in workers:
            worker_pos = np.array(worker["center"])
            distance = np.linalg.norm(worker_pos - crane_pos)
            if distance < danger_radius:
                safety_violations += 1
                # Graduated penalty: closer = more dangerous
                severity = (danger_radius - distance) / danger_radius
                safety_severity += severity
                violations.append("crane_danger_violation")
    
    # Entrance clearance - Graduated by how close facilities are
    entrance_violations = 0
    entrance_severity = 0.0
    for entrance in entrances:
        entrance_pos = np.array(entrance)
        for facility in facilities:
            facility_pos = np.array(facility["center"])
            distance = np.linalg.norm(facility_pos - entrance_pos)
            if distance < config.entrance_clearance:
                entrance_violations += 1
                # Graduated penalty: closer = worse access
                severity = (config.entrance_clearance - distance) / config.entrance_clearance
                entrance_severity += severity
                violations.append("entrance_clearance_violation")
    
    # Graduated safety penalty: combines count and severity
    total_safety_penalty = min(1.0, (safety_violations + entrance_violations) * 0.1 + 
                              (safety_severity + entrance_severity) * 0.3)
    safety_score = max(0.0, 1.0 - total_safety_penalty)
    
    # Combined score with graduated components
    final_score = 0.4 * boundary_score + 0.3 * overlap_score + 0.3 * safety_score
    feasible = len(violations) == 0
    
    return float(np.clip(final_score, 0.0, 1.0)), feasible, violations

# =============================================================================
# OBJECTIVE FUNCTION 2: OPERATIONAL EFFICIENCY
# =============================================================================

def calculate_worker_facility_clustering(facility_positions: Dict[str, List[np.ndarray]]) -> float:
    """Reward offices and rest areas forming a compact worker-support cluster."""
    worker_positions = facility_positions.get("office", []) + facility_positions.get("rest_area", [])
    if len(worker_positions) < 2:
        return 0.5

    worker_centroid = np.mean(worker_positions, axis=0)
    mean_cluster_radius = np.mean([
        np.linalg.norm(pos - worker_centroid)
        for pos in worker_positions
    ])
    compactness_score = max(0.0, 1.0 - mean_cluster_radius / 0.16)

    # Ground-truth layouts tend to intermix offices and rest areas in one support
    # zone, so each module should have the other worker type nearby where possible.
    cross_type_score = 0.5
    if facility_positions.get("office") and facility_positions.get("rest_area"):
        nearest_cross_distances = []
        for office_pos in facility_positions["office"]:
            nearest_cross_distances.append(min(
                np.linalg.norm(office_pos - rest_pos)
                for rest_pos in facility_positions["rest_area"]
            ))
        for rest_pos in facility_positions["rest_area"]:
            nearest_cross_distances.append(min(
                np.linalg.norm(rest_pos - office_pos)
                for office_pos in facility_positions["office"]
            ))
        avg_cross_distance = np.mean(nearest_cross_distances)
        cross_type_score = max(0.0, 1.0 - avg_cross_distance / 0.14)

    return float(np.clip(0.65 * compactness_score + 0.35 * cross_type_score, 0.0, 1.0))


def point_to_facility_rect_distance(point: np.ndarray, facility: Dict) -> float:
    """Distance from a point to the nearest point on a facility rectangle."""
    x, y = facility["center"]
    width, depth = get_facility_dimensions(facility)
    dx = max(abs(point[0] - x) - width / 2, 0.0)
    dy = max(abs(point[1] - y) - depth / 2, 0.0)
    return math.hypot(dx, dy)


def calculate_crane_core_coverage(facilities: List[Dict]) -> float:
    """Reward crane operating-radius overlap with every core footprint."""
    cranes = [f for f in facilities if f["type"] == "crane"]
    cores = [f for f in facilities if f["type"] == "core"]
    if not cranes or not cores:
        return 0.0

    crane_spec = FACILITY_SPECS["crane"]
    optimal_reach = crane_spec["optimal_reach"]
    operating_radius = crane_spec["operating_radius"]
    core_scores = []

    for core in cores:
        best_score = 0.0
        for crane in cranes:
            crane_pos = np.array(crane["center"])
            edge_distance = point_to_facility_rect_distance(crane_pos, core)
            if edge_distance <= optimal_reach:
                score = 1.0
            elif edge_distance <= operating_radius:
                score = 1.0 - 0.5 * (
                    (edge_distance - optimal_reach) / (operating_radius - optimal_reach)
                )
            else:
                score = 0.0
            best_score = max(best_score, score)
        core_scores.append(best_score)

    return float(np.mean(core_scores))


def calculate_operational_efficiency(facilities: List[Dict], entrances: List[Tuple[float, float]]) -> float:
    """O2: Operational Efficiency with enhanced crane model"""
    
    # Get facility positions by type
    facility_positions = {}
    for facility in facilities:
        ftype = facility["type"]
        if ftype not in facility_positions:
            facility_positions[ftype] = []
        facility_positions[ftype].append(np.array(facility["center"]))
    
    # Critical material flows (40%)
    flow_efficiency = 0.0
    critical_flows = [("storage", "core"), ("crane", "core"), ("storage", "crane")]
    
    total_flow_distance = 0.0
    flow_count = 0
    
    for source_type, target_type in critical_flows:
        if source_type in facility_positions and target_type in facility_positions:
            for source_pos in facility_positions[source_type]:
                for target_pos in facility_positions[target_type]:
                    distance = np.linalg.norm(target_pos - source_pos)
                    total_flow_distance += distance
                    flow_count += 1
    
    if flow_count > 0:
        avg_flow_distance = total_flow_distance / flow_count
        site_diagonal = math.sqrt(2) * 0.8
        flow_efficiency = max(0.0, 1.0 - avg_flow_distance / site_diagonal)
    
    # Enhanced Equipment accessibility with crane coverage analysis (40%)
    access_efficiency = 0.0
    if "crane" in facility_positions:
        crane_positions = facility_positions["crane"]
        work_areas = facility_positions.get("core", []) + facility_positions.get("storage", [])
        
        if work_areas:
            # Define crane parameters from configuration.
            crane_spec = FACILITY_SPECS["crane"]
            optimal_reach = crane_spec["optimal_reach"]
            max_reach = crane_spec["operating_radius"]
            overlap_bonus = 0.15  # Bonus for overlapping coverage
            
            total_coverage_score = 0.0
            
            for work_pos in work_areas:
                # Calculate coverage from each crane
                crane_coverages = []
                for crane_pos in crane_positions:
                    distance = np.linalg.norm(work_pos - crane_pos)
                    
                    if distance <= optimal_reach:
                        # Optimal zone: full efficiency
                        coverage = 1.0
                    elif distance <= max_reach:
                        # Reduced efficiency zone: linear decay
                        coverage = 1.0 - ((distance - optimal_reach) / (max_reach - optimal_reach)) * 0.4
                    else:
                        # Out of reach
                        coverage = 0.0
                    
                    crane_coverages.append(coverage)
                
                # Primary coverage: best crane
                primary_coverage = max(crane_coverages) if crane_coverages else 0.0
                
                # Redundancy bonus: multiple cranes covering the same area
                if len(crane_coverages) > 1:
                    # Count cranes providing meaningful coverage (>0.3)
                    effective_cranes = sum(1 for c in crane_coverages if c > 0.3)
                    if effective_cranes > 1:
                        # Bonus scales with number of covering cranes
                        redundancy_bonus = min(overlap_bonus * (effective_cranes - 1), 0.3)
                        primary_coverage = min(1.0, primary_coverage + redundancy_bonus)
                
                # Critical facility bonus: core facilities get higher weight
                facility_weight = 1.2 if any(np.array_equal(work_pos, core_pos) 
                                           for core_pos in facility_positions.get("core", [])) else 1.0
                
                total_coverage_score += primary_coverage * facility_weight
            
            # Normalize by weighted work areas
            total_weight = sum(1.2 if any(np.array_equal(work_pos, core_pos) 
                                        for core_pos in facility_positions.get("core", [])) else 1.0 
                             for work_pos in work_areas)
            access_efficiency = total_coverage_score / total_weight if total_weight > 0 else 0.0
            
            # Additional bonus for strategic crane placement
            if len(crane_positions) > 1:
                # Reward cranes that cover complementary areas
                coverage_overlap_penalty = 0.0
                for i, crane1 in enumerate(crane_positions):
                    for crane2 in crane_positions[i+1:]:
                        crane_distance = np.linalg.norm(crane1 - crane2)
                        if crane_distance < optimal_reach * 1.5:  # Too close
                            coverage_overlap_penalty += 0.1
                
                access_efficiency = max(0.0, access_efficiency - coverage_overlap_penalty)
    
    # Work sequence support: access to entrances plus worker-facility clustering.
    entrance_access_efficiency = 0.0
    if entrances and "office" in facility_positions:
        office_positions = facility_positions["office"]
        entrance_positions = [np.array(e) for e in entrances]
        
        total_entrance_access = 0.0
        for office_pos in office_positions:
            min_entrance_distance = min(np.linalg.norm(office_pos - entrance_pos) 
                                      for entrance_pos in entrance_positions)
            access_quality = max(0.0, 1.0 - min_entrance_distance / (math.sqrt(2) * 0.4))
            total_entrance_access += access_quality
        
        entrance_access_efficiency = total_entrance_access / len(office_positions)

    worker_cluster_efficiency = calculate_worker_facility_clustering(facility_positions)
    crane_core_coverage = calculate_crane_core_coverage(facilities)
    
    # Ground-truth-informed weighting: material/crane logic remains dominant,
    # with explicit crane-core coverage required for strong efficiency.
    final_score = (
        0.25 * flow_efficiency
        + 0.25 * access_efficiency
        + 0.20 * crane_core_coverage
        + 0.15 * entrance_access_efficiency
        + 0.15 * worker_cluster_efficiency
    )
    return float(np.clip(final_score, 0.0, 1.0))

# =============================================================================
# OBJECTIVE FUNCTION 3: LAYOUT ADAPTABILITY
# =============================================================================

def calculate_layout_adaptability(facilities: List[Dict], entrances: List[Tuple[float, float]], 
                                config: SiteConfig) -> float:
    """O3: Layout Adaptability"""
    
    # Expansion potential (40%)
    grid_size = 25
    cell_size = 1.0 / grid_size
    margin = config.boundary_margin
    
    occupied_cells = set()
    for facility in facilities:
        x, y = facility["center"]
        w, d = get_facility_dimensions(facility)
        
        cells_x = range(max(0, int((x - w/2) / cell_size)),
                       min(grid_size, int((x + w/2) / cell_size) + 1))
        cells_y = range(max(0, int((y - d/2) / cell_size)),
                       min(grid_size, int((y + d/2) / cell_size) + 1))
        
        for cx in cells_x:
            for cy in cells_y:
                occupied_cells.add((cx, cy))
    
    margin_cells = int(margin / cell_size) + 1
    total_usable_cells = (grid_size - 2 * margin_cells) ** 2
    available_cells = total_usable_cells - len(occupied_cells)
    expansion_potential = available_cells / max(1, total_usable_cells)
    
    # Route redundancy (35%)
    key_pairs = [("office", "core"), ("storage", "core"), ("crane", "storage")]
    redundancy_score = 0.0
    
    for source_type, target_type in key_pairs:
        source_facilities = [f for f in facilities if f["type"] == source_type]
        target_facilities = [f for f in facilities if f["type"] == target_type]
        
        if source_facilities and target_facilities:
            min_distances = []
            for source in source_facilities:
                for target in target_facilities:
                    distance = np.linalg.norm(np.array(source["center"]) - np.array(target["center"]))
                    min_distances.append(distance)
            
            if len(min_distances) > 1:
                distance_variance = np.var(min_distances)
                redundancy_score += 1.0 / (1.0 + distance_variance * 10)
            else:
                redundancy_score += 0.5
    
    route_redundancy = redundancy_score / len(key_pairs)
    
    # Reconfiguration ease (25%)
    reconfiguration_ease = 0.0
    
    for i, facility in enumerate(facilities):
        current_pos = np.array(facility["center"])
        w, d = get_facility_dimensions(facility)
        
        relocation_options = 0
        test_positions = 20
        
        for _ in range(test_positions):
            new_x = np.random.uniform(margin + w/2, 1.0 - margin - w/2)
            new_y = np.random.uniform(margin + d/2, 1.0 - margin - d/2)
            new_pos = np.array([new_x, new_y])
            
            test_facility = {
                "type": facility["type"],
                "center": (new_x, new_y),
                "rotation": facility.get("rotation", 0),
            }
            
            if not facility_within_site_boundary(test_facility, config):
                continue

            has_overlap = False
            for j, other_facility in enumerate(facilities):
                if i != j and rectangles_overlap(test_facility, other_facility):
                    has_overlap = True
                    break
            
            distance_from_original = np.linalg.norm(new_pos - current_pos)
            if not has_overlap and distance_from_original < 0.5:
                relocation_options += 1
        
        facility_flexibility = relocation_options / test_positions
        reconfiguration_ease += facility_flexibility
    
    reconfiguration_ease /= len(facilities)
    
    final_score = 0.4 * expansion_potential + 0.35 * route_redundancy + 0.25 * reconfiguration_ease
    return float(np.clip(final_score, 0.0, 1.0))

# =============================================================================
# COMBINED EVALUATION FUNCTION
# =============================================================================

def evaluate_individual(solution: List[Dict], entrances: List[Tuple[float, float]], 
                       config: SiteConfig, calculate_behaviors: bool = False) -> Dict:
    """
    Evaluate an individual solution with all three objectives
    
    Args:
        solution: List of facility dictionaries
        entrances: List of entrance positions
        config: Site configuration
        calculate_behaviors: If True, also calculate behavioral descriptors (for MAP-Elites)
    
    Returns:
        Dictionary with objectives, feasibility, violations, and optionally behaviors
    """
    safety_score, feasible, violations = calculate_safety_compliance(solution, entrances, config)
    efficiency_score = calculate_operational_efficiency(solution, entrances)
    adaptability_score = calculate_layout_adaptability(solution, entrances, config)
    
    result = {
        'objectives': (safety_score, efficiency_score, adaptability_score),
        'feasible': feasible,
        'violations': violations
    }
    
    if calculate_behaviors:
        from .behavioral_descriptors import calculate_spatial_organization, calculate_functional_integration
        spatial_org = calculate_spatial_organization(solution)
        functional_int = calculate_functional_integration(solution)
        result['behaviors'] = (spatial_org, functional_int)
    
    return result
