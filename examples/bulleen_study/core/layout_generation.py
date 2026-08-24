#!/usr/bin/env python3
"""
Layout Generation Module
========================

Functions for generating, mutating, and manipulating construction site layouts.
Includes entrance generation, facility placement, crossover, and mutation operations.
"""

import math
import random
from typing import List, Dict, Tuple
import numpy as np

from .config import (
    FACILITY_SPECS,
    FACILITY_CLEARANCE_BUFFER,
    SiteConfig,
    facility_within_site_boundary,
    get_facility_dimensions,
    get_site_bounds,
    rectangles_overlap,
)


def random_rotation_for_facility(facility_type: str) -> int:
    """Choose a discrete orientation for rectangular facilities."""
    spec = FACILITY_SPECS[facility_type]
    if abs(spec["w"] - spec["d"]) < 1e-9:
        return 0
    return random.choice([0, 90])

# =============================================================================
# ENTRANCE GENERATION
# =============================================================================

def generate_random_entrances(config: SiteConfig, seed: int = None) -> List[Tuple[float, float]]:
    """Generate random entrance locations"""
    if getattr(config, "fixed_entrances", None):
        return config.fixed_entrances[:]

    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random
    
    num_entrances = rng.randint(config.min_entrances, config.max_entrances)
    entrances = []
    margin = config.boundary_margin
    
    entrance_zones = [
        ('bottom', lambda: (rng.uniform(margin + 0.1, 1.0 - margin - 0.1), margin)),
        ('top', lambda: (rng.uniform(margin + 0.1, 1.0 - margin - 0.1), 1.0 - margin)),
        ('left', lambda: (margin, rng.uniform(margin + 0.1, 1.0 - margin - 0.1))),
        ('right', lambda: (1.0 - margin, rng.uniform(margin + 0.1, 1.0 - margin - 0.1)))
    ]
    
    min_separation = 0.2
    
    for i in range(num_entrances):
        attempts = 0
        max_attempts = 50
        
        while attempts < max_attempts:
            zone_name, generator = rng.choice(entrance_zones)
            entrance_pos = generator()
            
            too_close = False
            for existing_entrance in entrances:
                distance = math.sqrt((entrance_pos[0] - existing_entrance[0])**2 + 
                                   (entrance_pos[1] - existing_entrance[1])**2)
                if distance < min_separation:
                    too_close = True
                    break
            
            if not too_close:
                entrances.append(entrance_pos)
                break
            
            attempts += 1
        
        if len(entrances) <= i:
            fallback_x = rng.uniform(margin + 0.2, 1.0 - margin - 0.2)
            fallback_y = margin if i % 2 == 0 else 1.0 - margin
            entrances.append((fallback_x, fallback_y))
    
    return entrances

# =============================================================================
# FACILITY PLACEMENT
# =============================================================================

def create_random_layout(facility_types: List[str], boundary_margin: float,
                         config: SiteConfig = None) -> List[Dict]:
    """Create random layout with sequential placement"""
    facilities = []
    site_config = config or SiteConfig(boundary_margin=boundary_margin)
    
    priority_order = ["office", "rest_area", "core", "storage", "crane"]
    sorted_types = sorted(facility_types, 
                         key=lambda t: priority_order.index(t) if t in priority_order else len(priority_order))
    
    for ftype in sorted_types:
        facility = create_random_facility(ftype, boundary_margin, facilities, site_config)
        facilities.append(facility)
    
    return facilities

def create_random_facility(facility_type: str, boundary_margin: float, 
                          existing_facilities: List[Dict] = None,
                          config: SiteConfig = None) -> Dict:
    """Create random facility with overlap avoidance"""
    if existing_facilities is None:
        existing_facilities = []
    
    site_config = config or SiteConfig(boundary_margin=boundary_margin)
    rotation = random_rotation_for_facility(facility_type)
    probe_facility = {"type": facility_type, "center": (0.5, 0.5), "rotation": rotation}
    w, d = get_facility_dimensions(probe_facility)
    half_w, half_h = w/2, d/2
    
    safety_buffer = FACILITY_CLEARANCE_BUFFER
    safe_margin = boundary_margin + max(half_w, half_h) + safety_buffer
    site_min_x, site_max_x, site_min_y, site_max_y = get_site_bounds(site_config)
    min_x = site_min_x + safe_margin
    max_x = site_max_x - safe_margin
    min_y = site_min_y + safe_margin
    max_y = site_max_y - safe_margin
    
    if min_x >= max_x:
        min_x = site_min_x + boundary_margin + half_w + 0.01
        max_x = site_max_x - boundary_margin - half_w - 0.01
    if min_y >= max_y:
        min_y = site_min_y + boundary_margin + half_h + 0.01
        max_y = site_max_y - boundary_margin - half_h - 0.01

    worker_types = {"office", "rest_area"}
    worker_facilities = [f for f in existing_facilities if f["type"] in worker_types]
    worker_anchor = None
    if facility_type in worker_types and worker_facilities:
        worker_anchor = np.mean([np.array(f["center"]) for f in worker_facilities], axis=0)
    
    for attempt in range(100):
        if worker_anchor is not None and attempt < 70:
            # Offices and rest areas are usually organised as one support zone.
            x = random.gauss(worker_anchor[0], 0.055)
            y = random.gauss(worker_anchor[1], 0.055)
            x = max(min_x, min(x, max_x))
            y = max(min_y, min(y, max_y))
        else:
            x = random.uniform(min_x, max_x)
            y = random.uniform(min_y, max_y)
        
        candidate = {"type": facility_type, "center": (x, y), "rotation": rotation}
        
        if not facility_within_site_boundary(candidate, site_config, margin=0.01):
            continue
        
        has_overlap = any(rectangles_overlap(candidate, existing) 
                         for existing in existing_facilities)
        
        if not has_overlap:
            return candidate
        
        if attempt > 40 and existing_facilities:
            nearest_facility = min(existing_facilities, 
                                 key=lambda f: math.sqrt((f["center"][0] - x)**2 + (f["center"][1] - y)**2))
            nx, ny = nearest_facility["center"]
            
            push_distance = 0.15
            dx = x - nx if abs(x - nx) > 1e-6 else random.uniform(-0.1, 0.1)
            dy = y - ny if abs(y - ny) > 1e-6 else random.uniform(-0.1, 0.1)
            
            dist = math.sqrt(dx*dx + dy*dy)
            if dist > 1e-6:
                dx /= dist
                dy /= dist
                
                push_x = x + dx * push_distance
                push_y = y + dy * push_distance
                
                x = max(min_x, min(push_x, max_x))
                y = max(min_y, min(push_y, max_y))
    
    best_candidate = None
    best_overlap_count = float('inf')
    for _ in range(1000):
        x = random.uniform(min_x, max_x)
        y = random.uniform(min_y, max_y)
        candidate = {"type": facility_type, "center": (x, y), "rotation": rotation}
        if not facility_within_site_boundary(candidate, site_config, margin=0.01):
            continue

        overlap_count = sum(1 for existing in existing_facilities if rectangles_overlap(candidate, existing))
        if overlap_count < best_overlap_count:
            best_candidate = candidate
            best_overlap_count = overlap_count
            if overlap_count == 0:
                return candidate

    if best_candidate is not None:
        return best_candidate

    # Last-resort fallback for severely constrained polygons.
    centroid_x = sum(p[0] for p in site_config.boundary_polygon) / len(site_config.boundary_polygon) if site_config.boundary_polygon else 0.5
    centroid_y = sum(p[1] for p in site_config.boundary_polygon) / len(site_config.boundary_polygon) if site_config.boundary_polygon else 0.5
    return {"type": facility_type, "center": (centroid_x, centroid_y), "rotation": rotation}

def create_targeted_layout(facility_types: List[str], boundary_margin: float,
                          target_spatial_org: float = 0.5, 
                          target_functional_int: float = 0.5,
                          config: SiteConfig = None) -> List[Dict]:
    """Create layout targeting specific behavioral descriptors"""
    layout = []
    site_config = config or SiteConfig(boundary_margin=boundary_margin)
    center = np.array([0.5, 0.5])
    
    for i, facility_type in enumerate(facility_types):
        rotation = random_rotation_for_facility(facility_type)
        probe_facility = {"type": facility_type, "center": (0.5, 0.5), "rotation": rotation}
        w, d = get_facility_dimensions(probe_facility)
        half_w, half_h = w/2, d/2
        
        # For spatial organization: low = centralized, high = distributed
        if target_spatial_org < 0.3:  # Centralized
            radius = 0.1 + random.uniform(0, 0.1)
            angle = random.uniform(0, 2 * math.pi)
            x = 0.5 + radius * math.cos(angle)
            y = 0.5 + radius * math.sin(angle)
        elif target_spatial_org > 0.7:  # Distributed
            # Place facilities near corners/edges
            corners = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
            edges = [(0.2, 0.5), (0.8, 0.5), (0.5, 0.2), (0.5, 0.8)]
            positions = corners + edges
            base_pos = random.choice(positions)
            x = base_pos[0] + random.uniform(-0.05, 0.05)
            y = base_pos[1] + random.uniform(-0.05, 0.05)
        else:  # Moderate distribution
            x = random.uniform(0.3, 0.7)
            y = random.uniform(0.3, 0.7)
        
        # Ensure within boundaries
        site_min_x, site_max_x, site_min_y, site_max_y = get_site_bounds(site_config)
        min_x = site_min_x + boundary_margin + half_w
        max_x = site_max_x - boundary_margin - half_w
        min_y = site_min_y + boundary_margin + half_h
        max_y = site_max_y - boundary_margin - half_h
        
        x = max(min_x, min(x, max_x))
        y = max(min_y, min(y, max_y))

        candidate = {"type": facility_type, "center": (x, y), "rotation": rotation}
        if not facility_within_site_boundary(candidate, site_config, margin=0.01):
            candidate = create_random_facility(facility_type, boundary_margin, layout, site_config)
        
        layout.append(candidate)
    
    # Adjust for worker-operational separation.
    operational_types = {"core", "crane", "storage"}
    support_types = {"office", "rest_area"}
    
    # Separate by function for targeting
    operational_facilities = [f for f in layout if f["type"] in operational_types]
    support_facilities = [f for f in layout if f["type"] in support_types]
    
    if target_functional_int < 0.3 and operational_facilities and support_facilities:
        # Embedded: move support facilities near the operational centroid.
        operational_centroid = np.mean([np.array(f["center"]) for f in operational_facilities], axis=0)
        for facility in support_facilities:
            new_x = operational_centroid[0] + random.uniform(-0.08, 0.08)
            new_y = operational_centroid[1] + random.uniform(-0.08, 0.08)

            w, d = get_facility_dimensions(facility)
            half_w, half_h = w/2, d/2
            site_min_x, site_max_x, site_min_y, site_max_y = get_site_bounds(site_config)
            min_x = site_min_x + boundary_margin + half_w
            max_x = site_max_x - boundary_margin - half_w
            min_y = site_min_y + boundary_margin + half_h
            max_y = site_max_y - boundary_margin - half_h

            old_center = facility["center"]
            facility["center"] = (max(min_x, min(new_x, max_x)),
                                max(min_y, min(new_y, max_y)))
            if not facility_within_site_boundary(facility, site_config, margin=0.01):
                facility["center"] = old_center

    elif target_functional_int > 0.7 and operational_facilities and support_facilities:
        # Segregated: move support facilities toward the opposite side.
        for facility in support_facilities:
            if facility["center"][0] < 0.5:
                new_x = random.uniform(0.6, 0.9)
            else:
                new_x = random.uniform(0.1, 0.4)
            new_y = facility["center"][1] + random.uniform(-0.1, 0.1)
            
            w, d = get_facility_dimensions(facility)
            half_w, half_h = w/2, d/2
            site_min_x, site_max_x, site_min_y, site_max_y = get_site_bounds(site_config)
            min_x = site_min_x + boundary_margin + half_w
            max_x = site_max_x - boundary_margin - half_w
            min_y = site_min_y + boundary_margin + half_h
            max_y = site_max_y - boundary_margin - half_h
            
            old_center = facility["center"]
            facility["center"] = (max(min_x, min(new_x, max_x)),
                                max(min_y, min(new_y, max_y)))
            if not facility_within_site_boundary(facility, site_config, margin=0.01):
                facility["center"] = old_center
    
    return layout

# =============================================================================
# LAYOUT MUTATIONS
# =============================================================================

def mutate_layout(layout: List[Dict], boundary_margin: float,
                 p_mut: float = 0.4, sigma: float = 0.04,
                 config: SiteConfig = None) -> List[Dict]:
    """Enhanced mutation with constraint enforcement"""
    result = []
    site_config = config or SiteConfig(boundary_margin=boundary_margin)
    safety_buffer = FACILITY_CLEARANCE_BUFFER
    effective_margin = boundary_margin + safety_buffer
    
    for i, facility in enumerate(layout):
        if random.random() < p_mut:
            x, y = facility["center"]
            ftype = facility["type"]
            candidate_base = dict(facility)
            if random.random() < 0.2:
                current_rotation = int(candidate_base.get("rotation", 0)) % 180
                candidate_base["rotation"] = 90 if current_rotation == 0 else 0
            w, d = get_facility_dimensions(candidate_base)
            half_w, half_h = w/2, d/2
            
            best_candidate = facility
            best_score = float('inf')
            
            for attempt in range(8):
                new_x = x + random.gauss(0.0, sigma)
                new_y = y + random.gauss(0.0, sigma)
                
                site_min_x, site_max_x, site_min_y, site_max_y = get_site_bounds(site_config)
                min_x = site_min_x + effective_margin + half_w
                max_x = site_max_x - effective_margin - half_w
                min_y = site_min_y + effective_margin + half_h
                max_y = site_max_y - effective_margin - half_h
                
                new_x = max(min_x, min(new_x, max_x))
                new_y = max(min_y, min(new_y, max_y))
                
                candidate = {"type": ftype, "center": (new_x, new_y), "rotation": candidate_base.get("rotation", 0)}
                
                overlap_count = sum(1 for j, other_facility in enumerate(result + layout[i+1:])
                                  if rectangles_overlap(candidate, other_facility))
                
                boundary_violation = 0 if facility_within_site_boundary(candidate, site_config) else 1
                
                score = boundary_violation * 100 + overlap_count
                
                if score < best_score:
                    best_candidate = candidate
                    best_score = score
                    
                    if score == 0:
                        break
            
            result.append(best_candidate)
        else:
            result.append(dict(facility))
    
    return result

def mutate_toward_behavioral_diversity(layout: List[Dict], boundary_margin: float,
                                     current_spatial_org: float, current_functional_int: float,
                                     config: SiteConfig = None) -> List[Dict]:
    """Mutation that pushes solutions toward different behavioral regions"""
    result = []
    site_config = config or SiteConfig(boundary_margin=boundary_margin)
    
    # Determine target behavioral region (opposite quadrant)
    target_spatial_org = 1.0 - current_spatial_org + random.uniform(-0.2, 0.2)
    target_functional_int = 1.0 - current_functional_int + random.uniform(-0.2, 0.2)
    
    target_spatial_org = max(0.0, min(1.0, target_spatial_org))
    target_functional_int = max(0.0, min(1.0, target_functional_int))
    
    for facility in layout:
        if random.random() < 0.4:  # 40% chance to modify each facility
            candidate_base = dict(facility)
            if random.random() < 0.2:
                current_rotation = int(candidate_base.get("rotation", 0)) % 180
                candidate_base["rotation"] = 90 if current_rotation == 0 else 0
            w, d = get_facility_dimensions(candidate_base)
            half_w, half_h = w/2, d/2
            
            # Behavioral-directed position adjustment
            x, y = facility["center"]
            
            # Adjust for spatial organization
            center = np.array([0.5, 0.5])
            current_pos = np.array([x, y])
            
            if target_spatial_org < 0.5:  # Move toward center (centralized)
                direction = center - current_pos
                adjustment = direction * 0.15
            else:  # Move away from center (distributed)
                direction = current_pos - center
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)
                else:
                    direction = np.array([random.choice([-1, 1]), random.choice([-1, 1])])
                adjustment = direction * 0.15
            
            new_x = x + adjustment[0] + random.gauss(0, 0.03)
            new_y = y + adjustment[1] + random.gauss(0, 0.03)
            
            # Boundary constraints
            site_min_x, site_max_x, site_min_y, site_max_y = get_site_bounds(site_config)
            min_x = site_min_x + boundary_margin + half_w
            max_x = site_max_x - boundary_margin - half_w
            min_y = site_min_y + boundary_margin + half_h
            max_y = site_max_y - boundary_margin - half_h
            
            new_x = max(min_x, min(new_x, max_x))
            new_y = max(min_y, min(new_y, max_y))
            
            candidate = {"type": facility["type"], "center": (new_x, new_y), "rotation": candidate_base.get("rotation", 0)}
            if facility_within_site_boundary(candidate, site_config):
                result.append(candidate)
            else:
                result.append(dict(facility))
        else:
            result.append(dict(facility))
    
    return result

# =============================================================================
# CROSSOVER OPERATIONS
# =============================================================================

def crossover_layouts(layout1: List[Dict], layout2: List[Dict], 
                     p_swap: float = 0.5) -> Tuple[List[Dict], List[Dict]]:
    """Enhanced crossover with facility type preservation"""
    assert len(layout1) == len(layout2)
    
    child1, child2 = [], []
    
    for i, (f1, f2) in enumerate(zip(layout1, layout2)):
        if f1["type"] == f2["type"]:
            if random.random() < p_swap:
                child1.append({"type": f1["type"], "center": f2["center"], "rotation": f2.get("rotation", 0)})
                child2.append({"type": f2["type"], "center": f1["center"], "rotation": f1.get("rotation", 0)})
            else:
                child1.append(dict(f1))
                child2.append(dict(f2))
        else:
            child1.append(dict(f1))
            child2.append(dict(f2))
    
    return child1, child2

# =============================================================================
# CONSTRAINT REPAIR
# =============================================================================

def repair_layout_constraints(layout: List[Dict], boundary_margin: float, 
                             entrances: List[Tuple[float, float]], config: SiteConfig) -> List[Dict]:
    """Repair layout to satisfy all constraints with minimal behavioral change"""
    repaired_layout = []
    safety_buffer = FACILITY_CLEARANCE_BUFFER
    effective_margin = boundary_margin + safety_buffer
    
    for i, facility in enumerate(layout):
        ftype = facility["type"]
        w, d = get_facility_dimensions(facility)
        half_w, half_h = w/2, d/2
        
        # Try to repair the current position
        x, y = facility["center"]
        best_position = (x, y)
        best_violations = float('inf')
        
        # Check multiple positions around the current one
        for attempt in range(25):
            if attempt == 0:
                # First try the original position
                test_x, test_y = x, y
            else:
                # Try positions in smaller circles to preserve behavioral characteristics
                angle = random.uniform(0, 2 * math.pi)
                radius = (attempt / 25.0) * 0.15
                test_x = x + radius * math.cos(angle)
                test_y = y + radius * math.sin(angle)
            
            # Ensure within site bounding box before polygon validation
            site_min_x, site_max_x, site_min_y, site_max_y = get_site_bounds(config)
            min_x = site_min_x + effective_margin + half_w
            max_x = site_max_x - effective_margin - half_w
            min_y = site_min_y + effective_margin + half_h
            max_y = site_max_y - effective_margin - half_h
            
            test_x = max(min_x, min(test_x, max_x))
            test_y = max(min_y, min(test_y, max_y))
            
            test_facility = {"type": ftype, "center": (test_x, test_y), "rotation": facility.get("rotation", 0)}
            
            # Count violations with adjusted penalties
            violations = 0
            
            # Check boundary violations
            if not facility_within_site_boundary(test_facility, config):
                violations += 3
            
            # Check overlaps with already placed facilities
            for existing in repaired_layout:
                if rectangles_overlap(test_facility, existing):
                    violations += 5
            
            # Check crane safety
            if ftype == "crane":
                for existing in repaired_layout:
                    if existing["type"] in ("office", "rest_area"):
                        distance = math.sqrt((test_x - existing["center"][0])**2 + 
                                          (test_y - existing["center"][1])**2)
                        if distance < config.crane_safety_distance:
                            violations += 6
            
            # Check entrance clearance
            for entrance in entrances:
                distance = math.sqrt((test_x - entrance[0])**2 + (test_y - entrance[1])**2)
                if distance < config.entrance_clearance:
                    violations += 2
            
            # Update best position if better
            if violations < best_violations:
                best_violations = violations
                best_position = (test_x, test_y)
                
                # If perfect, use it
                if violations == 0:
                    break
        
        # Add the best position found
        repaired_facility = {"type": ftype, "center": best_position, "rotation": facility.get("rotation", 0)}
        repaired_layout.append(repaired_facility)
    
    return repaired_layout
