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

from .config import FACILITY_SPECS, SiteConfig, rectangles_overlap

# =============================================================================
# ENTRANCE GENERATION
# =============================================================================

def generate_random_entrances(config: SiteConfig, seed: int = None) -> List[Tuple[float, float]]:
    """Generate random entrance locations"""
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

def create_random_layout(facility_types: List[str], boundary_margin: float) -> List[Dict]:
    """Create random layout with sequential placement"""
    facilities = []
    
    priority_order = ["office", "rest_area", "core", "storage", "crane"]
    sorted_types = sorted(facility_types, 
                         key=lambda t: priority_order.index(t) if t in priority_order else len(priority_order))
    
    for ftype in sorted_types:
        facility = create_random_facility(ftype, boundary_margin, facilities)
        facilities.append(facility)
    
    return facilities

def create_random_facility(facility_type: str, boundary_margin: float, 
                          existing_facilities: List[Dict] = None) -> Dict:
    """Create random facility with overlap avoidance"""
    if existing_facilities is None:
        existing_facilities = []
    
    spec = FACILITY_SPECS[facility_type]
    half_w, half_h = spec["w"]/2, spec["d"]/2
    
    safety_buffer = 0.02
    safe_margin = boundary_margin + max(half_w, half_h) + safety_buffer
    min_coord = safe_margin
    max_coord = 1.0 - safe_margin
    
    if min_coord >= max_coord:
        min_coord = boundary_margin + half_w + 0.01
        max_coord = 1.0 - boundary_margin - half_w - 0.01
        if min_coord >= max_coord:
            min_coord = 0.1
            max_coord = 0.9
    
    for attempt in range(100):
        x = random.uniform(min_coord, max_coord)
        y = random.uniform(min_coord, max_coord)
        
        candidate = {"type": facility_type, "center": (x, y)}
        
        if (x - half_w < boundary_margin + 0.01 or x + half_w > 1.0 - boundary_margin - 0.01 or
            y - half_h < boundary_margin + 0.01 or y + half_h > 1.0 - boundary_margin - 0.01):
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
                
                x = max(min_coord, min(push_x, max_coord))
                y = max(min_coord, min(push_y, max_coord))
    
    x = max(min_coord, min(random.uniform(min_coord, max_coord), max_coord))
    y = max(min_coord, min(random.uniform(min_coord, max_coord), max_coord))
    
    return {"type": facility_type, "center": (x, y)}

def create_targeted_layout(facility_types: List[str], boundary_margin: float, 
                          target_spatial_org: float = 0.5, 
                          target_functional_int: float = 0.5) -> List[Dict]:
    """Create layout targeting specific behavioral descriptors"""
    layout = []
    center = np.array([0.5, 0.5])
    
    for i, facility_type in enumerate(facility_types):
        spec = FACILITY_SPECS[facility_type]
        half_w, half_h = spec["w"]/2, spec["d"]/2
        
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
        min_x = boundary_margin + half_w
        max_x = 1.0 - boundary_margin - half_w
        min_y = boundary_margin + half_h
        max_y = 1.0 - boundary_margin - half_h
        
        x = max(min_x, min(x, max_x))
        y = max(min_y, min(y, max_y))
        
        layout.append({"type": facility_type, "center": (x, y)})
    
    # Adjust for functional integration
    operational_types = {"core", "crane", "storage"}
    support_types = {"office", "rest_area"}
    
    # Separate by function for targeting
    operational_facilities = [f for f in layout if f["type"] in operational_types]
    support_facilities = [f for f in layout if f["type"] in support_types]
    
    if target_functional_int < 0.3 and operational_facilities and support_facilities:
        # Segregated: move support facilities to opposite side
        for facility in support_facilities:
            if facility["center"][0] < 0.5:
                new_x = random.uniform(0.6, 0.9)
            else:
                new_x = random.uniform(0.1, 0.4)
            new_y = facility["center"][1] + random.uniform(-0.1, 0.1)
            
            spec = FACILITY_SPECS[facility["type"]]
            half_w, half_h = spec["w"]/2, spec["d"]/2
            min_x = boundary_margin + half_w
            max_x = 1.0 - boundary_margin - half_w
            min_y = boundary_margin + half_h
            max_y = 1.0 - boundary_margin - half_h
            
            facility["center"] = (max(min_x, min(new_x, max_x)), 
                                max(min_y, min(new_y, max_y)))
    
    return layout

# =============================================================================
# LAYOUT MUTATIONS
# =============================================================================

def mutate_layout(layout: List[Dict], boundary_margin: float, 
                 p_mut: float = 0.4, sigma: float = 0.04) -> List[Dict]:
    """Enhanced mutation with constraint enforcement"""
    result = []
    safety_buffer = 0.02
    effective_margin = boundary_margin + safety_buffer
    
    for i, facility in enumerate(layout):
        if random.random() < p_mut:
            x, y = facility["center"]
            ftype = facility["type"]
            spec = FACILITY_SPECS[ftype]
            half_w, half_h = spec["w"]/2, spec["d"]/2
            
            best_candidate = facility
            best_score = float('inf')
            
            for attempt in range(8):
                new_x = x + random.gauss(0.0, sigma)
                new_y = y + random.gauss(0.0, sigma)
                
                min_x = effective_margin + half_w
                max_x = 1.0 - effective_margin - half_w
                min_y = effective_margin + half_h
                max_y = 1.0 - effective_margin - half_h
                
                new_x = max(min_x, min(new_x, max_x))
                new_y = max(min_y, min(new_y, max_y))
                
                candidate = {"type": ftype, "center": (new_x, new_y)}
                
                overlap_count = sum(1 for j, other_facility in enumerate(result + layout[i+1:])
                                  if rectangles_overlap(candidate, other_facility))
                
                boundary_violation = 0
                if (new_x - half_w < effective_margin or new_x + half_w > 1.0 - effective_margin or
                    new_y - half_h < effective_margin or new_y + half_h > 1.0 - effective_margin):
                    boundary_violation = 1
                
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
                                     current_spatial_org: float, current_functional_int: float) -> List[Dict]:
    """Mutation that pushes solutions toward different behavioral regions"""
    result = []
    
    # Determine target behavioral region (opposite quadrant)
    target_spatial_org = 1.0 - current_spatial_org + random.uniform(-0.2, 0.2)
    target_functional_int = 1.0 - current_functional_int + random.uniform(-0.2, 0.2)
    
    target_spatial_org = max(0.0, min(1.0, target_spatial_org))
    target_functional_int = max(0.0, min(1.0, target_functional_int))
    
    for facility in layout:
        if random.random() < 0.4:  # 40% chance to modify each facility
            spec = FACILITY_SPECS[facility["type"]]
            half_w, half_h = spec["w"]/2, spec["d"]/2
            
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
            min_x = boundary_margin + half_w
            max_x = 1.0 - boundary_margin - half_w
            min_y = boundary_margin + half_h
            max_y = 1.0 - boundary_margin - half_h
            
            new_x = max(min_x, min(new_x, max_x))
            new_y = max(min_y, min(new_y, max_y))
            
            result.append({"type": facility["type"], "center": (new_x, new_y)})
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
                child1.append({"type": f1["type"], "center": f2["center"]})
                child2.append({"type": f2["type"], "center": f1["center"]})
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
    safety_buffer = 0.01
    effective_margin = boundary_margin + safety_buffer
    
    for i, facility in enumerate(layout):
        ftype = facility["type"]
        spec = FACILITY_SPECS[ftype]
        half_w, half_h = spec["w"]/2, spec["d"]/2
        
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
            
            # Ensure within boundaries
            min_x = effective_margin + half_w
            max_x = 1.0 - effective_margin - half_w
            min_y = effective_margin + half_h
            max_y = 1.0 - effective_margin - half_h
            
            test_x = max(min_x, min(test_x, max_x))
            test_y = max(min_y, min(test_y, max_y))
            
            test_facility = {"type": ftype, "center": (test_x, test_y)}
            
            # Count violations with adjusted penalties
            violations = 0
            
            # Check boundary violations
            if (test_x - half_w < effective_margin or test_x + half_w > 1.0 - effective_margin or
                test_y - half_h < effective_margin or test_y + half_h > 1.0 - effective_margin):
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
        repaired_facility = {"type": ftype, "center": best_position}
        repaired_layout.append(repaired_facility)
    
    return repaired_layout