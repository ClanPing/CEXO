#!/usr/bin/env python3
"""
Configuration Module for Construction Site Layout Optimization
============================================================

Defines all facility specifications, configurations, and data structures
used by both MAP-Elites and NSGA-II algorithms.
"""

from dataclasses import dataclass
import math
from typing import List, Dict, Tuple, Optional

# =============================================================================
# FACILITY SPECIFICATIONS
# =============================================================================

FACILITY_SPECS = {
    "core":      {"w": 0.11, "d": 0.08, "category": "operational", "noise_level": 0.8},
    "crane":     {
        "w": 0.03,
        "d": 0.03,
        "category": "equipment",
        "noise_level": 0.95,
        "danger_radius": 0.14,
        "optimal_reach": 0.20,
        "operating_radius": 0.26,
    },
    "storage":   {"w": 0.04, "d": 0.025, "category": "operational", "noise_level": 0.6},
    "office":    {"w": 0.04, "d": 0.025, "category": "worker", "noise_level": 0.1},
    "rest_area": {"w": 0.035, "d": 0.025, "category": "worker", "noise_level": 0.1}
}

FACILITY_CLEARANCE_BUFFER = 0.006

FACILITY_COLORS = {
    "core":      "#4e79a7",  # Blue
    "crane":     "#e15759",  # Red
    "storage":   "#59a14f",  # Green
    "office":    "#af7aa1",  # Purple
    "rest_area": "#ff9d9a"   # Pink
}

FACILITY_TYPES = list(FACILITY_SPECS.keys())

PRACTICAL_BULLEEN_FACILITY_RANGES = {
    "core": (2, 3),
    "crane": (1, 2),
    "storage": (8, 14),
    "office": (3, 6),
    "rest_area": (3, 6),
}

PRACTICAL_BULLEEN_FACILITIES = [
    "core", "core",
    "crane", "crane",
    "storage", "storage", "storage", "storage", "storage",
    "storage", "storage", "storage", "storage", "storage",
    "office", "office", "office", "office", "office",
    "rest_area", "rest_area", "rest_area", "rest_area",
]

# Approximate normalized boundary traced from the top-down Bulleen Interchange
# example image. Coordinates are in the same [0, 1] space used by the optimizer.
BULLEEN_BOUNDARY_POLYGON = [
    (0.16, 0.36),
    (0.25, 0.30),
    (0.39, 0.27),
    (0.56, 0.21),
    (0.73, 0.18),
    (0.83, 0.26),
    (0.88, 0.40),
    (0.85, 0.55),
    (0.82, 0.70),
    (0.71, 0.82),
    (0.59, 0.76),
    (0.48, 0.68),
    (0.36, 0.71),
    (0.25, 0.74),
    (0.22, 0.61),
    (0.14, 0.51),
]

# Fixed practical-case entrances identified from the Bulleen boundary sketch.
# The optimizer uses points at the middle of each highlighted access opening.
BULLEEN_ENTRANCES = [
    (0.15, 0.43),  # west access
    (0.765, 0.77),  # north-east access
    (0.775, 0.225), # south-east access
]

def _segment_corridor(p1: Tuple[float, float], p2: Tuple[float, float],
                      width: float) -> List[Tuple[float, float]]:
    """Create a thin rectangular corridor around a road centreline segment."""
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length == 0:
        return [p1, p1, p1, p1]

    nx = -dy / length * width / 2
    ny = dx / length * width / 2
    return [
        (x1 + nx, y1 + ny),
        (x2 + nx, y2 + ny),
        (x2 - nx, y2 - ny),
        (x1 - nx, y1 - ny),
    ]


def _road_line_corridors(name: str, points: List[Tuple[float, float]],
                         width: float = 0.010) -> List[Dict]:
    """Convert a thin road centreline polyline into small exclusion corridors."""
    return [
        {
            "name": f"{name}_{i + 1:02d}",
            "polygon": _segment_corridor(points[i], points[i + 1], width),
        }
        for i in range(len(points) - 1)
    ]


# Approximate internal road/access lines from the Bulleen practical example.
# These are intentionally thin centreline buffers, matching the yellow-marked
# road positions rather than treating roads as large placement zones.
BULLEEN_ROAD_CENTERLINES = [
    # Upper-left approach into the internal network.
    [
        (0.145, 0.510),
        (0.205, 0.535),
        (0.285, 0.505),
        (0.380, 0.490),
    ],
    # Lower-left approach into the internal network.
    [
        (0.155, 0.405),
        (0.220, 0.405),
        (0.290, 0.400),
        (0.340, 0.405),
    ],
    # Left internal connector up to the top junction.
    [
        (0.340, 0.405),
        (0.360, 0.435),
        (0.375, 0.490),
        (0.390, 0.580),
        (0.430, 0.680),
    ],
    # Main central east-west road.
    [
        (0.340, 0.405),
        (0.450, 0.420),
        (0.560, 0.420),
        (0.640, 0.410),
        (0.690, 0.390),
    ],
    # Lower loop around the central/lower work zone.
    [
        (0.340, 0.405),
        (0.355, 0.370),
        (0.430, 0.335),
        (0.530, 0.300),
        (0.650, 0.295),
        (0.725, 0.320),
        (0.690, 0.390),
    ],
    # Upper road from top junction toward the right/top area.
    [
        (0.430, 0.680),
        (0.505, 0.655),
        (0.570, 0.620),
        (0.650, 0.590),
        (0.735, 0.570),
        (0.780, 0.600),
        (0.820, 0.700),
    ],
    # Right connector down from the upper road to the main central road.
    [
        (0.820, 0.700),
        (0.790, 0.620),
        (0.730, 0.535),
        (0.700, 0.465),
        (0.690, 0.390),
    ],
    # Short bottom-right access spur.
    [
        (0.690, 0.390),
        (0.735, 0.330),
        (0.840, 0.290),
    ],
]

BULLEEN_ROAD_POLYGONS = [
    corridor
    for line_index, centerline in enumerate(BULLEEN_ROAD_CENTERLINES)
    for corridor in _road_line_corridors(f"road_line_{line_index + 1:02d}", centerline)
]

# =============================================================================
# CONFIGURATION CLASSES
# =============================================================================

@dataclass
class SiteConfig:
    """Configuration for construction site layout generation"""
    facility_count: int = 5
    boundary_margin: float = 0.03
    boundary_polygon: Optional[List[Tuple[float, float]]] = None
    exclusion_zones: Optional[List[Dict]] = None
    fixed_entrances: Optional[List[Tuple[float, float]]] = None
    seed: int = 42
    min_entrances: int = 1
    max_entrances: int = 3
    pareto_size: int = 12
    entrance_clearance: float = 0.08
    crane_safety_distance: float = 0.18

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

@dataclass
class AutoencoderConfig:
    """Configuration for autoencoder-based behavioral descriptor learning"""
    use_learned_descriptors: bool = False  # Whether to use learned vs hand-crafted BDs
    latent_dim: int = 2  # Latent dimension (typically 2 for MAP-Elites grid)
    encoder_hidden: List[int] = None  # Hidden layers for encoder [128, 64, 32]
    decoder_hidden: List[int] = None  # Hidden layers for decoder [32, 64, 128]
    learning_rate: float = 0.001  # Adam optimizer learning rate
    batch_size: int = 32  # Training batch size
    training_epochs: int = 50  # Epochs per training session
    training_frequency: int = 1000  # Train every N iterations
    min_samples_for_training: int = 100  # Minimum samples before first training
    pretrain_iterations: int = 500  # Iterations before switching to learned BDs
    save_model_path: str = None  # Optional path to save trained model
    load_model_path: str = None  # Optional path to load pretrained model
    seed: int = None  # Random seed for reproducibility (uses SiteConfig.seed if None)
    
    def __post_init__(self):
        if self.encoder_hidden is None:
            self.encoder_hidden = [128, 64, 32]
        if self.decoder_hidden is None:
            self.decoder_hidden = [32, 64, 128]

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

def get_practical_bulleen_facility_mix() -> List[str]:
    """Representative fixed module mix for the Bulleen practical example."""
    return PRACTICAL_BULLEEN_FACILITIES[:]

def sample_practical_bulleen_facility_mix(seed: int = None) -> List[str]:
    """Sample a reproducible practical Bulleen module mix from type ranges."""
    import random

    rng = random.Random(seed) if seed is not None else random
    facilities = []
    for facility_type, (min_count, max_count) in PRACTICAL_BULLEEN_FACILITY_RANGES.items():
        facilities.extend([facility_type] * rng.randint(min_count, max_count))
    rng.shuffle(facilities)
    return facilities

def get_bulleen_boundary_polygon() -> List[Tuple[float, float]]:
    """Return the approximate normalized Bulleen practical-case boundary."""
    return BULLEEN_BOUNDARY_POLYGON[:]

def get_bulleen_entrances() -> List[Tuple[float, float]]:
    """Return fixed Bulleen practical-case entrance/access points."""
    return BULLEEN_ENTRANCES[:]

def get_bulleen_road_polygons() -> List[Dict]:
    """Return approximate road/access exclusion polygons for the Bulleen case."""
    return [
        {"name": zone["name"], "polygon": zone["polygon"][:]}
        for zone in BULLEEN_ROAD_POLYGONS
    ]

def get_site_bounds(config: SiteConfig) -> Tuple[float, float, float, float]:
    """Return min_x, max_x, min_y, max_y for rectangular or polygonal site."""
    if config.boundary_polygon:
        xs = [p[0] for p in config.boundary_polygon]
        ys = [p[1] for p in config.boundary_polygon]
        return min(xs), max(xs), min(ys), max(ys)
    return 0.0, 1.0, 0.0, 1.0

def point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test."""
    x, y = point
    inside = False
    j = len(polygon) - 1

    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i

    return inside

def facility_corners(facility: Dict) -> List[Tuple[float, float]]:
    """Return rectangle corners for a facility."""
    x, y = facility["center"]
    w, d = get_facility_dimensions(facility)
    half_w, half_h = w / 2, d / 2
    return [
        (x - half_w, y - half_h),
        (x + half_w, y - half_h),
        (x + half_w, y + half_h),
        (x - half_w, y + half_h),
    ]

def get_facility_dimensions(facility: Dict) -> Tuple[float, float]:
    """Return effective width/depth after discrete 0/90 degree rotation."""
    spec = FACILITY_SPECS[facility["type"]]
    rotation = int(facility.get("rotation", 0)) % 180
    if rotation == 90:
        return spec["d"], spec["w"]
    return spec["w"], spec["d"]

def point_in_rect(point: Tuple[float, float], corners: List[Tuple[float, float]]) -> bool:
    """Check point inside an axis-aligned facility rectangle."""
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    return min(xs) <= point[0] <= max(xs) and min(ys) <= point[1] <= max(ys)

def segments_intersect(a: Tuple[float, float], b: Tuple[float, float],
                       c: Tuple[float, float], d: Tuple[float, float]) -> bool:
    """Check whether two line segments intersect, including collinear contact."""
    def orientation(p, q, r):
        return (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])

    def on_segment(p, q, r):
        return (
            min(p[0], r[0]) - 1e-12 <= q[0] <= max(p[0], r[0]) + 1e-12
            and min(p[1], r[1]) - 1e-12 <= q[1] <= max(p[1], r[1]) + 1e-12
        )

    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)

    if (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0):
        return True

    if abs(o1) < 1e-12 and on_segment(a, c, b):
        return True
    if abs(o2) < 1e-12 and on_segment(a, d, b):
        return True
    if abs(o3) < 1e-12 and on_segment(c, a, d):
        return True
    if abs(o4) < 1e-12 and on_segment(c, b, d):
        return True

    return False

def facility_overlaps_polygon(facility: Dict, polygon: List[Tuple[float, float]]) -> bool:
    """Conservative axis-aligned facility rectangle vs polygon overlap check."""
    corners = facility_corners(facility)
    center = facility["center"]

    if point_in_polygon(center, polygon):
        return True
    if any(point_in_polygon(corner, polygon) for corner in corners):
        return True
    if any(point_in_rect(vertex, corners) for vertex in polygon):
        return True

    rect_edges = list(zip(corners, corners[1:] + corners[:1]))
    polygon_edges = list(zip(polygon, polygon[1:] + polygon[:1]))
    for rect_start, rect_end in rect_edges:
        for poly_start, poly_end in polygon_edges:
            if segments_intersect(rect_start, rect_end, poly_start, poly_end):
                return True

    return False

def facility_within_site_boundary(facility: Dict, config: SiteConfig, margin: float = 0.0) -> bool:
    """Check whether a facility rectangle lies inside the configured site."""
    corners = facility_corners(facility)

    if config.boundary_polygon:
        if margin > 0:
            # Conservative shrink approximation: require corners to be at least
            # margin away from the polygon bounding box as well as inside polygon.
            min_x, max_x, min_y, max_y = get_site_bounds(config)
            for x, y in corners:
                if x < min_x + margin or x > max_x - margin or y < min_y + margin or y > max_y - margin:
                    return False
        inside_boundary = all(point_in_polygon(corner, config.boundary_polygon) for corner in corners)
        if not inside_boundary:
            return False

        for zone in config.exclusion_zones or []:
            if facility_overlaps_polygon(facility, zone["polygon"]):
                return False

        return True

    for x, y in corners:
        if x < config.boundary_margin + margin or x > 1.0 - config.boundary_margin - margin:
            return False
        if y < config.boundary_margin + margin or y > 1.0 - config.boundary_margin - margin:
            return False
    return True

def rectangles_overlap(f1: Dict, f2: Dict) -> bool:
    """Check if two facility rectangles overlap"""
    x1, y1 = f1["center"]
    x2, y2 = f2["center"]
    
    w1, d1 = get_facility_dimensions(f1)
    w2, d2 = get_facility_dimensions(f2)
    
    left1, right1 = x1 - w1/2, x1 + w1/2
    bottom1, top1 = y1 - d1/2, y1 + d1/2
    
    left2, right2 = x2 - w2/2, x2 + w2/2
    bottom2, top2 = y2 - d2/2, y2 + d2/2
    
    buffer = FACILITY_CLEARANCE_BUFFER
    no_overlap = (right1 + buffer <= left2 or right2 + buffer <= left1 or 
                  top1 + buffer <= bottom2 or top2 + buffer <= bottom1)
    
    return not no_overlap

def calculate_overlap_area(f1: Dict, f2: Dict) -> float:
    """Calculate overlap area between two facilities"""
    if not rectangles_overlap(f1, f2):
        return 0.0
    
    x1, y1 = f1["center"]
    x2, y2 = f2["center"]
    
    w1, d1 = get_facility_dimensions(f1)
    w2, d2 = get_facility_dimensions(f2)
    
    left1, right1 = x1 - w1/2, x1 + w1/2
    bottom1, top1 = y1 - d1/2, y1 + d1/2
    
    left2, right2 = x2 - w2/2, x2 + w2/2
    bottom2, top2 = y2 - d2/2, y2 + d2/2
    
    overlap_left = max(left1, left2)
    overlap_right = min(right1, right2)
    overlap_bottom = max(bottom1, bottom2)
    overlap_top = min(top1, top2)
    
    if overlap_right > overlap_left and overlap_top > overlap_bottom:
        return (overlap_right - overlap_left) * (overlap_top - overlap_bottom)
    
    return 0.0
