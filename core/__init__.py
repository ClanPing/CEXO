"""Core package for CSLP-Elites optimization algorithms and utilities."""

from .config import (
    FACILITY_SPECS,
    FACILITY_COLORS,
    FACILITY_TYPES,
    SiteConfig,
    Individual,
    MapElitesConfig,
    NSGA2Config,
    generate_facility_mix,
    rectangles_overlap,
    calculate_overlap_area,
)

__all__ = [
    "FACILITY_SPECS",
    "FACILITY_COLORS",
    "FACILITY_TYPES",
    "SiteConfig",
    "Individual",
    "MapElitesConfig",
    "NSGA2Config",
    "generate_facility_mix",
    "rectangles_overlap",
    "calculate_overlap_area",
]
