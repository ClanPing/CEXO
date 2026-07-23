#!/usr/bin/env python3
"""Compatibility module for the historical optimizer import path."""

from .cexo_algorithm import (
    CEXOArchive,
    CEXOOptimizer,
    MapElitesArchive,
    MapElitesNSGA2Optimizer,
    ParetoFront,
    evaluate_mapelites_performance,
)

__all__ = [
    "ParetoFront",
    "CEXOArchive",
    "CEXOOptimizer",
    "MapElitesArchive",
    "MapElitesNSGA2Optimizer",
    "evaluate_mapelites_performance",
]
