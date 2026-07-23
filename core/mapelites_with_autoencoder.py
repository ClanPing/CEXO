#!/usr/bin/env python3
"""Compatibility wrapper for the CEXO implementation.

The old project used ``MapElitesWithAutoencoder`` for MAP-Elites plus learned
descriptors. The proposed method now lives in ``cexo_algorithm`` as
``CEXOOptimizer`` because it also includes per-cell NSGA-II/Pareto selection.
"""

from .cexo_algorithm import CEXOArchive, CEXOOptimizer


class MapElitesWithAutoencoder(CEXOOptimizer):
    """Backward-compatible alias for older imports."""


__all__ = ["CEXOArchive", "CEXOOptimizer", "MapElitesWithAutoencoder"]
