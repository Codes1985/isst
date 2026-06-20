"""
config.py — Re-exports all configuration symbols from settings.py.

All internal modules import from ``..config`` (core subpackage) or
``.config`` (pipeline level).  This shim allows settings.py to be the
single source of truth while satisfying those import paths.
"""

from .settings import (  # noqa: F401
    GenotyperConfig,
    KmerConfig,
    ClusteringConfig,
    ReassortmentConfig,
    DatabaseConfig,
    PerformanceConfig,
    SEGMENTS,
    SUBTYPES,
    SEGMENT_LENGTH_RANGES,
)
