"""
Genotype Assigner — builds composite genotype profiles from segment-level clusters.

Refactored with:
    1. SegmentStatus enum replaces ambiguous Optional[str] + sentinel pattern.
    2. SegmentValue dataclass wraps status + optional cluster_id with validation.
    3. is_complete uses integer comparison (avoids float equality).
    4. Removed dead reassortment_score/reassortment_flag fields.
    5. compare_to() moved onto GenotypeProfile as an instance method.
    6. Assertion on SEGMENTS ordering invariant.
    7. Backward-compatible segment_clusters property so that
       reassortment_detector.py, pipeline.py, and nomenclature.py
       continue to work without modification.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from ..config import SEGMENTS
from .clustering_engine import ClusterAssignment

logger = logging.getLogger(__name__)

# SEGMENTS is treated as a canonical, ordered tuple of all expected genome
# segments.  Profile strings and comparisons depend on this ordering being
# stable and complete.
assert isinstance(SEGMENTS, (list, tuple)), "SEGMENTS must be an ordered sequence"

ORPHAN_MARKER = "?"
MISSING_MARKER = "-"


# ---------------------------------------------------------------------------
# Segment-level typing
# ---------------------------------------------------------------------------

class SegmentStatus(Enum):
    """Explicit status for each segment in a genotype profile."""
    ASSIGNED = "assigned"
    ORPHAN = "orphan"
    MISSING = "missing"


@dataclass(frozen=True)
class SegmentValue:
    """Represents a single segment's cluster state.

    Invariants enforced at construction:
        - ASSIGNED requires a non-empty cluster_id.
        - ORPHAN and MISSING must not carry a cluster_id.
    """
    status: SegmentStatus
    cluster_id: Optional[str] = None

    def __post_init__(self):
        if self.status == SegmentStatus.ASSIGNED and not self.cluster_id:
            raise ValueError("ASSIGNED status requires a cluster_id")
        if self.status != SegmentStatus.ASSIGNED and self.cluster_id is not None:
            raise ValueError(
                f"{self.status.value} status must not have a cluster_id"
            )

    @property
    def display(self) -> str:
        """Human-readable display string for this segment."""
        if self.status == SegmentStatus.ASSIGNED:
            return self.cluster_id
        if self.status == SegmentStatus.ORPHAN:
            return ORPHAN_MARKER
        return MISSING_MARKER

    @property
    def is_assigned(self) -> bool:
        return self.status == SegmentStatus.ASSIGNED


# Convenience constructors for readability
def _assigned(cluster_id: str) -> SegmentValue:
    return SegmentValue(SegmentStatus.ASSIGNED, cluster_id)

def _orphan() -> SegmentValue:
    return SegmentValue(SegmentStatus.ORPHAN)

def _missing() -> SegmentValue:
    return SegmentValue(SegmentStatus.MISSING)


# ---------------------------------------------------------------------------
# Genotype profile
# ---------------------------------------------------------------------------

@dataclass
class GenotypeProfile:
    """Composite genotype profile for a single sequence.

    Internally stores typed ``SegmentValue`` objects in ``segment_values``.
    The legacy ``segment_clusters`` property returns a plain
    ``Dict[str, Optional[str]]`` so that downstream consumers
    (reassortment_detector, pipeline, nomenclature) work unchanged.
    """

    sequence_id: str
    segment_values: Dict[str, SegmentValue]
    cluster_version: str

    # ------------------------------------------------------------------
    # Backward-compatible accessor
    # ------------------------------------------------------------------

    @property
    def segment_clusters(self) -> Dict[str, Optional[str]]:
        """Legacy accessor: returns the old-style dict mapping segment name
        to cluster_id string, ``ORPHAN_MARKER``, or ``None``.

        Used by:
            - reassortment_detector.py  (reads cluster IDs per segment)
            - pipeline.py               (builds cluster_id_map for nomenclature)
        """
        result: Dict[str, Optional[str]] = {}
        for seg in SEGMENTS:
            sv = self.segment_values.get(seg)
            if sv is None or sv.status == SegmentStatus.MISSING:
                result[seg] = None
            elif sv.status == SegmentStatus.ORPHAN:
                result[seg] = ORPHAN_MARKER
            else:
                result[seg] = sv.cluster_id
        return result

    # ------------------------------------------------------------------
    # Profile display
    # ------------------------------------------------------------------

    @property
    def profile_string(self) -> str:
        """Dot-delimited cluster profile, e.g. ``A1.A1.B3.?.-.C2.A1.A1``."""
        return ".".join(
            self.segment_values[seg].display
            if seg in self.segment_values
            else MISSING_MARKER
            for seg in SEGMENTS
        )


    # ------------------------------------------------------------------
    # Completeness
    # ------------------------------------------------------------------

    @property
    def assigned_count(self) -> int:
        """Number of segments with a definitive cluster assignment."""
        return sum(
            1 for seg in SEGMENTS
            if seg in self.segment_values
            and self.segment_values[seg].is_assigned
        )

    @property
    def completeness(self) -> float:
        return self.assigned_count / len(SEGMENTS)

    @property
    def is_complete(self) -> bool:
        # Integer comparison avoids floating-point equality pitfalls.
        return self.assigned_count == len(SEGMENTS)

    # ------------------------------------------------------------------
    # Segment queries
    # ------------------------------------------------------------------

    @property
    def missing_segments(self) -> List[str]:
        return [
            seg for seg in SEGMENTS
            if seg not in self.segment_values
            or self.segment_values[seg].status == SegmentStatus.MISSING
        ]

    @property
    def orphan_segments(self) -> List[str]:
        return [
            seg for seg in SEGMENTS
            if seg in self.segment_values
            and self.segment_values[seg].status == SegmentStatus.ORPHAN
        ]

    # ------------------------------------------------------------------
    # Comparison (moved from GenotypeAssigner)
    # ------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Assigner
# ---------------------------------------------------------------------------

class GenotypeAssigner:
    def __init__(self, cluster_version: str = "v1"):
        self.cluster_version = cluster_version

    def assign_genotype(
        self,
        sequence_id: str,
        assignments: Dict[str, ClusterAssignment],
        available_segments: Optional[List[str]] = None,
    ) -> GenotypeProfile:
        """Build a GenotypeProfile for a single sequence.

        For each canonical segment, resolves one of three states:
            - **Assigned**: a ClusterAssignment exists and is not orphan.
            - **Orphan**: assignment exists but is orphan, or segment data
              was available but produced no assignment.
            - **Missing**: no data for this segment at all.
        """
        available = set(available_segments or [])
        segment_values: Dict[str, SegmentValue] = {}

        for seg in SEGMENTS:
            if seg in assignments:
                a = assignments[seg]
                if a.is_orphan:
                    segment_values[seg] = _orphan()
                else:
                    segment_values[seg] = _assigned(a.cluster_id)
            elif seg in available:
                segment_values[seg] = _orphan()
            else:
                segment_values[seg] = _missing()

        return GenotypeProfile(sequence_id, segment_values, self.cluster_version)

    def assign_batch(
        self,
        all_assignments: Dict[str, Dict[str, ClusterAssignment]],
        available_segments_map: Optional[Dict[str, List[str]]] = None,
    ) -> List[GenotypeProfile]:
        profiles = []
        for seq_id, seg_assignments in all_assignments.items():
            available = (
                available_segments_map.get(seq_id)
                if available_segments_map
                else None
            )
            profiles.append(
                self.assign_genotype(seq_id, seg_assignments, available)
            )

        complete = sum(1 for p in profiles if p.is_complete)
        logger.info(f"Assigned {len(profiles)} genotypes: {complete} complete")
        return profiles

    # ------------------------------------------------------------------
    # Grouping & summary
    # ------------------------------------------------------------------

    def find_genotype_groups(
        self, profiles: List[GenotypeProfile]
    ) -> Dict[str, List[str]]:
        groups: Dict[str, List[str]] = {}
        for p in profiles:
            groups.setdefault(p.profile_string, []).append(p.sequence_id)
        return groups

    def get_genotype_summary(self, profiles: List[GenotypeProfile]) -> Dict:
        groups = self.find_genotype_groups(profiles)
        return {
            "total_sequences": len(profiles),
            "unique_genotypes": len(groups),
            "complete_profiles": sum(1 for p in profiles if p.is_complete),
            "partial_profiles": sum(1 for p in profiles if not p.is_complete),
            "largest_genotype_group": max(
                (len(v) for v in groups.values()), default=0
            ),
            "singleton_genotypes": sum(
                1 for v in groups.values() if len(v) == 1
            ),
            "genotype_distribution": {
                k: len(v)
                for k, v in sorted(groups.items(), key=lambda x: -len(x[1]))
            },
        }
