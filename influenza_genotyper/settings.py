"""
Configuration Settings
======================
Central configuration for the influenza k-mer genotyping system.
"""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

SEGMENTS = ["PB2", "PB1", "PA", "HA", "NP", "NA", "M", "NS"]
SUBTYPES = ["H1N1pdm09", "H3N2"]

SEGMENT_LENGTH_RANGES = {
    "PB2": (2100, 2400), "PB1": (2100, 2400), "PA": (2100, 2300),
    "HA": (1600, 1800), "NP": (1400, 1600), "NA": (1350, 1470),
    "M": (900, 1050), "NS": (800, 920),
}


def jaccard_to_ani(jaccard: float, k: int) -> float:
    """Convert a Jaccard similarity to an ANI estimate (Mash formula).

    ``ANI = 1 + (1/k) * ln(2J / (1 + J))``, clamped to ``[0, 1]``.  Used to seed
    and reason about ANI thresholds from the legacy Jaccard thresholds.
    """
    if jaccard <= 0.0:
        return 0.0
    if jaccard >= 1.0:
        return 1.0
    return max(0.0, min(1.0, 1.0 + (1.0 / k) * math.log(2.0 * jaccard / (1.0 + jaccard))))


def ani_to_jaccard(ani: float, k: int) -> float:
    """Inverse of :func:`jaccard_to_ani`: ANI estimate -> Jaccard similarity.

    From ``ANI = 1 + (1/k) ln(2J/(1+J))``: let ``x = exp(k (ANI - 1))``; then
    ``J = x / (2 - x)``, clamped to ``[0, 1]``.  This is how an ANI cut height is
    turned into the Jaccard cut the linkage step actually operates on.
    """
    if ani >= 1.0:
        return 1.0
    if ani <= 0.0:
        return 0.0
    x = math.exp(k * (ani - 1.0))
    return max(0.0, min(1.0, x / (2.0 - x)))


@dataclass
class KmerConfig:
    default_k: int = 21
    segment_k: Dict[str, int] = field(default_factory=lambda: {
        "PB2": 21, "PB1": 21, "PA": 21, "HA": 21,
        "NP": 19, "NA": 19, "M": 17, "NS": 17,
    })
    num_hashes: int = 1024
    # Raised from 256 → 1024 (v4).  At 256 hashes the Jaccard standard error
    # near J=0.95 is ~0.014, roughly a third of the same-cluster distance cut
    # (0.05; 0.04 for HA/NA), so borderline pairs flipped cluster membership
    # stochastically and Stage-2 within-clade distance signals sat inside the
    # estimator noise.  1024 hashes cuts that SE to ~0.007.
    #
    # MIGRATION: signatures of different dimensions are NOT comparable
    # (jaccard_similarity raises on mismatch).  Any existing k-mer DB built at
    # 256 must be re-extracted (a fresh run()/run_recluster()) before mixing
    # with 1024-dim signatures.  To keep an existing DB as-is, set this back
    # to 256 explicitly.
    hash_seed: int = 42
    canonical: bool = True

    hash_backend: str = "mmh3"
    """Hash backend used to build MinHash signatures.

    Pinned to ``mmh3`` (MurmurHash3, x64 128-bit variant).  This is a
    *comparability* parameter, not just a performance one: signatures built
    with a different hash function are NOT comparable even at identical
    num_hashes / hash_seed / k.  It is recorded in the database signature
    fingerprint and validated on every run.  mmh3 is a hard install
    requirement; there is no silent fallback.
    """

    def get_k(self, segment: str) -> int:
        return self.segment_k.get(segment, self.default_k)

    def signature_fingerprint(self) -> Dict[str, object]:
        """Parameters that determine MinHash signature comparability.

        Two signatures are comparable iff they were built with identical values
        here. This is the single source of truth for the fingerprint recorded
        in the database on first write and validated on every run.
        ``hash_backend`` is included even though it is pinned to mmh3, so a
        future change is caught rather than silently assumed safe.
        """
        return {
            "num_hashes": self.num_hashes,
            "hash_seed": self.hash_seed,
            "canonical": self.canonical,
            "hash_backend": self.hash_backend,
            "segment_k": {seg: self.get_k(seg) for seg in SEGMENTS},
        }


@dataclass
class ClusteringConfig:
    same_cluster_threshold: float = 0.95
    related_cluster_threshold: float = 0.85
    segment_thresholds: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "PB2": {"same": 0.98, "related": 0.85},
        "PB1": {"same": 0.95, "related": 0.85},
        "PA":  {"same": 0.95, "related": 0.85},
        "HA":  {"same": 0.92, "related": 0.80},
        "NP":  {"same": 0.95, "related": 0.85},
        "NA":  {"same": 0.92, "related": 0.80},
        "M":   {"same": 0.96, "related": 0.88},
        "NS":  {"same": 0.96, "related": 0.88},
    })
    subtype_adjustments: Dict[str, float] = field(default_factory=lambda: {
        "H1N1pdm09": 0.0, "H3N2": -0.02,
    })

    # ── ANI thresholds (forward-looking canonical knobs) ──────────────────
    # Mechanically seeded from the Jaccard `segment_thresholds` above via
    # jaccard_to_ani(J, k) using each segment's own k (PB2/PB1/PA/HA=21,
    # NP/NA=19, M/NS=17).  These are the values to tweak going forward; the
    # Jaccard table is retained only because the current clustering path still
    # reads it, and is retired once formation/acceptance are rewired to ANI.
    segment_ani_thresholds: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "PB2": {"same": 0.99952, "related": 0.99597},
        "PB1": {"same": 0.99876, "related": 0.99597},
        "PA":  {"same": 0.99876, "related": 0.99597},
        "HA":  {"same": 0.99797, "related": 0.99439},
        "NP":  {"same": 0.99863, "related": 0.99555},
        "NA":  {"same": 0.99776, "related": 0.99380},
        "M":   {"same": 0.99879, "related": 0.99612},
        "NS":  {"same": 0.99879, "related": 0.99612},
    })
    subtype_ani_adjustments: Dict[str, float] = field(default_factory=lambda: {
        "H1N1pdm09": 0.0,
        # The legacy -0.02 Jaccard shift maps to a per-segment ANI delta of
        # -0.00050..-0.00064 (the mapping is nonlinear); -0.00057 is the
        # representative mean.  Approximate by construction — split per segment
        # later if a clade needs it.
        "H3N2": -0.00057,
    })
    same_ani_threshold: float = 0.998
    related_ani_threshold: float = 0.995
    min_cluster_size: int = 10
    max_cluster_diameter: float = 0.10
    linkage_method: str = "average"
    min_segments_for_constellation: int = 6
    """Minimum number of segments that must have an assigned allele for a
    constellation identifier to be issued.  Isolates with fewer assigned
    segments receive ``None`` instead of a constellation ID.

    The default of 6 (out of 8) mirrors the original hardcoded threshold and
    allows for occasional missing segments in real-world surveillance data
    while still requiring a majority of the genome to be typed.  Raise to 8
    for strict whole-genome typing; lower (with caution) for datasets where
    specific segments are systematically absent.
    """

    # ── Development mode ──────────────────────────────────────────────
    dev_mode: bool = False
    """When True, lowers ``min_cluster_size`` to ``DEV_MIN_CLUSTER_SIZE``
    so that small pilot datasets (< ~50 isolates) produce clusters rather
    than all-orphan results.

    **Never enable in production** — small clusters are statistically
    unreliable and will produce noisy genotype assignments.
    """
    DEV_MIN_CLUSTER_SIZE: int = 2

    @property
    def effective_min_cluster_size(self) -> int:
        """The min_cluster_size actually used by the clustering engine.

        Returns ``DEV_MIN_CLUSTER_SIZE`` when ``dev_mode=True``, otherwise
        returns ``min_cluster_size``.
        """
        return self.DEV_MIN_CLUSTER_SIZE if self.dev_mode else self.min_cluster_size

    def get_threshold(self, segment: str, subtype: str, level: str = "same") -> float:
        base = self.segment_thresholds.get(segment, {}).get(
            level, self.same_cluster_threshold if level == "same" else self.related_cluster_threshold
        )
        return base + self.subtype_adjustments.get(subtype, 0.0)

    def get_ani_threshold(self, segment: str, subtype: str, level: str = "same") -> float:
        """ANI threshold for a segment/subtype/level — the ANI-space mirror of
        :meth:`get_threshold`.

        This is the forward-looking accessor; cluster formation (via
        :func:`ani_to_jaccard`) and acceptance will read it once rewired.
        """
        base = self.segment_ani_thresholds.get(segment, {}).get(
            level,
            self.same_ani_threshold if level == "same" else self.related_ani_threshold,
        )
        return base + self.subtype_ani_adjustments.get(subtype, 0.0)


@dataclass
class ReassortmentConfig:
    """Configuration for the two-stage reassortment detector.

    Stage 1 (linkage disequilibrium testing):
        significance_level : alpha for Fisher's exact + Stouffer's Z
            combined p-value test.  Default 0.05.
        bonferroni : whether to apply Bonferroni correction across segments.

    Stage 2 (within-cluster distance refinement):
        distance_zscore_threshold : z-score cutoff for distance outliers.
        confidence_decay : z-score → confidence mapping: 1 - exp(-decay * z).
        min_constellation_mates : minimum group size for distance screening.
        min_segment_comparisons : minimum pairwise distances per segment.
        max_baseline_pairs : cap on comparisons per constellation group.
        baseline_std_floor : minimum std to prevent extreme z-scores.

    Co-reassortment coherence:
        coherence_boost : multiplier for biologically coherent reassortment.
        coherence_penalty : base multiplier for split co-reassortment groups.
        co_reassortment_groups : biological segment groups (overridable).

    Shared:
        min_discordant_segments : minimum segments flagged to report.
        min_confidence : minimum confidence to report an event.

    Legacy (backward compatibility):
        discordance_threshold : original heuristic threshold.  Retained for
            config files that still set it; NOT used by the Fisher's test path.
    """
    # Stage 1
    significance_level: float = 0.05
    bonferroni: bool = True

    # ── Stage 1 p-value combination & multiplicity (v4) ──────────────────
    combination_method: str = "cauchy"
    """How per-pair Fisher p-values are combined into a per-segment p-value.

    "cauchy"  — Cauchy combination test (ACAT).  Asymptotically valid under
                *arbitrary* dependence among the pair tests, which is the
                situation here (pair tests share segment marginals).  This
                replaces the anticonservatism of equal-weight Stouffer.
    "stouffer" — legacy equal-weight Stouffer's Z (assumes independence;
                 over-states significance).  Retained for reproducibility.
    """

    multiplicity: str = "global_fdr_bh"
    """Multiple-testing control across the whole screened population.

    "global_fdr_bh"  — Benjamini-Hochberg FDR over every (profile, segment)
                       combined p-value.  Controls the false-discovery rate
                       across all isolates, not just within one genome.
    "global_fdr_by"  — Benjamini-Yekutieli (dependence-safe, conservative).
    "per_profile_bonferroni" — legacy: Bonferroni across the segments of a
                       single profile only (no cross-profile control; ~alpha
                       false-positive floor per profile).
    """

    # ── Permutation gating (v4) ──────────────────────────────────────────
    gate_on_permutation: bool = False
    """When True, detect_and_validate() keeps a Stage 1 event only if the
    permutation test confirms it.  Off by default because the permutation
    pass is O(events × permutations × profiles); turn on for reported rates.
    """
    permutation_n: int = 1000
    permutation_seed: int = 42

    # Stage 2
    distance_zscore_threshold: float = 2.0
    confidence_decay: float = 0.3
    min_constellation_mates: int = 3
    min_segment_comparisons: int = 3
    max_baseline_pairs: int = 500
    baseline_std_floor: float = 0.01

    stage2_baseline: str = "per_constellation"
    """Baseline used to z-score a candidate's within-constellation distance.

    "per_constellation" — leave-one-out: the candidate's mean distance to its
                          mates is scored against the *mate-to-mate* distance
                          distribution of its own constellation, so a clade's
                          intrinsic diversity sets its own bar.  Falls back to
                          the global baseline when a constellation is too small.
    "global" — legacy: one pooled per-segment (mean, std) across all
               constellations (mis-calibrates for clades whose diversity
               differs from the population average).
    """

    # Coherence
    coherence_boost: float = 1.15
    coherence_penalty: float = 0.80
    co_reassortment_groups: Optional[List] = None

    # Shared
    min_discordant_segments: int = 1
    min_confidence: float = 0.7

    # Legacy — kept so old config files don't break
    discordance_threshold: float = 0.15


@dataclass
class DatabaseConfig:
    db_type: str = "sqlite"
    sqlite_path: Path = Path("data/influenza_genotyper.db")
    pg_host: str = "localhost"
    pg_port: int = 5432
    pg_database: str = "influenza_genotyper"
    pg_user: str = ""
    pg_password: str = ""

    def get_connection_string(self) -> str:
        if self.db_type == "sqlite":
            return f"sqlite:///{self.sqlite_path}"
        return f"postgresql://{self.pg_user}:{self.pg_password}@{self.pg_host}:{self.pg_port}/{self.pg_database}"


@dataclass
class PerformanceConfig:
    max_memory_gb: float = 4.0
    batch_size: int = 100
    num_workers: int = 4


@dataclass
class GenotyperConfig:
    kmer: KmerConfig = field(default_factory=KmerConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    reassortment: ReassortmentConfig = field(default_factory=ReassortmentConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    pipeline_version: str = "0.1.0"

    @classmethod
    def default(cls) -> "GenotyperConfig":
        return cls()
