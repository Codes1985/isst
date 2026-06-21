"""
Reassortment Detector — detects reassortment via segment discordance.

Detection is performed in three stages:

  Stage 0 (allele subtype discordance — deterministic):
      Reads the nomenclature allele assignments produced by NomenclatureManager.
      Extracts the subtype number encoded in each allele name (e.g. the '3'
      in 'PB1.3.0001') and compares it to the consensus subtype across all
      segments.  Any segment whose allele subtype differs from the genome
      majority is flagged as discordant with confidence 1.0.

      This is the primary path for catching cross-subtype reassortment in
      sequences that are fully or partially orphaned at the cluster level —
      including isolates where the reassorted segment never received a cluster
      assignment but did receive a cross-subtype allele name via the
      NomenclatureManager Stage 2b centroid search.

      Stage 0 requires the ``nomenclature`` dict to be passed to
      ``detect_reassortments()``.  When absent, Stage 0 is skipped and
      behaviour is identical to the original two-stage pipeline.

  Stage 1 (cluster discordance — linkage disequilibrium testing):
      For each segment pair, constructs a 2×2 contingency table and runs
      Fisher's exact test to determine whether the observed cluster combination
      departs from independence.  Per-pair p-values are combined across all
      partners for a given segment using Stouffer's Z method, yielding a
      single combined p-value per segment.  Segments whose combined p-value
      falls below the significance threshold (after optional Bonferroni
      correction) are flagged as discordant.

      Only sequences not already flagged by Stage 0 are screened here.

  Stage 2 (within-cluster distance refinement):
      For profiles that passed Stage 1, uses MinHash Jaccard distances to
      detect *within-cluster* reassortment — cases where a segment belongs
      to the right cluster but is genetically distant from constellation-mates.

Post-filter (co-reassortment coherence):
      Adjusts confidence based on whether discordant segments form biologically
      plausible reassortment units (e.g. the polymerase complex PB2/PB1/PA).

Optional validation mode (permutation test):
      For flagged events, shuffles cluster assignments across all profiles and
      rebuilds population statistics per permutation, producing assumption-free
      empirical p-values that account for correlated pair-tests.
"""

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, NamedTuple, Optional, Set, Tuple

import numpy as np
from scipy import stats as scipy_stats

from ..config import ReassortmentConfig, KmerConfig, SEGMENTS
from .genotype_assigner import GenotypeProfile, ORPHAN_MARKER
from .kmer_extractor import MinHashSignature

logger = logging.getLogger(__name__)

# ======================================================================
# Biological constants
# ======================================================================

SURFACE_SEGMENTS: Set[str] = {"HA", "NA"}
INTERNAL_SEGMENTS: Set[str] = {"PB2", "PB1", "PA", "NP", "M", "NS"}

# Segment groups that tend to reassort together due to functional coupling
# or packaging constraints.  Checked in order; each discordant segment is
# matched to the *first* group it belongs to, avoiding double-counting.
DEFAULT_CO_REASSORTMENT_GROUPS: List[FrozenSet[str]] = [
    frozenset({"PB2", "PB1", "PA"}),   # polymerase complex
    frozenset({"HA", "NA"}),            # surface glycoprotein pair
    frozenset({"M"}),                   # often reassorts independently
    frozenset({"NS"}),                  # often reassorts independently
    frozenset({"NP"}),                  # often reassorts independently
]

# ======================================================================
# Data structures
# ======================================================================


class PatternMap(NamedTuple):
    """Co-occurrence statistics computed from the dataset."""
    pair_counts: Dict[Tuple[str, str], Counter]
    marginal_counts: Dict[Tuple[str, str], Dict[str, Counter]]
    total_profiles: int


@dataclass
class ReassortmentEvent:
    """A single detected reassortment event.

    Backward-compatible with the original dataclass — ``detection_stage``
    and ``evidence`` are additive fields with safe defaults.
    """
    sequence_id: str
    discordant_segments: List[str]
    confidence: float
    event_type: str
    description: str
    detection_stage: int = 1
    evidence: Dict = field(default_factory=dict)


@dataclass
class ReassortmentReport:
    """Summary of reassortment detection across a dataset."""
    total_sequences: int
    sequences_analyzed: int
    events: List[ReassortmentEvent]
    flagged_sequences: int = 0

    @property
    def reassortment_rate(self) -> float:
        return (
            self.flagged_sequences / self.sequences_analyzed
            if self.sequences_analyzed
            else 0.0
        )


@dataclass
class PermutationResult:
    """Result of permutation-based validation for a single event."""
    sequence_id: str
    segment_empirical_pvalues: Dict[str, float]
    segment_stouffer_pvalues: Dict[str, float]
    concordant: bool
    n_permutations: int


# ======================================================================
# Helpers
# ======================================================================


def _sorted_pair(a: str, b: str) -> Tuple[str, str]:
    """Canonical ordered pair key — used everywhere to ensure consistency."""
    return (a, b) if a <= b else (b, a)


def _is_assigned(cluster_id: Optional[str]) -> bool:
    """Check if a cluster ID represents a valid assignment (not orphan/missing)."""
    return bool(cluster_id) and cluster_id != ORPHAN_MARKER


def _classify_event_type(discordant: List[str]) -> str:
    """Classify a reassortment event based on which segments are discordant."""
    disc_set = set(discordant)
    if disc_set.issubset(SURFACE_SEGMENTS) or disc_set.issubset(INTERNAL_SEGMENTS):
        return "surface_internal"
    if len(discordant) == 1:
        return "single_segment"
    return "complex"


def _co_reassortment_coherence(
    discordant: List[str],
    groups: List[FrozenSet[str]],
    coherence_boost: float,
    coherence_penalty: float,
) -> float:
    """Return a confidence multiplier based on biological plausibility.

    Each discordant segment is matched to the *first* group it belongs to,
    preventing double-counting when groups overlap.
    """
    disc_set = set(discordant)
    if not disc_set:
        return 1.0

    claimed: Set[str] = set()
    multiplier = 1.0

    for group in groups:
        unclaimed_overlap = (disc_set & group) - claimed
        if not unclaimed_overlap:
            continue
        claimed |= unclaimed_overlap

        if len(group) == 1:
            # Singleton group — always coherent
            multiplier *= coherence_boost
        elif unclaimed_overlap == group:
            # Full group is discordant — biologically coherent
            multiplier *= coherence_boost
        else:
            # Partial group — penalize proportionally
            fraction_present = len(unclaimed_overlap) / len(group)
            penalty = coherence_penalty + (1.0 - coherence_penalty) * fraction_present
            multiplier *= penalty

    return multiplier


def _clamp_confidence(value: float) -> float:
    """Clamp a confidence value to [0.0, 1.0]."""
    return max(0.0, min(1.0, value))


_ALLELE_RE = re.compile(r"^[A-Z0-9]+\.(\d+)\.\d+$")


def _allele_subtype_num(allele_name: Optional[str]) -> Optional[int]:
    """Extract the subtype number encoded in an allele name.

    Returns the integer subtype number (e.g. 1 for H1N1, 3 for H3N2) from
    an allele name like 'PB1.3.0001', or None if the name is absent or
    does not match the expected format.
    """
    if not allele_name:
        return None
    m = _ALLELE_RE.match(allele_name)
    return int(m.group(1)) if m else None


def _majority_subtype_num(allele_map: Dict[str, Optional[str]]) -> Optional[int]:
    """Return the most common subtype number across all segment alleles.

    Uses a simple plurality vote across all segments that have a parseable
    allele name.  Returns None when no alleles are present.
    """
    counts: Counter = Counter()
    for allele in allele_map.values():
        snum = _allele_subtype_num(allele)
        if snum is not None:
            counts[snum] += 1
    return counts.most_common(1)[0][0] if counts else None


def _screen_allele_subtypes(
    sequence_id: str,
    allele_map: Dict[str, Optional[str]],
    co_reassortment_groups: List[FrozenSet[str]],
    coherence_boost: float,
    coherence_penalty: float,
    min_discordant: int = 1,
    lineage_links: Optional[Dict[str, List[Dict]]] = None,
) -> Optional["ReassortmentEvent"]:
    """Stage 0: flag segments whose allele subtype differs from the genome majority.

    Parameters
    ----------
    sequence_id : str
    allele_map : dict
        segment_name -> allele_name (e.g. 'PB1.3.0001') or None.
    co_reassortment_groups, coherence_boost, coherence_penalty :
        Passed through to the coherence post-filter.
    min_discordant : int
        Minimum number of discordant segments required to raise an event.
    lineage_links : dict, optional
        segment_name -> list of lineage link dicts from db.get_allele_lineage().
        When provided, linked allele names are included in the event evidence,
        giving analysts the full cross-subtype naming picture.

    Returns
    -------
    ReassortmentEvent or None
    """
    majority = _majority_subtype_num(allele_map)
    if majority is None:
        return None  # no alleles at all — nothing to compare

    discordant = []
    segment_subtypes: Dict[str, Optional[int]] = {}
    for seg in SEGMENTS:
        snum = _allele_subtype_num(allele_map.get(seg))
        segment_subtypes[seg] = snum
        if snum is not None and snum != majority:
            discordant.append(seg)

    if len(discordant) < min_discordant:
        return None

    # Stage 0 is deterministic — confidence is 1.0 before coherence adjustment
    coherence = _co_reassortment_coherence(
        discordant, co_reassortment_groups,
        coherence_boost, coherence_penalty,
    )
    confidence = _clamp_confidence(1.0 * coherence)

    # Build a human-readable description of what was found
    disc_details = ", ".join(
        f"{seg}({allele_map.get(seg)})" for seg in discordant
    )
    majority_alleles = [
        allele_map.get(s) for s in SEGMENTS
        if _allele_subtype_num(allele_map.get(s)) == majority
    ]

    # Collect lineage links for discordant segments so the event evidence
    # records not just the mismatch but also the known aliases.
    linked: Dict[str, List[str]] = {}
    if lineage_links:
        for seg in discordant:
            links = lineage_links.get(seg, [])
            if links:
                linked[seg] = [lk["linked_allele"] for lk in links]

    return ReassortmentEvent(
        sequence_id=sequence_id,
        discordant_segments=discordant,
        confidence=confidence,
        event_type=_classify_event_type(discordant),
        description=(
            f"Allele subtype mismatch detected: {disc_details} "
            f"differ from genome-majority subtype {majority}"
        ),
        detection_stage=0,
        evidence={
            "majority_subtype_num": majority,
            "segment_subtype_nums": segment_subtypes,
            "discordant_alleles": {s: allele_map.get(s) for s in discordant},
            "concordant_allele_count": len(majority_alleles),
            "lineage_links": linked,
        },
    )


# ======================================================================
# Statistical functions
# ======================================================================


def _fisher_exact_pair(
    pair_key: Tuple[str, str],
    combo: Tuple[str, str],
    pair_counts: Dict[Tuple[str, str], Counter],
    marginal_counts: Dict[Tuple[str, str], Dict[str, Counter]],
) -> float:
    """Run Fisher's exact test for a single segment-pair combination.

    Constructs a 2×2 contingency table:

                       seg_B == cluster_B    seg_B != cluster_B
    seg_A == cA    [  observed (a)       |     b               ]
    seg_A != cA    [  c                  |     d               ]

    Returns the one-sided p-value testing whether this combination is
    *under-represented* (the reassortment signal — segments that should
    travel together but don't).
    """
    counts = pair_counts.get(pair_key, Counter())
    marginals = marginal_counts.get(pair_key)
    total = sum(counts.values())

    if total == 0 or marginals is None:
        return 1.0  # no data — cannot reject independence

    seg_a, seg_b = pair_key
    cluster_a, cluster_b = combo

    # Cell a: both clusters match the observed combo
    a = counts.get(combo, 0)

    # Marginal totals
    row_a_total = marginals[seg_a].get(cluster_a, 0)
    col_a_total = marginals[seg_b].get(cluster_b, 0)

    # Fill the 2×2 table
    b = row_a_total - a
    c = col_a_total - a
    d = total - row_a_total - col_a_total + a

    # Guard against impossible tables from data issues
    table = np.array([[max(a, 0), max(b, 0)],
                      [max(c, 0), max(d, 0)]])

    if table.sum() == 0:
        return 1.0

    try:
        result = scipy_stats.fisher_exact(table, alternative="less")
        return float(result.pvalue)
    except ValueError:
        # Degenerate table
        return 1.0


def _combine_pvalues_stouffer(pvalues: List[float]) -> float:
    """Combine p-values using Stouffer's Z method.

    Converts each p-value to a z-score via the inverse normal CDF,
    averages them (weighted equally), and converts back to a combined
    p-value.
    """
    if not pvalues:
        return 1.0

    # Clamp p-values away from 0 and 1 to avoid infinite z-scores
    eps = 1e-15
    clamped = [max(eps, min(1.0 - eps, p)) for p in pvalues]

    z_scores = [float(scipy_stats.norm.ppf(p)) for p in clamped]
    combined_z = np.sum(z_scores) / np.sqrt(len(z_scores))
    return float(scipy_stats.norm.cdf(combined_z))


def _combine_pvalues_cauchy(pvalues: List[float]) -> float:
    """Combine p-values using the Cauchy combination test (ACAT).

    Each p-value is mapped to a Cauchy variate via tan((0.5 - p) * pi); the
    equally-weighted mean is itself ~Cauchy under the null, *regardless of the
    dependence structure among the inputs* (Liu & Xie 2020).  That robustness
    to arbitrary dependence is exactly what the per-pair Fisher tests need —
    they share segment marginals and are correlated, which makes equal-weight
    Stouffer anticonservative.

    Small p-values use the stable tail form w/(p*pi) to avoid tan() overflow.
    """
    if not pvalues:
        return 1.0

    p = np.clip(np.asarray(pvalues, dtype=np.float64), 1e-300, 1.0 - 1e-15)
    k = p.size
    w = 1.0 / k

    # tan((0.5 - p)*pi); for very small p use cot(p*pi) ≈ 1/(p*pi)
    small = p < 1e-15
    terms = np.where(
        small,
        w / (p * np.pi),
        w * np.tan((0.5 - p) * np.pi),
    )
    stat = float(np.sum(terms))

    # Survival function of standard Cauchy.  Large positive stat → tail approx.
    if stat > 1e15:
        combined = (1.0 / stat) / np.pi
    else:
        combined = 0.5 - (np.arctan(stat) / np.pi)
    return float(min(max(combined, 0.0), 1.0))


def _combine_pvalues(pvalues: List[float], method: str = "cauchy") -> float:
    """Dispatch to the configured p-value combination method."""
    if method == "stouffer":
        return _combine_pvalues_stouffer(pvalues)
    return _combine_pvalues_cauchy(pvalues)


def _fdr_adjust(pvalues: List[float], method: str = "bh") -> List[float]:
    """Return BH (or BY) adjusted p-values (q-values), order preserved.

    method="bh"  — Benjamini-Hochberg (independence / positive dependence).
    method="by"  — Benjamini-Yekutieli (valid under arbitrary dependence;
                   more conservative by the harmonic factor c(m)=Σ 1/i).
    """
    p = np.asarray(pvalues, dtype=np.float64)
    m = p.size
    if m == 0:
        return []

    order = np.argsort(p)
    ranked = p[order]
    ranks = np.arange(1, m + 1, dtype=np.float64)

    c_m = float(np.sum(1.0 / ranks)) if method == "by" else 1.0
    q = ranked * m * c_m / ranks

    # Enforce monotonicity (step-up) and clamp to 1
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)

    out = np.empty(m, dtype=np.float64)
    out[order] = q
    return out.tolist()


def _compute_segment_scores(
    clusters: Dict[str, Optional[str]],
    pair_counts: Dict[Tuple[str, str], Counter],
    marginal_counts: Dict[Tuple[str, str], Dict[str, Counter]],
    method: str = "cauchy",
) -> Dict[str, float]:
    """Compute combined p-values for all segments in a profile.

    Shared by the main screening path and the permutation validator.  The
    per-pair Fisher p-values are combined via ``method`` ("cauchy" by default,
    "stouffer" for the legacy path).
    """
    segment_pvalues: Dict[str, float] = {}
    assigned_segs = [s for s in SEGMENTS if _is_assigned(clusters.get(s))]

    for seg in assigned_segs:
        pair_pvals: List[float] = []
        for other in assigned_segs:
            if other == seg:
                continue
            pair_key = _sorted_pair(seg, other)
            combo = (clusters[pair_key[0]], clusters[pair_key[1]])
            p = _fisher_exact_pair(pair_key, combo, pair_counts, marginal_counts)
            pair_pvals.append(p)
        if pair_pvals:
            segment_pvalues[seg] = _combine_pvalues(pair_pvals, method)

    return segment_pvalues


# ======================================================================
# Detector
# ======================================================================


class ReassortmentDetector:
    """Two-stage reassortment detector with statistical LD testing.

    Stage 1 uses Fisher's exact test per segment pair with Stouffer's Z
    combination to produce per-segment p-values.  Stage 2 uses MinHash
    distance z-scores for within-cluster refinement.

    Parameters
    ----------
    config : ReassortmentConfig, optional
        Full configuration.  All thresholds are read from this object.
    """

    def __init__(
        self,
        config: Optional[ReassortmentConfig] = None,
        db=None,
        kmer_config: Optional[KmerConfig] = None,
    ):
        self.config = config or ReassortmentConfig()
        self._db = db  # Optional DatabaseManager for lineage lookups
        self._kmer_config = kmer_config or KmerConfig()

        # Stage 1
        self._alpha = self.config.significance_level
        self.bonferroni = self.config.bonferroni

        # Stage 1 combination & multiplicity (v4)
        self.combination_method = getattr(self.config, "combination_method", "cauchy")
        self.multiplicity = getattr(self.config, "multiplicity", "global_fdr_bh")

        # Stage 2
        self.distance_zscore_threshold = self.config.distance_zscore_threshold
        self.confidence_decay = self.config.confidence_decay
        self.min_constellation_mates = self.config.min_constellation_mates
        self.min_segment_comparisons = self.config.min_segment_comparisons
        self.max_baseline_pairs = self.config.max_baseline_pairs
        self.baseline_std_floor = self.config.baseline_std_floor
        self.stage2_baseline = getattr(self.config, "stage2_baseline", "per_constellation")

        # Coherence
        self.coherence_boost = self.config.coherence_boost
        self.coherence_penalty = self.config.coherence_penalty
        self.co_reassortment_groups: List[FrozenSet[str]] = (
            [frozenset(g) for g in self.config.co_reassortment_groups]
            if self.config.co_reassortment_groups is not None
            else DEFAULT_CO_REASSORTMENT_GROUPS
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _segment_distance(
        self,
        segment_name: str,
        sig_a: MinHashSignature,
        sig_b: MinHashSignature,
    ) -> float:
        """Within-clade distance for Stage 2 as a containment-ANI distance
        (``1 - max-containment-ANI``), using the segment's k-mer length.

        Containment makes the distance length-tolerant: a truncated-but-identical
        segment reads as near-zero distance rather than an artefactual outlier,
        so short segments no longer produce false within-clade reassortment
        signals. For complete, equal-length segments it is a monotonic transform
        of Jaccard distance, so z-score behaviour is preserved.
        """
        k = self._kmer_config.get_k(segment_name)
        return 1.0 - MinHashSignature.containment_ani(sig_a, sig_b, k)

    def detect_reassortments(
        self,
        profiles: List[GenotypeProfile],
        signatures: Optional[Dict[str, Dict[str, MinHashSignature]]] = None,
        nomenclature: Optional[Dict[str, Dict]] = None,
    ) -> ReassortmentReport:
        """Detect reassortment events across a set of genotype profiles.

        Parameters
        ----------
        profiles : list of GenotypeProfile
            Per-sequence cluster assignments (one profile per virus).
        signatures : dict, optional
            MinHash signatures keyed by ``sequence_id -> segment -> signature``.
            When provided, enables Stage 2 within-cluster distance analysis.
        nomenclature : dict, optional
            Naming results from NomenclatureManager, keyed by sequence_id.
            Each value must have an ``alleles`` sub-dict mapping segment names
            to allele names (e.g. ``{'PB1': 'PB1.3.0001', ...}``).
            When provided, enables Stage 0 allele subtype discordance detection,
            which catches cross-subtype reassortment in fully- or partially-
            orphaned isolates that Stage 1 would miss entirely.
        """
        if len(profiles) < 2:
            return ReassortmentReport(len(profiles), 0, [])

        # All profiles are candidates for Stage 0; Stages 1+2 use completeness gate
        analyzable = [p for p in profiles if p.completeness >= 0.75]
        logger.info(
            f"Analyzing {len(profiles)} profiles for reassortment "
            f"(Stage 0: {'enabled' if nomenclature else 'disabled'}, "
            f"Stage 1/2 pool: {len(analyzable)}, "
            f"alpha={self._alpha}, bonferroni={self.bonferroni})"
        )

        # ── Stage 0: allele subtype discordance ──────────────────────────────
        stage0_events: Dict[str, ReassortmentEvent] = {}
        stage0_seq_ids: set = set()

        if nomenclature is not None:
            for profile in profiles:
                allele_map = (
                    nomenclature.get(profile.sequence_id, {}).get("alleles") or {}
                )
                # Build lineage link map for this profile's discordant segments.
                # Fetched per-allele from DB so the event evidence records aliases.
                lineage_links: Dict[str, List[Dict]] = {}
                if self._db is not None:
                    for seg, allele in allele_map.items():
                        if allele:
                            links = self._db.get_allele_lineage(allele)
                            if links:
                                lineage_links[seg] = links

                event = _screen_allele_subtypes(
                    sequence_id=profile.sequence_id,
                    allele_map=allele_map,
                    co_reassortment_groups=self.co_reassortment_groups,
                    coherence_boost=self.coherence_boost,
                    coherence_penalty=self.coherence_penalty,
                    min_discordant=self.config.min_discordant_segments,
                    lineage_links=lineage_links,
                )
                if event is not None:
                    stage0_events[profile.sequence_id] = event
                    stage0_seq_ids.add(profile.sequence_id)
            if stage0_events:
                logger.info(
                    f"Stage 0: {len(stage0_events)} allele-subtype reassortment "
                    f"event(s) detected"
                )
            else:
                logger.debug("Stage 0: no allele-subtype discordance detected")

        # Sequences flagged by Stage 0 are excluded from Stage 1 — they are
        # already fully resolved and adding a statistical test on top of a
        # deterministic allele match would be redundant and potentially wrong
        # (their orphan cluster IDs carry no LD information).
        stage1_pool = [
            p for p in analyzable if p.sequence_id not in stage0_seq_ids
        ]

        # ── Stage 1: LD-based discordance (combination + population FDR) ─────
        stage1_events, stage1_clear = self._run_stage1(stage1_pool)

        # ── Stage 2: within-cluster distance refinement ──────────────────────
        stage2_events: Dict[str, ReassortmentEvent] = {}
        if signatures is None:
            logger.debug("Stage 2 skipped: no MinHash signatures provided")
        elif not stage1_clear:
            logger.debug(
                "Stage 2 skipped: no profiles passed Stage 1 screening"
            )
        else:
            stage2_events = self._distance_refinement(
                stage1_clear, stage1_pool, signatures
            )

        all_events = (
            list(stage0_events.values())
            + list(stage1_events.values())
            + list(stage2_events.values())
        )
        report = ReassortmentReport(
            len(profiles), len(analyzable), all_events, len(all_events)
        )
        logger.info(
            f"Reassortment: {report.flagged_sequences} event(s) "
            f"({report.reassortment_rate:.1%} of analyzable) — "
            f"{len(stage0_events)} from Stage 0, "
            f"{len(stage1_events)} from Stage 1, "
            f"{len(stage2_events)} from Stage 2"
        )
        return report

    def detect_and_validate(
        self,
        profiles: List[GenotypeProfile],
        signatures: Optional[Dict[str, Dict[str, MinHashSignature]]] = None,
        nomenclature: Optional[Dict[str, Dict]] = None,
    ) -> ReassortmentReport:
        """Run detection and, when ``config.gate_on_permutation`` is True,
        keep a Stage 1 event only if the permutation test confirms it.

        This is the recommended entry point for *reported* reassortment rates:
        the screening combination (Cauchy) plus population FDR controls the
        default output, and the permutation null is the gold-standard
        verification layer for the Stage 1 calls that survive screening.
        Stage 0 and Stage 2 events are passed through unchanged (Stage 0 is
        deterministic; Stage 2 is distance-based, not LD-based).
        """
        report = self.detect_reassortments(profiles, signatures, nomenclature)
        if not getattr(self.config, "gate_on_permutation", False):
            return report

        stage1 = [e for e in report.events if e.detection_stage == 1]
        if not stage1:
            return report

        results = self.validate_events(
            stage1, profiles,
            n_permutations=getattr(self.config, "permutation_n", 1000),
            seed=getattr(self.config, "permutation_seed", 42),
        )
        confirmed = {r.sequence_id for r in results if r.concordant}

        kept = [
            e for e in report.events
            if e.detection_stage != 1 or e.sequence_id in confirmed
        ]
        dropped = len(report.events) - len(kept)
        if dropped:
            logger.info(
                f"Permutation gating removed {dropped} unconfirmed Stage 1 "
                f"event(s); {len(kept)} event(s) retained"
            )
        return ReassortmentReport(
            report.total_sequences,
            report.sequences_analyzed,
            kept,
            len(kept),
        )

    def validate_events(
        self,
        events: List[ReassortmentEvent],
        profiles: List[GenotypeProfile],
        n_permutations: int = 5000,
        seed: int = 42,
    ) -> List[PermutationResult]:
        """Validate Stage 1 events using a permutation test.

        For each flagged event, shuffles the target segment's cluster
        assignments across all profiles, rebuilds pair-level co-occurrence
        statistics under the permuted data, and recomputes the Stouffer-
        combined p-value.  This produces assumption-free empirical p-values
        that correctly account for correlated pair-tests.

        Only Stage 1 events are validated; Stage 2 events are skipped.

        Parameters
        ----------
        events : list of ReassortmentEvent
            Events to validate (typically from ``detect_reassortments``).
        profiles : list of GenotypeProfile
            The same profiles used for detection.
        n_permutations : int
            Number of shuffles per segment per event.  1000 for screening,
            5000+ for publication.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        list of PermutationResult
            One result per validated event.
        """
        analyzable = [p for p in profiles if p.completeness >= 0.75]
        pattern_map = self._build_pattern_map(analyzable)

        profile_map = {p.sequence_id: p for p in analyzable}
        profile_index = {p.sequence_id: i for i, p in enumerate(analyzable)}

        # Pre-extract per-profile data for fast pair-count rebuilding
        profiles_data: List[Tuple[List[str], Dict[str, Optional[str]]]] = []
        for p in analyzable:
            cl = p.segment_clusters
            assigned = [s for s in SEGMENTS if _is_assigned(cl.get(s))]
            profiles_data.append((assigned, dict(cl)))

        # Per-segment cluster vectors (one entry per analyzable profile)
        seg_vectors: Dict[str, np.ndarray] = {}
        for seg in SEGMENTS:
            seg_vectors[seg] = np.array([
                p.segment_clusters.get(seg) or ""
                for p in analyzable
            ])

        rng = np.random.default_rng(seed)
        results: List[PermutationResult] = []

        stage1_events = [e for e in events if e.detection_stage == 1]
        if not stage1_events:
            logger.info("Permutation validation: no Stage 1 events to validate")
            return results

        logger.info(
            f"Permutation validation: {len(stage1_events)} events × "
            f"{n_permutations} permutations"
        )

        for event in stage1_events:
            profile = profile_map.get(event.sequence_id)
            if profile is None:
                logger.warning(
                    f"Permutation: {event.sequence_id} not found in profiles"
                )
                continue

            idx = profile_index[event.sequence_id]
            clusters = profile.segment_clusters

            # Observed combined p-values
            observed_scores = _compute_segment_scores(
                clusters,
                pattern_map.pair_counts,
                pattern_map.marginal_counts,
                method=self.combination_method,
            )

            empirical_pvalues: Dict[str, float] = {}

            for seg in event.discordant_segments:
                if seg not in observed_scores:
                    continue

                observed_p = observed_scores[seg]
                vec = seg_vectors.get(seg)
                if vec is None or len(vec) == 0:
                    empirical_pvalues[seg] = 1.0
                    continue

                n_more_extreme = 0
                for _ in range(n_permutations):
                    # Shuffle the entire segment column across all profiles
                    shuffled_vec = rng.permutation(vec)

                    # Rebuild pair counts involving this segment under
                    # the permuted assignments
                    perm_pc: Dict[Tuple[str, str], Counter] = {}
                    perm_mc: Dict[Tuple[str, str], Dict[str, Counter]] = {}

                    for pi, (assigned, orig_cl) in enumerate(profiles_data):
                        perm_cid = str(shuffled_vec[pi])
                        if not perm_cid or perm_cid == ORPHAN_MARKER:
                            continue
                        for other in assigned:
                            if other == seg:
                                continue
                            ocid = orig_cl.get(other)
                            if not _is_assigned(ocid):
                                continue
                            pk = _sorted_pair(seg, other)
                            combo = (
                                (perm_cid, ocid) if pk[0] == seg
                                else (ocid, perm_cid)
                            )
                            perm_pc.setdefault(pk, Counter())[combo] += 1
                            pm = perm_mc.setdefault(
                                pk, {pk[0]: Counter(), pk[1]: Counter()}
                            )
                            pm[pk[0]][combo[0]] += 1
                            pm[pk[1]][combo[1]] += 1

                    # Score the target profile under permuted statistics
                    perm_clusters = dict(clusters)
                    perm_clusters[seg] = str(shuffled_vec[idx])

                    pair_pvals: List[float] = []
                    for other in SEGMENTS:
                        if other == seg:
                            continue
                        ocid = perm_clusters.get(other)
                        if not _is_assigned(ocid):
                            continue
                        pcid = perm_clusters.get(seg)
                        if not _is_assigned(pcid):
                            continue
                        pk = _sorted_pair(seg, other)
                        combo = (perm_clusters[pk[0]], perm_clusters[pk[1]])
                        pair_pvals.append(
                            _fisher_exact_pair(pk, combo, perm_pc, perm_mc)
                        )

                    null_p = _combine_pvalues(pair_pvals, self.combination_method)
                    if null_p <= observed_p:
                        n_more_extreme += 1

                # Conservative empirical p-value
                empirical_pvalues[seg] = (
                    (n_more_extreme + 1) / (n_permutations + 1)
                )

            # Concordance check
            n_tested = len(observed_scores)
            perm_alpha = self._alpha / n_tested if self.bonferroni else self._alpha

            concordant = all(
                empirical_pvalues.get(seg, 1.0) < perm_alpha
                for seg in event.discordant_segments
            )

            results.append(PermutationResult(
                sequence_id=event.sequence_id,
                segment_empirical_pvalues=empirical_pvalues,
                segment_stouffer_pvalues={
                    seg: observed_scores.get(seg, 1.0)
                    for seg in event.discordant_segments
                },
                concordant=concordant,
                n_permutations=n_permutations,
            ))

            logger.debug(
                f"Permutation: {event.sequence_id} — "
                f"concordant={concordant}, "
                f"empirical_p={empirical_pvalues}"
            )

        n_concordant = sum(1 for r in results if r.concordant)
        logger.info(
            f"Permutation validation: {n_concordant}/{len(results)} events "
            f"confirmed (concordant with Stouffer)"
        )
        return results

    # ------------------------------------------------------------------
    # Stage 1: LD testing via Fisher's exact + Stouffer's Z
    # ------------------------------------------------------------------

    def _build_pattern_map(self, profiles: List[GenotypeProfile]) -> PatternMap:
        """Build co-occurrence statistics for every segment pair.

        This is the linkage disequilibrium model of the population: it
        captures which segment cluster combinations travel together and
        how frequently.
        """
        pair_counts: Dict[Tuple[str, str], Counter] = {}
        marginal_counts: Dict[Tuple[str, str], Dict[str, Counter]] = {}

        for profile in profiles:
            clusters = profile.segment_clusters
            assigned = [
                s for s in SEGMENTS
                if _is_assigned(clusters.get(s))
            ]
            for i in range(len(assigned)):
                for j in range(i + 1, len(assigned)):
                    pair_key = _sorted_pair(assigned[i], assigned[j])
                    combo = (clusters[pair_key[0]], clusters[pair_key[1]])

                    pair_counts.setdefault(pair_key, Counter())[combo] += 1

                    pair_marginals = marginal_counts.setdefault(
                        pair_key,
                        {pair_key[0]: Counter(), pair_key[1]: Counter()},
                    )
                    pair_marginals[pair_key[0]][clusters[pair_key[0]]] += 1
                    pair_marginals[pair_key[1]][clusters[pair_key[1]]] += 1

        return PatternMap(pair_counts, marginal_counts, len(profiles))

    def _run_stage1(
        self, stage1_pool: List[GenotypeProfile]
    ) -> Tuple[Dict[str, ReassortmentEvent], List[GenotypeProfile]]:
        """Stage 1 orchestration with population-level multiple-testing control.

        Pass 1 computes a combined p-value per (profile, segment) using the
        configured combination method.  Pass 2 decides which of those tests
        are significant — either by global FDR across *every* profile (default)
        or by the legacy per-profile Bonferroni — then builds one event per
        profile from its rejected segments.
        """
        pattern_map = self._build_pattern_map(stage1_pool)

        # Pass 1 — per-segment combined p-values for every profile
        profile_pvals: Dict[str, Dict[str, float]] = {}
        for profile in stage1_pool:
            sp = _compute_segment_scores(
                profile.segment_clusters,
                pattern_map.pair_counts,
                pattern_map.marginal_counts,
                method=self.combination_method,
            )
            if sp:
                profile_pvals[profile.sequence_id] = sp

        # Pass 2 — significance decision across the whole population
        rejected, mult_info = self._select_significant(profile_pvals)

        stage1_events: Dict[str, ReassortmentEvent] = {}
        stage1_clear: List[GenotypeProfile] = []
        for profile in stage1_pool:
            discordant = list(rejected.get(profile.sequence_id, {}).keys())
            if len(discordant) >= self.config.min_discordant_segments:
                event = self._make_stage1_event(
                    profile,
                    discordant,
                    profile_pvals.get(profile.sequence_id, {}),
                    pattern_map,
                    mult_info.get(profile.sequence_id, {}),
                )
                if event is not None:
                    stage1_events[profile.sequence_id] = event
                    continue
            stage1_clear.append(profile)

        return stage1_events, stage1_clear

    def _select_significant(
        self, profile_pvals: Dict[str, Dict[str, float]]
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict]]:
        """Decide which (profile, segment) tests are discordant.

        Returns
        -------
        rejected : {seq_id: {segment: decision_statistic}}
        mult_info : {seq_id: evidence dict describing the correction applied}
        """
        rejected: Dict[str, Dict[str, float]] = defaultdict(dict)
        mult_info: Dict[str, Dict] = defaultdict(dict)

        if self.multiplicity == "per_profile_bonferroni":
            for sid, segp in profile_pvals.items():
                n = len(segp)
                eff = self._alpha / n if self.bonferroni else self._alpha
                mult_info[sid] = {"effective_alpha": eff, "bonferroni_corrected": self.bonferroni}
                for seg, p in segp.items():
                    if p < eff:
                        rejected[sid][seg] = p
            return rejected, mult_info

        # Global FDR across every (profile, segment) test
        keys: List[Tuple[str, str]] = []
        pvals: List[float] = []
        for sid, segp in profile_pvals.items():
            for seg, p in segp.items():
                keys.append((sid, seg))
                pvals.append(p)
        if not pvals:
            return rejected, mult_info

        fdr_method = "by" if self.multiplicity == "global_fdr_by" else "bh"
        qvals = _fdr_adjust(pvals, fdr_method)
        for (sid, seg), q in zip(keys, qvals):
            mult_info[sid].setdefault("q_values", {})[seg] = q
            if q < self._alpha:
                rejected[sid][seg] = q
        for sid in mult_info:
            mult_info[sid]["fdr_method"] = fdr_method
            mult_info[sid]["n_tests"] = len(pvals)
        return rejected, mult_info

    def _make_stage1_event(
        self,
        profile: GenotypeProfile,
        discordant: List[str],
        segment_pvalues: Dict[str, float],
        pattern_map: PatternMap,
        mult_info: Dict,
    ) -> Optional[ReassortmentEvent]:
        """Build a Stage 1 event from a profile's rejected segments."""
        clusters = profile.segment_clusters

        # Per-pair p-values for evidence
        segment_pair_pvalues: Dict[str, List[float]] = {}
        for seg in discordant:
            pair_pvals: List[float] = []
            for other in SEGMENTS:
                if other == seg or not _is_assigned(clusters.get(other)):
                    continue
                pair_key = _sorted_pair(seg, other)
                combo = (clusters[pair_key[0]], clusters[pair_key[1]])
                pair_pvals.append(
                    _fisher_exact_pair(
                        pair_key, combo,
                        pattern_map.pair_counts, pattern_map.marginal_counts,
                    )
                )
            segment_pair_pvalues[seg] = pair_pvals

        disc_pvals = [segment_pvalues[s] for s in discordant if s in segment_pvalues]
        raw_confidence = _clamp_confidence(
            1.0 - float(np.mean(disc_pvals)) if disc_pvals else 0.0
        )

        coherence = _co_reassortment_coherence(
            discordant, self.co_reassortment_groups,
            self.coherence_boost, self.coherence_penalty,
        )
        confidence = _clamp_confidence(raw_confidence * coherence)
        if confidence < self.config.min_confidence:
            return None

        evidence = {
            "segment_combined_pvalues": segment_pvalues,
            "segment_pair_pvalues": segment_pair_pvalues,
            "coherence_multiplier": coherence,
            "combination_method": self.combination_method,
            "multiplicity": self.multiplicity,
        }
        evidence.update(mult_info)

        return ReassortmentEvent(
            sequence_id=profile.sequence_id,
            discordant_segments=discordant,
            confidence=confidence,
            event_type=_classify_event_type(discordant),
            description=(
                f"Segments {discordant} show statistically significant "
                f"linkage disequilibrium breakdown"
            ),
            detection_stage=1,
            evidence=evidence,
        )

    # ------------------------------------------------------------------
    # Stage 2: Within-cluster distance refinement
    # ------------------------------------------------------------------

    def _distance_refinement(
        self,
        candidates: List[GenotypeProfile],
        all_profiles: List[GenotypeProfile],
        signatures: Dict[str, Dict[str, MinHashSignature]],
    ) -> Dict[str, ReassortmentEvent]:
        """Detect within-cluster reassortment using MinHash distances."""
        events: Dict[str, ReassortmentEvent] = {}

        # Log signature coverage
        total_candidates = len(candidates)
        candidates_with_sigs = sum(
            1 for p in candidates if p.sequence_id in signatures
        )
        candidates_missing_sigs = total_candidates - candidates_with_sigs
        if candidates_missing_sigs > 0:
            logger.warning(
                f"Stage 2: {candidates_missing_sigs}/{total_candidates} "
                f"candidates have no MinHash signatures and will be skipped"
            )

        # Check for incomplete signatures
        incomplete_count = 0
        for p in candidates:
            sigs = signatures.get(p.sequence_id, {})
            if sigs:
                assigned_segs = [
                    s for s in SEGMENTS
                    if _is_assigned(p.segment_clusters.get(s))
                ]
                missing_segs = [s for s in assigned_segs if s not in sigs]
                if missing_segs:
                    incomplete_count += 1
                    logger.debug(
                        f"Stage 2: {p.sequence_id} missing signatures "
                        f"for segments: {missing_segs}"
                    )
        if incomplete_count > 0:
            logger.info(
                f"Stage 2: {incomplete_count}/{candidates_with_sigs} "
                f"candidates have incomplete segment signatures"
            )

        # Group profiles by full constellation
        constellation_groups: Dict[
            Tuple[Optional[str], ...], List[GenotypeProfile]
        ] = defaultdict(list)
        for p in all_profiles:
            key = tuple(p.segment_clusters.get(s) for s in SEGMENTS)
            constellation_groups[key].append(p)

        group_sizes = [len(g) for g in constellation_groups.values()]
        small_groups = sum(
            1 for s in group_sizes if s <= self.min_constellation_mates
        )
        if small_groups > 0:
            logger.debug(
                f"Stage 2: {small_groups}/{len(constellation_groups)} "
                f"constellation groups have <= {self.min_constellation_mates} "
                f"members (too small for distance screening)"
            )

        seg_baselines = self._compute_distance_baselines(
            constellation_groups, signatures
        )
        if not seg_baselines:
            logger.warning(
                "Stage 2: could not compute distance baselines — "
                "insufficient signature coverage"
            )
            return events

        logger.debug(
            f"Stage 2: distance baselines computed for "
            f"{len(seg_baselines)}/{len(SEGMENTS)} segments"
        )

        skipped_no_sigs = 0
        for profile in candidates:
            if profile.sequence_id not in signatures:
                skipped_no_sigs += 1
                continue
            event = self._screen_profile_distances(
                profile, constellation_groups, signatures, seg_baselines
            )
            if event is not None:
                events[profile.sequence_id] = event

        if skipped_no_sigs:
            logger.debug(
                f"Stage 2: {skipped_no_sigs} candidates skipped (no signatures)"
            )
        if events:
            logger.info(
                f"Stage 2: {len(events)} within-cluster reassortment "
                f"candidates detected"
            )
        else:
            logger.debug("Stage 2: no within-cluster reassortment detected")

        return events

    def _compute_distance_baselines(
        self,
        constellation_groups: Dict[Tuple, List[GenotypeProfile]],
        signatures: Dict[str, Dict[str, MinHashSignature]],
    ) -> Dict[str, Tuple[float, float]]:
        """Compute per-segment (mean, std) of intra-constellation distances."""
        seg_distances: Dict[str, List[float]] = defaultdict(list)

        for group_profiles in constellation_groups.values():
            if len(group_profiles) < 2:
                continue
            ids = [p.sequence_id for p in group_profiles]

            pair_indices = [
                (i, j)
                for i in range(len(ids))
                for j in range(i + 1, len(ids))
            ]
            if len(pair_indices) > self.max_baseline_pairs:
                rng = np.random.default_rng(42)
                chosen = rng.choice(
                    len(pair_indices),
                    size=self.max_baseline_pairs,
                    replace=False,
                )
                pair_indices = [pair_indices[int(c)] for c in chosen]

            for i, j in pair_indices:
                sigs_i = signatures.get(ids[i], {})
                sigs_j = signatures.get(ids[j], {})
                for seg in SEGMENTS:
                    if seg in sigs_i and seg in sigs_j:
                        d = self._segment_distance(seg, sigs_i[seg], sigs_j[seg])
                        seg_distances[seg].append(d)

        baselines: Dict[str, Tuple[float, float]] = {}
        for seg, dists in seg_distances.items():
            arr = np.array(dists)
            baselines[seg] = (
                float(np.mean(arr)),
                max(float(np.std(arr)), self.baseline_std_floor),
            )
        return baselines

    def _local_baseline(
        self,
        seg: str,
        mate_ids: List[str],
        signatures: Dict[str, Dict[str, MinHashSignature]],
    ) -> Optional[Tuple[float, float]]:
        """Mate-to-mate (mean, std) distance for one segment within a single
        constellation.  ``mate_ids`` already excludes the candidate, so this is
        a leave-one-out baseline.  Returns None when too few mate pairs exist.
        """
        ids = [m for m in mate_ids if seg in signatures.get(m, {})]
        if len(ids) < 2:
            return None

        pairs = [
            (i, j) for i in range(len(ids)) for j in range(i + 1, len(ids))
        ]
        if len(pairs) < self.min_segment_comparisons:
            return None
        if len(pairs) > self.max_baseline_pairs:
            rng = np.random.default_rng(42)
            chosen = rng.choice(
                len(pairs), size=self.max_baseline_pairs, replace=False
            )
            pairs = [pairs[int(c)] for c in chosen]

        dists = [
            self._segment_distance(seg, signatures[ids[i]][seg], signatures[ids[j]][seg])
            for i, j in pairs
        ]
        return (
            float(np.mean(dists)),
            max(float(np.std(dists)), self.baseline_std_floor),
        )

    def _screen_profile_distances(
        self,
        profile: GenotypeProfile,
        constellation_groups: Dict[Tuple, List[GenotypeProfile]],
        signatures: Dict[str, Dict[str, MinHashSignature]],
        seg_baselines: Dict[str, Tuple[float, float]],
    ) -> Optional[ReassortmentEvent]:
        """Stage 2: flag segments that are distance outliers relative to
        constellation-mates."""
        my_sigs = signatures.get(profile.sequence_id, {})
        if not my_sigs:
            return None

        constellation_key = tuple(
            profile.segment_clusters.get(s) for s in SEGMENTS
        )
        mates = constellation_groups.get(constellation_key, [])
        mate_ids = [
            p.sequence_id for p in mates
            if p.sequence_id != profile.sequence_id
        ]
        if len(mate_ids) < self.min_constellation_mates:
            return None

        segment_z_scores: Dict[str, float] = {}
        segment_mean_dists: Dict[str, float] = {}

        use_local = (self.stage2_baseline == "per_constellation")

        for seg in SEGMENTS:
            if seg not in my_sigs:
                continue

            distances = []
            for mid in mate_ids:
                mate_sigs = signatures.get(mid, {})
                if seg in mate_sigs:
                    distances.append(
                        self._segment_distance(seg, my_sigs[seg], mate_sigs[seg])
                    )

            if len(distances) < self.min_segment_comparisons:
                continue

            mean_dist = float(np.mean(distances))

            # Per-constellation leave-one-out baseline (mate-to-mate distances,
            # candidate already excluded from mate_ids), falling back to the
            # global pooled baseline when the local one is too thin.
            baseline = None
            if use_local:
                baseline = self._local_baseline(seg, mate_ids, signatures)
            if baseline is None:
                baseline = seg_baselines.get(seg)
            if baseline is None:
                continue

            baseline_mean, baseline_std = baseline
            segment_mean_dists[seg] = mean_dist
            z = (mean_dist - baseline_mean) / baseline_std
            segment_z_scores[seg] = z

        if not segment_z_scores:
            return None

        discordant = [
            s for s, z in segment_z_scores.items()
            if z > self.distance_zscore_threshold
        ]
        if len(discordant) < self.config.min_discordant_segments:
            return None

        # Confidence from z-scores via exponential mapping
        disc_z = [segment_z_scores[s] for s in discordant]
        raw_confidence = float(
            1.0 - np.exp(-self.confidence_decay * np.mean(disc_z))
        )
        raw_confidence = _clamp_confidence(raw_confidence)

        coherence = _co_reassortment_coherence(
            discordant, self.co_reassortment_groups,
            self.coherence_boost, self.coherence_penalty,
        )
        confidence = _clamp_confidence(raw_confidence * coherence)

        if confidence < self.config.min_confidence:
            return None

        return ReassortmentEvent(
            sequence_id=profile.sequence_id,
            discordant_segments=discordant,
            confidence=confidence,
            event_type=_classify_event_type(discordant),
            description=(
                f"Segments {discordant} show elevated within-cluster "
                f"genetic distance to constellation-mates"
            ),
            detection_stage=2,
            evidence={
                "segment_z_scores": segment_z_scores,
                "segment_mean_distances": segment_mean_dists,
                "coherence_multiplier": coherence,
            },
        )
