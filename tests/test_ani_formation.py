"""Tests for ANI-driven cluster formation.

Formation now reads the per-segment ANI threshold and converts it to the
Jaccard cut height the linkage tree is built on. For the zero-adjustment
subtype this reproduces the legacy Jaccard cut (to table rounding); for H3N2
the representative ANI subtype adjustment introduces a small, bounded
difference (the known approximation, to be tuned later).
"""

import random

import pytest

from influenza_genotyper.config import (
    ClusteringConfig,
    KmerConfig,
    ani_to_jaccard,
    SEGMENTS,
)
from influenza_genotyper.core.kmer_extractor import KmerExtractor
from influenza_genotyper.core.clustering_engine import ClusteringEngine


def _new_cut(c, kc, seg, st):
    return 1.0 - ani_to_jaccard(c.get_ani_threshold(seg, st, "same"), kc.get_k(seg))


def _old_cut(c, seg, st):
    return 1.0 - c.get_threshold(seg, st, "same")


def test_formation_cut_matches_legacy_for_zero_adjustment_subtype():
    """H1N1pdm09 has no subtype adjustment, so the ANI-derived cut equals the
    legacy Jaccard cut up to ANI-table rounding."""
    c, kc = ClusteringConfig(), KmerConfig()
    for seg in SEGMENTS:
        assert _new_cut(c, kc, seg, "H1N1pdm09") == pytest.approx(
            _old_cut(c, seg, "H1N1pdm09"), abs=5e-4
        )


def test_h3n2_cut_difference_is_small_and_bounded():
    """The representative-mean H3N2 ANI adjustment differs from the legacy
    -0.02 Jaccard shift, but only slightly."""
    c, kc = ClusteringConfig(), KmerConfig()
    max_diff = max(
        abs(_new_cut(c, kc, seg, "H3N2") - _old_cut(c, seg, "H3N2"))
        for seg in SEGMENTS
    )
    assert max_diff < 5e-3


def _mutate(seq, n, rng):
    s = list(seq)
    for _ in range(n):
        i = rng.randrange(len(s))
        s[i] = rng.choice("ACGT")
    return "".join(s)


def test_clustering_groups_near_identical_sequences():
    """End-to-end: two tight groups of near-identical HA sequences form two
    clusters under the ANI-derived cut."""
    rng = random.Random(1)
    kc = KmerConfig()
    ex = KmerExtractor(kc)
    base1 = "".join(rng.choice("ACGT") for _ in range(1700))
    base2 = "".join(rng.choice("ACGT") for _ in range(1700))
    ids, sigs = [], []
    for i in range(3):
        ids.append(f"g1_{i}")
        sigs.append(ex.extract_signature(_mutate(base1, 1, rng), "HA"))
    for i in range(3):
        ids.append(f"g2_{i}")
        sigs.append(ex.extract_signature(_mutate(base2, 1, rng), "HA"))

    eng = ClusteringEngine(ClusteringConfig(dev_mode=True), kc)
    res = eng.cluster_signatures(ids, sigs, "HA", "H1N1pdm09", "v1")

    assert res.num_orphans == 0
    assert res.num_clusters == 2
    groups = {}
    for a in res.assignments:
        groups.setdefault(a.cluster_id, set()).add(a.sequence_id.split("_")[0])
    # each cluster is pure (all g1 or all g2)
    assert all(len(g) == 1 for g in groups.values())


def test_engine_defaults_kmer_config_when_omitted():
    """Backward-compatible construction: ClusteringEngine still works with only
    a ClusteringConfig (kmer_config defaults)."""
    eng = ClusteringEngine(ClusteringConfig())
    assert eng.kmer_config.get_k("HA") == 21
