"""Tests for the containment-ANI acceptance + naming lockstep (unit 2).

The same metric (max-containment-ANI, via the shared
MinHashSignature.max_containment_ani_vec helper) and the same per-segment ANI
threshold drive both cluster acceptance (engine) and allele naming (centroid
index). The defining new behaviour: a partial-but-identical segment is accepted
and named against its full-length lineage, where length-penalised Jaccard-ANI
would reject it.
"""

import random

import numpy as np
import pytest

from influenza_genotyper.config import ClusteringConfig, KmerConfig
from influenza_genotyper.core.kmer_extractor import KmerExtractor, MinHashSignature
from influenza_genotyper.core.clustering_engine import ClusteringEngine, ClusterDefinition
from influenza_genotyper.core.nomenclature import _CentroidIndex


@pytest.fixture
def env():
    kc = KmerConfig()
    return {
        "kc": kc,
        "ex": KmerExtractor(kc),
        "k": kc.get_k("HA"),
        "cfg": ClusteringConfig(),
        "T": ClusteringConfig().get_ani_threshold("HA", "H1N1pdm09", "same"),
    }


def _rnd(n, rng):
    return "".join(rng.choice("ACGT") for _ in range(n))


def _mut(s, n, rng):
    s = list(s)
    for _ in range(n):
        i = rng.randrange(len(s))
        s[i] = rng.choice("ACGT")
    return "".join(s)


def _cluster(centroid):
    return [ClusterDefinition(
        cluster_id="C1", segment_name="HA", subtype="H1N1pdm09",
        centroid_signature=centroid, centroid_global_index=0,
    )]


def test_vec_helper_matches_scalar_containment_ani(env):
    rng = random.Random(10)
    ex, k = env["ex"], env["k"]
    full = _rnd(1700, rng)
    a = ex.extract_signature(full, "HA")
    b = ex.extract_signature(full[:850], "HA")  # half subset
    j = MinHashSignature.jaccard_similarity(a, b)
    vec = float(MinHashSignature.max_containment_ani_vec(
        j, a.unique_kmer_count, b.unique_kmer_count, k))
    assert vec == pytest.approx(MinHashSignature.containment_ani(a, b, k), abs=1e-12)


def test_partial_subset_is_accepted_and_named(env):
    """The core property: a clean half-length subset joins and names against its
    full-length centroid, where Jaccard-ANI would fall well below threshold."""
    rng = random.Random(11)
    ex, k, cfg, T = env["ex"], env["k"], env["cfg"], env["T"]
    full = _rnd(1700, rng)
    centroid = ex.extract_signature(full, "HA")
    half = ex.extract_signature(full[:850], "HA")

    # Jaccard-ANI (length-penalised) would reject; containment-ANI accepts.
    assert MinHashSignature.jaccard_ani(half, centroid, k) < T
    assert MinHashSignature.containment_ani(half, centroid, k) >= T

    eng = ClusteringEngine(cfg, env["kc"])
    assert eng.assign_to_existing("q", half, _cluster(centroid), "HA", "H1N1pdm09").is_orphan is False

    idx = _CentroidIndex()
    idx.add("HA.1.0001", centroid)
    assert idx.best_match(half, T, k) == "HA.1.0001"


def test_acceptance_and_naming_share_the_boundary(env):
    """For the same query/centroid/threshold, 'accepted into cluster' and
    'matched to that allele name' agree across full, subset, and unrelated."""
    rng = random.Random(12)
    ex, k, cfg, T = env["ex"], env["k"], env["cfg"], env["T"]
    full = _rnd(1700, rng)
    centroid = ex.extract_signature(full, "HA")
    queries = {
        "full": ex.extract_signature(_mut(full, 1, rng), "HA"),
        "subset": ex.extract_signature(full[:850], "HA"),
        "unrelated": ex.extract_signature(_rnd(1700, rng), "HA"),
    }
    eng = ClusteringEngine(cfg, env["kc"])
    for label, q in queries.items():
        accepted_single = not eng.assign_to_existing("q", q, _cluster(centroid), "HA", "H1N1pdm09").is_orphan
        accepted_batch = not eng.assign_batch_to_existing(["q"], [q], _cluster(centroid), "HA", "H1N1pdm09")[0].is_orphan
        idx = _CentroidIndex()
        idx.add("HA.1.0001", centroid)
        named = idx.best_match(q, T, k) is not None
        assert accepted_single == accepted_batch == named, label


def test_single_and_batch_acceptance_agree_on_cluster(env):
    rng = random.Random(13)
    ex, cfg = env["ex"], env["cfg"]
    c1 = ex.extract_signature(_rnd(1700, rng), "HA")
    c2 = ex.extract_signature(_rnd(1700, rng), "HA")
    clusters = [
        ClusterDefinition("C1", "HA", "H1N1pdm09", centroid_signature=c1, centroid_global_index=0),
        ClusterDefinition("C2", "HA", "H1N1pdm09", centroid_signature=c2, centroid_global_index=1),
    ]
    # A query identical to C1's centroid must be accepted into C1 by both paths.
    eng = ClusteringEngine(cfg, env["kc"])
    single = eng.assign_to_existing("q", c1, clusters, "HA", "H1N1pdm09")
    batch = eng.assign_batch_to_existing(["q"], [c1], clusters, "HA", "H1N1pdm09")[0]
    assert single.cluster_id == batch.cluster_id == "C1"
    assert single.is_orphan == batch.is_orphan is False


def test_naming_tiebreak_is_lexicographic_and_order_independent(env):
    """Two identical centroids inserted in either order resolve to the
    lexicographically smallest allele name."""
    rng = random.Random(14)
    ex, k, T = env["ex"], env["k"], env["T"]
    centroid = ex.extract_signature(_rnd(1700, rng), "HA")
    for order in (["HA.1.0002", "HA.1.0001"], ["HA.1.0001", "HA.1.0002"]):
        idx = _CentroidIndex()
        for name in order:
            idx.add(name, centroid)  # identical signature, different names
        assert idx.best_match(centroid, T, k) == "HA.1.0001"


def test_best_match_with_score_returns_ani(env):
    rng = random.Random(15)
    ex, k, T = env["ex"], env["k"], env["T"]
    centroid = ex.extract_signature(_rnd(1700, rng), "HA")
    idx = _CentroidIndex()
    idx.add("HA.1.0001", centroid)
    name, score = idx.best_match_with_score(centroid, T, k)
    assert name == "HA.1.0001"
    assert score == pytest.approx(1.0, abs=1e-9)  # centroid vs itself
