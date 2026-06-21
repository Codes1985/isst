"""Tests that naming is independent of insertion / processing order.

These lock the two determinism fixes:
  * _CentroidIndex.best_match breaks ties on allele name, not array index,
  * constellation IDs are a pure function of the allele combination (wide hash,
    no order-dependent slide).
"""

import numpy as np
import pytest

from influenza_genotyper.core.nomenclature import (
    _CentroidIndex,
    NomenclatureManager,
    _CONSTELLATION_HASH_CHARS,
)
from influenza_genotyper.core.kmer_extractor import MinHashSignature
from influenza_genotyper.config import ClusteringConfig, SEGMENTS


def _sig(values, size=100):
    s = MinHashSignature(num_hashes=len(values), seed=42)
    s.signature = np.array(values, dtype=np.uint64)
    s.unique_kmer_count = size  # containment-ANI needs a nonzero set size
    return s


def test_best_match_tiebreak_is_order_independent():
    query = _sig([1, 2, 3, 4])
    a = _sig([1, 2, 3, 4])  # identical -> similarity 1.0
    b = _sig([1, 2, 3, 4])  # identical -> similarity 1.0 (a tie)

    forward = _CentroidIndex()
    forward.add("PB1.3.0005", a)
    forward.add("PB1.3.0002", b)

    reverse = _CentroidIndex()
    reverse.add("PB1.3.0002", b)
    reverse.add("PB1.3.0005", a)

    # Same winner regardless of insertion order, and it is the lexicographically
    # smallest name (lowest allele number), not the first inserted. Identical
    # signatures of equal size give containment-ANI 1.0, so any threshold < 1
    # matches; the assertion is about the tie-break, not the threshold.
    assert forward.best_match(query, 0.95, 21) == "PB1.3.0002"
    assert reverse.best_match(query, 0.95, 21) == "PB1.3.0002"


def test_best_match_below_threshold_returns_none():
    query = _sig([1, 2, 3, 4])
    far = _sig([9, 9, 9, 9])
    idx = _CentroidIndex()
    idx.add("PB1.3.0001", far)
    assert idx.best_match(query, 0.95, 21) is None


def _alleles(ha="HA.3.0001"):
    d = {seg: f"{seg}.3.0001" for seg in SEGMENTS}
    d["HA"] = ha
    return d


def test_constellation_id_is_pure_function_of_combination():
    combo_a = _alleles(ha="HA.3.0001")
    combo_b = _alleles(ha="HA.3.0099")

    nm1 = NomenclatureManager(db=None, clustering_config=ClusteringConfig())
    nm2 = NomenclatureManager(db=None, clustering_config=ClusteringConfig())

    # Insert in opposite orders across two managers.
    a1 = nm1.assign_constellation(combo_a, "H3N2")
    b1 = nm1.assign_constellation(combo_b, "H3N2")
    b2 = nm2.assign_constellation(combo_b, "H3N2")
    a2 = nm2.assign_constellation(combo_a, "H3N2")

    assert a1 == a2          # same combination -> same ID regardless of order
    assert b1 == b2
    assert a1 != b1          # different combinations -> different IDs


def test_constellation_id_format_and_width():
    nm = NomenclatureManager(db=None, clustering_config=ClusteringConfig())
    cid = nm.assign_constellation(_alleles(), "H3N2")
    prefix, _, suffix = cid.partition("-")
    assert prefix == "H3N2"
    assert len(suffix) == _CONSTELLATION_HASH_CHARS
    assert _CONSTELLATION_HASH_CHARS >= 12  # well past the old 4-char space


def test_constellation_below_min_segments_returns_none():
    nm = NomenclatureManager(db=None, clustering_config=ClusteringConfig())
    too_few = {seg: f"{seg}.3.0001" for seg in SEGMENTS[:5]}
    assert nm.assign_constellation(too_few, "H3N2") is None
