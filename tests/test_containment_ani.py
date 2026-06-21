"""Tests for containment and ANI estimators on MinHash signatures.

Synthetic k-mer sets (fixed strings, fixed seed) make these deterministic:
the signatures, and therefore the estimates, are reproducible run to run.
"""

import pytest

from influenza_genotyper.core.kmer_extractor import MinHashSignature


def _sig(strings, num_hashes=1024, seed=42):
    s = MinHashSignature(num_hashes=num_hashes, seed=seed)
    s.update_batch(set(strings))
    return s


# A is a perfect, half-size subset of B.
A = {f"kmer_{i}" for i in range(2000)}
B = A | {f"kmer_{i}" for i in range(2000, 4000)}
DISJOINT = {f"other_{i}" for i in range(2000)}


def test_identical_sets_are_fully_contained():
    s = _sig(A)
    s2 = _sig(A)
    assert MinHashSignature.max_containment(s, s2) == pytest.approx(1.0, abs=1e-9)
    assert MinHashSignature.containment_ani(s, s2, 21) == pytest.approx(1.0, abs=1e-9)
    assert MinHashSignature.jaccard_ani(s, s2, 21) == pytest.approx(1.0, abs=1e-9)


def test_perfect_subset_reads_full_ani_under_containment():
    sa, sb = _sig(A), _sig(B)
    c_a, c_b = MinHashSignature.containment(sa, sb)
    # A is contained in B; B is not contained in A.
    assert c_a > 0.95
    assert c_b < 0.65
    # The partial-but-identical segment scores ~1.0 under containment but is
    # penalised under Jaccard — the whole point of using containment.
    cani = MinHashSignature.containment_ani(sa, sb, 21)
    jani = MinHashSignature.jaccard_ani(sa, sb, 21)
    assert cani > 0.99
    assert cani > jani


def test_disjoint_sets_score_zero():
    sa, sd = _sig(A), _sig(DISJOINT)
    assert MinHashSignature.max_containment(sa, sd) == 0.0
    assert MinHashSignature.containment_ani(sa, sd, 21) == 0.0


def test_empty_signature_is_safe():
    empty = MinHashSignature(num_hashes=1024, seed=42)  # never updated
    assert MinHashSignature.containment(empty, _sig(A)) == (0.0, 0.0)
    assert MinHashSignature.containment_ani(empty, _sig(A), 21) == 0.0


def test_containment_survives_serialization():
    sa, sb = _sig(A), _sig(B)
    expected = MinHashSignature.max_containment(sa, sb)
    ra = MinHashSignature.from_bytes(sa.to_bytes())
    rb = MinHashSignature.from_bytes(sb.to_bytes())
    # cardinality (unique_kmer_count) is serialized, so containment is identical
    assert MinHashSignature.max_containment(ra, rb) == pytest.approx(expected)


def test_invalid_k_raises():
    sa, sb = _sig(A), _sig(B)
    with pytest.raises(ValueError):
        MinHashSignature.containment_ani(sa, sb, 0)


def test_comparability_guard_propagates():
    sa = _sig(A, seed=42)
    sb = _sig(B, seed=7)  # different seed -> incomparable
    with pytest.raises(ValueError):
        MinHashSignature.containment(sa, sb)
