"""Tests for the containment-ANI within-clade distance used by reassortment
Stage 2.

The point of the change: a truncated-but-identical segment must read as a
near-zero distance, not a large outlier, so segment length no longer fabricates
within-clade reassortment signals.
"""

import logging
import random

import pytest

from influenza_genotyper.config import (
    KmerConfig, ReassortmentConfig,
)
from influenza_genotyper.core.kmer_extractor import KmerExtractor
from influenza_genotyper.core.reassortment_detector import ReassortmentDetector

logging.disable(logging.CRITICAL)


def _rnd(n, rng):
    return "".join(rng.choice("ACGT") for _ in range(n))


def _mut(s, n, rng):
    s = list(s)
    for _ in range(n):
        i = rng.randrange(len(s))
        s[i] = rng.choice("ACGT")
    return "".join(s)


@pytest.fixture
def det_and_sigs():
    rng = random.Random(2)
    kc = KmerConfig()
    ex = KmerExtractor(kc)
    full = _rnd(1700, rng)
    sigs = {
        "identical": ex.extract_signature(full, "HA"),
        "truncated": ex.extract_signature(full[:850], "HA"),
        "near": ex.extract_signature(_mut(full, 5, rng), "HA"),
        "unrelated": ex.extract_signature(_rnd(1700, rng), "HA"),
    }
    ref = ex.extract_signature(full, "HA")
    det = ReassortmentDetector(ReassortmentConfig(), kmer_config=kc)
    return det, ref, sigs


def test_identical_is_zero_distance(det_and_sigs):
    det, ref, sigs = det_and_sigs
    assert det._segment_distance("HA", ref, sigs["identical"]) == pytest.approx(0.0, abs=1e-9)


def test_truncated_subset_is_near_zero(det_and_sigs):
    det, ref, sigs = det_and_sigs
    d = det._segment_distance("HA", ref, sigs["truncated"])
    assert d < 0.01  # length-tolerant: not the ~0.49 a Jaccard distance would give


def test_truncated_is_closer_than_a_few_mutations(det_and_sigs):
    """A truncated-but-identical segment is treated as at least as similar as a
    slightly mutated full-length one — the opposite of Jaccard's behaviour."""
    det, ref, sigs = det_and_sigs
    d_trunc = det._segment_distance("HA", ref, sigs["truncated"])
    d_near = det._segment_distance("HA", ref, sigs["near"])
    assert d_trunc <= d_near


def test_unrelated_is_far(det_and_sigs):
    det, ref, sigs = det_and_sigs
    assert det._segment_distance("HA", ref, sigs["unrelated"]) > 0.5
