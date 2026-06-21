"""Tests for the Jaccard<->ANI conversion math and the ANI threshold accessor.

The per-segment ANI thresholds are now canonical (set directly per divergence
rate), not seeded from a Jaccard table. These tests pin the conversion formula
and the ANI accessor (subtype adjustment + scalar fallback).
"""

import pytest

from influenza_genotyper.config import (
    ClusteringConfig,
    jaccard_to_ani,
    ani_to_jaccard,
)


@pytest.mark.parametrize("jaccard", [0.50, 0.80, 0.92, 0.95, 0.98, 0.999])
@pytest.mark.parametrize("k", [17, 19, 21])
def test_jaccard_ani_roundtrip(jaccard, k):
    ani = jaccard_to_ani(jaccard, k)
    assert ani_to_jaccard(ani, k) == pytest.approx(jaccard, abs=1e-9)


def test_conversion_is_monotonic_and_bounded():
    # higher Jaccard -> higher ANI, always within [0, 1]
    prev = -1.0
    for j in [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
        ani = jaccard_to_ani(j, 21)
        assert 0.0 <= ani <= 1.0
        assert ani > prev
        prev = ani


def test_conversion_edge_clamps():
    assert jaccard_to_ani(0.0, 21) == 0.0
    assert jaccard_to_ani(1.0, 21) == 1.0
    assert ani_to_jaccard(1.0, 21) == 1.0
    assert ani_to_jaccard(0.0, 21) == 0.0


def test_get_ani_threshold_applies_subtype_adjustment():
    c = ClusteringConfig()
    base = c.get_ani_threshold("HA", "H1N1pdm09", "same")
    h3n2 = c.get_ani_threshold("HA", "H3N2", "same")
    assert base == c.segment_ani_thresholds["HA"]["same"]
    assert h3n2 == pytest.approx(base - 0.00057, abs=1e-9)


def test_unknown_segment_falls_back_to_scalar_default():
    c = ClusteringConfig()
    assert c.get_ani_threshold("ZZZ", "H1N1pdm09", "same") == c.same_ani_threshold
    assert c.get_ani_threshold("ZZZ", "H1N1pdm09", "related") == c.related_ani_threshold
