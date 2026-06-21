"""Tests for Jaccard->ANI threshold conversion and the ANI accessor.

The ANI thresholds are a mechanical conversion of the legacy Jaccard thresholds
(to be tuned later). These tests pin the conversion math, verify the stored
table matches the formula, and confirm the legacy Jaccard path is untouched.
"""

import pytest

from influenza_genotyper.config import (
    ClusteringConfig,
    jaccard_to_ani,
    ani_to_jaccard,
    SEGMENTS,
)

SEGMENT_K = {"PB2": 21, "PB1": 21, "PA": 21, "HA": 21,
             "NP": 19, "NA": 19, "M": 17, "NS": 17}


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


def test_stored_ani_table_matches_formula():
    """Every stored ANI threshold equals jaccard_to_ani(J, k) of its Jaccard
    counterpart, to 5 dp — guards against transcription drift."""
    c = ClusteringConfig()
    for seg in SEGMENTS:
        k = SEGMENT_K[seg]
        for level in ("same", "related"):
            j = c.segment_thresholds[seg][level]
            expected = round(jaccard_to_ani(j, k), 5)
            assert c.segment_ani_thresholds[seg][level] == pytest.approx(expected, abs=1e-5), (
                f"{seg}/{level}: stored {c.segment_ani_thresholds[seg][level]} "
                f"!= formula {expected}"
            )


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


def test_legacy_jaccard_threshold_unchanged():
    """The Jaccard accessor must behave exactly as before (still used by the
    current clustering path)."""
    c = ClusteringConfig()
    assert c.get_threshold("HA", "H3N2", "same") == pytest.approx(0.90)   # 0.92 - 0.02
    assert c.get_threshold("PB2", "H1N1pdm09", "same") == pytest.approx(0.98)
