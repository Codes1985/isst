"""Dereplication thresholds are now a standalone, first-class per-segment table —
independent of clustering, with `derep >= same-cluster` enforced only as a
validated invariant, and the old `margin` knob removed.
"""
import pytest

from influenza_genotyper.settings import (
    DereplicationConfig,
    ClusteringConfig,
    DEFAULT_DEREP_ANI,
    SEGMENTS,
)


def test_default_table_is_populated_and_active():
    cfg = DereplicationConfig()
    assert cfg.enabled
    assert cfg.segment_ani == DEFAULT_DEREP_ANI
    assert set(cfg.segment_ani) == set(SEGMENTS)


def test_default_table_is_per_instance_copy():
    """Mutating one config's table must not leak into the shared constant or
    another instance (mutable-default hazard)."""
    a = DereplicationConfig()
    a.segment_ani["HA"] = 0.5
    assert DereplicationConfig().segment_ani["HA"] == DEFAULT_DEREP_ANI["HA"]
    assert DEFAULT_DEREP_ANI["HA"] != 0.5


def test_margin_knob_removed():
    """margin is no longer a configuration field — derep is set via the table."""
    with pytest.raises(TypeError):
        DereplicationConfig(margin=0.5)


def test_default_table_satisfies_invariant():
    """With clustering at ~1% (0.989) and derep at ~5 SNVs (>= 0.993 on every
    segment), the default config satisfies derep >= same-cluster everywhere."""
    assert DereplicationConfig().validate_against(ClusteringConfig()) == []


def test_validate_flags_a_looser_table():
    """A table deliberately looser than same-cluster is reported on every
    offending segment (the over-collapse direction)."""
    loose = DereplicationConfig(segment_ani={seg: 0.95 for seg in SEGMENTS})
    violations = loose.validate_against(ClusteringConfig())
    flagged = {seg for seg, _d, _s in violations}
    assert flagged == set(SEGMENTS)
    for _seg, derep, same in violations:
        assert derep < same


def test_validate_passes_when_tighter():
    """A table at/above same-cluster everywhere yields no violations."""
    tight = DereplicationConfig(segment_ani={seg: 0.99999 for seg in SEGMENTS})
    assert tight.validate_against(ClusteringConfig()) == []


def test_from_clustering_is_bootstrap_only():
    """from_clustering still works as a seed helper and returns a populated
    table, but no longer stores a margin."""
    cfg = DereplicationConfig.from_clustering(ClusteringConfig(), margin=0.5)
    assert set(cfg.segment_ani) == set(SEGMENTS)
    assert not hasattr(cfg, "margin")
    # seeded in the band above same-cluster -> invariant holds for a bootstrap
    assert cfg.validate_against(ClusteringConfig()) == []
