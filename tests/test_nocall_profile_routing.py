"""Regression tests for no-call routing into the genotype profile (fix A2).

A ``no_call`` segment is excluded from clustering and naming, so it must surface
in the genotype profile as **missing** (``-``), not as an **orphan** (``?``).
Before the fix the profile derived its "available" segments from
``valid_segments`` (a base-quality gate), which still contains a clean-but-short
no-call segment, so it was mislabelled an orphan — inflating the orphan view and
perturbing the constellation key (``?`` vs ``None``).

These run the real pipeline against a temporary SQLite database.
"""

import logging
import random
import tempfile

from influenza_genotyper.config import GenotyperConfig, DatabaseConfig, ClusteringConfig
from influenza_genotyper.pipeline import GenotypingPipeline

logging.disable(logging.CRITICAL)


def _rnd(n, rng):
    return "".join(rng.choice("ACGT") for _ in range(n))


def _mut(s, n, rng):
    s = list(s)
    for _ in range(n):
        i = rng.randrange(len(s))
        s[i] = rng.choice("ACGT")
    return "".join(s)


def _run(records, seed=7):
    fasta = "\n".join(f">{iso}|{seg}\n{seq}" for iso, seg, seq in records)
    fp = tempfile.mktemp(suffix=".fasta")
    with open(fp, "w") as fh:
        fh.write(fasta)
    cfg = GenotyperConfig.default()
    cfg.clustering = ClusteringConfig(dev_mode=True)  # min_cluster_size=2
    cfg.database = DatabaseConfig(sqlite_path=tempfile.mktemp(suffix=".db"))
    pipe = GenotypingPipeline(cfg)
    pipe.initialize()
    res = pipe.run(fp, subtype="H1N1pdm09", cluster_version="v1",
                   detect_reassortment=False)
    return {p.sequence_id: p for p in res["genotypes"]}, res


def test_no_call_segment_is_missing_not_orphan():
    rng = random.Random(7)
    full = _rnd(1700, rng)
    # iso1 carries a complete HA (clusters with iso2) and a no-call NA (400 bp;
    # NA min is 1350, floor 675, so 400 is well below the no-call floor).
    profiles, _ = _run([
        ("iso1", "HA", _mut(full, 1, rng)),
        ("iso1", "NA", full[:400]),
        ("iso2", "HA", _mut(full, 1, rng)),
    ])
    iso1 = profiles["iso1"]

    assert "NA" in iso1.missing_segments
    assert "NA" not in iso1.orphan_segments
    # Constellation key uses None (missing marker) for NA, not the orphan "?".
    assert iso1.segment_clusters["NA"] is None
    # HA still clustered normally.
    assert iso1.segment_clusters["HA"] is not None


def test_genuine_cluster_orphan_still_shows_as_orphan():
    """Guard the other direction: excluding no-calls must NOT swallow real
    cluster-orphans. A lone complete segment (no cluster mate) is a true orphan
    and must remain one (``?``, not ``-``)."""
    rng = random.Random(3)
    profiles, _ = _run([("solo", "HA", _rnd(1700, rng))])
    solo = profiles["solo"]

    assert "HA" in solo.orphan_segments
    assert "HA" not in solo.missing_segments
    assert solo.segment_clusters["HA"] == "?"


def test_no_call_does_not_count_toward_assignment_completeness():
    """A no-call segment is neither assigned nor orphaned, so it does not lift
    the profile's assigned-segment completeness."""
    rng = random.Random(11)
    full = _rnd(1700, rng)
    profiles, _ = _run([
        ("a", "HA", _mut(full, 1, rng)),
        ("a", "NA", full[:400]),   # no_call
        ("b", "HA", _mut(full, 1, rng)),
    ])
    # Only HA is assigned -> 1 of 8 segments.
    assert profiles["a"].assigned_count == 1
