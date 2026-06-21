"""End-to-end tests for completeness-based routing in the pipeline (unit 3).

Clusters form from complete sequences only; partials may join a formed cluster
via containment but never found one; no-call segments are excluded entirely.
These run the real pipeline against a temporary SQLite database.
"""

import logging
import random
import tempfile

import pytest

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


def _run(records, rng_seed=7):
    fasta = "\n".join(f">{iso}|{seg}\n{seq}" for iso, seg, seq in records)
    fp = tempfile.mktemp(suffix=".fasta")
    with open(fp, "w") as fh:
        fh.write(fasta)
    cfg = GenotyperConfig.default()
    cfg.clustering = ClusteringConfig(dev_mode=True)  # min_cluster_size=2 for small tests
    cfg.database = DatabaseConfig(sqlite_path=tempfile.mktemp(suffix=".db"))
    pipe = GenotypingPipeline(cfg)
    pipe.initialize()
    return pipe.run(fp, subtype="H1N1pdm09", cluster_version="v1", detect_reassortment=False)


def test_complete_forms_partial_joins_nocall_excluded():
    rng = random.Random(7)
    full = _rnd(1700, rng)
    res = _run([
        ("iso1", "HA", _mut(full, 1, rng)),
        ("iso2", "HA", _mut(full, 1, rng)),
        ("iso3", "HA", _mut(full, 1, rng)),
        ("iso4", "HA", full[:850]),   # partial subset -> joins
        ("iso5", "HA", full[:400]),   # no_call -> excluded
    ])
    asg = res["assignments"]
    nom = res["nomenclature"]

    cid = asg["iso1"]["HA"].cluster_id
    # completes cluster together
    for i in ("iso1", "iso2", "iso3"):
        assert not asg[i]["HA"].is_orphan and asg[i]["HA"].cluster_id == cid
    # partial joins the complete-founded cluster and shares its allele
    assert not asg["iso4"]["HA"].is_orphan
    assert asg["iso4"]["HA"].cluster_id == cid
    assert nom["iso4"]["alleles"]["HA"] == nom["iso1"]["alleles"]["HA"]
    # no_call excluded entirely
    assert "HA" not in asg.get("iso5", {})
    assert res["summary"]["total_orphans"] == 0


def test_partial_that_matches_nothing_becomes_orphan_not_a_cluster():
    """A partial that doesn't fit any complete-founded cluster orphans — it does
    not found a cluster of its own."""
    rng = random.Random(8)
    full = _rnd(1700, rng)
    other = _rnd(1700, rng)  # unrelated lineage
    res = _run([
        ("iso1", "HA", _mut(full, 1, rng)),
        ("iso2", "HA", _mut(full, 1, rng)),
        ("iso3", "HA", _mut(full, 1, rng)),
        ("isoP", "HA", other[:850]),   # partial of an unrelated lineage
    ])
    asg = res["assignments"]
    assert asg["isoP"]["HA"].is_orphan is True
    # only the complete lineage formed a cluster
    assert res["summary"]["total_clusters"] == 1


def test_partial_only_lineage_stays_orphan():
    """With no complete example, a partial cannot found a lineage; it stays an
    orphan until a complete arrives."""
    rng = random.Random(9)
    full = _rnd(1700, rng)
    res = _run([
        ("p1", "HA", full[:850]),
        ("p2", "HA", _mut(full, 1, rng)[:860]),
    ])
    asg = res["assignments"]
    assert asg["p1"]["HA"].is_orphan is True
    assert asg["p2"]["HA"].is_orphan is True
    assert res["summary"]["total_clusters"] == 0
