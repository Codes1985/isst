"""Tests for recluster-time orphan resolution (exit-door classification).

A recluster closes orphan episodes that it resolves:
  * absorbed              — joined an allele that already existed
  * minted_new            — a complete orphan now part of a freshly minted allele
  * resolved_by_completion — a partial orphan now part of a freshly minted allele
Sequences still orphan (or absent from the recluster) stay open.
"""

import logging
import random
import tempfile

import pytest

from influenza_genotyper.config import GenotyperConfig, DatabaseConfig, ClusteringConfig
from influenza_genotyper.pipeline import GenotypingPipeline
from influenza_genotyper.core.clustering_engine import ClusterAssignment

logging.disable(logging.CRITICAL)


def _rnd(n, rng):
    return "".join(rng.choice("ACGT") for _ in range(n))


def _mut(s, n, rng):
    s = list(s)
    for _ in range(n):
        i = rng.randrange(len(s))
        s[i] = rng.choice("ACGT")
    return "".join(s)


def _fasta(records):
    fp = tempfile.mktemp(suffix=".fasta")
    with open(fp, "w") as fh:
        fh.write("\n".join(f">{i}|{s}\n{q}" for i, s, q in records))
    return fp


def _pipe():
    cfg = GenotyperConfig.default()
    cfg.clustering = ClusteringConfig(dev_mode=True)
    cfg.database = DatabaseConfig(sqlite_path=tempfile.mktemp(suffix=".db"))
    p = GenotypingPipeline(cfg)
    p.initialize()
    return p


def test_recluster_minted_new():
    rng = random.Random(5)
    A, B = _rnd(1700, rng), _rnd(1700, rng)
    pipe = _pipe()
    # v1: A clusters; a lone complete of B is a complete orphan.
    pipe.run(_fasta([
        ("a1", "HA", _mut(A, 1, rng)), ("a2", "HA", _mut(A, 1, rng)),
        ("a3", "HA", _mut(A, 1, rng)), ("isoX", "HA", _mut(B, 1, rng)),
    ]), subtype="H1N1pdm09", cluster_version="v1", detect_reassortment=False)

    # recluster: B now has enough completes to found its own lineage.
    res = pipe.run_recluster(_fasta([
        ("a1", "HA", _mut(A, 1, rng)), ("a2", "HA", _mut(A, 1, rng)),
        ("a3", "HA", _mut(A, 1, rng)), ("isoX", "HA", _mut(B, 1, rng)),
        ("b2", "HA", _mut(B, 1, rng)), ("b3", "HA", _mut(B, 1, rng)),
    ]), subtype="H1N1pdm09", detect_reassortment=False)

    assert res["summary"]["orphans_resolved"]["minted_new"] == 1
    resolutions = {(r["sequence_id"], r["segment_name"]): r
                   for r in pipe.db.get_orphan_resolutions()}
    assert resolutions[("isoX", "HA")]["exit_reason"] == "minted_new"
    assert resolutions[("isoX", "HA")]["cluster_version"] == "v1"
    # no longer waiting
    assert not [o for o in pipe.db.get_open_orphans() if o["sequence_id"] == "isoX"]


def test_recluster_resolved_by_completion():
    rng = random.Random(6)
    A, B = _rnd(1700, rng), _rnd(1700, rng)
    pipe = _pipe()
    # v1: A clusters; a partial of B has no complete B to join -> partial orphan.
    pipe.run(_fasta([
        ("a1", "HA", _mut(A, 1, rng)), ("a2", "HA", _mut(A, 1, rng)),
        ("a3", "HA", _mut(A, 1, rng)), ("isoP", "HA", B[:850]),
    ]), subtype="H1N1pdm09", cluster_version="v1", detect_reassortment=False)
    assert pipe.db.get_open_orphans()  # isoP is waiting

    # recluster: complete B sequences arrive and found B; the partial joins.
    res = pipe.run_recluster(_fasta([
        ("a1", "HA", _mut(A, 1, rng)), ("a2", "HA", _mut(A, 1, rng)),
        ("a3", "HA", _mut(A, 1, rng)), ("isoP", "HA", B[:850]),
        ("b1", "HA", _mut(B, 1, rng)), ("b2", "HA", _mut(B, 1, rng)),
        ("b3", "HA", _mut(B, 1, rng)),
    ]), subtype="H1N1pdm09", detect_reassortment=False)

    assert res["summary"]["orphans_resolved"]["resolved_by_completion"] == 1
    resolutions = {(r["sequence_id"], r["segment_name"]): r
                   for r in pipe.db.get_orphan_resolutions()}
    assert resolutions[("isoP", "HA")]["exit_reason"] == "resolved_by_completion"


def test_resolution_classifier_all_doors_and_still_open():
    """Directly exercise the door classifier over every case, including absorbed
    (hard to trigger reliably end-to-end) and a still-orphan episode."""
    pipe = _pipe()
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat()
    seqs = ["s_mint", "s_compl", "s_abs", "s_open"]
    with pipe.db.connection() as c:
        for s in seqs:
            c.execute(
                "INSERT INTO sequences (sequence_id, subtype, created_at, updated_at) "
                "VALUES (?, 'H1N1pdm09', ?, ?)", (s, now, now))

    # Open episodes under v1.
    pipe.db.record_orphan_entry("s_mint", "HA", "v1", "complete", 1.0)
    pipe.db.record_orphan_entry("s_compl", "HA", "v1", "partial", 0.6)
    pipe.db.record_orphan_entry("s_abs", "HA", "v1", "complete", 1.0)
    pipe.db.record_orphan_entry("s_open", "HA", "v1", "complete", 1.0)

    def asg(sid, orphan):
        return ClusterAssignment(sid, "HA", None if orphan else "C1", 0.0, orphan)

    results = {
        "assignments": {
            "s_mint": {"HA": asg("s_mint", False)},
            "s_compl": {"HA": asg("s_compl", False)},
            "s_abs": {"HA": asg("s_abs", False)},
            "s_open": {"HA": asg("s_open", True)},   # still orphan
        },
        "nomenclature": {
            "s_mint": {"alleles": {"HA": "HA.1.0005"}},   # new allele
            "s_compl": {"alleles": {"HA": "HA.1.0005"}},  # new allele
            "s_abs": {"alleles": {"HA": "HA.1.0001"}},    # pre-existing
            "s_open": {"alleles": {"HA": None}},
        },
    }
    open_before = pipe.db.get_open_orphans()
    counts = pipe._resolve_orphans_after_recluster(
        open_before, results, pre_existing_alleles={"HA.1.0001"}
    )
    assert counts == {"minted_new": 1, "absorbed": 1, "resolved_by_completion": 1}

    doors = {r["sequence_id"]: r["exit_reason"]
             for r in pipe.db.get_orphan_resolutions()}
    assert doors == {
        "s_mint": "minted_new",
        "s_compl": "resolved_by_completion",
        "s_abs": "absorbed",
    }
    # the still-orphan episode remains open
    assert any(o["sequence_id"] == "s_open" for o in pipe.db.get_open_orphans())
