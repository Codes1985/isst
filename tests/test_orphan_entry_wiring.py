"""End-to-end tests for orphan-ledger entry wiring.

When a segment is flagged an orphan (a complete that founds nothing, or a
partial that joins nothing), the pipeline opens an orphan_events episode
carrying its category and completeness. No-call segments never reach assignment
and so are never recorded.
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


def _pipe():
    cfg = GenotyperConfig.default()
    cfg.clustering = ClusteringConfig(dev_mode=True)
    cfg.database = DatabaseConfig(sqlite_path=tempfile.mktemp(suffix=".db"))
    p = GenotypingPipeline(cfg)
    p.initialize()
    return p


def _fasta(records):
    fp = tempfile.mktemp(suffix=".fasta")
    with open(fp, "w") as fh:
        fh.write("\n".join(f">{i}|{s}\n{q}" for i, s, q in records))
    return fp


def test_batch_records_orphan_entries_with_category():
    rng = random.Random(7)
    A, B = _rnd(1700, rng), _rnd(1700, rng)
    fp = _fasta([
        ("iso1", "HA", _mut(A, 1, rng)),
        ("iso2", "HA", _mut(A, 1, rng)),
        ("iso3", "HA", _mut(A, 1, rng)),   # cluster
        ("isoX", "HA", _mut(B, 1, rng)),   # lone complete, unrelated -> complete orphan
        ("isoP", "HA", B[:850]),           # partial, unrelated -> partial orphan
        ("isoN", "HA", B[:400]),           # no_call -> excluded
    ])
    pipe = _pipe()
    pipe.run(fp, subtype="H1N1pdm09", cluster_version="v1", detect_reassortment=False)

    episodes = {(o["sequence_id"], o["segment_name"]): o
                for o in pipe.db.get_open_orphans("v1")}

    assert episodes[("isoX", "HA")]["category"] == "complete"
    assert episodes[("isoX", "HA")]["completeness"] == pytest.approx(1.0)
    assert episodes[("isoP", "HA")]["category"] == "partial"
    assert episodes[("isoP", "HA")]["completeness"] < 1.0
    # no_call never recorded
    assert ("isoN", "HA") not in episodes
    # clustered completes are not orphans
    for i in ("iso1", "iso2", "iso3"):
        assert (i, "HA") not in episodes


def test_orphan_entry_is_idempotent_across_reruns():
    rng = random.Random(7)
    A, B = _rnd(1700, rng), _rnd(1700, rng)
    records = [
        ("iso1", "HA", _mut(A, 1, rng)),
        ("iso2", "HA", _mut(A, 1, rng)),
        ("isoX", "HA", _mut(B, 1, rng)),  # complete orphan
    ]
    fp = _fasta(records)
    pipe = _pipe()
    pipe.run(fp, subtype="H1N1pdm09", cluster_version="v1", detect_reassortment=False)
    pipe.run(fp, subtype="H1N1pdm09", cluster_version="v1", detect_reassortment=False)
    episodes = [o for o in pipe.db.get_open_orphans("v1")
                if o["sequence_id"] == "isoX"]
    assert len(episodes) == 1  # one episode per (seq, seg, version)


def test_incremental_records_orphan_entries():
    rng = random.Random(11)
    A, B = _rnd(1700, rng), _rnd(1700, rng)
    pipe = _pipe()
    # Batch forms cluster C1 for lineage A under v1.
    pipe.run(_fasta([
        ("a1", "HA", _mut(A, 1, rng)),
        ("a2", "HA", _mut(A, 1, rng)),
        ("a3", "HA", _mut(A, 1, rng)),
    ]), subtype="H1N1pdm09", cluster_version="v1", detect_reassortment=False)
    # Incremental: an unrelated complete cannot join -> orphan recorded under v1.
    pipe.run_incremental(_fasta([
        ("b1", "HA", _mut(B, 1, rng)),
    ]), subtype="H1N1pdm09", cluster_version="v1", detect_reassortment=False)

    episodes = {(o["sequence_id"], o["segment_name"]): o
                for o in pipe.db.get_open_orphans("v1")}
    assert ("b1", "HA") in episodes
    assert episodes[("b1", "HA")]["category"] == "complete"
