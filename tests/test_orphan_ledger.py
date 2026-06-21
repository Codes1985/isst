"""Tests for the orphan lifecycle ledger (schema v5)."""

import datetime

import pytest

from influenza_genotyper.core.database_manager import DatabaseManager
from influenza_genotyper.config import DatabaseConfig


def _db(tmp_path):
    db = DatabaseManager(DatabaseConfig(sqlite_path=tmp_path / "g.db"))
    db.initialize()
    now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat()
    with db.connection() as c:
        for sid in ("seqA", "seqB", "seqC"):
            c.execute(
                "INSERT INTO sequences (sequence_id, subtype, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (sid, "H3N2", now, now),
            )
    return db


def test_entry_is_idempotent_per_version(tmp_path):
    db = _db(tmp_path)
    db.record_orphan_entry("seqA", "HA", "v1", "complete", 1.0, "C7", 0.06)
    db.record_orphan_entry("seqA", "HA", "v1", "complete", 1.0, "C7", 0.06)  # repeat
    assert len(db.get_open_orphans("v1")) == 1


def test_same_segment_can_reenter_under_new_version(tmp_path):
    db = _db(tmp_path)
    db.record_orphan_entry("seqA", "HA", "v1", "complete", 1.0)
    db.record_orphan_entry("seqA", "HA", "v2", "complete", 1.0)  # different version, allowed
    assert len(db.get_open_orphans()) == 2


def test_category_counts(tmp_path):
    db = _db(tmp_path)
    db.record_orphan_entry("seqA", "HA", "v1", "complete", 1.0)
    db.record_orphan_entry("seqC", "HA", "v1", "complete", 0.99)
    db.record_orphan_entry("seqB", "PB1", "v1", "partial", 0.62)
    counts = {(r["segment_name"], r["category"]): r["n"]
              for r in db.count_open_orphans_by_category("v1")}
    assert counts == {("HA", "complete"): 2, ("PB1", "partial"): 1}


def test_exit_closes_episode_and_records_door(tmp_path):
    db = _db(tmp_path)
    db.record_orphan_entry("seqA", "HA", "v1", "complete", 1.0)
    assert db.record_orphan_exit("seqA", "HA", "v1", "minted_new", "HA.3.0051") is True
    assert db.get_open_orphans("v1") == []
    res = db.get_orphan_resolutions("v1")
    assert len(res) == 1
    assert res[0]["exit_reason"] == "minted_new"
    assert res[0]["exit_allele"] == "HA.3.0051"
    assert res[0]["entered_at"] is not None and res[0]["exited_at"] is not None


def test_double_close_and_unknown_close_return_false(tmp_path):
    db = _db(tmp_path)
    db.record_orphan_entry("seqA", "HA", "v1", "complete", 1.0)
    assert db.record_orphan_exit("seqA", "HA", "v1", "absorbed") is True
    assert db.record_orphan_exit("seqA", "HA", "v1", "absorbed") is False   # already closed
    assert db.record_orphan_exit("nope", "NA", "v1", "absorbed") is False   # never recorded


def test_open_filters(tmp_path):
    db = _db(tmp_path)
    db.record_orphan_entry("seqA", "HA", "v1", "complete", 1.0)
    db.record_orphan_entry("seqB", "PB1", "v1", "partial", 0.62)
    assert len(db.get_open_orphans("v1", category="partial")) == 1
    assert len(db.get_open_orphans("v1", segment_name="HA")) == 1


def test_invalid_values_raise(tmp_path):
    db = _db(tmp_path)
    with pytest.raises(ValueError):
        db.record_orphan_entry("seqA", "HA", "v1", "bogus_category")
    db.record_orphan_entry("seqA", "HA", "v1", "complete", 1.0)
    with pytest.raises(ValueError):
        db.record_orphan_exit("seqA", "HA", "v1", "teleported")


def test_fk_cascade_removes_orphan_history(tmp_path):
    db = _db(tmp_path)
    db.record_orphan_entry("seqA", "HA", "v1", "complete", 1.0)
    with db.connection() as c:
        c.execute("DELETE FROM sequences WHERE sequence_id = 'seqA'")
    assert db.get_open_orphans("v1") == []
