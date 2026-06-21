"""Tests for the read-only orphan reporting surface."""

import tempfile

import pytest

from influenza_genotyper.config import DatabaseConfig
from influenza_genotyper.core.database_manager import DatabaseManager
from influenza_genotyper.core.orphan_report import OrphanReporter


def _db_with_ledger():
    db = DatabaseManager(DatabaseConfig(sqlite_path=tempfile.mktemp(suffix=".db")))
    db.initialize()
    with db.connection() as c:
        for s in ("sA", "sB", "sC", "sD"):
            c.execute(
                "INSERT INTO sequences (sequence_id, subtype, created_at, updated_at) "
                "VALUES (?, 'H1N1pdm09', '2026-06-01', '2026-06-01')", (s,))

        def ins(sid, seg, cat, comp, nc, nd, entered, reason=None, allele=None, exited=None):
            c.execute(
                """INSERT INTO orphan_events
                   (sequence_id, segment_name, cluster_version, category, completeness,
                    nearest_cluster, nearest_distance, entered_at, exit_reason,
                    exit_allele, exited_at)
                   VALUES (?,?,'v1',?,?,?,?,?,?,?,?)""",
                (sid, seg, cat, comp, nc, nd, entered, reason, allele, exited))

        # open episodes
        ins("sA", "HA", "complete", 1.0, "C1", 0.02, "2026-06-01T00:00:00")  # near miss
        ins("sA", "NA", "complete", 1.0, "C2", 0.30, "2026-06-01T00:00:00")  # coherence w/ sA
        ins("sB", "HA", "partial", 0.6, "C1", 0.10, "2026-06-05T00:00:00")   # partial waiting
        # resolved episodes
        ins("sC", "HA", "complete", 1.0, None, None, "2026-06-01T00:00:00",
            "minted_new", "HA.1.0009", "2026-06-10T00:00:00")               # 9 days
        ins("sD", "HA", "partial", 0.7, None, None, "2026-06-02T00:00:00",
            "resolved_by_completion", "HA.1.0009", "2026-06-04T00:00:00")   # 2 days
    return db


@pytest.fixture
def reporter():
    return OrphanReporter(_db_with_ledger())


def test_category_summary(reporter):
    cs = reporter.category_summary("v1")
    assert cs["totals"] == {"complete": 2, "partial": 1}
    assert cs["total_open"] == 3
    assert cs["by_segment"]["HA"] == {"complete": 1, "partial": 1}
    assert cs["by_segment"]["NA"]["complete"] == 1


def test_coherence_groups_complete_orphans_by_sequence(reporter):
    coh = reporter.coherence("v1")
    top = coh[0]
    assert top["sequence_id"] == "sA"
    assert top["segments"] == ["HA", "NA"]
    assert top["n_segments"] == 2
    # partial-only and resolved sequences are not complete-orphan coherence groups
    assert all(g["sequence_id"] != "sB" for g in coh)


def test_near_misses_sorted_by_distance(reporter):
    nm = reporter.near_misses("v1")
    assert nm[0]["sequence_id"] == "sA" and nm[0]["segment_name"] == "HA"
    assert nm[0]["nearest_distance"] == pytest.approx(0.02)
    dists = [o["nearest_distance"] for o in nm]
    assert dists == sorted(dists)


def test_partial_waiting_grouped_by_segment(reporter):
    pw = reporter.partial_waiting("v1")
    assert set(pw.keys()) == {"HA"}
    assert pw["HA"][0]["sequence_id"] == "sB"


def test_cohort_by_entry_date(reporter):
    cohort = reporter.cohort("v1")
    assert cohort["2026-06-01"] == 2  # sA/HA, sA/NA
    assert cohort["2026-06-05"] == 1  # sB


def test_resolution_outcomes(reporter):
    ro = reporter.resolution_outcomes()
    assert ro["by_door"]["minted_new"] == 1
    assert ro["by_door"]["resolved_by_completion"] == 1
    assert ro["by_door"]["absorbed"] == 0
    assert ro["total"] == 2


def test_time_to_resolution_days(reporter):
    ttr = reporter.time_to_resolution()
    assert ttr["overall"]["count"] == 2
    assert ttr["overall"]["max_days"] == pytest.approx(9.0)
    assert ttr["overall"]["min_days"] == pytest.approx(2.0)
    assert ttr["by_door"]["minted_new"]["median_days"] == pytest.approx(9.0)
    assert ttr["by_door"]["resolved_by_completion"]["median_days"] == pytest.approx(2.0)


def test_persistent_waiters_oldest_first(reporter):
    pw = reporter.persistent_waiters(as_of="2026-06-15T00:00:00")
    assert pw[0]["age_days"] == pytest.approx(14.0)  # sA entries from 06-01
    assert pw[0]["sequence_id"] == "sA"
    ages = [r["age_days"] for r in pw]
    assert ages == sorted(ages, reverse=True)


def test_build_and_render(reporter):
    report = reporter.build("v1")
    assert set(report["snapshot"]) == {
        "category_summary", "coherence", "near_misses", "partial_waiting"
    }
    assert set(report["history"]) == {
        "cohort", "resolution_outcomes", "time_to_resolution", "persistent_waiters"
    }
    text = OrphanReporter.render_text(report)
    assert "Orphan report" in text
    assert "Resolved to date: 2" in text
