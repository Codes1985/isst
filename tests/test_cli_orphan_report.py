"""End-to-end tests for the `orphan-report` CLI subcommand.

These keep the command wired: they exercise parsing, dispatch, and the actual
render against a seeded ledger, so the reporting layer cannot silently drift
back into being unreachable from the CLI.
"""

import datetime
import json
import logging

from influenza_genotyper import cli
from influenza_genotyper.config import DatabaseConfig
from influenza_genotyper.core.database_manager import DatabaseManager


def _seed(tmp_path):
    """A small ledger: one open near-miss, one open partial, one resolved."""
    db = DatabaseManager(DatabaseConfig(sqlite_path=tmp_path / "g.db"))
    db.initialize()
    now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat()
    with db.connection() as c:
        for sid in ("seqA", "seqB"):
            c.execute(
                "INSERT INTO sequences (sequence_id, subtype, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (sid, "H3N2", now, now),
            )
    db.record_orphan_entry("seqA", "HA", "v1", "complete", 1.0, "HA.3.0007", 0.012)
    db.record_orphan_entry("seqB", "NA", "v1", "partial", 0.60, None, None)
    db.record_orphan_entry("seqB", "PB1", "v1", "complete", 1.0, "C1", 0.03)
    db.record_orphan_exit("seqB", "PB1", "v1", "minted_new", "PB1.3.0019")
    return tmp_path / "g.db"


def test_subcommand_parses_and_dispatches(tmp_path):
    args = cli.build_parser().parse_args(
        ["orphan-report", "--db", str(tmp_path / "x.db"), "--cluster-version", "v1"]
    )
    assert args.subcommand == "orphan-report"
    assert args.cluster_version == "v1"
    assert args.limit == 20
    assert args.json is False


def test_renders_text_report(tmp_path, capsys):
    db_path = _seed(tmp_path)
    args = cli.build_parser().parse_args(
        ["orphan-report", "--db", str(db_path), "--cluster-version", "v1"]
    )
    rc = cli.cmd_orphan_report(args, logging.getLogger("test"))
    out = capsys.readouterr().out

    assert rc == 0
    assert "Orphan report (v1)" in out
    assert "Open: 2" in out                 # seqA/HA complete + seqB/NA partial
    assert "Resolved to date: 1" in out     # seqB/PB1 minted_new
    assert "minted_new 1" in out
    assert "seqA/HA" in out                  # the near-miss line


def test_json_output_is_parseable(tmp_path, capsys):
    db_path = _seed(tmp_path)
    args = cli.build_parser().parse_args(
        ["orphan-report", "--db", str(db_path), "--cluster-version", "v1", "--json"]
    )
    rc = cli.cmd_orphan_report(args, logging.getLogger("test"))
    out = capsys.readouterr().out

    assert rc == 0
    report = json.loads(out)
    assert set(report) >= {"snapshot", "history", "cluster_version"}
    assert report["snapshot"]["category_summary"]["total_open"] == 2


def test_missing_db_returns_error(tmp_path, capsys):
    args = cli.build_parser().parse_args(
        ["orphan-report", "--db", str(tmp_path / "nope.db")]
    )
    rc = cli.cmd_orphan_report(args, logging.getLogger("test"))
    assert rc == 1
