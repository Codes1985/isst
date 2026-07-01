"""Regression tests for the --allow-param-change CLI wiring (fix A3).

The signature-fingerprint guard aborts a run when k-mer/hash parameters differ
from what the database was stamped with. ``--allow-param-change`` is the
supported escape hatch that re-stamps instead of aborting. These tests lock in
that the flag exists, defaults off, and is actually forwarded to
``GenotypingPipeline.initialize(allow_param_change=...)`` by ``cmd_run`` — the
one line of behaviour the fix adds.
"""

import logging
import tempfile

from influenza_genotyper.cli import build_parser, cmd_run
import influenza_genotyper as ig
import influenza_genotyper.cli as cli


def test_flag_parses_and_defaults_off():
    parser = build_parser()
    off = parser.parse_args(["run", "x.fasta"])
    on = parser.parse_args(["run", "x.fasta", "--allow-param-change"])
    assert off.allow_param_change is False
    assert on.allow_param_change is True


class _FakePipeline:
    """Records the kwargs passed to initialize(); no-ops everything else."""
    last_init_kwargs: dict = {}

    def __init__(self, config=None):
        pass

    def initialize(self, allow_param_change=False):
        _FakePipeline.last_init_kwargs = {"allow_param_change": allow_param_change}

    def run(self, **kwargs):
        return {"summary": {}, "genotypes": [], "nomenclature": {}}


def _run_cmd(monkeypatch, argv):
    monkeypatch.setattr(ig, "GenotypingPipeline", _FakePipeline)
    monkeypatch.setattr(cli, "print_summary", lambda *a, **k: None)
    fasta = tempfile.mktemp(suffix=".fasta")
    with open(fasta, "w") as fh:
        fh.write(">iso1|HA\nACGTACGTACGT\n")
    args = build_parser().parse_args(
        ["run", fasta, "--no-tsv", "--no-reassortment", *argv]
    )
    rc = cmd_run(args, logging.getLogger("test"))
    assert rc == 0
    return _FakePipeline.last_init_kwargs


def test_flag_is_forwarded_to_initialize(monkeypatch):
    kwargs = _run_cmd(monkeypatch, ["--allow-param-change"])
    assert kwargs == {"allow_param_change": True}


def test_absent_flag_forwards_false(monkeypatch):
    kwargs = _run_cmd(monkeypatch, [])
    assert kwargs == {"allow_param_change": False}
