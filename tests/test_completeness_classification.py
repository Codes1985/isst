"""Tests for length-based segment completeness classification (no-call gate).

This module is purely additive: it classifies each segment by length into
complete / partial / no_call without changing validity or which segments the
existing pipeline sees.
"""

import pytest

from influenza_genotyper.core.sequence_processor import (
    classify_completeness,
    SequenceProcessor,
    NO_CALL_LENGTH_FRACTION,
)


# HA expected range is (1600, 1800); floor 0.5 -> boundary at 800 bp.
@pytest.mark.parametrize(
    "length,expected_category",
    [
        (1700, "complete"),   # within range
        (1600, "complete"),   # exactly min length
        (1599, "partial"),    # just below min
        (800, "partial"),     # exactly at the floor (inclusive)
        (799, "no_call"),     # just below the floor
        (400, "no_call"),     # well below
    ],
)
def test_classification_boundaries(length, expected_category):
    completeness, category = classify_completeness(length, "HA")
    assert category == expected_category


def test_completeness_is_capped_at_one():
    c, _ = classify_completeness(5000, "HA")  # far longer than max
    assert c == 1.0


def test_unknown_segment_is_treated_as_complete():
    c, cat = classify_completeness(100, "NOT_A_SEGMENT")
    assert cat == "complete"
    assert c is None


def _seq(n):
    return ("ACGT" * (n // 4 + 1))[:n]


def _process(*segments):
    fasta = "\n".join(f">{sid}|HA\n{_seq(length)}" for sid, length in segments)
    return {r.sequence_id: r for r in SequenceProcessor().process_string(fasta)}


def test_segment_records_carry_category_and_completeness():
    recs = _process(("full", 1700), ("part", 1000), ("tiny", 500))
    assert recs["full"].segments["HA"].category == "complete"
    assert recs["part"].segments["HA"].category == "partial"
    assert recs["tiny"].segments["HA"].category == "no_call"
    assert recs["part"].segments["HA"].completeness == pytest.approx(1000 / 1600)


def test_classification_does_not_change_validity_or_visibility():
    """A no-call segment must remain valid and visible to the existing pipeline;
    routing on category is a separate, later concern."""
    recs = _process(("tiny", 500))
    seg = recs["tiny"].segments["HA"]
    assert seg.is_valid is True              # length never flips validity
    assert "HA" in recs["tiny"].valid_segments  # still seen by existing flow


def test_category_views():
    recs = _process(("full", 1700), ("part", 1000), ("tiny", 500))
    assert "HA" in recs["full"].complete_segments
    assert "HA" in recs["part"].partial_segments
    assert "HA" in recs["tiny"].no_call_segments


def test_summary_counts_categories():
    recs = _process(("full", 1700), ("part", 1000), ("tiny", 500))
    summary = SequenceProcessor().get_processing_summary(list(recs.values()))
    assert summary["segment_categories"] == {"complete": 1, "partial": 1, "no_call": 1}


def test_floor_is_configurable_per_instance():
    # Raise the floor so a 1000 bp HA (0.625) becomes a no-call.
    proc = SequenceProcessor(no_call_length_fraction=0.7)
    fasta = f">x|HA\n{_seq(1000)}"
    rec = proc.process_string(fasta)[0]
    assert rec.segments["HA"].category == "no_call"
