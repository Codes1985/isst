"""Stage 0 must run independent of population size, and the report must keep
the deterministic count separate from the statistical rate (Option B).

These pin two coupled behaviours:
  * a single fully-named genome with an internal subtype contradiction is
    flagged by Stage 0 even though n<2 (no population for Stages 1-2), and
  * an incomplete-but-discordant genome reports as a deterministic flag with a
    statistical rate of 0.0, never the old "1 flagged but rate 0%" contradiction
    masquerading as a real statistical rate.
"""
from influenza_genotyper.settings import (
    ReassortmentConfig,
    DereplicationConfig,
    KmerConfig,
    SEGMENTS,
)
from influenza_genotyper.core.genotype_assigner import (
    GenotypeProfile,
    SegmentValue,
    SegmentStatus,
)
from influenza_genotyper.core.reassortment_detector import ReassortmentDetector


def _profile(sid, present_segments):
    """Profile assigned on `present_segments`, missing elsewhere."""
    return GenotypeProfile(
        sequence_id=sid,
        segment_values={
            seg: SegmentValue(SegmentStatus.ASSIGNED, f"c_{seg}")
            for seg in present_segments
        },
        cluster_version="v1",
    )


def _nomen(subtype_by_seg):
    return {"alleles": {seg: f"{seg}.{n}.0001" for seg, n in subtype_by_seg.items()}}


def _detector():
    # Dereplication off keeps these tests about Stage 0 / reporting only.
    cfg = ReassortmentConfig(dereplication=DereplicationConfig(enabled=False))
    return ReassortmentDetector(cfg, kmer_config=KmerConfig())


def test_stage0_flags_single_complete_genome():
    """One fully-named genome, HA disagreeing with the 7-segment majority —
    flagged by Stage 0 despite n<2."""
    subtypes = {seg: 3 for seg in SEGMENTS}
    subtypes["HA"] = 1  # the contradiction
    prof = _profile("solo", SEGMENTS)
    report = _detector().detect_reassortments(
        [prof], signatures=None, nomenclature={"solo": _nomen(subtypes)}
    )
    assert report.flagged_sequences == 1
    assert report.deterministic_flags == 1
    ev = report.events[0]
    assert ev.detection_stage == 0
    assert ev.discordant_segments == ["HA"]


def test_single_genome_no_nomenclature_is_empty():
    """Without naming there is nothing for Stage 0 to compare, and Stages 1-2
    can't run at n<2 — clean empty report, no crash."""
    report = _detector().detect_reassortments([_profile("solo", SEGMENTS)])
    assert report.flagged_sequences == 0
    assert report.events == []
    assert report.reassortment_rate == 0.0


def test_empty_input_is_clean():
    report = _detector().detect_reassortments([])
    assert report.total_sequences == 0
    assert report.flagged_sequences == 0
    assert report.reassortment_rate == 0.0


def test_incomplete_discordant_reports_count_not_false_rate():
    """The case that exposed the old inconsistency: a single incomplete genome
    (below the 0.75 Stage 1/2 gate) that Stage 0 still flags. It must surface as
    a deterministic flag with statistical rate 0.0 — not '1 flagged, 0% rate'
    pretending to be a statistical measurement."""
    present = SEGMENTS[:5]  # 5/8 assigned -> completeness 0.625, below gate
    subtypes = {seg: 3 for seg in present}
    subtypes["HA"] = 1
    prof = _profile("partial", present)
    report = _detector().detect_reassortments(
        [prof], signatures=None, nomenclature={"partial": _nomen(subtypes)}
    )
    assert report.flagged_sequences == 1
    assert report.deterministic_flags == 1
    assert report.statistical_flags == 0
    assert report.sequences_analyzed == 0      # nothing was statistically screened
    assert report.reassortment_rate == 0.0     # honest: no statistical rate to give


def test_rate_is_statistical_only():
    """flagged - deterministic = statistical, and the rate uses only that."""
    from influenza_genotyper.core.reassortment_detector import ReassortmentReport
    r = ReassortmentReport(
        total_sequences=10, sequences_analyzed=4,
        events=[], flagged_sequences=3, deterministic_flags=1,
    )
    assert r.statistical_flags == 2
    assert r.reassortment_rate == 0.5  # 2 statistical / 4 screened, Stage 0 excluded
