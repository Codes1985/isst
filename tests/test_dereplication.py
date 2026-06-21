"""Dereplication tests for the reassortment detector.

Covers the two claims that motivate the feature:

  * a near-identical outbreak collapses to a single representative, and
  * a reassortant (distinct on its donor segment) survives as its own genome,

plus the ``_same_copy`` predicate's all-segments rule and its partial-segment
guard, and the end-to-end clutter fix (one event with a count, not N duplicate
calls).
"""
import random

import pytest

from influenza_genotyper.settings import (
    ClusteringConfig,
    ReassortmentConfig,
    DereplicationConfig,
    KmerConfig,
    SEGMENTS,
    SEGMENT_LENGTH_RANGES,
)
from influenza_genotyper.core.kmer_extractor import KmerExtractor
from influenza_genotyper.core.genotype_assigner import (
    GenotypeProfile,
    SegmentValue,
    SegmentStatus,
)
from influenza_genotyper.core.reassortment_detector import ReassortmentDetector


# --------------------------------------------------------------------------
# Synthetic sequence helpers
# --------------------------------------------------------------------------

_EXTRACTOR = KmerExtractor(KmerConfig())


def _mid_len(seg: str) -> int:
    lo, hi = SEGMENT_LENGTH_RANGES[seg]
    return (lo + hi) // 2


def _randseq(n: int, rng: random.Random) -> str:
    return "".join(rng.choice("ACGT") for _ in range(n))


def _mutate(seq: str, frac: float, rng: random.Random) -> str:
    if frac <= 0:
        return seq
    s = list(seq)
    for i in rng.sample(range(len(s)), int(len(s) * frac)):
        s[i] = rng.choice("ACGT")
    return "".join(s)


def _base_genome(rng: random.Random) -> dict:
    """One sequence per segment, lengths at the midpoint of the expected range."""
    return {seg: _randseq(_mid_len(seg), rng) for seg in SEGMENTS}


def _sigs(seqs: dict) -> dict:
    """segment -> MinHashSignature for a genome's sequences."""
    return {seg: _EXTRACTOR.extract_signature(seq, seg) for seg, seq in seqs.items()}


def _profile(sid: str, clusters: dict) -> GenotypeProfile:
    """A fully-assigned profile with the given per-segment cluster ids."""
    return GenotypeProfile(
        sequence_id=sid,
        segment_values={
            seg: SegmentValue(SegmentStatus.ASSIGNED, clusters[seg]) for seg in SEGMENTS
        },
        cluster_version="v1",
    )


_CONSTELLATION_A = {seg: f"cA_{seg}" for seg in SEGMENTS}


def _detector(margin: float = 0.5, enabled: bool = True) -> ReassortmentDetector:
    derep = DereplicationConfig.from_clustering(
        ClusteringConfig(), margin=margin
    )
    derep.enabled = enabled
    return ReassortmentDetector(
        ReassortmentConfig(dereplication=derep), kmer_config=KmerConfig()
    )


# --------------------------------------------------------------------------
# Core claim: outbreak collapses, reassortant survives
# --------------------------------------------------------------------------

def test_outbreak_collapses_reassortant_survives():
    rng = random.Random(7)
    base = _base_genome(rng)

    profiles, signatures = [], {}

    # (1) Outbreak: 5 exact copies of `base` — same constellation, identical seq.
    base_sig = _sigs(base)
    for i in range(5):
        sid = f"out_{i}"
        profiles.append(_profile(sid, _CONSTELLATION_A))
        signatures[sid] = base_sig  # identical signatures

    # (2) Independents: same constellation as the outbreak, but ~2% divergent
    #     sequence — same clade, genuinely different copies.
    for i in range(3):
        sid = f"ind_{i}"
        seqs = {seg: _mutate(base[seg], 0.02, rng) for seg in SEGMENTS}
        profiles.append(_profile(sid, _CONSTELLATION_A))
        signatures[sid] = _sigs(seqs)

    # (3) Reassortant: lineage-A on 7 segments, a divergent donor on HA placed in
    #     a different cluster -> different constellation -> its own bucket.
    reass_clusters = dict(_CONSTELLATION_A)
    reass_clusters["HA"] = "cB_HA"
    reass_seqs = dict(base)
    reass_seqs["HA"] = _randseq(_mid_len("HA"), rng)  # unrelated donor segment
    profiles.append(_profile("reass", reass_clusters))
    signatures["reass"] = _sigs(reass_seqs)

    random.Random(1).shuffle(profiles)  # order independence

    det = _detector()
    reps, members = det.dereplicate(profiles, signatures)

    rep_ids = {p.sequence_id for p in reps}

    # 9 genomes -> 5 representatives: 1 outbreak + 3 independents + 1 reassortant
    assert len(reps) == 5

    # the outbreak collapsed to exactly one representative covering all 5 copies
    outbreak_groups = [
        ids for ids in members.values() if all(i.startswith("out_") for i in ids)
    ]
    assert len(outbreak_groups) == 1
    assert sorted(outbreak_groups[0]) == [f"out_{i}" for i in range(5)]

    # the reassortant is its own representative, never merged into lineage A
    assert "reass" in rep_ids
    assert members["reass"] == ["reass"]

    # each independent stands alone (same clade != same copy)
    for i in range(3):
        assert f"ind_{i}" in rep_ids
        assert members[f"ind_{i}"] == [f"ind_{i}"]


def test_disabled_is_identity():
    rng = random.Random(3)
    base = _base_genome(rng)
    base_sig = _sigs(base)
    profiles, signatures = [], {}
    for i in range(4):  # an outbreak that would collapse if enabled
        sid = f"out_{i}"
        profiles.append(_profile(sid, _CONSTELLATION_A))
        signatures[sid] = base_sig

    det = _detector(enabled=False)
    reps, members = det.dereplicate(profiles, signatures)
    assert len(reps) == 4
    assert all(members[p.sequence_id] == [p.sequence_id] for p in profiles)


def test_no_signatures_is_identity():
    rng = random.Random(4)
    base = _base_genome(rng)
    profiles = [_profile(f"out_{i}", _CONSTELLATION_A) for i in range(3)]
    det = _detector()
    reps, members = det.dereplicate(profiles, signatures=None)
    assert len(reps) == 3


# --------------------------------------------------------------------------
# The _same_copy predicate: all-segments rule and partial guard
# --------------------------------------------------------------------------

def test_same_copy_blocks_on_a_real_segment_difference():
    """Identical on 7 segments but genuinely different (full-length) on HA -> not
    the same copy: a comparable segment over the line disqualifies the pair."""
    rng = random.Random(11)
    base = _base_genome(rng)
    other = dict(base)
    other["HA"] = _randseq(_mid_len("HA"), rng)  # full-length but unrelated HA

    det = _detector()
    thr = det.config.dereplication.segment_ani
    assert det._same_copy(_sigs(base), _sigs(other), thr) is False


def test_same_copy_skips_truncated_segment():
    """A truncated segment is untrustworthy for containment, so it is skipped
    rather than allowed to block or falsely confirm; the other matching
    segments still satisfy the rule."""
    rng = random.Random(12)
    base = _base_genome(rng)
    trunc = dict(base)
    trunc["HA"] = base["HA"][: int(len(base["HA"]) * 0.6)]  # 60% prefix

    det = _detector()
    thr = det.config.dereplication.segment_ani
    # HA k-mer ratio ~0.6 < min_kmer_ratio (0.85) -> skipped; 7 others match.
    assert det._same_copy(_sigs(base), _sigs(trunc), thr) is True


def test_same_copy_requires_min_shared_segments():
    """Too few comparable matching segments -> not collapsed even if all present
    ones agree."""
    rng = random.Random(13)
    base = _base_genome(rng)
    # keep only 4 segments present in both
    keep = SEGMENTS[:4]
    a = {seg: _sigs(base)[seg] for seg in keep}
    b = {seg: _sigs(base)[seg] for seg in keep}
    det = _detector()
    thr = det.config.dereplication.segment_ani
    assert det._same_copy(a, b, thr) is False  # 4 < min_shared_segments (6)


# --------------------------------------------------------------------------
# End-to-end: the clutter fix (one event with a count, not N duplicates)
# --------------------------------------------------------------------------

def _reassortant_outbreak(n: int, rng: random.Random):
    """n identical copies of a reassortant: lineage-A majority with a discordant
    (different-subtype) HA allele. Returns (profiles, signatures, nomenclature)."""
    base = _base_genome(rng)
    base_sig = _sigs(base)
    clusters = dict(_CONSTELLATION_A)
    clusters["HA"] = "cB_HA"
    # subtype 1 everywhere except HA (subtype 3) -> Stage 0 discordance
    alleles = {seg: f"{seg}.1.0001" for seg in SEGMENTS}
    alleles["HA"] = "HA.3.0001"

    profiles, signatures, nomenclature = [], {}, {}
    for i in range(n):
        sid = f"rx_{i}"
        profiles.append(_profile(sid, clusters))
        signatures[sid] = base_sig
        nomenclature[sid] = {"alleles": dict(alleles)}
    return profiles, signatures, nomenclature


def test_detection_collapses_duplicate_calls():
    rng = random.Random(21)
    profiles, signatures, nomenclature = _reassortant_outbreak(5, rng)

    # With dereplication ON: one event standing for all 5 copies.
    on = _detector(enabled=True)
    rep_on = on.detect_reassortments(profiles, signatures, nomenclature)
    assert len(rep_on.events) == 1
    ev = rep_on.events[0]
    assert ev.detection_stage == 0
    assert ev.represented_count == 5
    assert sorted(ev.represented_ids) == [f"rx_{i}" for i in range(5)]

    # With dereplication OFF: the same outbreak produces 5 duplicate calls.
    off = _detector(enabled=False)
    rep_off = off.detect_reassortments(profiles, signatures, nomenclature)
    assert len(rep_off.events) == 5
    assert all(e.represented_count == 1 for e in rep_off.events)
