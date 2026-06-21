"""True-positive tests for the statistical reassortment stages.

Stage 0 (deterministic) and the dereplication front-end are tested elsewhere.
These pin the two statistical detection paths, which previously had no test that
asserted they actually *fire* — only the reporting arithmetic and the distance
metric were covered:

  * Stage 1 — linkage-disequilibrium depletion (Fisher's exact -> ACAT -> BH).
  * Stage 2 — within-constellation distance outlier (containment-ANI z-score).

A regression in either would otherwise pass the suite silently, the same shape
of gap that masked the cross-subtype Stage 0 wiring bug.
"""

import random

from influenza_genotyper.settings import (
    ReassortmentConfig, DereplicationConfig, KmerConfig, SEGMENTS,
)
from influenza_genotyper.core.genotype_assigner import (
    GenotypeProfile, SegmentValue, SegmentStatus,
)
from influenza_genotyper.core.kmer_extractor import KmerExtractor
from influenza_genotyper.core.reassortment_detector import ReassortmentDetector

SEG_LEN = {"PB2": 2100, "PB1": 2100, "PA": 2100, "HA": 1600,
           "NP": 1400, "NA": 1350, "M": 900, "NS": 800}


def _detector():
    # Dereplication off: these tests isolate the statistical stages.
    cfg = ReassortmentConfig(dereplication=DereplicationConfig(enabled=False))
    return ReassortmentDetector(cfg, kmer_config=KmerConfig())


def _profile(sid, clusters):
    return GenotypeProfile(
        sequence_id=sid,
        segment_values={seg: SegmentValue(SegmentStatus.ASSIGNED, clusters[seg])
                        for seg in SEGMENTS},
        cluster_version="v1",
    )


def test_stage1_flags_linkage_depleted_combination():
    """Two lineages travel together; a genome with a backbone-from-A / HA-from-B
    pairing is an under-represented combination Stage 1 must flag."""
    profiles = []
    for i in range(20):
        profiles.append(_profile(f"A{i}", {seg: "cA" for seg in SEGMENTS}))
    for i in range(20):
        profiles.append(_profile(f"B{i}", {seg: "cB" for seg in SEGMENTS}))
    reass = {seg: "cA" for seg in SEGMENTS}
    reass["HA"] = "cB"
    profiles.append(_profile("REASS", reass))

    report = _detector().detect_reassortments(profiles, signatures=None, nomenclature=None)

    stage1 = [e for e in report.events
              if e.detection_stage == 1 and e.sequence_id == "REASS"]
    assert stage1, "Stage 1 did not flag the linkage-depleted reassortant"
    assert "HA" in stage1[0].discordant_segments
    assert report.statistical_flags >= 1


def test_stage2_flags_within_constellation_outlier():
    """All genomes share one constellation (so Stage 1 sees no depletion), but
    one genome's HA signature is genetically distant from its mates — a
    within-cluster reassortment Stage 2 must flag."""
    rng = random.Random(7)
    ex = KmerExtractor(KmerConfig())

    def rseq(n):
        return "".join(rng.choice("ACGT") for _ in range(n))

    def mut(s, n):
        s = list(s)
        for _ in range(n):
            i = rng.randrange(len(s))
            s[i] = rng.choice([b for b in "ACGT" if b != s[i]])
        return "".join(s)

    base = {seg: rseq(SEG_LEN[seg]) for seg in SEGMENTS}
    outlier_ha = rseq(SEG_LEN["HA"])  # unrelated HA, but labelled the same cluster

    profiles, signatures = [], {}
    for i in range(5):
        sid = f"g{i}"
        profiles.append(_profile(sid, {seg: "c1" for seg in SEGMENTS}))
        sigs = {}
        for seg in SEGMENTS:
            seq = outlier_ha if (seg == "HA" and i == 4) else mut(base[seg], 3)
            sigs[seg] = ex.extract_signature(seq, seg)
        signatures[sid] = sigs

    report = _detector().detect_reassortments(profiles, signatures=signatures, nomenclature=None)

    stage2 = [e for e in report.events
              if e.detection_stage == 2 and e.sequence_id == "g4"]
    assert stage2, "Stage 2 did not flag the within-constellation distance outlier"
    assert "HA" in stage2[0].discordant_segments


def test_clean_population_raises_nothing():
    """A single homogeneous lineage produces no statistical flags."""
    profiles = [_profile(f"A{i}", {seg: "cA" for seg in SEGMENTS}) for i in range(15)]
    report = _detector().detect_reassortments(profiles, signatures=None, nomenclature=None)
    assert report.statistical_flags == 0
