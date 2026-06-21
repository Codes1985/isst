"""End-to-end regression test for cross-subtype reassortment detection.

A reassortant — an isolate declared one subtype but carrying a segment from a
different subtype's lineage — must be caught by Stage 0. That requires the
pipeline to feed the reassorted (cluster-orphan) segment's *own* signature into
nomenclature so the cross-subtype centroid search (Stage 2b) can name it under
the donor subtype; that cross-subtype allele name is the discordance Stage 0
reads.

This guards the pipeline→nomenclature seam: both layers are correct in
isolation, but the wiring that carries orphan signatures across it was missing,
silently disabling Stage 0 for the exact case it was built for. The companion
test pins the invariant that a *non*-matching orphan (a genuinely novel lineage)
is still left unnamed and is not minted at batch time.
"""

import random
import tempfile

from influenza_genotyper import GenotypingPipeline, GenotyperConfig
from influenza_genotyper.settings import ClusteringConfig, DatabaseConfig

SEG_LEN = {"PB2": 2100, "PB1": 2100, "PA": 2100, "HA": 1600,
           "NP": 1400, "NA": 1350, "M": 900, "NS": 800}
SEGS = list(SEG_LEN)


def _rnd(n, rng):
    return "".join(rng.choice("ACGT") for _ in range(n))


def _mut(s, n, rng):
    s = list(s)
    for _ in range(n):
        i = rng.randrange(len(s))
        s[i] = rng.choice([b for b in "ACGT" if b != s[i]])
    return "".join(s)


def _write_fasta(path):
    rng = random.Random(20260621)
    A = {seg: _rnd(SEG_LEN[seg], rng) for seg in SEGS}   # H3N2 lineage
    B = {seg: _rnd(SEG_LEN[seg], rng) for seg in SEGS}   # H1N1pdm09 lineage

    recs = []

    def emit(iso, subtype, base, snps=2, override=None):
        for seg in SEGS:
            src = base[seg] if (override is None or seg not in override) else override[seg]
            recs.append((f"{iso}|{seg}|{subtype}", _mut(src, snps, rng)))

    # H3N2 backbone population (>=2 so segments cluster in dev-mode)
    for i in range(3):
        emit(f"ISL_H3_{i}", "H3N2", A)
    # H1N1pdm09 donor population — registers HA.1.xxxx (the donor centroid)
    for i in range(3):
        emit(f"ISL_H1_{i}", "H1N1pdm09", B)
    # Reassortant: declared H3N2, backbone from lineage A, HA from lineage B
    emit("ISL_REASS", "H3N2", A, override={"HA": B["HA"]})
    # Divergent novel singleton: matches nothing — must stay unnamed ('?')
    D = {seg: _mut(A[seg], int(SEG_LEN[seg] * 0.05), rng) for seg in SEGS}
    emit("ISL_DIV", "H3N2", D, snps=0)

    with open(path, "w") as fh:
        for h, s in recs:
            fh.write(f">{h}\n{s}\n")


def _run():
    fp = tempfile.mktemp(suffix=".fasta")
    _write_fasta(fp)
    cfg = GenotyperConfig.default()
    cfg.clustering = ClusteringConfig(dev_mode=True)
    cfg.database = DatabaseConfig(sqlite_path=tempfile.mktemp(suffix=".db"))
    pipe = GenotypingPipeline(cfg)
    pipe.initialize()
    return pipe.run(fp, cluster_version="v1", detect_reassortment=True)


def test_reassortant_HA_named_under_donor_subtype():
    res = _run()
    ha = res["nomenclature"]["ISL_REASS"]["alleles"].get("HA")
    # Must be named under the donor subtype (H1N1pdm09 -> '.1.'), not left
    # unnamed ('?') and not minted under the host subtype ('.3.').
    assert ha is not None, "reassortant HA left unnamed — cross-subtype naming did not fire"
    assert ha.split(".")[1] == "1", f"reassortant HA named under wrong subtype: {ha}"


def test_reassortant_flagged_by_stage0():
    res = _run()
    report = res["reassortment"]
    stage0 = [e for e in report.events
              if e.detection_stage == 0 and e.sequence_id == "ISL_REASS"]
    assert stage0, "no Stage 0 event raised for the cross-subtype reassortant"
    assert "HA" in stage0[0].discordant_segments
    assert report.deterministic_flags >= 1


def test_novel_orphan_stays_unnamed_not_minted():
    """The fix must not mint alleles for non-matching orphans: a divergent novel
    isolate's segments stay '?' (waiting for a recluster), and no clean isolate
    receives a spurious cross-subtype name."""
    res = _run()
    naming = res["nomenclature"]
    div = naming["ISL_DIV"]["alleles"]
    # The divergent singleton matches no cluster and no cross-subtype donor, so
    # its segments must remain unnamed.
    assert all(v is None for v in div.values()), \
        f"divergent orphan was named (should stay '?'): {div}"
    # Clean H3N2 isolates carry only subtype-3 alleles.
    for i in range(3):
        for seg, name in naming[f"ISL_H3_{i}"]["alleles"].items():
            if name is not None:
                assert name.split(".")[1] == "3", f"clean H3N2 {seg} got {name}"
