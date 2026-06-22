"""End-to-end test for ``repair_allele_subtype`` (the ``repair`` subcommand).

Repair is a destructive maintenance operation — it renames a donor population's
mis-subtyped allele, records a permanent lineage link, and rewrites
``segment_kmers`` / ``genotypes`` in place — yet had no test. This pins its
contract and guards its DB dependency chain (several reader/writer methods it
calls were nearly removed in a dead-code pass).
"""

import random
import tempfile

from influenza_genotyper import GenotypingPipeline, GenotyperConfig
from influenza_genotyper.settings import ClusteringConfig, DatabaseConfig, KmerConfig
from influenza_genotyper.core.kmer_extractor import KmerExtractor


def _pipe():
    cfg = GenotyperConfig.default()
    cfg.clustering = ClusteringConfig(dev_mode=True)
    cfg.database = DatabaseConfig(sqlite_path=tempfile.mktemp(suffix=".db"))
    pipe = GenotypingPipeline(cfg)
    pipe.initialize()
    return pipe


def _seed_misnamed(pipe, allele="PB1.1.0003"):
    """An H3N2 donor sequence whose PB1 carries a subtype-1 (H1N1pdm09) allele."""
    rng = random.Random(5)
    ex = KmerExtractor(KmerConfig())
    sig = ex.extract_signature("".join(rng.choice("ACGT") for _ in range(2200)), "PB1")
    now = "2026-01-01T00:00:00"
    with pipe.db.connection() as c:
        c.execute("INSERT INTO sequences (sequence_id, subtype, created_at, updated_at) "
                  "VALUES (?,?,?,?)", ("donor1", "H3N2", now, now))
        c.execute("INSERT INTO segment_kmers (sequence_id, segment_name, k_value, "
                  "kmer_signature, sequence_length, cluster_version, allele_id, is_orphan, created_at) "
                  "VALUES (?,?,?,?,?,?,?,?,?)",
                  ("donor1", "PB1", 21, sig.to_bytes(), 2200, "v1", allele, 1, now))
        c.execute("INSERT INTO genotypes (sequence_id, genotype_profile, allele_profile, "
                  "cluster_version, created_at) VALUES (?,?,?,?,?)",
                  ("donor1", "-.PB1:c1.-.-.-.-.-.-", allele, "v1", now))


def test_repair_corrects_subtype_and_links_lineage():
    pipe = _pipe()
    _seed_misnamed(pipe)
    assert pipe.db.get_genotype("donor1")["allele_profile"] == "PB1.1.0003"

    res = pipe.repair_allele_subtype("PB1", "H3N2", "PB1.1.0003", "v1")

    assert res["affected_sequences"] == ["donor1"]
    assert res["old_allele"] == "PB1.1.0003"
    # corrected to the donor subtype (H3N2 -> '.3.')
    assert res["new_allele"].split(".")[1] == "3"
    assert res["lineage_link_recorded"] is True
    # both the genotype profile and the underlying segment_kmers row are rewritten
    assert pipe.db.get_genotype("donor1")["allele_profile"] == res["new_allele"]
    with pipe.db.connection() as c:
        row = c.execute("SELECT allele_id FROM segment_kmers WHERE sequence_id='donor1' "
                        "AND segment_name='PB1'").fetchone()
    assert row[0] == res["new_allele"]


def test_repair_is_noop_when_no_sequence_carries_the_allele():
    pipe = _pipe()
    _seed_misnamed(pipe, allele="PB1.1.0003")
    res = pipe.repair_allele_subtype("PB1", "H3N2", "PB1.9.9999", "v1")  # absent allele
    assert res["affected_sequences"] == []
    assert res["lineage_link_recorded"] is False
    # the real misnamed row is untouched
    assert pipe.db.get_genotype("donor1")["allele_profile"] == "PB1.1.0003"


# ---------------------------------------------------------------------------
# Full cycle: drive the misnaming into existence through pipeline.run, then
# repair it. This exercises the real upstream path (the reassortants' foreign
# PB1 clusters and mints under the recipient subtype; the H3N2 donor's PB1
# orphans and inherits it cross-subtype, producing a false Stage 0 event) and
# confirms repair undoes the whole thing — including the persisted event.
# ---------------------------------------------------------------------------

_SEGS = ["PB2", "PB1", "PA", "HA", "NP", "NA", "M", "NS"]
_LEN = {"PB2": 2100, "PB1": 2200, "PA": 2100, "HA": 1600,
        "NP": 1400, "NA": 1350, "M": 900, "NS": 800}


def _write_misnaming_fasta(path):
    rng = random.Random(11)

    def rseq(n):
        return "".join(rng.choice("ACGT") for _ in range(n))

    def mut(s, n):
        s = list(s)
        for _ in range(n):
            i = rng.randrange(len(s))
            s[i] = rng.choice([b for b in "ACGT" if b != s[i]])
        return "".join(s)

    F = rseq(_LEN["PB1"])                       # foreign PB1 lineage
    H1 = {s: rseq(_LEN[s]) for s in _SEGS}       # H1N1 backbone lineage
    H3 = {s: rseq(_LEN[s]) for s in _SEGS}       # H3N2 lineage
    recs = []

    def emit(iso, sub, pb1_base, back):
        for seg in _SEGS:
            src = pb1_base if seg == "PB1" else back[seg]
            recs.append((f"{iso}|{seg}|{sub}", mut(src, 2)))

    # Two H1N1 reassortants (H1 backbone + foreign PB1): their PB1 clusters and
    # is minted PB1.1.0001 — the allele misnamed under the recipient subtype.
    emit("REASS1", "H1N1pdm09", F, H1)
    emit("REASS2", "H1N1pdm09", F, H1)
    # Two pure H3N2 isolates: lineage H3 throughout (so the H3N2 backbone forms).
    emit("PURE1", "H3N2", H3["PB1"], H3)
    emit("PURE2", "H3N2", H3["PB1"], H3)
    # The donor: H3N2 backbone but the foreign PB1 — its PB1 orphans in the H3N2
    # pool and cross-matches PB1.1.0001, producing a false Stage 0 event.
    emit("DONOR", "H3N2", F, H3)

    with open(path, "w") as fh:
        for h, s in recs:
            fh.write(f">{h}\n{s}\n")


def _donor_event_segments(pipe):
    with pipe.db.connection() as c:
        return [r[0] for r in c.execute(
            "SELECT discordant_segments FROM reassortment_events WHERE sequence_id='DONOR'"
        ).fetchall()]


def test_repair_clears_misnaming_produced_by_a_full_pipeline_run(tmp_path):
    fp = str(tmp_path / "misnaming.fasta")
    _write_misnaming_fasta(fp)
    pipe = _pipe()
    pipe.run(fp, cluster_version="v1", detect_reassortment=True)

    # Precondition: the run itself produced the misnaming and a false event.
    donor_pb1 = pipe.db.get_genotype("DONOR")["allele_profile"].split(" | ")[1]
    assert donor_pb1 == "PB1.1.0001", f"expected misnamed PB1.1.0001, got {donor_pb1}"
    assert _donor_event_segments(pipe) == ["PB1"], "expected a false Stage 0 event on DONOR/PB1"

    # Repair the donor population's PB1.
    res = pipe.repair_allele_subtype("PB1", "H3N2", "PB1.1.0001", "v1")
    assert res["affected_sequences"] == ["DONOR"]
    assert res["new_allele"].split(".")[1] == "3"   # corrected to H3N2
    assert res["lineage_link_recorded"] is True

    # Postcondition: allele corrected, false event removed.
    fixed_pb1 = pipe.db.get_genotype("DONOR")["allele_profile"].split(" | ")[1]
    assert fixed_pb1 == res["new_allele"]
    assert fixed_pb1.split(".")[1] == "3"
    assert _donor_event_segments(pipe) == [], "false Stage 0 event was not removed by repair"
