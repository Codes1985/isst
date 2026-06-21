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
