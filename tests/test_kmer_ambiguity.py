"""Regression tests for k-mer ambiguity handling (fix A1).

Locks in two coupled guarantees:

  * ``extract_kmers`` drops any window containing a base outside the concrete
    ACGT alphabet — ``N`` *and* every other IUPAC ambiguity code — so an
    ambiguity-bearing sequence never contributes foreign k-mers that would
    silently deflate containment/Jaccard against an otherwise-identical genome.
  * The signature fingerprint records the k-mer alphabet, so a database built
    under the previous (N-only) filtering is flagged on the next run rather than
    silently mixing incomparable signatures.
"""

import pytest

from influenza_genotyper.core.kmer_extractor import (
    extract_kmers,
    extract_kmer_set,
    KmerExtractor,
)
from influenza_genotyper.core.database_manager import (
    DatabaseManager,
    SignatureFingerprintMismatch,
)
from influenza_genotyper.config import KmerConfig, DatabaseConfig


_AMBIGUITY_CODES = list("NRYSWKMBDHV")


# ── k-mer extraction ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("code", _AMBIGUITY_CODES)
def test_ambiguity_codes_are_dropped_from_kmers(code):
    """No emitted k-mer may contain a base outside ACGT, for any IUPAC code."""
    seq = "ACGTACGTACGT" + code + "ACGTACGTACGT"
    kmers = extract_kmers(seq, 7)
    assert kmers, "expected some pure-ACGT k-mers to survive"
    assert all(set(km) <= set("ACGT") for km in kmers)


@pytest.mark.parametrize("code", _AMBIGUITY_CODES)
def test_ambiguous_base_behaves_like_N(code):
    """A single ambiguity code yields the same k-mer set as an N in the same
    position: both drop exactly the windows spanning that base and keep the
    rest. This is the property that stops ambiguity codes from polluting the
    signature."""
    left, right = "ACGTACGTACGT", "ACGTACGTACGT"
    with_code = extract_kmer_set(left + code + right, 7)
    with_n = extract_kmer_set(left + "N" + right, 7)
    assert with_code == with_n


def test_pure_acgt_sequence_is_not_over_filtered():
    """A clean sequence still yields every window (regression guard against the
    stricter filter dropping valid k-mers)."""
    seq = "ACGTACGTACGTACGTACGT"  # no N, no ambiguity
    k = 7
    assert len(extract_kmers(seq, k, canonical=False)) == len(seq) - k + 1


def test_ambiguous_kmer_set_is_subset_of_resolved():
    """The k-mers from an ambiguity-bearing sequence must be a subset of those
    from the fully-resolved sequence — never a foreign k-mer."""
    resolved = "ACGTACGTACGTAACGTACGTACGT"
    ambiguous = resolved[:12] + "R" + resolved[13:]  # one base -> R
    assert extract_kmer_set(ambiguous, 7) <= extract_kmer_set(resolved, 7)


def test_signature_size_ignores_ambiguity_windows():
    """The signature's unique-k-mer count reflects only real ACGT k-mers."""
    ex = KmerExtractor(KmerConfig())
    clean = ex.extract_signature("ACGT" * 200, "HA")
    # Insert a stretch of ambiguity codes; those windows must not inflate the set.
    dirty_seq = ("ACGT" * 100) + "RYSWK" + ("ACGT" * 100)
    dirty = ex.extract_signature(dirty_seq, "HA")
    assert dirty.unique_kmer_count <= clean.unique_kmer_count


# ── fingerprint coupling ─────────────────────────────────────────────────────

def test_fingerprint_records_kmer_alphabet():
    fp = KmerConfig().signature_fingerprint()
    assert fp.get("kmer_alphabet") == "ACGT"


def test_legacy_fingerprint_without_alphabet_is_rejected(tmp_path):
    """A database stamped before the ambiguity fix (no ``kmer_alphabet`` key)
    must be flagged on the next run, not silently accepted."""
    db = DatabaseManager(DatabaseConfig(sqlite_path=tmp_path / "g.db"))
    db.initialize()

    legacy = KmerConfig().signature_fingerprint()
    legacy.pop("kmer_alphabet")           # simulate a pre-fix stamp
    db.ensure_signature_fingerprint(legacy)

    with pytest.raises(SignatureFingerprintMismatch):
        db.ensure_signature_fingerprint(KmerConfig().signature_fingerprint())
