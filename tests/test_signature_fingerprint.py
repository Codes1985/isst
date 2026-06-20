"""Tests for signature determinism and the database fingerprint guard.

These lock the comparability invariants hardened in schema v4:
  * the mmh3 backend reproduces a known golden vector,
  * jaccard_similarity refuses mismatched parameters,
  * the database stamps its signature parameters on first write and refuses
    mismatched runs thereafter (the regression guard for the num_hashes bug).
"""

import pytest

from influenza_genotyper.core.kmer_extractor import (
    selftest,
    MinHashSignature,
    KmerExtractor,
)
from influenza_genotyper.core.database_manager import (
    DatabaseManager,
    SignatureFingerprintMismatch,
)
from influenza_genotyper.config import KmerConfig, DatabaseConfig


def test_backend_selftest_passes():
    """The pinned mmh3 backend must reproduce the golden vector."""
    assert selftest() is True


def test_identical_sequences_are_maximally_similar():
    ex = KmerExtractor(KmerConfig())
    a = ex.extract_signature("ACGT" * 200, "HA")
    b = ex.extract_signature("ACGT" * 200, "HA")
    assert MinHashSignature.jaccard_similarity(a, b) == pytest.approx(1.0)


def test_jaccard_rejects_seed_mismatch():
    a = MinHashSignature(num_hashes=256, seed=42)
    b = MinHashSignature(num_hashes=256, seed=7)
    with pytest.raises(ValueError):
        MinHashSignature.jaccard_similarity(a, b)


def test_jaccard_rejects_num_hashes_mismatch():
    a = MinHashSignature(num_hashes=256, seed=42)
    b = MinHashSignature(num_hashes=1024, seed=42)
    with pytest.raises(ValueError):
        MinHashSignature.jaccard_similarity(a, b)


def test_signature_roundtrip_preserves_params():
    ex = KmerExtractor(KmerConfig())
    sig = ex.extract_signature("ACGT" * 200, "HA")
    restored = MinHashSignature.from_bytes(sig.to_bytes())
    assert restored.num_hashes == sig.num_hashes
    assert restored.seed == sig.seed


def test_fingerprint_stamps_then_validates(tmp_path):
    db = DatabaseManager(DatabaseConfig(sqlite_path=tmp_path / "g.db"))
    db.initialize()
    base = KmerConfig()

    # First write stamps; an identical run is accepted.
    db.ensure_signature_fingerprint(base.signature_fingerprint())
    db.ensure_signature_fingerprint(base.signature_fingerprint())
    assert db.get_signature_fingerprint()["num_hashes"] == base.num_hashes


def test_fingerprint_blocks_num_hashes_mismatch(tmp_path):
    """Regression guard: an incremental run at a different num_hashes is refused."""
    db = DatabaseManager(DatabaseConfig(sqlite_path=tmp_path / "g.db"))
    db.initialize()
    db.ensure_signature_fingerprint(KmerConfig().signature_fingerprint())  # stamp at default

    drifted = KmerConfig()
    drifted.num_hashes = 256
    with pytest.raises(SignatureFingerprintMismatch):
        db.ensure_signature_fingerprint(drifted.signature_fingerprint())


def test_fingerprint_override_restamps(tmp_path):
    db = DatabaseManager(DatabaseConfig(sqlite_path=tmp_path / "g.db"))
    db.initialize()
    db.ensure_signature_fingerprint(KmerConfig().signature_fingerprint())

    drifted = KmerConfig()
    drifted.num_hashes = 256
    db.ensure_signature_fingerprint(drifted.signature_fingerprint(), allow_change=True)
    assert db.get_signature_fingerprint()["num_hashes"] == 256
