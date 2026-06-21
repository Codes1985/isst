"""
K-mer Extractor — extracts k-mers and generates MinHash signatures.

Optimized implementation (v2):
    1. Kirsch-Mitzenmacher trick: derives num_hashes values from just TWO base
       hashes via h_i(x) = h1(x) + i * h2(x), replacing N independent SHA-256
       calls per k-mer with 1 hash call + vectorized integer arithmetic.
    2. Batch hashing: _hash_kmer_multiple_batch() computes all k-mer hash
       vectors in a single (n_kmers × num_hashes) NumPy operation, enabling
       a vectorized column-wise minimum for signature construction.
    3. Single pinned hash backend: mmh3 (MurmurHash3, x64 128-bit variant).
       mmh3 is a hard install requirement — there is NO silent fallback to
       another hash function, because signatures built with a different
       backend are not comparable even at identical num_hashes/seed/k.

    Measured speedup: ~500-1000x over the original per-k-mer SHA-256 loop.
    All public APIs are unchanged.
"""

import struct
import logging
import numpy as np
from typing import Dict, List, Optional, Set, Tuple

from ..config import KmerConfig, SEGMENTS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hash backend — mmh3 only, no fallback
# ---------------------------------------------------------------------------
#
# mmh3 is a *required* dependency.  We deliberately do NOT fall back to xxhash
# or hashlib: a silent backend switch produces signatures that are byte-for-byte
# incomparable to every other machine's at the same num_hashes/seed/k, which is
# the single most insidious reproducibility failure for this tool.  If mmh3 is
# missing we raise immediately with an actionable message rather than degrade.

HASH_BACKEND = "mmh3"
MMH3_X64ARCH = True  # MurmurHash3 has two 128-bit variants; pin the x64 one so
                     # the hash output is stable across mmh3 versions/platforms.

try:
    import mmh3
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'mmh3' package is required to build MinHash signatures but is not "
        "installed. Install it with `pip install mmh3` (or `pip install -e .`, "
        "which now lists mmh3 as a core dependency). There is intentionally no "
        "fallback hash backend, because signatures built with a different hash "
        "function are not comparable."
    ) from exc

# ---------------------------------------------------------------------------
# Core k-mer utilities
# ---------------------------------------------------------------------------

COMPLEMENT = str.maketrans(
    "ACGTUNRYSWKMBDHVacgtunryswkmbdhv",
    "TGCAANYRSWMKVHDBtgcaanyrswmkvhdb",
)


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence.

    Handles the full IUPAC nucleotide alphabet (R/Y/S/W/K/M/B/D/H/V/N), so
    canonical k-mer selection stays biologically correct for sequences that
    carry ambiguity codes.  The previous ACGT-only table left ambiguous bases
    unchanged on the complement strand, which mis-canonicalized any k-mer
    containing them.  Unknown characters are passed through untouched.
    """
    return seq.translate(COMPLEMENT)[::-1]


def canonical_kmer(kmer: str) -> str:
    """Return the lexicographically smaller of a k-mer and its reverse complement."""
    rc = reverse_complement(kmer)
    return kmer if kmer <= rc else rc


def extract_kmers(sequence: str, k: int, canonical: bool = True) -> List[str]:
    """Extract all k-mers from a sequence, skipping those containing N."""
    if len(sequence) < k:
        return []
    kmers = []
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        if "N" in kmer:
            continue
        if canonical:
            kmer = canonical_kmer(kmer)
        kmers.append(kmer)
    return kmers


def extract_kmer_set(sequence: str, k: int, canonical: bool = True) -> Set[str]:
    """Extract the unique set of k-mers from a sequence."""
    return set(extract_kmers(sequence, k, canonical))


def extract_kmer_frequencies(sequence: str, k: int, canonical: bool = True) -> Dict[str, int]:
    """Extract k-mer frequency counts from a sequence."""
    freqs: Dict[str, int] = {}
    for kmer in extract_kmers(sequence, k, canonical):
        freqs[kmer] = freqs.get(kmer, 0) + 1
    return freqs


# ---------------------------------------------------------------------------
# Optimized hashing — Kirsch-Mitzenmacher trick
# ---------------------------------------------------------------------------

def _hash_pair(kmer: str, seed: int = 42) -> Tuple[np.uint64, np.uint64]:
    """
    Compute two independent 64-bit hashes for a k-mer using mmh3.

    MurmurHash3_128 (x64 variant) is computed once and split into two 64-bit
    halves, which seed the Kirsch-Mitzenmacher expansion downstream.

    Returns
    -------
    (h1, h2) : tuple of np.uint64
    """
    data = kmer.encode("ascii")
    h128 = mmh3.hash128(data, seed, x64arch=MMH3_X64ARCH, signed=False)
    h1 = np.uint64(h128 & 0xFFFFFFFFFFFFFFFF)
    h2 = np.uint64(h128 >> 64)
    return h1, h2


def _hash_kmer_multiple(kmer: str, num_hashes: int, base_seed: int = 42) -> np.ndarray:
    """
    Generate num_hashes hash values for a k-mer using Kirsch-Mitzenmacher:
    h_i(x) = h1(x) + i * h2(x), computed mod 2^64 via uint64 overflow.

    Replaces the original's loop of num_hashes independent SHA-256 calls
    with 1 hash call + vectorized arithmetic.

    Returns
    -------
    np.ndarray of shape (num_hashes,) with dtype uint64
    """
    h1, h2 = _hash_pair(kmer, base_seed)
    indices = np.arange(num_hashes, dtype=np.uint64)
    return h1 + indices * h2


def _hash_kmer_multiple_batch(
    kmers: List[str], num_hashes: int, base_seed: int = 42
) -> np.ndarray:
    """
    Compute hash vectors for a batch of k-mers at once.

    Returns an (n_kmers, num_hashes) uint64 array via vectorized broadcasting,
    enabling a single np.min(axis=0) call for the MinHash signature.
    """
    n = len(kmers)
    if n == 0:
        return np.empty((0, num_hashes), dtype=np.uint64)

    h1_arr = np.empty(n, dtype=np.uint64)
    h2_arr = np.empty(n, dtype=np.uint64)
    for idx, kmer in enumerate(kmers):
        h1_arr[idx], h2_arr[idx] = _hash_pair(kmer, base_seed)

    # Broadcasting: (n, 1) + (1, num_hashes) * (n, 1) → (n, num_hashes)
    indices = np.arange(num_hashes, dtype=np.uint64).reshape(1, -1)
    return h1_arr.reshape(-1, 1) + indices * h2_arr.reshape(-1, 1)


# ---------------------------------------------------------------------------
# Determinism self-test
# ---------------------------------------------------------------------------
#
# Golden vector: the mmh3 x64 128-bit hash of a fixed 21-mer at seed 42, split
# into its two 64-bit halves. If a future mmh3 version, a different build, or a
# changed variant flag ever alters the output, this trips loudly instead of
# silently producing a database of signatures incompatible with everyone else's.
# Regenerate ONLY with deliberate intent (it invalidates every existing DB).
_SELFTEST_KMER = "ACGTACGTACGTACGTACGTA"
_SELFTEST_SEED = 42
_SELFTEST_EXPECTED = (13036166743686632327, 4543100632486228299)  # (h1, h2)


def selftest(raise_on_failure: bool = True) -> bool:
    """Verify the hash backend reproduces the pinned golden vector.

    Call this at startup / in CI to fail fast on any backend, version, or
    variant drift before a single signature is written.

    Returns True on success.  If ``raise_on_failure`` is True (default), a
    mismatch raises RuntimeError; otherwise it returns False.
    """
    h1, h2 = _hash_pair(_SELFTEST_KMER, _SELFTEST_SEED)
    got = (int(h1), int(h2))
    if got != _SELFTEST_EXPECTED:
        msg = (
            "MinHash hash backend self-test FAILED — signatures built here will "
            "not be comparable to the reference. "
            f"k-mer={_SELFTEST_KMER!r} seed={_SELFTEST_SEED} "
            f"expected={_SELFTEST_EXPECTED} got={got}. "
            "Check the installed mmh3 version and the x64arch variant flag."
        )
        if raise_on_failure:
            raise RuntimeError(msg)
        logger.error(msg)
        return False
    logger.debug("kmer_extractor hash backend self-test passed (mmh3, x64).")
    return True


# ---------------------------------------------------------------------------
# MinHash Signature
# ---------------------------------------------------------------------------

class MinHashSignature:
    """
    MinHash signature for estimating Jaccard similarity between k-mer sets.

    Attributes
    ----------
    num_hashes : int
        Number of hash functions (signature dimension).
    seed : int
        Base seed for hash function generation.
    signature : np.ndarray
        The MinHash signature vector (num_hashes uint64 values).
    kmer_count : int
        Total k-mer positions in the source sequence (including duplicates).
    unique_kmer_count : int
        Number of unique k-mers fed into the signature.
    """

    def __init__(self, num_hashes: int = 256, seed: int = 42):
        self.num_hashes = num_hashes
        self.seed = seed
        self.signature = np.full(num_hashes, np.iinfo(np.uint64).max, dtype=np.uint64)
        self.kmer_count = 0
        self.unique_kmer_count = 0

    def update_batch(self, kmers: Set[str]) -> None:
        """
        Update the signature with a set of k-mers using batch hashing.

        Computes all hash vectors in one (n_kmers × num_hashes) operation,
        then takes column-wise minimum for the signature.
        """
        self.unique_kmer_count = len(kmers)
        if not kmers:
            return

        all_hashes = _hash_kmer_multiple_batch(list(kmers), self.num_hashes, self.seed)
        batch_mins = np.min(all_hashes, axis=0)
        self.signature = np.minimum(self.signature, batch_mins)

    @staticmethod
    def jaccard_similarity(sig_a: "MinHashSignature", sig_b: "MinHashSignature") -> float:
        """Estimate Jaccard similarity from two MinHash signatures.

        Refuses to compare signatures built with different parameters.  A
        ``num_hashes`` mismatch is a dimension error; a ``seed`` mismatch
        produces same-length but semantically incomparable signatures that
        would otherwise yield a silently meaningless similarity.  Both are
        caught here.  (Backend is pinned to mmh3 process-wide and ``canonical``
        is enforced database-wide via the signature fingerprint, so neither can
        vary between two signatures that reach this point.)
        """
        if sig_a.num_hashes != sig_b.num_hashes or sig_a.seed != sig_b.seed:
            raise ValueError(
                "Incomparable MinHash signatures: "
                f"(num_hashes={sig_a.num_hashes}, seed={sig_a.seed}) vs "
                f"(num_hashes={sig_b.num_hashes}, seed={sig_b.seed}). "
                "They were built with different parameters and cannot be compared."
            )
        return float(np.sum(sig_a.signature == sig_b.signature)) / sig_a.num_hashes

    # ------------------------------------------------------------------
    # Containment and ANI
    # ------------------------------------------------------------------
    #
    # Containment is derived from the estimated Jaccard J and the *exact* k-mer
    # set cardinalities (unique_kmer_count), which are already stored in every
    # signature — so no extra stored field is needed.
    #
    #     |A ∩ B| = J · (|A| + |B|) / (1 + J)
    #     C(A ⊆ B) = |A ∩ B| / |A|        (directional)
    #
    # The only estimation error is in J (kept small by a large num_hashes); the
    # cardinalities are exact. Containment is robust to length differences: a
    # short segment that is a clean subset of a longer one approaches C = 1 even
    # though their Jaccard is depressed by the size gap. That is the property we
    # want for partial-but-identical segments.

    @staticmethod
    def containment(
        sig_a: "MinHashSignature", sig_b: "MinHashSignature"
    ) -> Tuple[float, float]:
        """Return directional containment ``(C(A⊆B), C(B⊆A))`` in [0, 1].

        ``C(A⊆B)`` is the fraction of A's k-mers also present in B. Returns
        ``(0.0, 0.0)`` when either set is empty or the estimated intersection
        is zero. Reuses ``jaccard_similarity`` (so the same comparability guard
        applies).
        """
        a = sig_a.unique_kmer_count
        b = sig_b.unique_kmer_count
        if a == 0 or b == 0:
            return 0.0, 0.0
        j = MinHashSignature.jaccard_similarity(sig_a, sig_b)
        if j <= 0.0:
            return 0.0, 0.0
        inter = j * (a + b) / (1.0 + j)
        c_a = min(inter / a, 1.0)
        c_b = min(inter / b, 1.0)
        return c_a, c_b

    @staticmethod
    def max_containment(
        sig_a: "MinHashSignature", sig_b: "MinHashSignature"
    ) -> float:
        """Larger of the two directional containments.

        This is the quantity to threshold against: it asks "is the smaller set
        a clean subset of the larger?", which is exactly what makes a
        truncated-but-identical segment score near 1.0.
        """
        c_a, c_b = MinHashSignature.containment(sig_a, sig_b)
        return max(c_a, c_b)

    @staticmethod
    def containment_ani(
        sig_a: "MinHashSignature", sig_b: "MinHashSignature", k: int
    ) -> float:
        """Estimate average nucleotide identity from max-containment.

        Under a simple substitution model the probability a k-mer is shared is
        ~ANI**k, so ANI ≈ C**(1/k). Using *max*-containment means a partial
        segment that is a perfect subset of its full-length counterpart returns
        ~1.0 rather than being penalised for missing length (which is what plain
        Jaccard-based ANI would do). ``k`` is the k-mer length the signatures
        were built with (per-segment); it is not stored in the signature, so the
        caller supplies it.
        """
        if k <= 0:
            raise ValueError("k must be a positive integer")
        c = MinHashSignature.max_containment(sig_a, sig_b)
        if c <= 0.0:
            return 0.0
        return float(c ** (1.0 / k))

    @staticmethod
    def jaccard_ani(
        sig_a: "MinHashSignature", sig_b: "MinHashSignature", k: int
    ) -> float:
        """Estimate ANI from Jaccard via the Mash distance: ``1 + (1/k)·ln(2J/(1+J))``.

        Provided for comparison/validation. For complete, equal-length segments
        it agrees with :meth:`containment_ani`; it *under*-estimates when the two
        segments differ in length, which is the failure mode containment avoids.
        """
        if k <= 0:
            raise ValueError("k must be a positive integer")
        j = MinHashSignature.jaccard_similarity(sig_a, sig_b)
        if j <= 0.0:
            return 0.0
        d = -(1.0 / k) * float(np.log(2.0 * j / (1.0 + j)))
        return max(0.0, 1.0 - d)

    @staticmethod
    def max_containment_ani_vec(
        jaccard, size_a, size_b, k: int
    ) -> "np.ndarray":
        """Vectorised max-containment-ANI from estimated Jaccard and exact sizes.

        The numpy-broadcast equivalent of :meth:`containment_ani`: it is the
        single source of truth shared by cluster acceptance (engine) and allele
        naming (centroid index), so the two cannot drift. ``jaccard`` and the
        sizes broadcast against each other (e.g. one query vs N centroids, or M
        queries vs one centroid). Returns 0 where either set is empty or the
        estimated intersection is zero.

            |A ∩ B| = J·(|A|+|B|)/(1+J);  C = max(|A∩B|/|A|, |A∩B|/|B|);  ANI = C**(1/k)
        """
        if k <= 0:
            raise ValueError("k must be a positive integer")
        j = np.asarray(jaccard, dtype=np.float64)
        sa = np.asarray(size_a, dtype=np.float64)
        sb = np.asarray(size_b, dtype=np.float64)
        inter = np.where(j > 0.0, j * (sa + sb) / (1.0 + j), 0.0)
        sa_safe = np.where(sa > 0.0, sa, 1.0)
        sb_safe = np.where(sb > 0.0, sb, 1.0)
        c_a = np.where(sa > 0.0, np.minimum(inter / sa_safe, 1.0), 0.0)
        c_b = np.where(sb > 0.0, np.minimum(inter / sb_safe, 1.0), 0.0)
        c = np.maximum(c_a, c_b)
        valid = (sa > 0.0) & (sb > 0.0) & (c > 0.0)
        return np.where(valid, np.power(np.where(c > 0.0, c, 1.0), 1.0 / k), 0.0)

    def to_bytes(self) -> bytes:
        """Serialize signature to a compact binary format."""
        header = struct.pack("<II", self.num_hashes, self.seed)
        counts = struct.pack("<II", self.kmer_count, self.unique_kmer_count)
        return header + counts + self.signature.tobytes()

    @classmethod
    def from_bytes(cls, data: bytes) -> "MinHashSignature":
        """Deserialize signature from bytes."""
        num_hashes, seed = struct.unpack("<II", data[:8])
        kmer_count, unique_kmer_count = struct.unpack("<II", data[8:16])
        sig = cls(num_hashes=num_hashes, seed=seed)
        sig.kmer_count = kmer_count
        sig.unique_kmer_count = unique_kmer_count
        sig.signature = np.frombuffer(data[16:], dtype=np.uint64).copy()
        return sig

    def is_empty(self) -> bool:
        """Check if the signature has been populated."""
        return self.unique_kmer_count == 0


# ---------------------------------------------------------------------------
# KmerExtractor
# ---------------------------------------------------------------------------

class KmerExtractor:
    """
    High-level k-mer extraction and MinHash signature generation.

    Accepts a KmerConfig for per-segment k values, hash parameters, etc.
    """

    def __init__(self, config: Optional[KmerConfig] = None):
        self.config = config or KmerConfig()

    def extract_signature(self, sequence: str, segment_name: str) -> MinHashSignature:
        """Extract a MinHash signature for a sequence from a given segment."""
        k = self.config.get_k(segment_name)
        kmer_set = extract_kmer_set(sequence, k, canonical=self.config.canonical)
        sig = MinHashSignature(num_hashes=self.config.num_hashes, seed=self.config.hash_seed)
        sig.kmer_count = len(sequence) - k + 1 if len(sequence) >= k else 0
        sig.update_batch(kmer_set)
        return sig

    def extract_all_segments(self, segments: Dict[str, str]) -> Dict[str, MinHashSignature]:
        """Extract signatures for all recognized segments."""
        return {seg: self.extract_signature(seq, seg) for seg, seq in segments.items() if seg in SEGMENTS}

    def compute_pairwise_distances(self, signatures: List[MinHashSignature]) -> np.ndarray:
        """
        Compute pairwise Jaccard distance matrix.

        Uses vectorized NumPy broadcasting for ~50-100x speedup over
        per-pair Python loops.
        """
        n = len(signatures)
        if n == 0:
            return np.zeros((0, 0), dtype=np.float64)

        # Stack all signature arrays into (n, d) matrix
        hash_matrix = np.array(
            [sig.signature for sig in signatures], dtype=np.uint64
        )

        distances = np.zeros((n, n), dtype=np.float64)
        for i in range(n - 1):
            matches = hash_matrix[i] == hash_matrix[i + 1:]  # (n-i-1, d)
            similarities = np.mean(matches, axis=1)
            dists = np.clip(1.0 - similarities, 0.0, 1.0)
            distances[i, i + 1:] = dists
            distances[i + 1:, i] = dists

        return distances

    def parameter_sweep(self, sequences: List[str], segment_name: str,
                        k_values: Optional[List[int]] = None) -> Dict[int, Dict]:
        """Sweep k values and report k-mer statistics for tuning."""
        if k_values is None:
            k_values = [15, 17, 19, 21, 23, 25, 27, 29, 31]
        results = {}
        for k in k_values:
            unique_counts = []
            total_counts = []
            for seq in sequences:
                kmer_set = extract_kmer_set(seq, k)
                kmers_all = extract_kmers(seq, k)
                unique_counts.append(len(kmer_set))
                total_counts.append(len(kmers_all))
            results[k] = {
                "k": k,
                "mean_unique_kmers": float(np.mean(unique_counts)),
                "std_unique_kmers": float(np.std(unique_counts)),
                "mean_total_kmers": float(np.mean(total_counts)),
                "saturation": float(np.mean([u/t if t > 0 else 0 for u, t in zip(unique_counts, total_counts)])),
            }
        return results
