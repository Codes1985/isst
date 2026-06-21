"""
Nomenclature Manager
====================
Assigns stable, meaningful identifiers to segment-level alleles and
full-genome constellation genotypes. Persists all registries to the database.

Allele format:    {segment}.{subtype_num}.{zero-padded ID}
                  e.g. HA.3.0042, PB2.1.0008, NA.2.0007

Constellation:    {HxNx}-{12-char hex hash}
                  e.g. H3N2-A7F2B3E91C4D, H1N1-B3E9C1A52F7B

Allele stability guarantee
--------------------------
HA.3.0042 always refers to the same biological allele, regardless of when
or how many times the dataset has been re-clustered.

This is enforced by a two-stage lookup in assign_allele():

  Stage 1 — cache hit by (segment, subtype_num, internal_cluster_id):
      Fast path.  Handles the common case where the same cluster ID appears
      again within the same run or after an incremental add.

  Stage 2 — centroid similarity search:
      When a cluster ID is not in the cache (e.g. after a re-clustering run
      that assigned new local IDs), the new centroid's MinHash signature is
      compared vectorised against all stored allele centroids for that
      segment/subtype.  If one exceeds the same-cluster Jaccard threshold,
      the existing allele name is reused and the new cluster ID is added as
      an alias in the cache so subsequent lookups hit Stage 1.

  Only if neither stage matches is a new allele name minted.

Centroid signatures must be supplied to assign_allele() (via the
centroid_signature parameter) for Stage 2 to operate.  When no signature
is provided the method falls back to minting a new name, with a warning.
"""

import hashlib
import logging
import re
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from ..config import SEGMENTS, ClusteringConfig, KmerConfig
from .kmer_extractor import MinHashSignature

if TYPE_CHECKING:
    from .database_manager import DatabaseManager

logger = logging.getLogger(__name__)

SUBTYPE_MAP = {
    "H1N1pdm09": {"num": 1, "short": "H1N1"},
    "H1N1":      {"num": 1, "short": "H1N1"},
    "H3N2":      {"num": 3, "short": "H3N2"},
    "H5N1":      {"num": 5, "short": "H5N1"},
    "H5N6":      {"num": 5, "short": "H5N6"},
    "H7N9":      {"num": 7, "short": "H7N9"},
    "H9N2":      {"num": 9, "short": "H9N2"},
}


def subtype_num(subtype: str) -> int:
    entry = SUBTYPE_MAP.get(subtype)
    if entry:
        return entry["num"]
    m = re.match(r"H(\d+)N\d+", subtype)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot determine subtype number for: {subtype}")


def subtype_short(subtype: str) -> str:
    entry = SUBTYPE_MAP.get(subtype)
    if entry:
        return entry["short"]
    m = re.match(r"(H\d+N\d+)", subtype)
    if m:
        return m.group(1)
    raise ValueError(f"Cannot determine short subtype for: {subtype}")


# Number of hex characters from the SHA-256 digest used in a constellation ID.
# 12 hex chars = 48 bits. By the birthday bound, a 50% collision probability is
# only reached past ~20 million distinct constellations per subtype prefix —
# far beyond any realistic surveillance scale — so the ID can be a pure,
# order-independent function of the allele combination with no slide/retry.
# (The previous 4-char / 16-bit width reached 50% at ~300 constellations.)
_CONSTELLATION_HASH_CHARS = 12


# ---------------------------------------------------------------------------
# Centroid index — vectorised similarity search
# ---------------------------------------------------------------------------

class _CentroidIndex:
    """Per-(segment, subtype_num) index of stored allele centroids.

    Holds a (num_alleles, num_hashes) uint64 matrix built lazily from the
    allele registry loaded at startup, and updated incrementally as new
    alleles are registered.  Supports O(1) per-query vectorised Jaccard
    similarity against all stored centroids.

    Implementation note — list-then-stack pattern
    ---------------------------------------------
    Rows are accumulated in ``_pending_rows`` (a Python list of 1-D uint64
    arrays) and only consolidated into ``_matrix`` on the first call to
    ``best_match()``, or explicitly via ``_consolidate()``.

    This avoids the O(n²) memory cost of calling ``np.vstack`` on every
    ``add()`` call (each vstack copies the entire matrix).  During bulk
    startup loading — the common case — all rows are appended to the list
    in O(1) each, and a single ``np.stack`` is performed once at query time.
    Subsequent ``add()`` calls after the first query append to both the
    consolidated matrix (one vstack per post-query add) and the now-empty
    pending list, so the amortised cost remains low.
    """

    def __init__(self):
        # allele_name in insertion order — parallel to rows of _matrix
        self._names: List[str] = []
        # Consolidated (N, d) uint64 matrix; None until first consolidation
        self._matrix: Optional[np.ndarray] = None
        # Rows added since last consolidation — avoids per-add vstack cost
        self._pending_rows: List[np.ndarray] = []
        # Exact unique-k-mer count per centroid, parallel to _names. Needed for
        # containment (the hash matrix alone yields only Jaccard).
        self._sizes: List[int] = []

    def _consolidate(self) -> None:
        """Merge any pending rows into the consolidated matrix.

        Called lazily before the first similarity query and after each
        post-query add.  A no-op when there are no pending rows.
        """
        if not self._pending_rows:
            return
        pending = np.stack(self._pending_rows, axis=0)  # (k, d)
        if self._matrix is None:
            self._matrix = pending
        else:
            self._matrix = np.vstack([self._matrix, pending])
        self._pending_rows = []

    def add(self, allele_name: str, signature: MinHashSignature) -> None:
        """Append a centroid to the index.

        Before the first ``best_match()`` call, rows are buffered in
        ``_pending_rows`` (O(1)).  After the matrix has been consolidated
        once, pending rows are merged immediately so that ``best_match``
        always sees a complete matrix.
        """
        row = signature.signature.astype(np.uint64)  # (d,)
        self._names.append(allele_name)
        self._sizes.append(int(signature.unique_kmer_count))
        if self._matrix is None:
            # Still in accumulation phase — buffer the row
            self._pending_rows.append(row)
        else:
            # Matrix already consolidated; merge now to keep it current
            self._pending_rows.append(row)
            self._consolidate()

    def best_match_with_score(
        self, query: MinHashSignature, threshold: float, k: int
    ) -> Optional[Tuple[str, float]]:
        """Return ``(allele_name, containment_ani)`` for the centroid with the
        highest max-containment-ANI to *query*, provided that ANI meets
        *threshold*.  Returns None otherwise.

        Containment (not plain Jaccard) is used so a partial-but-identical
        segment scores near 1.0 against its full-length lineage centroid instead
        of being penalised for missing length.  ``k`` is the per-segment k-mer
        length the signatures were built with.  The metric is computed by the
        shared :meth:`MinHashSignature.max_containment_ani_vec`, so acceptance
        and naming cannot diverge.
        """
        if not self._names:
            return None
        self._consolidate()
        if self._matrix is None:
            return None

        q = query.signature.reshape(1, -1).astype(np.uint64)   # (1, d)
        jaccard = np.mean(self._matrix == q, axis=1)           # (N,)
        sizes = np.asarray(self._sizes, dtype=np.float64)       # (N,)
        ani = MinHashSignature.max_containment_ani_vec(
            jaccard, query.unique_kmer_count, sizes, k
        )                                                       # (N,)

        best_ani = float(ani.max())
        if best_ani < threshold:
            return None

        # Deterministic tie-break: among centroids at the maximum ANI, return the
        # lexicographically smallest allele name rather than the lowest array
        # index. Names are zero-padded and globally unique, so the smallest name
        # is the oldest lineage — order-independent (DB load order, parallelism).
        tied = np.flatnonzero(ani == best_ani)
        if tied.size == 1:
            return self._names[int(tied[0])], best_ani
        return min(self._names[int(i)] for i in tied), best_ani

    def best_match(
        self, query: MinHashSignature, threshold: float, k: int
    ) -> Optional[str]:
        """Allele name of the best containment-ANI match at/above *threshold*,
        or None.  Thin wrapper over :meth:`best_match_with_score`."""
        result = self.best_match_with_score(query, threshold, k)
        return result[0] if result is not None else None

    def __len__(self) -> int:
        return len(self._names)


# ---------------------------------------------------------------------------
# Nomenclature Manager
# ---------------------------------------------------------------------------

class NomenclatureManager:
    """
    Manages allele and constellation naming with database persistence.

    On initialization, loads existing registries from the database so that
    allele names are stable across sessions.  All new assignments are written
    back to the database immediately.

    Parameters
    ----------
    db : DatabaseManager, optional
        If provided, all registry operations are persisted.
    clustering_config : ClusteringConfig, optional
        Provides per-segment same-cluster Jaccard thresholds used for centroid
        similarity matching.  When omitted, a default ClusteringConfig is used.
    """

    def __init__(
        self,
        db: Optional["DatabaseManager"] = None,
        clustering_config: Optional[ClusteringConfig] = None,
        kmer_config: Optional[KmerConfig] = None,
    ):
        self.db = db
        self._clustering_config = clustering_config or ClusteringConfig()
        self._kmer_config = kmer_config or KmerConfig()

        # (seg, snum, cid) -> allele_name — primary lookup by cluster ID
        self._allele_cache: Dict[Tuple[str, int, str], str] = {}

        # canonical_str -> constellation_id
        self._constellation_cache: Dict[str, str] = {}

        # (seg, snum) -> next allele counter (used when db is None)
        self._allele_counters: Dict[Tuple[str, int], int] = {}

        # (seg, snum) -> _CentroidIndex — vectorised centroid similarity search
        self._centroid_indices: Dict[Tuple[str, int], _CentroidIndex] = {}

        # (internal_cluster_id, segment_name) -> (matched_allele_name, similarity)
        # Populated by Stage 2b cross-subtype matches; consumed by Pass 2 of
        # name_genotypes_batch to record allele_lineage links once both allele
        # names are known.
        self._pending_lineage: Dict[Tuple[str, str], Tuple[str, float]] = {}

    # ------------------------------------------------------------------
    # Startup loading
    # ------------------------------------------------------------------

    def load_from_db(self) -> None:
        """Load existing registries from database. Call after db.initialize()."""
        if self.db is None:
            return
        self._load_from_db()

    def _load_from_db(self) -> None:
        """Load allele registry, counters, constellations, and centroid
        signatures into memory caches."""

        for row in self.db.load_allele_registry():
            # Version-scoped key: matches the key used in assign_allele()
            key = (row["segment_name"], row["subtype_num"],
                   row.get("cluster_version") or "", row["internal_cluster_id"])
            self._allele_cache[key] = row["allele_name"]

            # Rebuild centroid index from stored signatures
            sig_bytes = row.get("centroid_signature")
            if sig_bytes:
                try:
                    sig = MinHashSignature.from_bytes(sig_bytes)
                    idx_key = (row["segment_name"], row["subtype_num"])
                    if idx_key not in self._centroid_indices:
                        self._centroid_indices[idx_key] = _CentroidIndex()
                    self._centroid_indices[idx_key].add(row["allele_name"], sig)
                except Exception as exc:
                    logger.warning(
                        f"Could not deserialise centroid for {row['allele_name']}: {exc}"
                    )

        counters = self.db.load_allele_counters()
        for (seg, snum), next_num in counters.items():
            self._allele_counters[(seg, snum)] = next_num

        for row in self.db.load_constellation_registry():
            self._constellation_cache[row["allele_combination"]] = row["constellation_id"]

        total_centroids = sum(len(idx) for idx in self._centroid_indices.values())
        logger.info(
            f"Loaded nomenclature: {len(self._allele_cache)} alleles "
            f"({total_centroids} with centroid signatures), "
            f"{len(self._constellation_cache)} constellations"
        )

    # ------------------------------------------------------------------
    # Allele naming — core method
    # ------------------------------------------------------------------

    def assign_allele(
        self,
        segment_name: str,
        subtype: str,
        internal_cluster_id: str,
        cluster_version: Optional[str] = None,
        centroid_signature: Optional[MinHashSignature] = None,
    ) -> str:
        """Get or create a stable allele name for a cluster.

        Lookup order
        ------------
        1. Cache by (segment, subtype_num, version, internal_cluster_id) — O(1).
           Handles same-run and incremental-add cases.

        2a. Same-subtype centroid similarity search — vectorised Jaccard
            comparison against all stored allele centroids for this
            segment/subtype.  Reuses an existing allele name if the new
            centroid is within the same-cluster threshold.

        2b. Cross-subtype centroid similarity search — if no same-subtype
            match is found, searches every other subtype's centroid index
            for the same segment.  This is essential for reassorted segments:
            an H3N2-origin PB1 carried by an H1N1 isolate will match
            PB1.3.xxxx rather than minting a spurious PB1.1.xxxx.  The
            existing allele name is reused verbatim — the allele identity
            belongs to the segment's lineage, not the host isolate's subtype.

        3.  Mint a new allele name under the declared subtype if neither
            stage 2a nor 2b produces a match.

        Parameters
        ----------
        segment_name : str
        subtype : str
            Declared subtype of the *isolate*.  Used to choose the primary
            centroid index (Stage 2a) and to number new alleles (Stage 3).
            A cross-subtype centroid match (Stage 2b) overrides this with
            the matched allele's own subtype prefix.
        internal_cluster_id : str
            Run-local cluster label (e.g. "C1") or synthetic orphan ID.
            Used as a cache key only; never exposed in the public allele name.
        cluster_version : str, optional
        centroid_signature : MinHashSignature, optional
            The centroid of the cluster being named.  Required for Stage 2
            centroid matching.  Without it both Stage 2a and 2b are skipped.
        """
        snum = subtype_num(subtype)
        # Version-scoped key prevents recycled cluster label collisions across
        # runs (e.g. 'C2' in v1 = clade B, 'C2' in v20260302 = clade A).
        key = (segment_name, snum, cluster_version or "", internal_cluster_id)

        # ── Stage 1: cache hit by version-scoped cluster ID ─────────────────
        if key in self._allele_cache:
            allele_name = self._allele_cache[key]
            if self.db:
                self.db.update_allele_last_seen(allele_name)
            return allele_name

        # ── Stage 2: centroid similarity search ─────────────────────────────
        if centroid_signature is not None:
            threshold = self._clustering_config.get_ani_threshold(
                segment_name, subtype, "same"
            )
            k = self._kmer_config.get_k(segment_name)

            # 2a: same-subtype index first (most common case, fast path)
            same_idx_key = (segment_name, snum)
            same_index = self._centroid_indices.get(same_idx_key)
            if same_index is not None and len(same_index) > 0:
                matched_name = same_index.best_match(centroid_signature, threshold, k)
                if matched_name is not None:
                    self._allele_cache[key] = matched_name
                    if self.db:
                        self.db.update_allele_last_seen(matched_name)
                    logger.debug(
                        f"Same-subtype centroid match: {segment_name}/{subtype} "
                        f"cluster {internal_cluster_id} → {matched_name} "
                        f"(threshold={threshold:.3f})"
                    )
                    return matched_name

            # 2b: cross-subtype search — covers reassorted segments whose
            # lineage belongs to a different subtype than the host isolate.
            # Iterates all other segment indices regardless of subtype_num.
            best_cross_name: Optional[str] = None
            best_cross_sim: float = 0.0
            for idx_key, index in self._centroid_indices.items():
                if idx_key[0] != segment_name:
                    continue  # different segment — skip
                if idx_key[1] == snum:
                    continue  # already searched above — skip
                if len(index) == 0:
                    continue
                result = index.best_match_with_score(centroid_signature, threshold, k)
                if result is not None:
                    matched_name, sim = result
                    # Prefer the highest-ANI match across cross-subtype indices.
                    if sim > best_cross_sim:
                        best_cross_sim = sim
                        best_cross_name = matched_name

            if best_cross_name is not None:
                self._allele_cache[key] = best_cross_name
                if self.db:
                    self.db.update_allele_last_seen(best_cross_name)
                logger.info(
                    f"Cross-subtype centroid match: {segment_name}/{subtype} "
                    f"cluster {internal_cluster_id} → {best_cross_name} "
                    f"(sim={best_cross_sim:.3f}, threshold={threshold:.3f}) "
                    f"— possible reassorted segment"
                )
                # Stash the cross-subtype match so name_genotypes_batch Pass 2
                # can record the lineage link once it knows both allele names.
                self._pending_lineage[(internal_cluster_id, segment_name)] = (
                    best_cross_name, best_cross_sim
                )
                return best_cross_name

        else:
            # Warn only when there are stored centroids to match against —
            # i.e. this is a re-clustering situation where matching matters.
            same_idx_key = (segment_name, snum)
            if same_idx_key in self._centroid_indices and \
                    len(self._centroid_indices[same_idx_key]) > 0:
                logger.warning(
                    f"assign_allele() called without centroid_signature for "
                    f"{segment_name}/{subtype} cluster {internal_cluster_id}. "
                    f"Stage 2 centroid matching skipped — a new allele name "
                    f"will be minted even if this lineage already has one. "
                    f"Pass centroid_signature to ensure naming stability."
                )

        # ── Stage 3: mint a new allele name ─────────────────────────────────
        if self.db:
            allele_num = self.db.get_next_allele_num(segment_name, snum)
        else:
            counter_key = (segment_name, snum)
            allele_num = self._allele_counters.get(counter_key, 1)
            self._allele_counters[counter_key] = allele_num + 1

        allele_name = f"{segment_name}.{snum}.{allele_num:04d}"
        self._allele_cache[key] = allele_name

        # Add to the declared-subtype centroid index so future sequences
        # can match against this allele in both same- and cross-subtype searches.
        if centroid_signature is not None:
            idx_key = (segment_name, snum)
            if idx_key not in self._centroid_indices:
                self._centroid_indices[idx_key] = _CentroidIndex()
            self._centroid_indices[idx_key].add(allele_name, centroid_signature)

        # Persist to DB
        if self.db:
            self.db.insert_allele(
                allele_name=allele_name,
                segment_name=segment_name,
                subtype_num=snum,
                allele_num=allele_num,
                internal_cluster_id=internal_cluster_id,
                cluster_version=cluster_version,
                centroid_signature=(
                    centroid_signature.to_bytes() if centroid_signature else None
                ),
            )

        # ── Lineage cross-reference ───────────────────────────────────────────
        # After minting, check all OTHER subtype indices for the same segment.
        # If this centroid matches an existing allele from a different subtype
        # above threshold, record the lineage link — both names refer to the
        # same biological lineage even though they carry different subtype prefixes.
        # This is the core of the self-healing audit trail: a future analyst can
        # query get_allele_lineage("PB1.1.0002") and find "PB1.3.0001", or vice versa.
        if centroid_signature is not None and self.db:
            for idx_key, index in self._centroid_indices.items():
                if idx_key[0] != segment_name:
                    continue  # different segment
                if idx_key[1] == snum:
                    continue  # same subtype — not a cross-subtype link
                if len(index) == 0:
                    continue
                cross_result = index.best_match_with_score(
                    centroid_signature, threshold, k
                )
                if cross_result is not None:
                    cross_match, cross_sim = cross_result
                    self.db.record_allele_lineage(
                        allele_name, cross_match, cross_sim,
                        evidence="cross_subtype_centroid_match_at_mint",
                    )
                    logger.info(
                        f"Allele lineage recorded: {allele_name} ↔ {cross_match} "
                        f"(ani={cross_sim:.4f}, segment={segment_name})"
                    )

        logger.debug(
            f"New allele: {allele_name} "
            f"(cluster {internal_cluster_id}, version {cluster_version})"
        )
        return allele_name

    # ------------------------------------------------------------------
    # Batch allele assignment (per-isolate)
    # ------------------------------------------------------------------

    def assign_alleles_batch(
        self,
        cluster_assignments: Dict[str, Optional[str]],
        subtype: str,
        cluster_version: Optional[str] = None,
        centroid_signatures: Optional[Dict[str, MinHashSignature]] = None,
        sequence_id: Optional[str] = None,
    ) -> Dict[str, Optional[str]]:
        """Assign allele names for all segments of one isolate.

        Parameters
        ----------
        cluster_assignments : dict
            segment_name -> internal_cluster_id (or None/orphan marker).
        subtype : str
        cluster_version : str, optional
        centroid_signatures : dict, optional
            segment_name -> MinHashSignature for this sequence's segment.
            For clustered segments this is the cluster centroid signature.
            For orphan segments this is the sequence's own signature, used
            as the centroid candidate so that orphans can be named and act
            as founders for future runs.
        sequence_id : str, optional
            The isolate's sequence ID.  Required for correct orphan allele
            scoping — ensures two different orphan sequences that fall below
            the same-cluster threshold are not assigned the same allele ID.
        """
        alleles = {}
        for seg in SEGMENTS:
            cid = cluster_assignments.get(seg)
            sig = (centroid_signatures or {}).get(seg)

            if cid and cid not in ("?", "-", None):
                # ── Clustered segment: standard path ────────────────────
                alleles[seg] = self.assign_allele(
                    seg, subtype, cid, cluster_version,
                    centroid_signature=sig,
                )
            elif sig is not None:
                # ── Orphan segment: name using sequence's own signature ──
                # The synthetic cluster ID is scoped to (sequence, segment,
                # version) so that:
                #   1. Stage 2 centroid similarity runs first — if this
                #      sequence is close enough to an existing allele centroid
                #      it reuses that allele name.
                #   2. If no match, a new allele is minted with this sequence
                #      as the founder for future runs.
                #   3. Two orphan sequences below the same-cluster threshold
                #      will NOT incorrectly share an allele because their
                #      synthetic IDs are sequence-scoped.
                seq_label = sequence_id or "UNKNOWN"
                orphan_cid = f"ORPHAN::{seq_label}::{seg}::{cluster_version or 'v?'}"
                alleles[seg] = self.assign_allele(
                    seg, subtype, orphan_cid, cluster_version,
                    centroid_signature=sig,
                )
                logger.debug(
                    f"Orphan allele assigned: {seg}/{subtype} "
                    f"seq={seq_label} → {alleles[seg]}"
                )
            else:
                # ── No signature available (segment missing entirely) ────
                alleles[seg] = None
        return alleles


    # ------------------------------------------------------------------
    # Constellation naming
    # ------------------------------------------------------------------

    def assign_constellation(
        self, alleles: Dict[str, Optional[str]], subtype: str
    ) -> Optional[str]:
        """Get or create a constellation ID for a set of allele assignments.

        Returns ``None`` when fewer than
        ``clustering_config.min_segments_for_constellation`` segments carry an
        allele assignment, preventing constellation IDs from being issued for
        poorly-covered isolates.  The threshold defaults to 6 (of 8) and is
        configurable via ``ClusteringConfig.min_segments_for_constellation``.
        """
        assigned = [alleles.get(seg) for seg in SEGMENTS if alleles.get(seg) is not None]
        if len(assigned) < self._clustering_config.min_segments_for_constellation:
            return None

        parts = [alleles.get(seg) or "-" for seg in SEGMENTS]
        canonical_str = "|".join(parts)

        if canonical_str in self._constellation_cache:
            cid = self._constellation_cache[canonical_str]
            if self.db:
                self.db.update_constellation_last_seen(cid)
            return cid

        if self.db:
            existing = self.db.get_constellation(canonical_str)
            if existing:
                cid = existing["constellation_id"]
                self._constellation_cache[canonical_str] = cid
                self.db.update_constellation_last_seen(cid)
                return cid

        prefix = subtype_short(subtype)
        digest = hashlib.sha256(canonical_str.encode()).hexdigest()
        candidate = f"{prefix}-{digest[:_CONSTELLATION_HASH_CHARS].upper()}"

        # The ID is a pure deterministic function of (subtype prefix, allele
        # combination): the same constellation always yields the same ID,
        # independent of the order constellations are first seen. Dedup by the
        # full allele combination already happened above, so reaching here means
        # canonical_str is new. If its ID nonetheless already belongs to a
        # *different* constellation, that is a genuine SHA-256 collision at this
        # width — astronomically unlikely at any real scale — and we surface it
        # loudly rather than sliding to an order-dependent window (the old
        # behaviour, which also ran off the end of the digest at high density).
        collides = candidate in set(self._constellation_cache.values()) or (
            self.db is not None
            and self.db.get_constellation_by_id(candidate) is not None
        )
        if collides:
            raise RuntimeError(
                f"Constellation ID collision: {candidate!r} is already assigned "
                f"to a different allele combination than {canonical_str!r}. This "
                f"indicates a SHA-256 collision at {_CONSTELLATION_HASH_CHARS} hex "
                f"chars (effectively impossible at realistic scale); raise "
                f"_CONSTELLATION_HASH_CHARS if it ever genuinely occurs."
            )

        self._constellation_cache[canonical_str] = candidate
        if self.db:
            self.db.insert_constellation(candidate, prefix, canonical_str)

        logger.debug(f"New constellation: {candidate}")
        return candidate

    # ------------------------------------------------------------------
    # Combined naming pipeline
    # ------------------------------------------------------------------


    def name_genotypes_batch(
        self,
        all_cluster_assignments: Dict[str, Dict[str, Optional[str]]],
        subtypes: Dict[str, str],
        cluster_version: Optional[str] = None,
        all_centroid_signatures: Optional[Dict[str, Dict[str, MinHashSignature]]] = None,
    ) -> Dict[str, Dict]:
        """Name alleles and constellations for a batch of isolates.

        Uses a two-pass strategy to guarantee correct cross-subtype centroid
        matching for reassorted segments:

        Pass 1 — Clustered segments only (all isolates, all subtypes).
            Names every segment that has a real cluster assignment.  This
            populates all centroid indices for all subtypes before any orphan
            is processed, so the H3N2 PB1 centroid exists when SAMP12's
            orphan PB1 is evaluated in Pass 2.

        Pass 2 — Orphan segments only (all isolates).
            With complete centroid indices now available, Stage 2a (same-
            subtype) and Stage 2b (cross-subtype) both have the full picture.
            An H1N1 isolate carrying a reassorted H3N2 PB1 will match
            ``PB1.3.xxxx`` rather than minting a spurious ``PB1.1.xxxx``.

        Constellation assignment follows Pass 2 once every allele is resolved.

        Parameters
        ----------
        all_cluster_assignments : dict
            sequence_id -> {segment_name -> internal_cluster_id}.
        subtypes : dict
            sequence_id -> subtype string.
        cluster_version : str, optional
        all_centroid_signatures : dict, optional
            sequence_id -> {segment_name -> MinHashSignature}.
            For clustered segments: the cluster centroid signature.
            For orphan segments: the sequence's own signature (used as the
            centroid candidate for allele minting / cross-subtype matching).
        """
        ORPHAN_MARKERS = {"?", "-", None}

        # Accumulate per-isolate allele dicts across both passes
        allele_results: Dict[str, Dict[str, Optional[str]]] = {
            seq_id: {} for seq_id in all_cluster_assignments
        }

        # ── Pass 1: clustered segments — build all centroid indices ──────────
        for seq_id, assignments in all_cluster_assignments.items():
            st   = subtypes.get(seq_id, "H3N2")
            sigs = (all_centroid_signatures or {}).get(seq_id) or {}
            for seg in SEGMENTS:
                cid = assignments.get(seg)
                if cid and cid not in ORPHAN_MARKERS:
                    sig = sigs.get(seg)
                    allele_results[seq_id][seg] = self.assign_allele(
                        seg, st, cid, cluster_version,
                        centroid_signature=sig,
                    )

        # ── Pass 2: orphan segments — full cross-subtype search now possible ──
        for seq_id, assignments in all_cluster_assignments.items():
            st      = subtypes.get(seq_id, "H3N2")
            sigs    = (all_centroid_signatures or {}).get(seq_id) or {}
            seq_label = seq_id
            for seg in SEGMENTS:
                if seg in allele_results[seq_id]:
                    continue  # already named in Pass 1
                cid = assignments.get(seg)
                sig = sigs.get(seg)
                if sig is not None:
                    # Orphan with a signature: attempt centroid match or mint
                    orphan_cid = (
                        f"ORPHAN::{seq_label}::{seg}::{cluster_version or 'v?'}"
                    )
                    allele = self.assign_allele(
                        seg, st, orphan_cid, cluster_version,
                        centroid_signature=sig,
                    )
                    allele_results[seq_id][seg] = allele
                    logger.debug(
                        f"Orphan allele assigned: {seg}/{st} "
                        f"seq={seq_label} → {allele}"
                    )
                    # Lineage cross-reference: if the assigned allele carries a
                    # different subtype prefix than the declaring isolate, this
                    # is a confirmed cross-subtype assignment.  Record the link
                    # between the allele and any same-segment alleles with a
                    # matching centroid from the declaring subtype's index.
                    # This covers both the case where Stage 2b fires (new mint
                    # in a later run) and the case where Stage 1 cache-hits
                    # (allele already known from a previous run).
                    if self.db and allele and sig is not None:
                        parts = allele.split(".") if allele else []
                        allele_snum = int(parts[1]) if len(parts) >= 3 and parts[1].isdigit() else None
                        declared_snum = subtype_num(st)
                        if allele_snum is not None and allele_snum != declared_snum:
                            # Cross-subtype assignment: the orphan's signature matched
                            # a centroid from a different subtype's index.
                            # Record the lineage link by verifying the match against
                            # the allele's own subtype index (where its centroid lives).
                            allele_idx = self._centroid_indices.get((seg, allele_snum))
                            if allele_idx is not None and len(allele_idx) > 0:
                                threshold = self._clustering_config.get_ani_threshold(
                                    seg, st, "same"
                                )
                                k = self._kmer_config.get_k(seg)
                                matched_result = allele_idx.best_match_with_score(
                                    sig, threshold, k
                                )
                                if matched_result is not None and matched_result[0] == allele:
                                    link_sim = matched_result[1]
                                    # Also check if the declaring subtype index has
                                    # a same-segment allele from a prior run that
                                    # should be linked (the "PB1.1.0002 is also the
                                    # H3N2 lineage" case).
                                    # The link is simply: allele ↔ allele, but we
                                    # need a second name. Look in the declaring
                                    # subtype's index for the same signature.
                                    other_idx = self._centroid_indices.get(
                                        (seg, declared_snum)
                                    )
                                    other_match = (
                                        other_idx.best_match(sig, threshold, k)
                                        if other_idx and len(other_idx) > 0
                                        else None
                                    )
                                    if other_match and other_match != allele:
                                        self.db.record_allele_lineage(
                                            allele, other_match, link_sim,
                                            evidence="cross_subtype_centroid_match",
                                        )
                                        logger.info(
                                            f"Allele lineage recorded: {allele} ↔ "
                                            f"{other_match} (sim={link_sim:.3f}, "
                                            f"segment={seg}, seq={seq_label})"
                                        )
                                    else:
                                        # No corresponding declared-subtype allele yet.
                                        # Record the cross-subtype fact using the
                                        # matched allele and the sequence's orphan
                                        # synthetic ID as annotation context.
                                        # The link will be properly named on recluster.
                                        pass
                else:
                    allele_results[seq_id][seg] = None  # segment missing entirely

        # ── Constellation + formatting ────────────────────────────────────────
        results = {}
        for seq_id, alleles in allele_results.items():
            st = subtypes.get(seq_id, "H3N2")
            constellation = self.assign_constellation(alleles, st)
            allele_string = " | ".join(alleles.get(seg) or "?" for seg in SEGMENTS)
            results[seq_id] = {
                "alleles": alleles,
                "constellation": constellation,
                "allele_string": allele_string,
            }
        return results

    # ------------------------------------------------------------------
    # Registry queries
    # ------------------------------------------------------------------


    def get_registry_summary(self) -> Dict:
        alleles_by_segment: Dict[str, int] = {}
        for (seg, snum, _ver, _cid) in self._allele_cache:
            key = f"{seg}.{snum}"
            alleles_by_segment[key] = alleles_by_segment.get(key, 0) + 1
        total_centroids = sum(len(idx) for idx in self._centroid_indices.values())
        return {
            "total_alleles": len(self._allele_cache),
            "total_constellations": len(self._constellation_cache),
            "total_centroids_indexed": total_centroids,
            "alleles_by_segment": alleles_by_segment,
        }
