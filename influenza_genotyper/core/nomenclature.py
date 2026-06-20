"""
Nomenclature Manager
====================
Assigns stable, meaningful identifiers to segment-level alleles and
full-genome constellation genotypes. Persists all registries to the database.

Allele format:    {segment}.{subtype_num}.{zero-padded ID}
                  e.g. HA.3.0042, PB2.1.0008, NA.2.0007

Constellation:    {HxNx}-{4-char hex hash}
                  e.g. H3N2-A7F2, H1N1-B3E9

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

from ..config import SEGMENTS, ClusteringConfig
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
        # allele_name -> observed (distance) radius of that lineage.  Grows as
        # member-clusters are assigned; consumed by nearest-lineage assignment.
        self._radius: Dict[str, float] = {}

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

    def add(self, allele_name: str, signature: MinHashSignature,
            radius: float = 0.0) -> None:
        """Append a centroid to the index.

        Before the first ``best_match()`` call, rows are buffered in
        ``_pending_rows`` (O(1)).  After the matrix has been consolidated
        once, pending rows are merged immediately so that ``best_match``
        always sees a complete matrix.  ``radius`` is the lineage's observed
        distance spread (0.0 for a freshly-minted, single-member lineage).
        """
        row = signature.signature.astype(np.uint64)  # (d,)
        self._names.append(allele_name)
        self._radius[allele_name] = float(radius)
        if self._matrix is None:
            # Still in accumulation phase — buffer the row
            self._pending_rows.append(row)
        else:
            # Matrix already consolidated; merge now to keep it current
            self._pending_rows.append(row)
            self._consolidate()

    def get_radius(self, allele_name: str) -> float:
        return self._radius.get(allele_name, 0.0)

    @staticmethod
    def _allele_num(name: str) -> float:
        """Parse the trailing zero-padded counter from an allele name.

        ``"PB1.3.0042"`` -> ``42``.  Returns ``inf`` for unparseable names so
        they sort *after* well-formed ones in the oldest-wins tie-break.
        """
        try:
            return float(int(name.rsplit(".", 1)[1]))
        except (IndexError, ValueError):
            return float("inf")

    def best_match(
        self, query: MinHashSignature, threshold: float
    ) -> Optional[Tuple[str, float]]:
        """Return ``(allele_name, similarity)`` for the best allele to reuse,
        or ``None`` if no stored centroid clears *threshold*.

        Selection rule — **oldest wins, not most similar.**  Among *every*
        centroid whose Jaccard similarity meets the threshold, the allele with
        the lowest allele number (i.e. the earliest minted) is returned.  This
        is the stability-preserving choice: when a lineage is represented by
        more than one near-duplicate centroid in the registry, a re-clustered
        cluster must re-attach to the original allele name rather than drift to
        a marginally-more-similar but newer duplicate.  Ties on allele number
        (which should not occur within a single index) fall back to insertion
        order.

        Uses vectorised broadcasting: all Jaccard similarities are computed in
        a single (N, d) == (1, d) comparison, avoiding Python loops over
        alleles.  Returning the similarity lets callers avoid recomputing it.
        """
        if not self._names:
            return None

        # Consolidate any buffered rows before querying
        self._consolidate()

        if self._matrix is None:
            return None

        q = query.signature.reshape(1, -1).astype(np.uint64)  # (1, d)
        # Broadcasting: (N, d) == (1, d) → (N, d) bool
        matches = self._matrix == q
        similarities = np.mean(matches, axis=1)                # (N,)

        qualifying = np.nonzero(similarities >= threshold)[0]
        if qualifying.size == 0:
            return None

        # Oldest-wins: lowest allele number among all above-threshold matches,
        # breaking ties by insertion order.
        best_idx = min(
            (int(i) for i in qualifying),
            key=lambda i: (self._allele_num(self._names[i]), i),
        )
        return self._names[best_idx], float(similarities[best_idx])

    def nearest_match(
        self, query: MinHashSignature, tie_eps: float = 1e-9
    ) -> Optional[Tuple[str, float, float, float]]:
        """Return the *nearest* stored lineage (no threshold gate).

        Returns ``(allele_name, similarity, radius, runner_up_similarity)`` or
        ``None`` if the index is empty.  Unlike ``best_match`` (which applies an
        absolute threshold), this answers the relative question "which lineage
        is closest" — the basis of nearest-lineage assignment.  The caller then
        decides drift-vs-novelty using the returned lineage radius.

        Oldest-wins tie-break: among centroids within ``tie_eps`` of the maximum
        similarity (near-duplicate lineages), the lowest allele number is
        returned, preserving name stability.  ``runner_up_similarity`` is the
        best similarity to any *other* lineage, used for boundary-ambiguity
        (confidence) flagging.
        """
        if not self._names:
            return None
        self._consolidate()
        if self._matrix is None:
            return None

        q = query.signature.reshape(1, -1).astype(np.uint64)
        similarities = np.mean(self._matrix == q, axis=1)   # (N,)
        top = float(similarities.max())

        near_top = np.nonzero(similarities >= top - tie_eps)[0]
        best_idx = min(
            (int(i) for i in near_top),
            key=lambda i: (self._allele_num(self._names[i]), i),
        )

        # Runner-up: best similarity to a lineage other than the chosen one.
        if len(similarities) > 1:
            masked = similarities.copy()
            masked[best_idx] = -1.0
            runner_up = float(masked.max())
        else:
            runner_up = -1.0

        return (
            self._names[best_idx],
            float(similarities[best_idx]),
            self._radius.get(self._names[best_idx], 0.0),
            runner_up,
        )

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
        Provides per-segment same-cluster Jaccard thresholds used as the *floor*
        of each lineage's novelty margin in nearest-lineage assignment.  When
        omitted, a default ClusteringConfig is used.

    Nearest-lineage assignment
    ---------------------------
    A query cluster is assigned to the *nearest* existing lineage rather than to
    any lineage clearing a fixed similarity threshold.  A new lineage is minted
    only when the query's distance to the nearest lineage exceeds that lineage's
    novelty margin, where

        margin = novelty_factor x clamp(observed_radius, floor, ceiling)
        floor  = 1 - same_cluster_threshold(segment, subtype)
        ceiling = radius_ceiling_factor x floor

    The margin is therefore *relative* to each lineage's own observed spread —
    not a single global cutoff — with the per-segment same-cluster threshold as
    a sane lower bound and a ceiling to bound runaway lumping.  These knobs are
    read from ClusteringConfig if present (``novelty_factor``,
    ``radius_ceiling_factor``, ``boundary_confidence_band``) and otherwise
    default to 1.0 / 2.0 / 0.25.
    """

    def __init__(
        self,
        db: Optional["DatabaseManager"] = None,
        clustering_config: Optional[ClusteringConfig] = None,
    ):
        self.db = db
        self._clustering_config = clustering_config or ClusteringConfig()

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

        # subtype_num -> canonical subtype string, restricted to subtypes that
        # carry a threshold adjustment.  Used so that cross-subtype centroid
        # search applies the *matched* subtype's threshold rather than the
        # declaring isolate's (whose adjustment was calibrated for a different
        # subtype's within-clade diversity).  Falls back to the declaring
        # subtype when a stored subtype_num is not in this map.
        self._snum_to_subtype: Dict[int, str] = {}
        for _st in self._clustering_config.subtype_adjustments:
            try:
                self._snum_to_subtype.setdefault(subtype_num(_st), _st)
            except ValueError:
                continue

        # Allele names minted from a single orphan sequence ("provisional
        # founders").  Such alleles are statistically unverified — they bypass
        # the clustering engine's min_cluster_size gate — so they are tracked
        # here (and persisted via the registry's `provisional` column) so they
        # are distinguishable and prunable.  A provisional allele is promoted
        # to established the first time a real (clustered) assignment reuses it.
        self._provisional_alleles: set = set()

        # Nearest-lineage novelty knobs (read from config if present).  These
        # are sensitivity dials applied *relative* to each lineage's spread —
        # not boundary-placement constants.
        cc = self._clustering_config
        self._novelty_factor = float(getattr(cc, "novelty_factor", 1.0))
        self._radius_ceiling_factor = float(getattr(cc, "radius_ceiling_factor", 2.0))
        self._boundary_confidence_band = float(
            getattr(cc, "boundary_confidence_band", 0.25)
        )

        # seq_id -> list of segments whose latest assignment was boundary-
        # ambiguous (near-equidistant between two lineages).  Reset per batch.
        self._batch_low_confidence: Dict[str, List[str]] = {}
        # Set transiently by assign_allele so the batch driver can attribute a
        # low-confidence assignment to the right (seq_id, segment).
        self._last_assignment_low_confidence: bool = False

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

            if row.get("provisional"):
                self._provisional_alleles.add(row["allele_name"])

            # Rebuild centroid index from stored signatures, carrying the
            # observed per-lineage radius used by nearest-lineage assignment.
            sig_bytes = row.get("centroid_signature")
            if sig_bytes:
                try:
                    sig = MinHashSignature.from_bytes(sig_bytes)
                    idx_key = (row["segment_name"], row["subtype_num"])
                    if idx_key not in self._centroid_indices:
                        self._centroid_indices[idx_key] = _CentroidIndex()
                    self._centroid_indices[idx_key].add(
                        row["allele_name"], sig,
                        radius=float(row.get("radius") or 0.0),
                    )
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
        provisional: bool = False,
        cluster_radius: float = 0.0,
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
            if not provisional:
                self._promote_if_provisional(allele_name)
            if self.db:
                self.db.update_allele_last_seen(allele_name)
            return allele_name

        # ── Stage 2: nearest-lineage assignment ─────────────────────────────
        # Assign to the *nearest* existing lineage (same- or cross-subtype),
        # minting only when the query is beyond that lineage's novelty margin.
        # The margin is relative to each lineage's observed spread, floored by
        # the per-segment same-cluster threshold and capped by the ceiling.
        if centroid_signature is not None:
            self._last_assignment_low_confidence = False

            # Collect each candidate index's nearest lineage + its margin.
            # candidate = (sim, name, owning_idx_key, dist, margin, is_cross)
            candidates = []
            per_index_nearest_sims = []      # for global runner-up / confidence
            within_index_runner_up = -1.0    # runner-up inside the chosen index
            for idx_key, index in self._centroid_indices.items():
                if idx_key[0] != segment_name or len(index) == 0:
                    continue
                nm = index.nearest_match(centroid_signature)
                if nm is None:
                    continue
                name, sim, radius, runner_up = nm
                is_cross = (idx_key[1] != snum)
                cand_subtype = (
                    subtype if not is_cross
                    else self._snum_to_subtype.get(idx_key[1], subtype)
                )
                floor_d = 1.0 - self._clustering_config.get_threshold(
                    segment_name, cand_subtype, "same"
                )
                ceiling_d = self._radius_ceiling_factor * floor_d
                eff_radius = min(max(radius, floor_d), ceiling_d)
                margin = self._novelty_factor * eff_radius
                dist = 1.0 - sim
                candidates.append((sim, name, idx_key, dist, margin, is_cross, runner_up))
                per_index_nearest_sims.append(sim)

            if candidates:
                # Nearest overall = highest similarity; oldest-wins on near-ties.
                top_sim = max(c[0] for c in candidates)
                near_top = [c for c in candidates if c[0] >= top_sim - 1e-9]
                best = min(near_top, key=lambda c: (_CentroidIndex._allele_num(c[1])))
                sim, name, idx_key, dist, margin, is_cross, runner_up = best

                # Global runner-up for boundary-ambiguity: the best similarity to
                # any *other* lineage, whether in this index or another.
                others = [s for s in per_index_nearest_sims if s < sim - 1e-9]
                second = max(others + [runner_up]) if (others or runner_up >= 0) else -1.0

                if dist <= margin:
                    # Within the lineage's basin → drift, not novelty. Assign.
                    # NOTE: the lineage radius is *not* mutated here. It is
                    # write-once cluster geometry (set at mint from the founding
                    # cluster's member spread, recomputed only at recluster),
                    # exactly like the centroid and mean_diameter. Keeping it
                    # read-only in the naming layer is what makes within-version
                    # assignment order-independent and machine-independent, and
                    # makes single-linkage chaining structurally impossible
                    # between reclusters (a basin that never grows can't creep
                    # across a boundary).

                    # Boundary-ambiguity flag: near-equidistant to a 2nd lineage.
                    if second >= 0 and (sim - second) < self._boundary_confidence_band * max(margin, 1e-9):
                        self._last_assignment_low_confidence = True

                    self._allele_cache[key] = name
                    if not provisional:
                        self._promote_if_provisional(name)
                    if self.db:
                        self.db.update_allele_last_seen(name)

                    if is_cross:
                        self._pending_lineage[(internal_cluster_id, segment_name)] = (
                            name, sim
                        )
                        logger.info(
                            f"Nearest-lineage (cross-subtype) match: "
                            f"{segment_name}/{subtype} cluster {internal_cluster_id} "
                            f"→ {name} (sim={sim:.3f}, dist={dist:.3f} ≤ "
                            f"margin={margin:.3f}) — possible reassorted segment"
                        )
                    else:
                        logger.debug(
                            f"Nearest-lineage match: {segment_name}/{subtype} "
                            f"cluster {internal_cluster_id} → {name} "
                            f"(sim={sim:.3f}, dist={dist:.3f} ≤ margin={margin:.3f}, "
                            f"low_conf={self._last_assignment_low_confidence})"
                        )
                    return name
                # else: nearest is beyond its novelty margin → fall through to mint
                logger.debug(
                    f"Novelty: {segment_name}/{subtype} cluster {internal_cluster_id} "
                    f"nearest {name} dist={dist:.3f} > margin={margin:.3f} → mint"
                )

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

        # The new lineage's basin radius is its cluster's own member spread,
        # clamped to the per-segment ceiling.  This is what makes nearest-lineage
        # assignment tolerate drift: a genuinely broad lineage starts with a wide
        # margin, a tight one starts narrow.  (Singletons/orphans have radius 0
        # and fall back to the per-segment floor at decision time.)
        mint_floor_d = 1.0 - self._clustering_config.get_threshold(
            segment_name, subtype, "same"
        )
        mint_ceiling_d = self._radius_ceiling_factor * mint_floor_d
        mint_radius = min(max(cluster_radius, 0.0), mint_ceiling_d)

        # Add to the declared-subtype centroid index so future sequences
        # can match against this allele in both same- and cross-subtype searches.
        if centroid_signature is not None:
            idx_key = (segment_name, snum)
            if idx_key not in self._centroid_indices:
                self._centroid_indices[idx_key] = _CentroidIndex()
            self._centroid_indices[idx_key].add(
                allele_name, centroid_signature, radius=mint_radius
            )

        # Persist to DB
        if self.db:
            self.db.insert_allele(
                allele_name=allele_name,
                segment_name=segment_name,
                subtype_num=snum,
                internal_cluster_id=internal_cluster_id,
                allele_num=allele_num,
                cluster_version=cluster_version,
                centroid_signature=(
                    centroid_signature.to_bytes() if centroid_signature else None
                ),
                provisional=provisional,
                radius=mint_radius,
            )

        if provisional:
            # Founded from a single orphan sequence — unverified until a real
            # clustered assignment corroborates it (see _promote_if_provisional).
            self._provisional_alleles.add(allele_name)
            logger.info(
                f"Provisional allele minted from orphan: {allele_name} "
                f"({segment_name}/{subtype}, cluster {internal_cluster_id}). "
                f"Flagged unverified until a clustered assignment reuses it."
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
                cross_subtype = self._snum_to_subtype.get(idx_key[1], subtype)
                cross_threshold = self._clustering_config.get_threshold(
                    segment_name, cross_subtype, "same"
                )
                cross = index.best_match(centroid_signature, cross_threshold)
                if cross is not None:
                    cross_match, cross_sim = cross
                    self.db.record_allele_lineage(
                        allele_name, cross_match, cross_sim,
                        evidence="cross_subtype_centroid_match_at_mint",
                    )
                    logger.info(
                        f"Allele lineage recorded: {allele_name} ↔ {cross_match} "
                        f"(sim={cross_sim:.3f}, segment={segment_name})"
                    )

        logger.debug(
            f"New allele: {allele_name} "
            f"(cluster {internal_cluster_id}, version {cluster_version})"
        )
        return allele_name

    # ------------------------------------------------------------------
    # Provisional (orphan-founded) allele handling
    # ------------------------------------------------------------------

    def _promote_if_provisional(self, allele_name: str) -> None:
        """Mark a provisional allele as established.

        Called when a *clustered* (non-provisional) assignment reuses an allele
        that was originally founded by a single orphan sequence.  Corroboration
        by a real cluster is what graduates an orphan founder to an established
        allele.  No-op for alleles that are already established.
        """
        if allele_name in self._provisional_alleles:
            self._provisional_alleles.discard(allele_name)
            if self.db:
                self.db.update_allele_provisional(allele_name, False)
            logger.info(
                f"Allele promoted (orphan founder corroborated by a clustered "
                f"assignment): {allele_name}"
            )

    def is_provisional(self, allele_name: str) -> bool:
        """True if *allele_name* was founded by a single orphan sequence and
        has not since been corroborated by a clustered assignment."""
        return allele_name in self._provisional_alleles

    def provisional_alleles(self) -> List[str]:
        """Return the currently-provisional (orphan-founded, uncorroborated)
        allele names, sorted — these are the prunable founders."""
        return sorted(self._provisional_alleles)



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
                    provisional=True,
                )
                logger.debug(
                    f"Orphan allele assigned: {seg}/{subtype} "
                    f"seq={seq_label} → {alleles[seg]}"
                )
            else:
                # ── No signature available (segment missing entirely) ────
                alleles[seg] = None
        return alleles

    def get_allele_name(self, segment_name: str, subtype: str,
                        internal_cluster_id: str,
                        cluster_version: Optional[str] = None) -> Optional[str]:
        """Look up an existing allele without creating a new one."""
        key = (segment_name, subtype_num(subtype),
               cluster_version or "", internal_cluster_id)
        return self._allele_cache.get(key)

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
        short_hash = digest[:4].upper()
        candidate = f"{prefix}-{short_hash}"

        in_memory_ids = set(self._constellation_cache.values())
        offset = 0
        while True:
            if candidate in in_memory_ids:
                offset += 1
                short_hash = digest[offset:offset + 4].upper()
                candidate = f"{prefix}-{short_hash}"
                continue
            if self.db:
                db_row = self.db.get_constellation_by_id(candidate)
                if db_row is not None:
                    offset += 1
                    short_hash = digest[offset:offset + 4].upper()
                    candidate = f"{prefix}-{short_hash}"
                    continue
            break

        self._constellation_cache[canonical_str] = candidate
        if self.db:
            self.db.insert_constellation(candidate, prefix, canonical_str)

        logger.debug(f"New constellation: {candidate}")
        return candidate

    # ------------------------------------------------------------------
    # Combined naming pipeline
    # ------------------------------------------------------------------

    def name_genotype(
        self,
        cluster_assignments: Dict[str, Optional[str]],
        subtype: str,
        cluster_version: Optional[str] = None,
        centroid_signatures: Optional[Dict[str, MinHashSignature]] = None,
        sequence_id: Optional[str] = None,
    ) -> Dict:
        """Full naming pipeline for one isolate: alleles + constellation.

        Parameters
        ----------
        cluster_assignments : dict
            segment_name -> internal_cluster_id.
        subtype : str
        cluster_version : str, optional
        centroid_signatures : dict, optional
            segment_name -> MinHashSignature.  Pass this to enable Stage 2
            centroid matching and orphan allele naming.
        sequence_id : str, optional
            Used to scope synthetic orphan cluster IDs so different sequences
            do not incorrectly share an allele below the similarity threshold.
        """
        alleles = self.assign_alleles_batch(
            cluster_assignments, subtype, cluster_version,
            centroid_signatures, sequence_id=sequence_id,
        )
        constellation = self.assign_constellation(alleles, subtype)
        allele_string = " | ".join(alleles.get(seg) or "?" for seg in SEGMENTS)
        return {
            "alleles": alleles,
            "constellation": constellation,
            "allele_string": allele_string,
        }

    def name_genotypes_batch(
        self,
        all_cluster_assignments: Dict[str, Dict[str, Optional[str]]],
        subtypes: Dict[str, str],
        cluster_version: Optional[str] = None,
        all_centroid_signatures: Optional[Dict[str, Dict[str, MinHashSignature]]] = None,
        all_cluster_radii: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Dict[str, Dict]:
        """Name alleles and constellations for a batch of isolates.

        Two-pass strategy for correct cross-subtype matching of reassorted
        segments:

        Pass 1 — Clustered segments (all isolates).  Populates every centroid
            index before any orphan is processed.

        Pass 2 — Orphan segments.  With complete indices, nearest-lineage
            assignment (same- and cross-subtype) has the full picture, so a
            reassortant orphan matches the correct existing lineage rather than
            minting a spurious one.

        Constellation assignment follows once every allele is resolved.

        Parameters
        ----------
        all_cluster_assignments : dict
            sequence_id -> {segment_name -> internal_cluster_id}.
        subtypes : dict
            sequence_id -> subtype string.
        cluster_version : str, optional
        all_centroid_signatures : dict, optional
            sequence_id -> {segment_name -> MinHashSignature} (cluster centroid
            for clustered segments; the sequence's own signature for orphans).
        """
        ORPHAN_MARKERS = {"?", "-", None}

        # Accumulate per-isolate allele dicts across both passes
        allele_results: Dict[str, Dict[str, Optional[str]]] = {
            seq_id: {} for seq_id in all_cluster_assignments
        }
        # Per-isolate list of segments whose assignment landed near a lineage
        # boundary (ambiguous) — surfaced for downstream confidence reporting.
        self._batch_low_confidence = {seq_id: [] for seq_id in all_cluster_assignments}

        # ── Pass 1: clustered segments — build all centroid indices ──────────
        # Iterate isolates in a deterministic (sorted) order so that the
        # arbitrary allele numbers minted on a cold-start run are reproducible:
        # the same input data always yields the same {0001, 0002, ...} mapping
        # regardless of dict insertion order.
        for seq_id, assignments in sorted(all_cluster_assignments.items()):
            st   = subtypes.get(seq_id, "H3N2")
            sigs = (all_centroid_signatures or {}).get(seq_id) or {}
            radii = (all_cluster_radii or {}).get(seq_id) or {}
            for seg in SEGMENTS:
                cid = assignments.get(seg)
                if cid and cid not in ORPHAN_MARKERS:
                    sig = sigs.get(seg)
                    allele_results[seq_id][seg] = self.assign_allele(
                        seg, st, cid, cluster_version,
                        centroid_signature=sig,
                        cluster_radius=radii.get(seg, 0.0),
                    )
                    if self._last_assignment_low_confidence:
                        self._batch_low_confidence[seq_id].append(seg)

        # ── Pass 2: orphan segments — full cross-subtype search now possible ──
        for seq_id, assignments in sorted(all_cluster_assignments.items()):
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
                        provisional=True,
                    )
                    allele_results[seq_id][seg] = allele
                    if self._last_assignment_low_confidence:
                        self._batch_low_confidence[seq_id].append(seg)
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
                                allele_subtype = self._snum_to_subtype.get(
                                    allele_snum, st
                                )
                                allele_threshold = self._clustering_config.get_threshold(
                                    seg, allele_subtype, "same"
                                )
                                matched = allele_idx.best_match(sig, allele_threshold)
                                if matched is not None and matched[0] == allele:
                                    link_sim = matched[1]
                                    # Also check if the declaring subtype index has
                                    # a same-segment allele from a prior run that
                                    # should be linked (the "PB1.1.0002 is also the
                                    # H3N2 lineage" case).  Look in the declaring
                                    # subtype's index for the same signature.
                                    other_idx = self._centroid_indices.get(
                                        (seg, declared_snum)
                                    )
                                    declared_threshold = self._clustering_config.get_threshold(
                                        seg, st, "same"
                                    )
                                    other = (
                                        other_idx.best_match(sig, declared_threshold)
                                        if other_idx and len(other_idx) > 0
                                        else None
                                    )
                                    other_match = other[0] if other is not None else None
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
            low_conf = sorted(set(self._batch_low_confidence.get(seq_id, [])))
            results[seq_id] = {
                "alleles": alleles,
                "constellation": constellation,
                "allele_string": allele_string,
                # Segments whose assignment was near a lineage boundary (the
                # query sat almost equidistant between two lineages).  These are
                # the calls to treat as low-confidence for reassortment reporting.
                "low_confidence_segments": low_conf,
            }
        return results

    # ------------------------------------------------------------------
    # Registry queries
    # ------------------------------------------------------------------

    def get_allele_count(self) -> int:
        return len(self._allele_cache)

    def get_constellation_count(self) -> int:
        return len(self._constellation_cache)

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
