"""
Genotyping Pipeline — end-to-end orchestration.
FASTA → k-mer extraction → clustering → genotype assignment → reassortment detection

Three operating modes
---------------------
run(fasta_path, ...)
    Full batch clustering.  Clusters all sequences in the FASTA from scratch
    and writes a new cluster set to the database.  Use for the initial
    reference run and for periodic re-clustering.

run_incremental(fasta_path, ...)
    Incremental assignment.  Loads existing cluster centroids from the
    database (from a previous ``run()`` or ``run_recluster()`` call) and
    assigns each new sequence to its nearest cluster without altering the
    cluster definitions.  Sequences that fall outside all clusters are
    flagged as orphans.  Naming (alleles, constellations) is fully stable
    because the same cluster IDs are reused.

run_recluster(fasta_path, ...)
    Periodic re-clustering.  Equivalent to ``run()`` but automatically
    derives a new cluster_version string, marks all previous clusters for
    that segment/subtype as inactive, and logs the transition.  Use when
    orphan accumulation indicates the reference clustering is becoming stale.
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .config import GenotyperConfig, SEGMENTS
from .core import (
    DatabaseManager, SequenceProcessor, KmerExtractor, MinHashSignature,
    ClusteringEngine, ClusteringResult, ClusterAssignment, ClusterDefinition,
    GenotypeAssigner, GenotypeProfile,
    ReassortmentDetector, ReassortmentReport, NomenclatureManager,
)
from .core.reassortment_detector import PermutationResult

logger = logging.getLogger(__name__)


class GenotypingPipeline:
    def __init__(self, config: Optional[GenotyperConfig] = None):
        self.config = config or GenotyperConfig.default()
        self.db = DatabaseManager(self.config.database)
        self.processor = SequenceProcessor()
        self.extractor = KmerExtractor(self.config.kmer)
        self.clusterer = ClusteringEngine(self.config.clustering)
        self.assigner = GenotypeAssigner()
        self.reassortment = ReassortmentDetector(self.config.reassortment, db=self.db)
        self.nomenclature = NomenclatureManager(
            db=self.db,
            clustering_config=self.config.clustering,
        )

    def initialize(self) -> None:
        self.db.initialize()
        self.nomenclature.load_from_db()
        logger.info("Pipeline initialized")

    # ------------------------------------------------------------------
    # Shared internal helpers
    # ------------------------------------------------------------------

    def _parse_and_store_sequences(
        self,
        fasta_path: str,
        subtype: Optional[str],
        cluster_version: str,
        skip_existing: bool = False,
    ):
        """Parse FASTA, optionally skip already-processed sequences, write to DB.

        Returns (records, all_signatures, timing_seconds).
        """
        t0 = time.time()
        if subtype:
            self.processor.default_subtype = subtype
        records = self.processor.process_file(fasta_path)

        if skip_existing:
            existing_ids = self.db.get_sequence_ids_in_db()
            new_records = [r for r in records if r.sequence_id not in existing_ids]
            skipped = len(records) - len(new_records)
            if skipped:
                logger.info(
                    f"Incremental mode: skipping {skipped} sequences already "
                    f"present in the database."
                )
            records = new_records

        with self.db.bulk_operation() as conn:
            for rec in records:
                self.db.insert_sequence_conn(
                    conn, rec.sequence_id, rec.subtype,
                    rec.collection_date, rec.metadata,
                )
                status = (
                    "failed" if rec.segments_found == 0
                    else "processed" if rec.is_complete
                    else "partial"
                )
                self.db.update_sequence_status_conn(
                    conn, rec.sequence_id, status, rec.segments_found
                )

        all_signatures: Dict[str, Dict[str, MinHashSignature]] = {}
        with self.db.bulk_operation() as conn:
            for rec in records:
                seq_sigs = {}
                for seg_name, seg_rec in rec.valid_segments.items():
                    sig = self.extractor.extract_signature(seg_rec.sequence, seg_name)
                    seq_sigs[seg_name] = sig
                    self.db.insert_segment_kmer_conn(
                        conn, rec.sequence_id, seg_name,
                        self.config.kmer.get_k(seg_name),
                        sig.to_bytes(), seg_rec.length, cluster_version,
                    )
                all_signatures[rec.sequence_id] = seq_sigs

        return records, all_signatures, time.time() - t0

    def _assign_nomenclature_and_genotypes(
        self,
        records,
        all_assignments: Dict[str, Dict[str, ClusterAssignment]],
        cluster_version: str,
        all_cluster_defs: Optional[Dict[str, Dict[str, "ClusterDefinition"]]] = None,
    ):
        """Run naming and genotype profile construction, write results to DB.

        Parameters
        ----------
        all_cluster_defs : dict, optional
            {subtype: {segment: [ClusterDefinition]}} — used to extract centroid
            signatures so NomenclatureManager can perform Stage 2 allele matching.
            When provided, every cluster assignment is accompanied by its centroid
            signature, enabling stable allele names across re-clustering runs.

        Returns (genotypes, naming_results, timing_seconds).
        """
        t0 = time.time()
        subtype_map = {rec.sequence_id: rec.subtype for rec in records}

        cluster_id_map: Dict[str, Dict[str, Optional[str]]] = {}
        for seq_id, seg_assignments in all_assignments.items():
            cluster_id_map[seq_id] = {
                seg: (None if a.is_orphan else a.cluster_id)
                for seg, a in seg_assignments.items()
            }

        # Build centroid signature map: sequence_id -> segment -> centroid sig
        # This allows NomenclatureManager Stage 2 to match new cluster IDs to
        # existing allele names by centroid similarity rather than by ID string.
        centroid_sig_map: Dict[str, Dict[str, MinHashSignature]] = {}
        # Parallel map of per-cluster basin radius (data-derived novelty margin).
        cluster_radius_map: Dict[str, Dict[str, float]] = {}
        if all_cluster_defs is not None:
            # Build lookup: (subtype, segment, cluster_id) -> centroid signature
            centroid_lookup: Dict[Tuple[str, str, str], MinHashSignature] = {}
            radius_lookup: Dict[Tuple[str, str, str], float] = {}
            for st, seg_results in all_cluster_defs.items():
                for seg, cluster_list in seg_results.items():
                    for cdef in cluster_list:
                        if cdef.centroid_signature is not None:
                            centroid_lookup[(st, seg, cdef.cluster_id)] = (
                                cdef.centroid_signature
                            )
                            radius_lookup[(st, seg, cdef.cluster_id)] = (
                                getattr(cdef, "radius", 0.0)
                            )

            for rec in records:
                st = rec.subtype
                seq_sigs: Dict[str, MinHashSignature] = {}
                seq_radii: Dict[str, float] = {}
                for seg, a in all_assignments.get(rec.sequence_id, {}).items():
                    if not a.is_orphan and a.cluster_id:
                        sig = centroid_lookup.get((st, seg, a.cluster_id))
                        if sig is not None:
                            seq_sigs[seg] = sig
                            seq_radii[seg] = radius_lookup.get((st, seg, a.cluster_id), 0.0)
                if seq_sigs:
                    centroid_sig_map[rec.sequence_id] = seq_sigs
                    cluster_radius_map[rec.sequence_id] = seq_radii

        naming_results = self.nomenclature.name_genotypes_batch(
            cluster_id_map, subtype_map, cluster_version,
            all_centroid_signatures=centroid_sig_map if centroid_sig_map else None,
            all_cluster_radii=cluster_radius_map if cluster_radius_map else None,
        )

        available_map = {
            rec.sequence_id: list(rec.valid_segments.keys()) for rec in records
        }
        genotypes = self.assigner.assign_batch(
            all_assignments, available_map
        )

        with self.db.bulk_operation() as conn:
            for seq_id, naming in naming_results.items():
                for seg, allele_name in naming["alleles"].items():
                    if allele_name:
                        self.db.update_allele_assignment_conn(
                            conn, seq_id, seg, allele_name, cluster_version
                        )
            for gt in genotypes:
                naming = naming_results.get(gt.sequence_id, {})
                self.db.insert_genotype_conn(
                    conn,
                    sequence_id=gt.sequence_id,
                    genotype_profile=gt.profile_string,
                    cluster_version=cluster_version,
                    allele_profile=naming.get("allele_string"),
                    constellation_id=naming.get("constellation"),
                    completeness=gt.completeness,
                )

        # ── Update member counts in allele_registry and constellation_registry
        # Tally how many sequences were assigned to each allele and constellation
        # in this run, then write the counts back to the registry tables.
        # These are denormalised counters — ground truth is always in
        # segment_kmers and genotypes — but they are useful for quick inspection.
        allele_counts: Dict[str, int] = {}
        for naming in naming_results.values():
            for allele_name in naming.get("alleles", {}).values():
                if allele_name:
                    allele_counts[allele_name] = allele_counts.get(allele_name, 0) + 1
        for allele_name, count in allele_counts.items():
            self.db.update_allele_last_seen(allele_name, member_count=count)

        constellation_counts: Dict[str, int] = {}
        for naming in naming_results.values():
            cid = naming.get("constellation")
            if cid:
                constellation_counts[cid] = constellation_counts.get(cid, 0) + 1
        for cid, count in constellation_counts.items():
            self.db.update_constellation_last_seen(cid, member_count=count)

        return genotypes, naming_results, time.time() - t0

    def _detect_reassortment(
        self,
        genotypes: List[GenotypeProfile],
        all_signatures: Dict[str, Dict[str, MinHashSignature]],
        naming_results: Optional[Dict] = None,
    ):
        """Run two-stage reassortment detection and persist events.

        Parameters
        ----------
        genotypes : list of GenotypeProfile
        all_signatures : dict
            MinHash signatures keyed by sequence_id -> segment -> signature.
        naming_results : dict, optional
            Output of NomenclatureManager.name_genotypes_batch(), keyed by
            sequence_id.  When provided, enables Stage 0 allele subtype
            discordance detection — including lineage-resolved discordance
            for alleles whose cross-subtype origin was established by a
            prior repair_allele_subtype() call.

        Returns (report, timing_seconds).
        """
        t0 = time.time()
        # Clear any existing events for these sequences before detecting and
        # re-inserting — prevents duplicate rows when the same sequences are
        # re-run in batch mode. allele_lineage and allele_registry are
        # unaffected; the knowledge anchored there is never touched here.
        seq_ids = [g.sequence_id for g in genotypes]
        self.db.delete_reassortment_events(seq_ids)

        report = self.reassortment.detect_reassortments(
            genotypes, all_signatures, nomenclature=naming_results
        )
        for ev in report.events:
            desc = f"[Stage 2] {ev.description}" if ev.detection_stage == 2 else ev.description
            self.db.insert_reassortment_event(
                ev.sequence_id, ev.discordant_segments, ev.confidence, desc
            )
        return report, time.time() - t0

    def _load_reference_clusters(
        self,
        subtypes: List[str],
        ref_version: Optional[str],
    ) -> Dict[str, Dict[str, List[ClusterDefinition]]]:
        """Load cluster centroids from the DB for all subtypes and segments.

        Parameters
        ----------
        subtypes:
            Subtypes present in the new sequences.
        ref_version:
            Explicit cluster_version to load from.  When ``None``, the most
            recently created active version for each segment/subtype pair is
            used automatically.

        Returns
        -------
        Nested dict: ``{subtype: {segment: [ClusterDefinition, ...]}}``
        """
        ref_clusters: Dict[str, Dict[str, List[ClusterDefinition]]] = {}

        for st in subtypes:
            ref_clusters[st] = {}
            for seg in SEGMENTS:
                if ref_version:
                    rows = self.db.get_active_clusters_by_version(seg, st, ref_version)
                else:
                    latest = self.db.get_latest_cluster_version(seg, st)
                    if latest is None:
                        logger.warning(
                            f"No active clusters found for {seg}/{st}. "
                            f"All sequences for this segment will be orphans."
                        )
                        ref_clusters[st][seg] = []
                        continue
                    rows = self.db.get_active_clusters_by_version(seg, st, latest)

                cluster_defs = []
                for row in rows:
                    sig_bytes = row.get("centroid_signature")
                    centroid_sig = (
                        MinHashSignature.from_bytes(sig_bytes)
                        if sig_bytes
                        else None
                    )
                    cdef = ClusterDefinition(
                        cluster_id=row["cluster_id"],
                        segment_name=seg,
                        subtype=st,
                        centroid_signature=centroid_sig,
                        mean_diameter=row.get("mean_diameter", 0.0),
                        version=row["version"],
                    )
                    cluster_defs.append(cdef)

                ref_clusters[st][seg] = cluster_defs
                logger.debug(
                    f"Loaded {len(cluster_defs)} reference clusters for {seg}/{st}"
                )

        return ref_clusters

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        fasta_path: str,
        subtype: Optional[str] = None,
        cluster_version: str = "v1",
        detect_reassortment: bool = True,
    ) -> Dict:
        """Full batch clustering run.

        Clusters all sequences in ``fasta_path`` from scratch and writes a
        complete new cluster set to the database.  All existing clusters for
        the same ``cluster_version`` are overwritten.

        Use this for the initial reference run on a baseline dataset, or for a
        one-off re-analysis with different parameters (use a new
        ``cluster_version`` to avoid clobbering production data).

        For adding new sequences to an existing clustering, use
        ``run_incremental()``.  For periodic supervised re-clustering that
        supersedes previous versions, use ``run_recluster()``.
        """
        timing: Dict[str, float] = {}
        results: Dict = {}

        # Steps 1 & 2: Parse, store, extract signatures
        records, all_signatures, t = self._parse_and_store_sequences(
            fasta_path, subtype, cluster_version, skip_existing=False
        )
        results["records"] = records
        results["signatures"] = all_signatures
        timing["sequence_processing_and_kmer"] = t

        # Step 3: Full clustering per segment per subtype
        t0 = time.time()
        by_subtype: Dict[str, List] = {}
        for rec in records:
            by_subtype.setdefault(rec.subtype, []).append(rec)

        all_clustering: Dict[str, Dict[str, ClusteringResult]] = {}
        all_assignments: Dict[str, Dict[str, ClusterAssignment]] = {}

        for st, st_records in by_subtype.items():
            all_clustering[st] = {}
            for seg in SEGMENTS:
                seq_ids, sigs = [], []
                for rec in st_records:
                    if rec.sequence_id in all_signatures and seg in all_signatures[rec.sequence_id]:
                        seq_ids.append(rec.sequence_id)
                        sigs.append(all_signatures[rec.sequence_id][seg])
                if len(sigs) < 2:
                    continue
                cr = self.clusterer.cluster_signatures(
                    seq_ids, sigs, seg, st, cluster_version
                )
                all_clustering[st][seg] = cr
                for a in cr.assignments:
                    all_assignments.setdefault(a.sequence_id, {})[seg] = a

        with self.db.bulk_operation() as conn:
            for st, seg_results in all_clustering.items():
                for seg, cr in seg_results.items():
                    for cdef in cr.clusters:
                        self.db.insert_cluster_conn(
                            conn, cdef.cluster_id, seg, st,
                            cdef.centroid_signature.to_bytes() if cdef.centroid_signature else b"",
                            cdef.size, cdef.mean_diameter, cluster_version,
                            radius=getattr(cdef, "radius", 0.0),
                        )
                    for a in cr.assignments:
                        if not a.is_orphan:
                            self.db.update_cluster_assignment_conn(
                                conn, a.sequence_id, seg,
                                a.cluster_id, cluster_version, a.distance_to_centroid,
                            )
                        else:
                            self.db.flag_orphan_conn(
                                conn, a.sequence_id, seg,
                                a.nearest_cluster, a.nearest_distance,
                            )

        results["clustering"] = all_clustering
        timing["clustering"] = time.time() - t0

        # Step 4: Nomenclature + genotypes
        # Pass cluster definitions so centroid signatures reach the nomenclature
        # manager, enabling Stage 2 allele matching on future re-clustering runs.
        genotypes, naming_results, t = self._assign_nomenclature_and_genotypes(
            records, all_assignments, cluster_version,
            all_cluster_defs={
                st: {seg: cr.clusters for seg, cr in seg_results.items()}
                for st, seg_results in all_clustering.items()
            },
        )
        results["genotypes"] = genotypes
        results["nomenclature"] = naming_results
        timing["genotype_assignment"] = t

        # Step 5: Reassortment
        if detect_reassortment:
            report, t = self._detect_reassortment(
                genotypes, all_signatures, naming_results=naming_results
            )
            results["reassortment"] = report
            timing["reassortment"] = t

        timing["total"] = sum(timing.values())
        results["timing"] = timing
        results["summary"] = self._build_summary(results, mode="batch")
        return results

    def run_incremental(
        self,
        fasta_path: str,
        subtype: Optional[str] = None,
        cluster_version: Optional[str] = None,
        detect_reassortment: bool = True,
    ) -> Dict:
        """Incremental assignment of new sequences to existing clusters.

        Loads reference cluster centroids from the database — from the version
        specified by ``cluster_version``, or the most recently created active
        version if ``cluster_version`` is ``None`` — and assigns each new
        sequence to its nearest cluster using vectorised centroid comparison.
        Cluster definitions are never modified.  Sequences that fall outside
        all clusters (Jaccard distance > threshold) are flagged as orphans.

        Because the same cluster IDs are reused, allele names and constellation
        IDs produced by this run are directly comparable to those from the
        original batch run.

        Sequences already present in the database (matched by sequence_id) are
        silently skipped, making the method safe to call repeatedly on
        overlapping FASTA files.

        Parameters
        ----------
        fasta_path:
            FASTA file containing the new sequences to assign.
        subtype:
            Default subtype override (same semantics as ``run()``).
        cluster_version:
            The cluster version to assign sequences into.  Pass ``None`` to
            automatically use the most recently created active version.
        detect_reassortment:
            Whether to run reassortment detection on the newly assigned
            sequences.
        """
        timing: Dict[str, float] = {}
        results: Dict = {}

        # Steps 1 & 2: Parse new sequences only, store, extract signatures
        records, all_signatures, t = self._parse_and_store_sequences(
            fasta_path, subtype,
            cluster_version or "incremental",
            skip_existing=True,
        )
        results["records"] = records
        results["signatures"] = all_signatures
        timing["sequence_processing_and_kmer"] = t

        if not records:
            logger.info("Incremental run: no new sequences to process.")
            results["assignments"] = {}
            results["genotypes"] = []
            results["nomenclature"] = {}
            timing["total"] = sum(timing.values())
            results["timing"] = timing
            results["summary"] = self._build_summary(results, mode="incremental")
            return results

        # Step 3: Load reference clusters and assign
        t0 = time.time()
        by_subtype: Dict[str, List] = {}
        for rec in records:
            by_subtype.setdefault(rec.subtype, []).append(rec)

        ref_clusters = self._load_reference_clusters(
            list(by_subtype.keys()), cluster_version
        )

        # Resolve the actual version string we're writing into
        resolved_version: str = cluster_version or ""
        if not resolved_version:
            for st_clusters in ref_clusters.values():
                for seg_clusters in st_clusters.values():
                    if seg_clusters:
                        resolved_version = seg_clusters[0].version
                        break
                if resolved_version:
                    break
        if not resolved_version:
            resolved_version = "incremental"

        all_assignments: Dict[str, Dict[str, ClusterAssignment]] = {}

        for st, st_records in by_subtype.items():
            st_ref = ref_clusters.get(st, {})
            for seg in SEGMENTS:
                seg_clusters = st_ref.get(seg, [])
                seq_ids, sigs = [], []
                for rec in st_records:
                    if rec.sequence_id in all_signatures and seg in all_signatures[rec.sequence_id]:
                        seq_ids.append(rec.sequence_id)
                        sigs.append(all_signatures[rec.sequence_id][seg])
                if not seq_ids:
                    continue

                if not seg_clusters:
                    for sid in seq_ids:
                        all_assignments.setdefault(sid, {})[seg] = ClusterAssignment(
                            sid, seg, None, 1.0, True
                        )
                    continue

                assignments = self.clusterer.assign_batch_to_existing(
                    seq_ids, sigs, seg_clusters, seg, st
                )
                for a in assignments:
                    all_assignments.setdefault(a.sequence_id, {})[seg] = a

        # Persist assignments and update cluster member counts
        cluster_member_deltas: Dict[tuple, int] = {}

        with self.db.bulk_operation() as conn:
            for st, st_records in by_subtype.items():
                for seg in SEGMENTS:
                    for rec in st_records:
                        a = all_assignments.get(rec.sequence_id, {}).get(seg)
                        if a is None:
                            continue
                        if not a.is_orphan:
                            self.db.update_cluster_assignment_conn(
                                conn, a.sequence_id, seg,
                                a.cluster_id, resolved_version,
                                a.distance_to_centroid,
                            )
                            key = (a.cluster_id, seg, st, resolved_version)
                            cluster_member_deltas[key] = cluster_member_deltas.get(key, 0) + 1
                        else:
                            self.db.flag_orphan_conn(
                                conn, a.sequence_id, seg,
                                a.nearest_cluster, a.nearest_distance,
                            )

        # Update member counts on cluster rows (outside the bulk transaction
        # to avoid reading within the same write transaction)
        for (cluster_id, seg, st, version), delta in cluster_member_deltas.items():
            rows = self.db.get_active_clusters_by_version(seg, st, version)
            for row in rows:
                if row["cluster_id"] == cluster_id:
                    self.db.update_cluster_member_count(
                        cluster_id, seg, version, row["member_count"] + delta
                    )
                    break

        timing["incremental_assignment"] = time.time() - t0

        # Step 4: Nomenclature + genotypes
        # Pass reference cluster definitions so centroid signatures are available
        # for Stage 2 allele matching — the same centroids as the original batch run.
        genotypes, naming_results, t = self._assign_nomenclature_and_genotypes(
            records, all_assignments, resolved_version,
            all_cluster_defs={
                st: seg_clusters
                for st, seg_clusters in ref_clusters.items()
            },
        )
        results["genotypes"] = genotypes
        results["nomenclature"] = naming_results
        timing["genotype_assignment"] = t

        # Step 5: Reassortment
        if detect_reassortment:
            report, t = self._detect_reassortment(
                genotypes, all_signatures, naming_results=naming_results
            )
            results["reassortment"] = report
            timing["reassortment"] = t

        orphan_counts: Dict[str, int] = {}
        for seg_assignments in all_assignments.values():
            for seg, a in seg_assignments.items():
                if a.is_orphan:
                    orphan_counts[seg] = orphan_counts.get(seg, 0) + 1

        timing["total"] = sum(timing.values())
        results["timing"] = timing
        results["orphan_counts"] = orphan_counts
        results["summary"] = self._build_summary(results, mode="incremental")
        return results

    def run_recluster(
        self,
        fasta_path: str,
        subtype: Optional[str] = None,
        detect_reassortment: bool = True,
    ) -> Dict:
        """Periodic re-clustering that supersedes the previous reference clustering.

        Equivalent to ``run()`` but:
            1. Automatically derives a new ``cluster_version`` string from the
               current UTC date (``v{YYYYMMDD}``).
            2. Marks all previous active clusters for each affected
               segment/subtype as inactive in the database before writing the
               new ones, preventing stale centroids from being loaded by future
               incremental runs.
            3. Logs a clear audit message recording the new version.

        Use this when orphan accumulation from incremental runs indicates that
        the existing clusters no longer represent circulating diversity well.
        A common trigger is an orphan rate above 10–15% over a surveillance
        period.

        After ``run_recluster()``, subsequent ``run_incremental()`` calls will
        automatically pick up the new version (as it will be the most recently
        created active version).
        """
        new_version = f"v{datetime.utcnow().strftime('%Y%m%d')}"
        logger.info(f"Re-clustering run started. New cluster_version: {new_version}")

        results = self.run(
            fasta_path=fasta_path,
            subtype=subtype,
            cluster_version=new_version,
            detect_reassortment=detect_reassortment,
        )

        # Retire all previous active clusters for affected segment/subtype pairs
        records = results.get("records", [])
        affected_subtypes = {rec.subtype for rec in records}

        with self.db.bulk_operation() as conn:
            for st in affected_subtypes:
                for seg in SEGMENTS:
                    conn.execute(
                        """UPDATE clusters SET is_active=0
                           WHERE segment_name=? AND subtype=? AND version!=? AND is_active=1""",
                        (seg, st, new_version),
                    )

        logger.info(
            f"Re-clustering complete. Previous cluster versions retired. "
            f"Active version is now: {new_version}"
        )

        results["cluster_version"] = new_version
        results["summary"]["mode"] = "recluster"
        results["summary"]["new_cluster_version"] = new_version
        return results

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_reassortments(
        self,
        report: ReassortmentReport,
        genotypes: List[GenotypeProfile],
        n_permutations: int = 5000,
        seed: int = 42,
    ) -> List[PermutationResult]:
        """Validate reassortment events using permutation testing.

        Wraps ``ReassortmentDetector.validate_events``.  Call after any of
        the three run methods to confirm borderline events or produce
        publication-quality empirical p-values.
        """
        return self.reassortment.validate_events(
            report.events, genotypes, n_permutations, seed
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _build_summary(self, results: Dict, mode: str = "batch") -> Dict:
        records = results.get("records", [])
        genotypes = results.get("genotypes", [])
        gt_summary = self.assigner.get_genotype_summary(genotypes) if genotypes else {}
        clustering = results.get("clustering", {})
        reassortment = results.get("reassortment")

        total_clusters = sum(
            cr.num_clusters for st in clustering.values() for cr in st.values()
        )
        total_orphans = sum(
            cr.num_orphans for st in clustering.values() for cr in st.values()
        )

        if mode == "incremental":
            orphan_counts = results.get("orphan_counts", {})
            total_orphans = sum(orphan_counts.values())
            total_clusters = 0

        stage1_count = stage2_count = 0
        if reassortment:
            stage1_count = sum(1 for e in reassortment.events if e.detection_stage == 1)
            stage2_count = sum(1 for e in reassortment.events if e.detection_stage == 2)

        return {
            "mode": mode,
            "total_isolates": len(records),
            "complete_genomes": sum(1 for r in records if r.is_complete),
            "by_subtype": (
                {st: sum(1 for r in records if r.subtype == st)
                 for st in {r.subtype for r in records}}
                if records else {}
            ),
            "unique_genotypes": gt_summary.get("unique_genotypes", 0),
            "total_clusters": total_clusters,
            "total_orphans": total_orphans,
            "reassortment_events": reassortment.flagged_sequences if reassortment else 0,
            "reassortment_stage1": stage1_count,
            "reassortment_stage2": stage2_count,
            "nomenclature": self.nomenclature.get_registry_summary(),
            "timing": results.get("timing", {}),
        }

    # ------------------------------------------------------------------
    # Allele repair
    # ------------------------------------------------------------------

    def repair_allele_subtype(
        self,
        segment_name: str,
        correct_subtype: str,
        misnamed_allele: str,
        cluster_version: str,
    ) -> Dict:
        """Correct a segment allele that was minted under the wrong subtype.

        Background
        ----------
        When a reassortant isolate is processed before any isolates from the
        donor population, its foreign segment is minted as an allele under the
        recipient subtype (e.g. ``PB1.1.0003`` for an H1N1-background isolate
        carrying an H3N2-origin PB1).  Subsequent sequences from the true donor
        population then match that centroid via Stage 2b and inherit the
        wrong-subtype allele name, causing false-positive Stage 0 reassortment
        events for the entire donor population.

        This method corrects the situation **without renaming or invalidating
        the existing allele**.  It:

        1. Loads the stored k-mer signatures for all sequences of
           ``correct_subtype`` that currently carry ``misnamed_allele``.
        2. Creates a fresh nomenclature context that excludes ``misnamed_allele``
           from the centroid index so Stage 2 cannot match back to it.
        3. Runs those signatures through ``assign_allele()`` — they will either
           match a same-subtype centroid (if one exists from a separate run) or
           mint a new correctly-named allele (e.g. ``PB1.3.0004``).
        4. Records an ``allele_lineage`` link between ``misnamed_allele`` and
           the newly assigned name so the relationship is permanently auditable.
        5. Updates ``segment_kmers.allele_id`` and ``genotypes.allele_profile``
           in place for all affected sequences.

        The ``misnamed_allele`` remains in ``allele_registry`` as an active
        allele — it is the correct name for the original reassortant isolate.
        Only the donor-population sequences have their allele_id updated.

        Parameters
        ----------
        segment_name : str
            The segment whose allele is wrong (e.g. ``"PB1"``).
        correct_subtype : str
            The subtype string of the donor population whose sequences are
            being mis-flagged (e.g. ``"H3N2"``).
        misnamed_allele : str
            The allele name that was minted under the wrong subtype
            (e.g. ``"PB1.1.0003"``).
        cluster_version : str
            The cluster version to operate on (e.g. ``"v1"``).

        Returns
        -------
        dict with keys:
            affected_sequences : list[str]
                Sequence IDs whose allele_id was updated.
            old_allele : str
                The misnamed allele (unchanged in the registry).
            new_allele : str
                The correctly-named allele now assigned to donor sequences.
            lineage_link_recorded : bool
                True if a new allele_lineage row was written.
            similarity : float
                Jaccard similarity between the two allele centroids.
        """
        from .core.nomenclature import NomenclatureManager, subtype_num
        from .core.kmer_extractor import MinHashSignature

        logger.info(
            f"repair_allele_subtype: segment={segment_name}, "
            f"correct_subtype={correct_subtype}, "
            f"misnamed_allele={misnamed_allele}, "
            f"cluster_version={cluster_version}"
        )

        # ── Step 1: find affected sequences ──────────────────────────────────
        rows = self.db.get_segment_signatures_for_subtype(
            segment_name, correct_subtype, cluster_version
        )
        affected_rows = [r for r in rows if r["allele_id"] == misnamed_allele]
        if not affected_rows:
            logger.info(
                f"repair_allele_subtype: no {correct_subtype} sequences carry "
                f"{misnamed_allele} — nothing to repair."
            )
            return {
                "affected_sequences": [],
                "old_allele": misnamed_allele,
                "new_allele": misnamed_allele,
                "lineage_link_recorded": False,
                "similarity": 0.0,
            }

        logger.info(
            f"repair_allele_subtype: {len(affected_rows)} {correct_subtype} sequences "
            f"carry {misnamed_allele} — rebuilding allele for correct subtype."
        )

        # ── Step 2: build a fresh nomenclature context ────────────────────────
        # Load the full registry, then *remove* the misnamed allele from the
        # centroid index so the affected signatures can't match back to it.
        # This forces Stage 2 to either find a legitimate same-subtype centroid
        # or mint a new correctly-named allele.
        repair_nom = NomenclatureManager(
            db=self.db, clustering_config=self.config.clustering
        )
        repair_nom.load_from_db()

        snum = subtype_num(correct_subtype)
        idx_key = (segment_name, snum)
        # The misnamed allele was stored under the *wrong* subtype number,
        # so it lives in a different index — but scan all segment indices and
        # remove it cleanly from any that contain it.
        import numpy as np
        for key, index in repair_nom._centroid_indices.items():
            if key[0] != segment_name:
                continue
            if misnamed_allele not in index._names:
                continue
            # Consolidate first so _matrix is current
            index._consolidate()
            idx = index._names.index(misnamed_allele)
            index._names.pop(idx)
            if index._matrix is not None and index._matrix.shape[0] > idx:
                index._matrix = np.delete(index._matrix, idx, axis=0)
            logger.debug(
                f"repair_allele_subtype: removed {misnamed_allele} from "
                f"centroid index {key}"
            )

        # ── Step 3: assign allele to a representative signature ──────────────
        # Use the first affected row's signature as the representative centroid.
        # All affected sequences should be from the same lineage (they all
        # matched the misnamed allele centroid), so the first is sufficient.
        rep_row = affected_rows[0]
        rep_sig = MinHashSignature.from_bytes(rep_row["kmer_signature"])
        synthetic_cid = f"REPAIR::{correct_subtype}::{segment_name}::{cluster_version}"

        new_allele = repair_nom.assign_allele(
            segment_name=segment_name,
            subtype=correct_subtype,
            internal_cluster_id=synthetic_cid,
            cluster_version=cluster_version,
            centroid_signature=rep_sig,
        )

        logger.info(
            f"repair_allele_subtype: {misnamed_allele} → {new_allele} "
            f"for {len(affected_rows)} {correct_subtype} sequences."
        )

        # ── Step 4: record the lineage link ──────────────────────────────────
        lineage_recorded = False
        similarity = 0.0
        if new_allele != misnamed_allele:
            # Compute similarity between the two centroids
            old_allele_row = self.db.get_allele_by_name(misnamed_allele)
            if old_allele_row and old_allele_row.get("centroid_signature"):
                old_sig = MinHashSignature.from_bytes(
                    old_allele_row["centroid_signature"]
                )
                similarity = MinHashSignature.jaccard_similarity(rep_sig, old_sig)
            self.db.record_allele_lineage(
                misnamed_allele,
                new_allele,
                similarity,
                evidence=f"repair_allele_subtype:{misnamed_allele}→{new_allele}",
            )
            lineage_recorded = True
            logger.info(
                f"repair_allele_subtype: lineage link recorded — "
                f"{misnamed_allele} ↔ {new_allele} (sim={similarity:.3f})"
            )

        # ── Step 5: update segment_kmers and genotypes in DB ─────────────────
        affected_ids = [r["sequence_id"] for r in affected_rows]
        with self.db.bulk_operation() as conn:
            for seq_id in affected_ids:
                conn.execute(
                    """UPDATE segment_kmers
                       SET allele_id = ?
                       WHERE sequence_id = ? AND segment_name = ?
                         AND cluster_version = ?""",
                    (new_allele, seq_id, segment_name, cluster_version),
                )

        # Rebuild allele_profile strings for affected genotype rows
        for seq_id in affected_ids:
            geno = self.db.get_genotype(seq_id)
            if not geno or not geno.get("allele_profile"):
                continue
            old_profile = geno["allele_profile"]
            new_profile = old_profile.replace(misnamed_allele, new_allele)
            if new_profile == old_profile:
                continue
            # Recompute constellation from the updated allele map
            allele_map = {}
            for seg_allele in new_profile.split(" | "):
                parts = seg_allele.split(".")
                if len(parts) >= 1:
                    seg = parts[0]
                    if seg in SEGMENTS:
                        allele_map[seg] = seg_allele
            new_constellation = self.nomenclature.assign_constellation(
                allele_map, correct_subtype
            )
            self.db.update_genotype_allele_profile(
                seq_id, cluster_version, new_profile, new_constellation
            )

        # ── Step 5b: remove stale reassortment events for affected sequences ───
        # The affected sequences were previously flagged as reassortants because
        # they carried a cross-subtype allele name.  Now that the allele has been
        # correctly renamed, those events are false positives and must be removed.
        # The original reassortant isolate is NOT in affected_ids so its
        # legitimate reassortment event is preserved.
        stale_removed = self.db.delete_reassortment_events(affected_ids)
        if stale_removed:
            logger.info(
                f"repair_allele_subtype: removed {stale_removed} stale "
                f"reassortment event(s) for affected sequences."
            )

        # ── Step 5c: retire constellations made empty by the repair ────────────
        # Constellations whose allele_combination contained the misnamed allele
        # now have zero members — the affected sequences were reassigned to new
        # constellations reflecting the corrected allele name.  Retire them so
        # they are excluded from active lookups without deleting the audit trail.
        constellations_retired = self.db.retire_empty_constellations(
            containing_allele=misnamed_allele
        )
        if constellations_retired:
            logger.info(
                f"repair_allele_subtype: retired {constellations_retired} empty "
                f"constellation(s) containing {misnamed_allele}."
            )

        logger.info(
            f"repair_allele_subtype: complete. "
            f"Updated {len(affected_ids)} sequences: {affected_ids}"
        )

        return {
            "affected_sequences": affected_ids,
            "old_allele": misnamed_allele,
            "new_allele": new_allele,
            "lineage_link_recorded": lineage_recorded,
            "similarity": similarity,
            "stale_events_removed": stale_removed,
            "constellations_retired": constellations_retired,
        }
