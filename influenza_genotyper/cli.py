#!/usr/bin/env python3
"""
cli.py — Command-line wrapper for the influenza genotyper pipeline.

Subcommands
-----------

  run          Run the genotyping pipeline (batch / incremental / recluster).

  repair       Correct a segment allele that was minted under the wrong subtype.
               Use when a reassortant isolate was processed before any isolates
               from the true donor population, causing the foreign segment to be
               named under the recipient subtype (e.g. PB1.1.0002 for an H3N2
               PB1 segment in an H1N1-background isolate).  The repair re-names
               all donor-population sequences to a correctly-typed allele, records
               a permanent lineage link between the two names, and removes the
               stale false-positive reassortment events from the database.

Typical workflow
----------------

  # 1. Initial reference run on baseline dataset
  run-genotyper run sequences_baseline.fasta --mode batch --cluster-version v1

  # 2. Weekly incremental ingestion of new sequences
  run-genotyper run sequences_week42.fasta --mode incremental

  # 3. Annual re-clustering when orphan rate is high
  run-genotyper run sequences_all.fasta --mode recluster

  # 4. Repair a misnamed allele (dry-run first, then commit)
  run-genotyper repair --segment PB1 --correct-subtype H3N2 \\
      --misnamed-allele PB1.1.0002 --cluster-version v1 --dry-run

  run-genotyper repair --segment PB1 --correct-subtype H3N2 \\
      --misnamed-allele PB1.1.0002 --cluster-version v1
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Argument parsers
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run-genotyper",
        description="K-mer based genotyping and reassortment detection for influenza whole-genome sequences.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    sub = p.add_subparsers(dest="subcommand", metavar="SUBCOMMAND")
    sub.required = True

    _add_run_parser(sub)
    _add_repair_parser(sub)

    return p


def _add_run_parser(sub):
    p = sub.add_parser(
        "run",
        help="Run the genotyping pipeline (batch / incremental / recluster).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("fasta", metavar="FASTA",
        help="Input FASTA file. Segment identity is inferred from sequence headers.")

    p.add_argument("--mode", choices=["batch", "incremental", "recluster"], default="batch",
        help=(
            "Operating mode. "
            "'batch': full clustering from scratch (use for initial reference run). "
            "'incremental': assign new sequences to existing clusters (routine surveillance). "
            "'recluster': full re-clustering that retires old clusters and assigns a new dated version. "
            "(default: batch)"
        ))

    p.add_argument("--subtype", metavar="SUBTYPE", default=None,
        help="Default subtype when not encoded in FASTA headers (e.g. H3N2, H1N1pdm09).")

    p.add_argument("--cluster-version", metavar="VERSION", default=None,
        help=(
            "Cluster version label. "
            "batch: sets the version for the new cluster set (default: v1). "
            "incremental: which existing version to assign into (default: auto-detect latest). "
            "recluster: ignored — version is set from the current date."
        ))

    p.add_argument("--db", metavar="PATH", default="data/influenza_genotyper.db",
        help="Path to the SQLite database. The same DB must be used across batch and incremental runs.")

    p.add_argument("--out-dir", metavar="DIR", default=".",
        help="Directory for output TSV files and JSON summary. Created if absent.")

    p.add_argument("--no-tsv", action="store_true", default=False,
        help="Skip writing TSV output files. Summary is still printed to stdout.")

    p.add_argument("--num-hashes", metavar="N", type=int, default=None,
        help="MinHash signature size. Must be identical across all compared runs. "
             "When omitted, the KmerConfig default is used. (default: 1024)")

    p.add_argument("--hash-seed", metavar="SEED", type=int, default=42,
        help="MinHash base seed. Must be identical across all compared runs. (default: 42)")

    p.add_argument("--min-cluster-size", metavar="N", type=int, default=10,
        help="Minimum cluster members. Only applies in batch/recluster modes. (default: 10)")

    p.add_argument("--linkage", metavar="METHOD", default="average",
        choices=["average", "complete", "single", "ward"],
        help="Hierarchical clustering linkage method. Only applies in batch/recluster modes. (default: average)")

    p.add_argument("--dev-mode", action="store_true", default=False,
        help="Lower min-cluster-size to 2 for small pilot datasets. NOT for production.")

    p.add_argument("--no-reassortment", action="store_true", default=False,
        help="Skip reassortment detection entirely.")

    p.add_argument("--alpha", metavar="FLOAT", type=float, default=0.05,
        help="Significance level for Stage 1 Fisher exact + Stouffer Z test. (default: 0.05)")

    p.add_argument("--no-bonferroni", action="store_true", default=False,
        help="Disable Bonferroni correction across segments in Stage 1.")

    p.add_argument("--zscore-threshold", metavar="FLOAT", type=float, default=2.0,
        help="Stage 2 z-score cutoff for within-cluster distance outliers. (default: 2.0)")

    p.add_argument("--min-confidence", metavar="FLOAT", type=float, default=0.7,
        help="Minimum confidence score (0-1) to report a reassortment event. (default: 0.7)")

    p.add_argument("--validate-reassortment", action="store_true", default=False,
        help="Run permutation-based validation on Stage 1 reassortment events.")

    p.add_argument("--permutations", metavar="N", type=int, default=1000,
        help="Permutations per segment per event for validation. (default: 1000)")

    p.add_argument("--log-level", metavar="LEVEL", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity. (default: INFO)")

    return p


def _add_repair_parser(sub):
    p = sub.add_parser(
        "repair",
        help="Correct a segment allele minted under the wrong subtype.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Correct a segment allele that was minted under the wrong subtype.\n\n"
            "Use this when a reassortant isolate was processed before any isolates\n"
            "from the true donor population, causing the foreign segment to be named\n"
            "under the recipient subtype (e.g. PB1.1.0002 for a true H3N2 PB1).\n\n"
            "The repair:\n"
            "  1. Re-names all donor-population sequences to a correctly-typed allele\n"
            "     (e.g. PB1.3.0002), leaving the original reassortant isolate unchanged.\n"
            "  2. Records a permanent allele_lineage link between the two names.\n"
            "  3. Removes stale false-positive reassortment events for the affected\n"
            "     sequences from the database.\n\n"
            "Use --dry-run first to confirm which sequences will be affected.\n\n"
            "Example\n"
            "-------\n"
            "  run-genotyper repair \\\n"
            "      --segment PB1 \\\n"
            "      --correct-subtype H3N2 \\\n"
            "      --misnamed-allele PB1.1.0002 \\\n"
            "      --cluster-version v1 \\\n"
            "      --dry-run\n"
        ),
    )

    p.add_argument("--segment", metavar="SEG", required=True,
        help="Segment whose allele is misnamed (e.g. PB1, HA, NA).")

    p.add_argument("--correct-subtype", metavar="SUBTYPE", required=True,
        help=(
            "Subtype of the donor population whose sequences are currently "
            "carrying the misnamed allele (e.g. H3N2)."
        ))

    p.add_argument("--misnamed-allele", metavar="ALLELE", required=True,
        help="The allele name to repair (e.g. PB1.1.0002).")

    p.add_argument("--cluster-version", metavar="VERSION", required=True,
        help="Cluster version the allele belongs to (e.g. v1).")

    p.add_argument("--db", metavar="PATH", default="data/influenza_genotyper.db",
        help="Path to the SQLite database. (default: data/influenza_genotyper.db)")

    p.add_argument("--dry-run", action="store_true", default=False,
        help=(
            "Report which sequences would be affected without making any changes. "
            "Recommended before committing a repair."
        ))

    p.add_argument("--log-level", metavar="LEVEL", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity. (default: INFO)")

    return p


# ---------------------------------------------------------------------------
# TSV / output helpers
# ---------------------------------------------------------------------------

def write_genotypes_tsv(genotypes, naming_results, out_path):
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["sequence_id", "profile_string", "allele_string",
                    "constellation_id", "completeness", "is_complete",
                    "missing_segments", "orphan_segments"])
        for gt in genotypes:
            naming = naming_results.get(gt.sequence_id, {})
            w.writerow([
                gt.sequence_id, gt.profile_string,
                naming.get("allele_string", ""), naming.get("constellation", ""),
                f"{gt.completeness:.3f}", gt.is_complete,
                ",".join(gt.missing_segments) or "",
                ",".join(gt.orphan_segments) or "",
            ])


def write_clusters_tsv(clustering_results, out_path):
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["subtype", "segment", "cluster_id", "member_count", "mean_diameter"])
        for subtype, seg_results in clustering_results.items():
            for seg, cr in seg_results.items():
                for cl in cr.clusters:
                    w.writerow([subtype, seg, cl.cluster_id, cl.size, f"{cl.mean_diameter:.4f}"])


def write_orphans_tsv(clustering_results, out_path):
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["sequence_id", "segment", "nearest_cluster", "nearest_distance"])
        for subtype, seg_results in clustering_results.items():
            for seg, cr in seg_results.items():
                for a in cr.assignments:
                    if a.is_orphan:
                        w.writerow([
                            a.sequence_id, seg,
                            a.nearest_cluster or "",
                            f"{a.nearest_distance:.4f}" if a.nearest_distance is not None else "",
                        ])


def write_incremental_orphans_tsv(orphan_counts, out_path):
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["segment", "orphan_count"])
        for seg, count in sorted(orphan_counts.items()):
            w.writerow([seg, count])


def write_reassortment_tsv(report, out_path):
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["sequence_id", "detection_stage", "event_type",
                    "discordant_segments", "confidence", "description"])
        for ev in report.events:
            w.writerow([
                ev.sequence_id, ev.detection_stage, ev.event_type,
                ",".join(ev.discordant_segments), f"{ev.confidence:.4f}", ev.description,
            ])


def write_validation_tsv(validation_results, out_path):
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["sequence_id", "concordant", "n_permutations",
                    "segment", "empirical_p", "stouffer_p"])
        for v in validation_results:
            for seg, emp_p in v.segment_empirical_pvalues.items():
                stouffer_p = v.segment_stouffer_pvalues.get(seg, "")
                w.writerow([
                    v.sequence_id, v.concordant, v.n_permutations, seg,
                    f"{emp_p:.6f}",
                    f"{stouffer_p:.6f}" if stouffer_p != "" else "",
                ])


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def print_summary(summary, mode):
    mode_labels = {
        "batch":       "BATCH — full clustering",
        "incremental": "INCREMENTAL — assign to existing clusters",
        "recluster":   "RE-CLUSTER — new reference, old clusters retired",
    }
    timing = summary.get("timing", {})
    nom = summary.get("nomenclature", {})

    print("\n" + "=" * 64)
    print(f"  INFLUENZA GENOTYPER — {mode_labels.get(mode, mode.upper())}")
    print("=" * 64)

    if summary.get("new_cluster_version"):
        print(f"  New cluster version : {summary['new_cluster_version']}")
    print(f"  Isolates processed  : {summary.get('total_isolates', 0)}")
    print(f"  Complete genomes    : {summary.get('complete_genomes', 0)}")
    if summary.get("by_subtype"):
        print(f"  By subtype          : " +
              ", ".join(f"{st} ({n})" for st, n in sorted(summary["by_subtype"].items())))
    print(f"  Unique genotypes    : {summary.get('unique_genotypes', 0)}")
    if mode in ("batch", "recluster"):
        print(f"  Clusters created    : {summary.get('total_clusters', 0)}")
    print(f"  Orphan segments     : {summary.get('total_orphans', 0)}")
    if summary.get("reassortment_events") is not None:
        print(f"  Reassortment events : {summary['reassortment_events']} "
              f"(Stage 1: {summary.get('reassortment_stage1', 0)}, "
              f"Stage 2: {summary.get('reassortment_stage2', 0)})")
    print(f"  Alleles registered  : {nom.get('total_alleles', 0)}")
    print(f"  Constellations      : {nom.get('total_constellations', 0)}")
    if timing:
        print()
        print("  Timing (seconds)")
        for stage, secs in timing.items():
            if stage != "total":
                print(f"    {stage:<34} {secs:.2f}s")
        print(f"    {'total':<34} {timing.get('total', 0):.2f}s")
    print("=" * 64 + "\n")


def print_repair_summary(result, dry_run=False):
    tag = "DRY RUN — " if dry_run else ""
    print("\n" + "=" * 64)
    print(f"  INFLUENZA GENOTYPER — {tag}ALLELE REPAIR")
    print("=" * 64)
    print(f"  Misnamed allele     : {result['old_allele']}")
    print(f"  Correct allele      : {result['new_allele']}")
    if not dry_run:
        print(f"  Similarity          : {result['similarity']:.3f}")
        print(f"  Lineage link        : {'recorded' if result['lineage_link_recorded'] else 'not recorded (same allele)'}")
    print(f"  Sequences affected  : {len(result['affected_sequences'])}")
    for sid in result["affected_sequences"]:
        print(f"    {sid}")
    if not dry_run:
        print(f"  Stale events removed: {result.get('stale_events_removed', 0)}")
    print("=" * 64 + "\n")


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def cmd_run(args, log) -> int:
    if args.mode == "incremental" and args.cluster_version is None:
        log.info("Incremental mode: will auto-detect the most recently created active cluster version.")
    if args.mode == "recluster" and args.cluster_version is not None:
        log.warning("--cluster-version is ignored in 'recluster' mode (version set from current date).")
    if args.mode == "incremental" and args.dev_mode:
        log.warning("--dev-mode has no effect in 'incremental' mode.")

    fasta_path = Path(args.fasta)
    if not fasta_path.exists():
        log.error(f"FASTA file not found: {fasta_path}")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from influenza_genotyper import GenotypingPipeline, GenotyperConfig
        from influenza_genotyper.settings import (
            KmerConfig, ClusteringConfig, ReassortmentConfig, DatabaseConfig,
        )
    except ImportError as exc:
        log.error(f"Could not import influenza_genotyper: {exc}\n"
                  "Ensure the package is installed or PYTHONPATH is set correctly.")
        return 1

    # num_hashes is intentionally NOT passed as a hardcoded CLI default. When
    # the user omits --num-hashes, KmerConfig's own default (the single source
    # of truth in settings.py) governs, so the CLI and library can never drift.
    kmer_config = KmerConfig(hash_seed=args.hash_seed)
    if args.num_hashes is not None:
        kmer_config.num_hashes = args.num_hashes

    config = GenotyperConfig(
        kmer=kmer_config,
        clustering=ClusteringConfig(
            min_cluster_size=args.min_cluster_size,
            linkage_method=args.linkage,
            dev_mode=args.dev_mode,
        ),
        reassortment=ReassortmentConfig(
            significance_level=args.alpha,
            bonferroni=not args.no_bonferroni,
            distance_zscore_threshold=args.zscore_threshold,
            min_confidence=args.min_confidence,
        ),
        database=DatabaseConfig(sqlite_path=Path(args.db)),
    )

    pipeline = GenotypingPipeline(config=config)
    pipeline.initialize()

    log.info(f"Mode: {args.mode} | Input: {fasta_path}")

    if args.mode == "batch":
        results = pipeline.run(
            fasta_path=str(fasta_path),
            subtype=args.subtype,
            cluster_version=args.cluster_version or "v1",
            detect_reassortment=not args.no_reassortment,
        )
    elif args.mode == "incremental":
        results = pipeline.run_incremental(
            fasta_path=str(fasta_path),
            subtype=args.subtype,
            cluster_version=args.cluster_version,
            detect_reassortment=not args.no_reassortment,
        )
    elif args.mode == "recluster":
        results = pipeline.run_recluster(
            fasta_path=str(fasta_path),
            subtype=args.subtype,
            detect_reassortment=not args.no_reassortment,
        )

    validation_results = []
    if args.validate_reassortment and not args.no_reassortment:
        report = results.get("reassortment")
        if report and report.events:
            log.info(f"Running permutation validation ({args.permutations} permutations)...")
            validation_results = pipeline.validate_reassortments(
                report=report,
                genotypes=results["genotypes"],
                n_permutations=args.permutations,
            )
        else:
            log.info("No reassortment events to validate.")

    if not args.no_tsv:
        stem = fasta_path.stem
        mode = args.mode

        geno_path = out_dir / f"{stem}_{mode}_genotypes.tsv"
        write_genotypes_tsv(results.get("genotypes", []), results.get("nomenclature", {}), geno_path)
        log.info(f"Genotypes: {geno_path}")

        if mode in ("batch", "recluster") and results.get("clustering"):
            cl_path = out_dir / f"{stem}_{mode}_clusters.tsv"
            write_clusters_tsv(results["clustering"], cl_path)
            log.info(f"Clusters: {cl_path}")

            orp_path = out_dir / f"{stem}_{mode}_orphans.tsv"
            write_orphans_tsv(results["clustering"], orp_path)
            log.info(f"Orphans: {orp_path}")

        if mode == "incremental" and results.get("orphan_counts"):
            orp_path = out_dir / f"{stem}_incremental_orphans.tsv"
            write_incremental_orphans_tsv(results["orphan_counts"], orp_path)
            log.info(f"Orphan summary: {orp_path}")

        if not args.no_reassortment and results.get("reassortment"):
            reas_path = out_dir / f"{stem}_{mode}_reassortment.tsv"
            write_reassortment_tsv(results["reassortment"], reas_path)
            log.info(f"Reassortment: {reas_path}")

        if validation_results:
            val_path = out_dir / f"{stem}_{mode}_validation.tsv"
            write_validation_tsv(validation_results, val_path)
            log.info(f"Validation: {val_path}")

        summ_path = out_dir / f"{stem}_{mode}_summary.json"
        with open(summ_path, "w") as fh:
            json.dump(results.get("summary", {}), fh, indent=2, default=str)
        log.info(f"Summary: {summ_path}")

    print_summary(results.get("summary", {}), args.mode)
    return 0


def cmd_repair(args, log) -> int:
    db_path = Path(args.db)
    if not db_path.exists():
        log.error(f"Database not found: {db_path}")
        return 1

    try:
        from influenza_genotyper import GenotypingPipeline, GenotyperConfig
        from influenza_genotyper.settings import DatabaseConfig
    except ImportError as exc:
        log.error(f"Could not import influenza_genotyper: {exc}\n"
                  "Ensure the package is installed or PYTHONPATH is set correctly.")
        return 1

    config = GenotyperConfig(
        database=DatabaseConfig(sqlite_path=db_path),
    )

    pipeline = GenotypingPipeline(config=config)
    pipeline.initialize()

    if args.dry_run:
        # Inspect the DB without making any changes: report which sequences
        # currently carry the misnamed allele for the given subtype/version.
        rows = pipeline.db.get_segment_signatures_for_subtype(
            args.segment, args.correct_subtype, args.cluster_version
        )
        affected = [r for r in rows if r["allele_id"] == args.misnamed_allele]
        dry_result = {
            "old_allele": args.misnamed_allele,
            "new_allele": "<would be assigned on repair>",
            "similarity": 0.0,
            "lineage_link_recorded": False,
            "affected_sequences": [r["sequence_id"] for r in affected],
            "stale_events_removed": 0,
        }
        log.info(
            f"Dry run: {len(affected)} {args.correct_subtype} sequence(s) carry "
            f"{args.misnamed_allele} — no changes made."
        )
        print_repair_summary(dry_result, dry_run=True)
        return 0

    result = pipeline.repair_allele_subtype(
        segment_name=args.segment,
        correct_subtype=args.correct_subtype,
        misnamed_allele=args.misnamed_allele,
        cluster_version=args.cluster_version,
    )

    print_repair_summary(result)
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger("run-genotyper")

    if args.subcommand == "run":
        return cmd_run(args, log)
    elif args.subcommand == "repair":
        return cmd_repair(args, log)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
