# Influenza Genotyper (`isst`)

K-mer based genotyping and reassortment detection for influenza whole-genome
sequences. Sequences are reduced to per-segment MinHash signatures, clustered
hierarchically, assigned stable allele and constellation names, and screened
for reassortment using a staged statistical pipeline.

## Features

- **MinHash signatures** per segment with auto-selected hash backend
  (`mmh3` > `xxhash` > `hashlib` fallback).
- **Hierarchical clustering** (scipy) with per-segment and per-subtype
  similarity thresholds.
- **Stable nomenclature** — an allele name (e.g. `HA.3.0042`) always refers to
  the same biological allele across re-clustering runs, via centroid-similarity
  matching rather than ID strings.
- **Three-stage reassortment detection** — deterministic cross-subtype allele
  discordance, linkage-disequilibrium testing, and within-cluster distance
  refinement, with optional permutation validation.
- **Persistent state** in SQLite, enabling incremental surveillance ingestion
  against a fixed reference clustering.

## Installation

Requires Python 3.9+.

```bash
git clone https://github.com/Codes1985/isst.git
cd isst
pip install -e ".[fast-hash]"     # fast-hash pulls in mmh3 + xxhash
```

The `fast-hash` extra is optional — without it the tool falls back to a
SHA-256 backend (correct, just slower). For development:

```bash
pip install -e ".[dev,fast-hash]"
```

## Quickstart

The pipeline has two subcommands: `run` (batch / incremental / recluster) and
`repair`.

```bash
# 1. Initial reference run on a baseline dataset
run-genotyper run sequences_baseline.fasta --mode batch --cluster-version v1

# 2. Routine weekly ingestion of new sequences against the existing clustering
run-genotyper run sequences_week42.fasta --mode incremental

# 3. Periodic re-clustering when the orphan rate climbs
run-genotyper run sequences_all.fasta --mode recluster
```

Segment identity is inferred from FASTA headers. Outputs are written to
`--out-dir` as TSV files plus a JSON summary (use `--no-tsv` to print the
summary only).

### Operating modes

| Mode          | What it does                                                                 |
| ------------- | --------------------------------------------------------------------------- |
| `batch`       | Full clustering from scratch. Use for the initial reference run.            |
| `incremental` | Assigns new sequences to existing clusters without altering definitions.   |
| `recluster`   | Full re-clustering; retires old clusters and assigns a new dated version.  |

### Repairing a misnamed allele

When a reassortant isolate is processed before any isolate from the true donor
population, a foreign segment can be minted under the recipient subtype. The
`repair` subcommand renames the donor-population sequences to a correctly-typed
allele, records a permanent lineage link, and removes the resulting
false-positive reassortment events. Always dry-run first:

```bash
run-genotyper repair \
    --segment PB1 --correct-subtype H3N2 \
    --misnamed-allele PB1.1.0002 --cluster-version v1 --dry-run

# then commit by dropping --dry-run
```

## Reproducibility note

MinHash signatures are only comparable when built with identical parameters.
`--num-hashes`, `--hash-seed`, and the per-segment `k` values **must match
across every run that shares a database** — mismatched signatures are not
comparable and will raise at comparison time. Use one database per parameter
set, and keep the same flags across batch and incremental runs.

For publication-grade reproducibility, pin exact dependency versions (e.g. with
a `requirements.txt` lockfile or `pip freeze`) rather than relying on the lower
bounds declared in `pyproject.toml`.

## Architecture

```
run_genotyper.py            CLI entry point (argparse, all file I/O)
influenza_genotyper/
├── settings.py             Configuration dataclasses (single source of truth)
├── config.py               Re-export shim over settings
├── pipeline.py             GenotypingPipeline — end-to-end orchestration
└── core/
    ├── sequence_processor.py   FASTA parsing, validation, segment ID
    ├── kmer_extractor.py       K-mer extraction + MinHash signatures
    ├── clustering_engine.py    Hierarchical clustering
    ├── genotype_assigner.py    Composite genotype profiles
    ├── nomenclature.py         Stable allele & constellation naming
    ├── reassortment_detector.py  Three-stage reassortment detection
    └── database_manager.py     SQLite schema + CRUD
```

Dependency direction runs strictly downward: the CLI depends on the pipeline,
the pipeline on the core engines, and the engines on config.

## Development

```bash
pip install -e ".[dev,fast-hash]"
pytest                 # run the test suite
ruff check .           # lint
```

## License

See [LICENSE](LICENSE). <!-- TODO: confirm license choice -->
