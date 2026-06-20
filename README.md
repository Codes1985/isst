# Influenza Genotyper (`isst`)

K-mer based genotyping and reassortment detection for influenza whole-genome
sequences. Sequences are reduced to per-segment MinHash signatures, clustered
hierarchically, assigned stable allele and constellation names, and screened
for reassortment using a staged statistical pipeline.

## How it works

The pipeline takes a FASTA of influenza whole-genome sequences and moves each
one through these steps:

1. **Parse & validate** — segments are identified from FASTA headers and
   length-checked, producing one sequence per genome segment.
2. **Extract k-mers** — each segment is broken into overlapping k-mers, with
   `k` chosen per segment (21 for the long segments, down to 17 for the
   shortest). K-mers containing an ambiguous base (`N`) are dropped, and each is
   canonicalized — folded to the smaller of itself and its reverse complement —
   so the result is strand-independent. The unique set of canonical k-mers is
   what carries forward.
3. **Signature** — that k-mer set is reduced to a fixed-size MinHash signature (1024),
   hashed with mmh3 (pinned x64 MurmurHash3), which lets later steps estimate
   Jaccard similarity.
4. **Cluster** — signatures are clustered hierarchically (scipy), using
   per-segment and per-subtype similarity thresholds.
5. **Allele and Constellation Naming** — each segment gets a stable allele name (e.g. `HA.3.0042`) and the
   genome gets a whole-genome constellation; names persist across re-clustering
   via centroid matching.
6. **Detect reassortment** — a three-stage screen flags cross-subtype (e.g., H3N2 and H1N1) allele
   discordance, then linkage-disequilibrium departures, then within-cluster
   distance outliers, with optional permutation validation.
7. **Persist** — sequences, signatures, and genotypes are stored in SQLite,
   supporting incremental ingestion of new sequences against a fixed reference
   clustering. The signature parameters are stamped on first write and validated
   every run, so a mismatch fails loudly rather than silently corrupting results
   (see [Reproducibility note](#reproducibility-note)).

## Installation

Requires Python 3.9+.

```bash
git clone https://github.com/Codes1985/isst.git
cd isst
```

Then install with either pip or conda.

**pip:**

```bash
pip install -e .
```

**conda** (recommended on systems where building the compiled dependencies is
awkward — conda supplies numpy, scipy, and mmh3 as prebuilt binaries):

```bash
conda env create -f environment.yml
conda activate isst
```

`mmh3` is a required dependency either way — it is the single pinned hash
backend (there is intentionally no fallback, since a different hash function
would silently produce incomparable signatures).

For development, install the `dev` extra (pytest, pytest-cov, ruff):

```bash
pip install -e ".[dev]"
```

The conda environment installs the `dev` extra by default, so no extra step is
needed there.

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

MinHash signatures are only comparable when built with identical parameters:
`num_hashes`, `hash_seed`, per-segment `k`, the `canonical` flag, and the hash
backend. The first run against a database **stamps these into a signature
fingerprint**; every later run validates against it and refuses to proceed on
any mismatch, with a message naming the offending parameter. This turns a
silent incompatibility into a clear error before any data is written.

The hash backend is pinned to `mmh3` (MurmurHash3, x64 128-bit variant) with no
fallback, and a startup self-test checks it against a known vector, so a backend
or library-version change can't quietly alter signatures either.

To deliberately change a parameter you must re-stamp the database (an explicit
override that abandons comparability with existing signatures) — which on a
populated database means re-extracting it. For publication-grade reproducibility,
also pin exact dependency versions (e.g. a `requirements.txt` lockfile or
`pip freeze`) rather than relying on the lower bounds in `pyproject.toml`.

## Architecture

```
influenza_genotyper/
├── cli.py                 CLI entry point (argparse, all file I/O)
├── settings.py            Configuration dataclasses (single source of truth)
├── config.py              Re-export shim over settings
├── pipeline.py            GenotypingPipeline — end-to-end orchestration
└── core/
    ├── sequence_processor.py   FASTA parsing, validation, segment ID
    ├── kmer_extractor.py       K-mer extraction + MinHash signatures
    ├── clustering_engine.py    Hierarchical clustering
    ├── genotype_assigner.py    Composite genotype profiles
    ├── nomenclature.py         Stable allele & constellation naming
    ├── reassortment_detector.py  Three-stage reassortment detection
    └── database_manager.py     SQLite schema + CRUD
```

The `run-genotyper` console command maps to `influenza_genotyper.cli:main`.
Dependency direction runs strictly downward: the CLI depends on the pipeline,
the pipeline on the core engines, and the engines on config.

## Development

```bash
pip install -e ".[dev]"
pytest                 # run the test suite
ruff check .           # lint
```

## License

See [LICENSE](LICENSE). <!-- TODO: confirm license choice -->
