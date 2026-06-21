# Influenza Genotyper (`isst`)

K-mer based genotyping and reassortment detection for influenza whole-genome
sequences. Sequences are reduced to per-segment MinHash signatures, compared by
**containment-ANI**, clustered hierarchically, assigned stable allele and
constellation names, and screened for reassortment using a staged statistical
pipeline. Sequences that cannot yet be placed confidently are tracked through an
**orphan lifecycle ledger** until a later re-clustering resolves them.

## How it works

The pipeline takes a FASTA of influenza whole-genome sequences and moves each
one through these steps:

1. **Parse, validate & classify completeness** — segments are identified from
   FASTA headers and length-checked, producing one record per genome segment.
   Each segment is also assigned a **completeness category** from its length
   relative to the expected length for that segment:
   `complete` (at or above the expected minimum), `partial` (present but short),
   or `no_call` (below half the expected length — too little sequence to trust).
   This category drives how the segment is routed downstream.
2. **Extract k-mers** — each segment is broken into overlapping k-mers, with
   `k` chosen per segment (21 for the long segments, down to 17 for the
   shortest). K-mers containing an ambiguous base (`N`) are dropped, and each is
   canonicalized — folded to the smaller of itself and its reverse complement —
   so the result is strand-independent. The unique set of canonical k-mers is
   what carries forward.
3. **Signature** — that k-mer set is reduced to a fixed-size MinHash signature
   (1024), hashed with mmh3 (pinned x64 MurmurHash3). Downstream comparison is
   done in terms of **average nucleotide identity (ANI)**, estimated from MinHash
   *containment* rather than raw Jaccard. Containment asks "what fraction of the
   smaller sequence's k-mers appear in the larger?", so a partial or truncated
   segment that is otherwise a clean subset of a full-length one reads as nearly
   identical — where symmetric Jaccard would have penalised it purely for being
   shorter. ANI is the control surface throughout: every threshold below is
   reasoned and stored as an ANI value (with a per-subtype adjustment) and only
   converted to the underlying distance at the point of use.
4. **Cluster** — signatures are clustered hierarchically (scipy), cutting at
   per-segment, per-subtype thresholds expressed in ANI (set to admit roughly
   1% within-cluster divergence). Routing depends on
   completeness: clusters **form from complete segments only**; **partial**
   segments do not seed or alter cluster definitions but may **join** an existing
   cluster when their containment-ANI clears the threshold; **no-call** segments
   are excluded from clustering entirely. This keeps cluster geometry anchored on
   trustworthy full-length sequence while still letting good partial data be
   placed.
5. **Allele and constellation naming** — each segment gets a stable allele name
   (e.g. `HA.3.0042`) and the genome gets a whole-genome constellation. Names
   persist across re-clustering via centroid matching, which uses the same
   containment-ANI metric as acceptance — so the decision to *accept* a sequence
   into a lineage and the decision to *name* it are driven by a single shared
   measure and cannot drift apart. A partial segment that is a clean subset of an
   established lineage is therefore named into that lineage rather than being left
   unnamed for want of length.
6. **Dereplicate, then detect reassortment** — near-identical genomes are first
   collapsed to a single representative each (see [Dereplication](#dereplication)),
   so an outbreak of duplicate samples cannot inflate the statistics or clutter
   the output; each flagged event is later expanded back to the genomes it stands
   for. A three-stage screen then flags cross-subtype (e.g. H3N2 and H1N1) allele
   discordance, then linkage-disequilibrium departures, then within-cluster
   distance outliers, with optional permutation validation. The distance stage
   uses containment-ANI distance, so a truncated-but-identical segment no longer
   registers as a spurious outlier. Stage 0 is deterministic and runs even for a
   single genome; its finds are reported as a count, kept separate from the
   population-statistical rate of the later stages.
7. **Persist** — sequences, signatures, and genotypes are stored in SQLite,
   supporting incremental ingestion of new sequences against a fixed reference
   clustering. The signature parameters are stamped on first write and validated
   every run, so a mismatch fails loudly rather than silently corrupting results
   (see [Reproducibility note](#reproducibility-note)). Sequences that cannot be
   confidently placed are recorded in the orphan lifecycle ledger (see
   [Orphan lifecycle](#orphan-lifecycle) below).

## Completeness and routing

The completeness category assigned in step 1 is the single knob that decides a
segment's path:

| Category   | Definition                                  | Clustering role                          |
| ---------- | ------------------------------------------- | ---------------------------------------- |
| `complete` | length >= expected minimum for the segment  | seeds and shapes clusters                |
| `partial`  | between the no-call floor and the minimum   | may join an existing cluster, never forms one |
| `no_call`  | below half the expected length              | excluded from clustering and naming      |

Because containment-ANI is subset-tolerant, a partial segment is judged on the
identity of the sequence it *does* have, not penalised for the sequence it is
missing. The completeness gate and the containment metric work together: the gate
decides whether a segment is allowed to participate, and containment decides where
it belongs once it is.

## Dereplication

Surveillance datasets often contain many near-identical genomes — copies of a
single virus sampled repeatedly through an outbreak. Counted naively, those
duplicates bias the reassortment statistics (an outbreak of *N* copies votes *N*
times when it is really one independent observation) and clutter the report with
*N* duplicate calls. Before any reassortment counting, the detector **collapses**
each group of near-identical genomes to a single representative.

Two genomes are treated as the same copy only when they agree on **every**
comparable segment at once — an all-segments conjunction — so a reassortant,
which differs on its donor segment, is never folded into a lineage it only
partly resembles. The threshold is a standalone per-segment ANI table, set
independently of clustering: clustering granularity is a surveillance *policy*
(how fine a strain to track, ~1% divergence), whereas the dereplication
threshold is a *measurement* fact (how much divergence still means "one virus,
sampled twice", ~5 SNVs per segment). The only constraint between them is that
dereplication must be at least as tight as clustering, which the pipeline
validates and warns about at startup.

Each reassortment event then carries the count of genomes its representative
stands for, so one outbreak reports as one event with a multiplicity rather than
many duplicate rows.

## Orphan lifecycle

A sequence that clears the completeness gate but matches no existing cluster
closely enough becomes an **orphan**. Rather than being silently dropped, each
orphan is opened as an episode in a dedicated ledger, capturing the entry context
(completeness category, the nearest cluster and its distance, and a timestamp).
An episode stays open — "still waiting" — until a later **re-clustering** places
the sequence, at which point it is closed with one of three exit reasons:

- `absorbed` — the sequence joined an allele that already existed.
- `minted_new` — a complete orphan founded a genuinely new allele.
- `resolved_by_completion` — a partial orphan was resolved into a new allele once
  enough surrounding data accumulated.

This turns "orphan rate" from an opaque number into an auditable history: you can
see which sequences are waiting, how long they have waited, what they were near,
and how past orphans were ultimately resolved. The `OrphanReporter` class (and the
`GenotypingPipeline.orphan_report(...)` convenience method) assembles this into a
set of read-only panels — current snapshot, candidate novel lineages, near-misses,
partial sequences waiting by segment, and resolution outcomes over time — with a
plain-text renderer surfaced by the `orphan-report` subcommand (see [Inspecting orphans](#inspecting-orphans)).

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

The pipeline has three subcommands: `run` (batch / incremental / recluster),
`repair`, and `orphan-report`.

```bash
# 1. Initial reference run on a baseline dataset
run-genotyper run sequences_baseline.fasta --mode batch --cluster-version v1

# 2. Routine weekly ingestion of new sequences against the existing clustering
run-genotyper run sequences_week42.fasta --mode incremental

# 3. Periodic re-clustering when the orphan rate climbs
run-genotyper run sequences_all.fasta --mode recluster

# 4. Inspect the orphan ledger at any time (read-only; makes no changes)
run-genotyper orphan-report --cluster-version v1
```

Segment identity is inferred from FASTA headers. Outputs are written to
`--out-dir` as TSV files plus a JSON summary (use `--no-tsv` to print the
summary only). A `recluster` run additionally reports how many previously-open
orphan episodes it resolved, broken down by exit reason.

### Operating modes

| Mode          | What it does                                                                 |
| ------------- | --------------------------------------------------------------------------- |
| `batch`       | Full clustering from scratch. Use for the initial reference run.            |
| `incremental` | Assigns new sequences to existing clusters without altering definitions.   |
| `recluster`   | Full re-clustering; retires old clusters, assigns a new dated version, and resolves outstanding orphan episodes. |

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

### Inspecting orphans

The `orphan-report` subcommand prints a read-only summary of the orphan ledger —
how many sequences are waiting, which are closest to joining a cluster
(threshold-review candidates), candidate novel lineages (one isolate orphaning on
several segments at once), partial segments awaiting a full-length example, and
how past orphans were resolved with their wait times. It changes nothing and does
not trigger a re-clustering; it is a monitoring view you read to decide whether a
recluster or a threshold review is warranted.

```bash
# Snapshot windowed to a cluster version (history panels always span versions)
run-genotyper orphan-report --cluster-version v1

# Full report as JSON, for piping into other tools
run-genotyper orphan-report --json
```

A typical run looks like:

```
Orphan report (v3)
  Open: 7 (5 complete, 2 partial)
  Multi-segment complete orphans (candidate novel lineages): 1
    A/Manitoba/12/2026: HA, NA, PB1
  Nearest near-misses:
    A/Saskatchewan/7/2026/HA → HA.3.0007 (dist 0.0120)
    A/Alberta/3/2026/NA → NA.2.0011 (dist 0.0180)
  Resolved to date: 5 (minted_new 2, absorbed 2, by_completion 1)
  Time-to-resolution (days): median 29.0, max 35.0 (n=5)
  Oldest waiter: A/Nunavut/2/2026/PB1 (63.0 days)
```

It is most useful after incremental ingestion (to see what is accumulating) and
after a recluster (to see what resolved and how long it took).

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
    ├── sequence_processor.py   FASTA parsing, validation, completeness, segment ID
    ├── kmer_extractor.py       K-mer extraction + MinHash signatures + containment-ANI
    ├── clustering_engine.py    Hierarchical clustering (ANI-thresholded)
    ├── genotype_assigner.py    Composite genotype profiles
    ├── nomenclature.py         Stable allele & constellation naming
    ├── reassortment_detector.py  Dereplication + three-stage reassortment detection
    ├── orphan_report.py        Read-only orphan lifecycle reporting
    └── database_manager.py     SQLite schema + CRUD (incl. orphan ledger)
```

The `run-genotyper` console command maps to `influenza_genotyper.cli:main`.
Dependency direction runs strictly downward: the CLI depends on the pipeline,
the pipeline on the core engines, and the engines on config. ANI thresholds and
the containment metric are defined in `settings.py`/`kmer_extractor.py` and
shared by clustering, naming, and reassortment detection, so the three never
diverge on how similarity is measured.

## Development

```bash
pip install -e ".[dev]"
pytest                 # run the test suite
ruff check .           # lint
```

## License

Released under the MIT License — see [LICENSE](LICENSE).
Copyright (c) 2026 Public Health Agency of Canada.
