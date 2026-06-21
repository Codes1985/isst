"""
Database Manager — schema creation and CRUD for the genotyping system.
Supports SQLite (dev) and PostgreSQL (prod).
"""

import json
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from contextlib import contextmanager

from ..config import DatabaseConfig, SEGMENTS

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 5
# v4: signature fingerprint stored in schema_info and enforced on every run.
#     No table changes — the fingerprint lives in the existing key/value
#     schema_info table — so there is nothing to migrate for an existing v3 DB
#     beyond the version stamp. The fingerprint itself is stamped lazily by the
#     pipeline on first write via ensure_signature_fingerprint().
# v5: orphan_events lifecycle ledger — an append-style log (one row per orphan
#     episode) recording when a segment entered the orphan state and when/how it
#     left. Powers the orphan reporting surface. Defined in CREATE_TABLES_SQL
#     for fresh DBs and recreated idempotently in _migrate for upgrades.


class SignatureFingerprintMismatch(ValueError):
    """Raised when a run's signature parameters differ from the database's.

    MinHash signatures are only comparable when built with identical
    parameters.  This database was stamped with one fingerprint on first write;
    any later run must match it, or every comparison against existing
    signatures is silently meaningless.  Re-extract under the original
    parameters, point at a fresh database, or pass an explicit override to
    deliberately re-stamp (which abandons comparability with existing data).
    """

    def __init__(self, stored: Dict[str, Any], incoming: Dict[str, Any], diffs: str):
        self.stored = stored
        self.incoming = incoming
        self.diffs = diffs
        super().__init__(
            "Signature parameter mismatch with the existing database.\n"
            f"  Differences (database → this run): {diffs}\n"
            "  Signatures built now would not be comparable to those already "
            "stored. Use the original parameters, a fresh --db path, or an "
            "explicit override to re-stamp."
        )

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS schema_info (
    key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sequences (
    sequence_id     TEXT PRIMARY KEY,
    collection_date TEXT,
    subtype         TEXT NOT NULL,
    metadata_json   TEXT DEFAULT '{}',
    status          TEXT DEFAULT 'pending' CHECK (status IN ('pending','processed','failed','partial')),
    segments_found  INTEGER DEFAULT 0,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS segment_kmers (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    sequence_id     TEXT NOT NULL REFERENCES sequences(sequence_id) ON DELETE CASCADE,
    segment_name    TEXT NOT NULL,
    k_value         INTEGER NOT NULL,
    kmer_signature  BLOB,
    sequence_length INTEGER,
    cluster_id      TEXT,
    allele_id       TEXT,
    cluster_version TEXT,
    distance_to_centroid REAL,
    is_orphan       INTEGER DEFAULT 0,
    created_at      TEXT NOT NULL,
    UNIQUE(sequence_id, segment_name, cluster_version)
);

CREATE TABLE IF NOT EXISTS genotypes (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    sequence_id         TEXT NOT NULL REFERENCES sequences(sequence_id) ON DELETE CASCADE,
    genotype_profile    TEXT NOT NULL,
    allele_profile      TEXT,
    constellation_id    TEXT,
    cluster_version     TEXT NOT NULL,
    reassortment_score  REAL DEFAULT 0.0,
    reassortment_flag   INTEGER DEFAULT 0,
    completeness        REAL DEFAULT 1.0,
    created_at          TEXT NOT NULL,
    UNIQUE(sequence_id, cluster_version)
);

CREATE TABLE IF NOT EXISTS clusters (
    cluster_id      TEXT NOT NULL,
    segment_name    TEXT NOT NULL,
    subtype         TEXT NOT NULL,
    centroid_signature BLOB,
    member_count    INTEGER DEFAULT 0,
    mean_diameter   REAL,
    version         TEXT NOT NULL,
    is_active       INTEGER DEFAULT 1,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    PRIMARY KEY (cluster_id, segment_name, version)
);

CREATE TABLE IF NOT EXISTS allele_registry (
    allele_name         TEXT PRIMARY KEY,
    segment_name        TEXT NOT NULL,
    subtype_num         INTEGER NOT NULL,
    allele_num          INTEGER NOT NULL,
    internal_cluster_id TEXT NOT NULL,
    cluster_version     TEXT,
    centroid_signature  BLOB,
    member_count        INTEGER DEFAULT 0,
    first_seen          TEXT NOT NULL,
    last_seen           TEXT NOT NULL,
    is_active           INTEGER DEFAULT 1,
    UNIQUE(segment_name, subtype_num, internal_cluster_id)
);

CREATE TABLE IF NOT EXISTS allele_lineage (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    allele_a        TEXT NOT NULL,
    allele_b        TEXT NOT NULL,
    similarity      REAL NOT NULL,
    evidence        TEXT NOT NULL DEFAULT 'cross_subtype_centroid_match',
    created_at      TEXT NOT NULL,
    UNIQUE(allele_a, allele_b),
    CHECK(allele_a < allele_b)
);

CREATE TABLE IF NOT EXISTS constellation_registry (
    constellation_id    TEXT PRIMARY KEY,
    subtype_short       TEXT NOT NULL,
    allele_combination  TEXT NOT NULL UNIQUE,
    member_count        INTEGER DEFAULT 0,
    first_seen          TEXT NOT NULL,
    last_seen           TEXT NOT NULL,
    is_active           INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS allele_counters (
    segment_name    TEXT NOT NULL,
    subtype_num     INTEGER NOT NULL,
    next_num        INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (segment_name, subtype_num)
);

CREATE TABLE IF NOT EXISTS orphan_sequences (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    sequence_id     TEXT NOT NULL REFERENCES sequences(sequence_id) ON DELETE CASCADE,
    segment_name    TEXT NOT NULL,
    nearest_cluster TEXT,
    nearest_distance REAL,
    flagged_at      TEXT NOT NULL,
    resolved        INTEGER DEFAULT 0,
    resolved_at     TEXT,
    UNIQUE(sequence_id, segment_name)
);

CREATE TABLE IF NOT EXISTS reassortment_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    sequence_id     TEXT NOT NULL REFERENCES sequences(sequence_id) ON DELETE CASCADE,
    discordant_segments TEXT NOT NULL,
    confidence      REAL NOT NULL,
    description     TEXT,
    detected_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS clustering_runs (
    run_id          TEXT PRIMARY KEY,
    version         TEXT NOT NULL,
    parameters_json TEXT NOT NULL,
    subtype         TEXT NOT NULL,
    segment_name    TEXT,
    sequences_processed INTEGER DEFAULT 0,
    clusters_created    INTEGER DEFAULT 0,
    orphans_flagged     INTEGER DEFAULT 0,
    started_at      TEXT NOT NULL,
    completed_at    TEXT
);

CREATE INDEX IF NOT EXISTS idx_sequences_subtype ON sequences(subtype);
CREATE INDEX IF NOT EXISTS idx_sequences_date ON sequences(collection_date);
CREATE INDEX IF NOT EXISTS idx_segment_kmers_seq ON segment_kmers(sequence_id);
CREATE INDEX IF NOT EXISTS idx_segment_kmers_cluster ON segment_kmers(cluster_id);
CREATE INDEX IF NOT EXISTS idx_segment_kmers_allele ON segment_kmers(allele_id);
CREATE INDEX IF NOT EXISTS idx_genotypes_seq ON genotypes(sequence_id);
CREATE INDEX IF NOT EXISTS idx_genotypes_profile ON genotypes(genotype_profile);
CREATE INDEX IF NOT EXISTS idx_genotypes_constellation ON genotypes(constellation_id);
CREATE INDEX IF NOT EXISTS idx_clusters_segment ON clusters(segment_name, subtype);
CREATE INDEX IF NOT EXISTS idx_orphan_unresolved ON orphan_sequences(resolved);
CREATE INDEX IF NOT EXISTS idx_allele_segment ON allele_registry(segment_name, subtype_num);
CREATE INDEX IF NOT EXISTS idx_constellation_subtype ON constellation_registry(subtype_short);
CREATE INDEX IF NOT EXISTS idx_allele_lineage_a ON allele_lineage(allele_a);
CREATE INDEX IF NOT EXISTS idx_allele_lineage_b ON allele_lineage(allele_b);

CREATE TABLE IF NOT EXISTS orphan_events (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    sequence_id      TEXT NOT NULL REFERENCES sequences(sequence_id) ON DELETE CASCADE,
    segment_name     TEXT NOT NULL,
    cluster_version  TEXT NOT NULL,
    -- entry context: immutable, written when the orphan episode opens
    category         TEXT NOT NULL CHECK (category IN ('complete','partial')),
    completeness     REAL,
    nearest_cluster  TEXT,
    nearest_distance REAL,
    entered_at       TEXT NOT NULL,
    -- resolution: written once when the episode closes; NULL while still waiting
    exit_reason      TEXT CHECK (exit_reason IN ('minted_new','absorbed','resolved_by_completion')),
    exit_allele      TEXT,
    exited_at        TEXT,
    -- one episode per segment per cluster version; a segment may re-enter the
    -- orphan state under a later version as a distinct row
    UNIQUE(sequence_id, segment_name, cluster_version)
);
CREATE INDEX IF NOT EXISTS idx_orphan_events_open ON orphan_events(cluster_version, exited_at);
CREATE INDEX IF NOT EXISTS idx_orphan_events_seq ON orphan_events(sequence_id);
"""


class DatabaseManager:
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()

    def initialize(self) -> None:
        if self.config.db_type == "sqlite":
            Path(self.config.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as conn:
            conn.executescript(CREATE_TABLES_SQL)
            now = datetime.utcnow().isoformat()
            self._migrate(conn, now)
            conn.execute(
                "INSERT OR REPLACE INTO schema_info (key, value, updated_at) VALUES (?, ?, ?)",
                ("schema_version", str(SCHEMA_VERSION), now),
            )
        logger.info("Database initialized (schema v%d)", SCHEMA_VERSION)

    def _migrate(self, conn: Any, now: str) -> None:
        """Apply incremental migrations for existing databases."""
        row = conn.execute(
            "SELECT value FROM schema_info WHERE key='schema_version'"
        ).fetchone()
        existing = int(row["value"]) if row else 1
        if existing < 3:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS allele_lineage (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    allele_a        TEXT NOT NULL,
                    allele_b        TEXT NOT NULL,
                    similarity      REAL NOT NULL,
                    evidence        TEXT NOT NULL DEFAULT 'cross_subtype_centroid_match',
                    created_at      TEXT NOT NULL,
                    UNIQUE(allele_a, allele_b),
                    CHECK(allele_a < allele_b)
                );
                CREATE INDEX IF NOT EXISTS idx_allele_lineage_a ON allele_lineage(allele_a);
                CREATE INDEX IF NOT EXISTS idx_allele_lineage_b ON allele_lineage(allele_b);
            """)
            logger.info("Database migrated to schema v3 (allele_lineage)")
        if existing < 5:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS orphan_events (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    sequence_id      TEXT NOT NULL REFERENCES sequences(sequence_id) ON DELETE CASCADE,
                    segment_name     TEXT NOT NULL,
                    cluster_version  TEXT NOT NULL,
                    category         TEXT NOT NULL CHECK (category IN ('complete','partial')),
                    completeness     REAL,
                    nearest_cluster  TEXT,
                    nearest_distance REAL,
                    entered_at       TEXT NOT NULL,
                    exit_reason      TEXT CHECK (exit_reason IN ('minted_new','absorbed','resolved_by_completion')),
                    exit_allele      TEXT,
                    exited_at        TEXT,
                    UNIQUE(sequence_id, segment_name, cluster_version)
                );
                CREATE INDEX IF NOT EXISTS idx_orphan_events_open ON orphan_events(cluster_version, exited_at);
                CREATE INDEX IF NOT EXISTS idx_orphan_events_seq ON orphan_events(sequence_id);
            """)
            logger.info("Database migrated to schema v5 (orphan_events ledger)")

    # ------------------------------------------------------------------
    # Orphan lifecycle ledger
    # ------------------------------------------------------------------

    def record_orphan_entry(
        self,
        sequence_id: str,
        segment_name: str,
        cluster_version: str,
        category: str,
        completeness: Optional[float] = None,
        nearest_cluster: Optional[str] = None,
        nearest_distance: Optional[float] = None,
    ) -> None:
        """Open an orphan episode for a segment under a cluster version.

        Idempotent per (sequence_id, segment_name, cluster_version): re-running
        the same version does not create a duplicate episode. ``category`` is
        'complete' or 'partial'. No-calls (segments below the length floor) are a
        data-quality metric, not orphan episodes, and are not recorded here.
        """
        if category not in ("complete", "partial"):
            raise ValueError(
                f"orphan category must be 'complete' or 'partial', got {category!r}"
            )
        with self.connection() as conn:
            self.record_orphan_entry_conn(
                conn, sequence_id, segment_name, cluster_version, category,
                completeness, nearest_cluster, nearest_distance,
            )

    def record_orphan_entry_conn(
        self,
        conn: Any,
        sequence_id: str,
        segment_name: str,
        cluster_version: str,
        category: str,
        completeness: Optional[float] = None,
        nearest_cluster: Optional[str] = None,
        nearest_distance: Optional[float] = None,
    ) -> None:
        """Connection-bound form of :meth:`record_orphan_entry`, for use inside
        an existing ``bulk_operation`` write transaction (opening a second
        connection there would deadlock on the SQLite write lock)."""
        if category not in ("complete", "partial"):
            raise ValueError(
                f"orphan category must be 'complete' or 'partial', got {category!r}"
            )
        now = datetime.utcnow().isoformat()
        conn.execute(
            """INSERT INTO orphan_events
                   (sequence_id, segment_name, cluster_version, category,
                    completeness, nearest_cluster, nearest_distance, entered_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(sequence_id, segment_name, cluster_version)
               DO NOTHING""",
            (sequence_id, segment_name, cluster_version, category,
             completeness, nearest_cluster, nearest_distance, now),
        )

    def record_orphan_exit(
        self,
        sequence_id: str,
        segment_name: str,
        cluster_version: str,
        exit_reason: str,
        exit_allele: Optional[str] = None,
    ) -> bool:
        """Close an open orphan episode, recording how it left the orphan state.

        ``exit_reason`` is one of 'minted_new', 'absorbed',
        'resolved_by_completion'. ``exit_allele`` is the allele the segment
        became or joined, where applicable. Returns True if an open episode was
        closed, False if there was none to close (already resolved, or never
        recorded).
        """
        if exit_reason not in ("minted_new", "absorbed", "resolved_by_completion"):
            raise ValueError(
                "exit_reason must be one of 'minted_new', 'absorbed', "
                f"'resolved_by_completion', got {exit_reason!r}"
            )
        with self.connection() as conn:
            return self.record_orphan_exit_conn(
                conn, sequence_id, segment_name, cluster_version,
                exit_reason, exit_allele,
            )

    def record_orphan_exit_conn(
        self,
        conn: Any,
        sequence_id: str,
        segment_name: str,
        cluster_version: str,
        exit_reason: str,
        exit_allele: Optional[str] = None,
    ) -> bool:
        """Connection-bound form of :meth:`record_orphan_exit`, for closing
        several episodes inside one ``bulk_operation`` transaction."""
        if exit_reason not in ("minted_new", "absorbed", "resolved_by_completion"):
            raise ValueError(
                "exit_reason must be one of 'minted_new', 'absorbed', "
                f"'resolved_by_completion', got {exit_reason!r}"
            )
        now = datetime.utcnow().isoformat()
        cur = conn.execute(
            """UPDATE orphan_events
                   SET exit_reason = ?, exit_allele = ?, exited_at = ?
                 WHERE sequence_id = ? AND segment_name = ?
                   AND cluster_version = ? AND exited_at IS NULL""",
            (exit_reason, exit_allele, now,
             sequence_id, segment_name, cluster_version),
        )
        return cur.rowcount > 0

    def count_open_orphans_by_category(
        self, cluster_version: str
    ) -> List[Dict[str, Any]]:
        """Per-segment counts of currently-open orphan episodes for a version.

        Returns rows of {segment_name, category, n} — the snapshot top-line.
        """
        with self.connection() as conn:
            rows = conn.execute(
                """SELECT segment_name, category, COUNT(*) AS n
                       FROM orphan_events
                      WHERE cluster_version = ? AND exited_at IS NULL
                      GROUP BY segment_name, category
                      ORDER BY segment_name, category""",
                (cluster_version,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_open_orphans(
        self,
        cluster_version: Optional[str] = None,
        category: Optional[str] = None,
        segment_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Open (still-waiting) orphan episodes, newest first.

        Optional filters narrow by version, category, and segment. Backs the
        persistent-waiters, cohort, and coherence panels (the latter joins these
        rows back to their signatures in ``segment_kmers`` at report time).
        """
        clauses = ["exited_at IS NULL"]
        params: List[Any] = []
        if cluster_version is not None:
            clauses.append("cluster_version = ?")
            params.append(cluster_version)
        if category is not None:
            clauses.append("category = ?")
            params.append(category)
        if segment_name is not None:
            clauses.append("segment_name = ?")
            params.append(segment_name)
        where = " AND ".join(clauses)
        with self.connection() as conn:
            rows = conn.execute(
                f"""SELECT * FROM orphan_events
                        WHERE {where}
                        ORDER BY entered_at DESC, id DESC""",
                params,
            ).fetchall()
        return [dict(r) for r in rows]

    def get_orphan_resolutions(
        self,
        cluster_version: Optional[str] = None,
        since: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Closed orphan episodes, for the resolution-outcome and
        time-to-resolution panels.

        Each row carries both ``entered_at`` and ``exited_at`` so wait time is
        computed at report time rather than stored. ``since`` filters on
        ``exited_at`` (ISO timestamp); ``cluster_version`` narrows to one version.
        """
        clauses = ["exited_at IS NOT NULL"]
        params: List[Any] = []
        if cluster_version is not None:
            clauses.append("cluster_version = ?")
            params.append(cluster_version)
        if since is not None:
            clauses.append("exited_at >= ?")
            params.append(since)
        where = " AND ".join(clauses)
        with self.connection() as conn:
            rows = conn.execute(
                f"""SELECT * FROM orphan_events
                        WHERE {where}
                        ORDER BY exited_at DESC, id DESC""",
                params,
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Signature fingerprint — comparability guard
    # ------------------------------------------------------------------

    def get_signature_fingerprint(self) -> Optional[Dict[str, Any]]:
        """Return the stored signature fingerprint, or None if unstamped."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT value FROM schema_info WHERE key='signature_fingerprint'"
            ).fetchone()
        return json.loads(row["value"]) if row else None

    def ensure_signature_fingerprint(
        self, fingerprint: Dict[str, Any], allow_change: bool = False
    ) -> None:
        """Stamp the database's signature fingerprint, or validate against it.

        On a fresh database the fingerprint is recorded. On every subsequent
        run the incoming fingerprint must match the stored one exactly, or a
        SignatureFingerprintMismatch is raised — turning a silent
        incompatibility into a clear, immediate error before any signatures are
        written. ``allow_change=True`` re-stamps instead of raising, for
        deliberate parameter migrations (which abandon comparability with
        existing data).
        """
        incoming = json.dumps(fingerprint, sort_keys=True)
        stored = self.get_signature_fingerprint()
        now = datetime.utcnow().isoformat()

        if stored is None:
            with self.connection() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO schema_info (key, value, updated_at) "
                    "VALUES (?, ?, ?)",
                    ("signature_fingerprint", incoming, now),
                )
            logger.info("Stamped signature fingerprint: %s", incoming)
            return

        if json.dumps(stored, sort_keys=True) == incoming:
            logger.debug("Signature fingerprint matches the database.")
            return

        diffs = self._diff_fingerprints(stored, fingerprint)
        if allow_change:
            with self.connection() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO schema_info (key, value, updated_at) "
                    "VALUES (?, ?, ?)",
                    ("signature_fingerprint", incoming, now),
                )
            logger.warning(
                "Signature fingerprint CHANGED with allow_change=True (%s). "
                "Signatures already in this database are NOT comparable to new "
                "ones.", diffs,
            )
            return

        raise SignatureFingerprintMismatch(stored, fingerprint, diffs)

    @staticmethod
    def _diff_fingerprints(stored: Dict[str, Any], incoming: Dict[str, Any]) -> str:
        """Human-readable summary of which fingerprint fields differ."""
        keys = sorted(set(stored) | set(incoming))
        parts = [
            f"{k}: {stored.get(k)!r} → {incoming.get(k)!r}"
            for k in keys
            if stored.get(k) != incoming.get(k)
        ]
        return "; ".join(parts) if parts else "(values differ)"

    @contextmanager
    def connection(self):
        conn = sqlite3.connect(str(self.config.sqlite_path), detect_types=sqlite3.PARSE_DECLTYPES)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @contextmanager
    def bulk_operation(self):
        """Context manager for batching many writes in a single transaction.

        Use this when inserting hundreds of sequences, segment k-mers, or
        cluster assignments in a tight loop.  All writes within the block
        share one connection and one commit, reducing per-call overhead
        from ~7ms to < 0.1ms per write.

        Example::

            with db.bulk_operation() as conn:
                for rec in records:
                    db._insert_sequence_conn(conn, rec.sequence_id, rec.subtype, ...)
                    for seg, sig in sigs.items():
                        db._insert_segment_kmer_conn(conn, rec.sequence_id, seg, ...)
        """
        conn = sqlite3.connect(str(self.config.sqlite_path), detect_types=sqlite3.PARSE_DECLTYPES)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("BEGIN")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── Sequence CRUD ──────────────────────────────────────────────────

    def insert_sequence(self, sequence_id: str, subtype: str,
                        collection_date: Optional[str] = None,
                        metadata: Optional[Dict] = None) -> None:
        now = datetime.utcnow().isoformat()
        with self.connection() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO sequences
                   (sequence_id, collection_date, subtype, metadata_json, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (sequence_id, collection_date, subtype, json.dumps(metadata or {}), now, now),
            )

    def insert_sequence_conn(self, conn: Any, sequence_id: str, subtype: str,
                             collection_date: Optional[str] = None,
                             metadata: Optional[Dict] = None) -> None:
        """Bulk-operation variant — uses a caller-supplied connection."""
        now = datetime.utcnow().isoformat()
        conn.execute(
            """INSERT OR IGNORE INTO sequences
               (sequence_id, collection_date, subtype, metadata_json, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (sequence_id, collection_date, subtype, json.dumps(metadata or {}), now, now),
        )

    def update_sequence_status(self, sequence_id: str, status: str,
                               segments_found: Optional[int] = None) -> None:
        now = datetime.utcnow().isoformat()
        with self.connection() as conn:
            if segments_found is not None:
                conn.execute("UPDATE sequences SET status=?, segments_found=?, updated_at=? WHERE sequence_id=?",
                             (status, segments_found, now, sequence_id))
            else:
                conn.execute("UPDATE sequences SET status=?, updated_at=? WHERE sequence_id=?",
                             (status, now, sequence_id))

    def update_sequence_status_conn(self, conn: Any, sequence_id: str, status: str,
                                    segments_found: Optional[int] = None) -> None:
        """Bulk-operation variant — uses a caller-supplied connection."""
        now = datetime.utcnow().isoformat()
        if segments_found is not None:
            conn.execute("UPDATE sequences SET status=?, segments_found=?, updated_at=? WHERE sequence_id=?",
                         (status, segments_found, now, sequence_id))
        else:
            conn.execute("UPDATE sequences SET status=?, updated_at=? WHERE sequence_id=?",
                         (status, now, sequence_id))

    def get_sequence(self, sequence_id: str) -> Optional[Dict]:
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM sequences WHERE sequence_id=?", (sequence_id,)).fetchone()
            return dict(row) if row else None

    def get_sequences_by_subtype(self, subtype: str, status: Optional[str] = None) -> List[Dict]:
        with self.connection() as conn:
            if status:
                rows = conn.execute("SELECT * FROM sequences WHERE subtype=? AND status=?", (subtype, status)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM sequences WHERE subtype=?", (subtype,)).fetchall()
            return [dict(r) for r in rows]

    # ── Segment k-mer CRUD ─────────────────────────────────────────────

    def insert_segment_kmer(self, sequence_id: str, segment_name: str, k_value: int,
                            kmer_signature: bytes, sequence_length: int,
                            cluster_version: str = "v0") -> None:
        now = datetime.utcnow().isoformat()
        with self.connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO segment_kmers
                   (sequence_id, segment_name, k_value, kmer_signature, sequence_length, cluster_version, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (sequence_id, segment_name, k_value, kmer_signature, sequence_length, cluster_version, now),
            )

    def insert_segment_kmer_conn(self, conn: Any, sequence_id: str, segment_name: str,
                                 k_value: int, kmer_signature: bytes, sequence_length: int,
                                 cluster_version: str = "v0") -> None:
        """Bulk-operation variant — uses a caller-supplied connection."""
        now = datetime.utcnow().isoformat()
        conn.execute(
            """INSERT OR REPLACE INTO segment_kmers
               (sequence_id, segment_name, k_value, kmer_signature, sequence_length, cluster_version, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (sequence_id, segment_name, k_value, kmer_signature, sequence_length, cluster_version, now),
        )

    def update_cluster_assignment(self, sequence_id: str, segment_name: str,
                                  cluster_id: str, cluster_version: str,
                                  distance_to_centroid: float, is_orphan: bool = False) -> None:
        with self.connection() as conn:
            conn.execute(
                """UPDATE segment_kmers SET cluster_id=?, distance_to_centroid=?, is_orphan=?
                   WHERE sequence_id=? AND segment_name=? AND cluster_version=?""",
                (cluster_id, distance_to_centroid, int(is_orphan),
                 sequence_id, segment_name, cluster_version),
            )

    def update_cluster_assignment_conn(self, conn: Any, sequence_id: str, segment_name: str,
                                       cluster_id: str, cluster_version: str,
                                       distance_to_centroid: float, is_orphan: bool = False) -> None:
        """Bulk-operation variant — uses a caller-supplied connection."""
        conn.execute(
            """UPDATE segment_kmers SET cluster_id=?, distance_to_centroid=?, is_orphan=?
               WHERE sequence_id=? AND segment_name=? AND cluster_version=?""",
            (cluster_id, distance_to_centroid, int(is_orphan),
             sequence_id, segment_name, cluster_version),
        )

    def update_allele_assignment(self, sequence_id: str, segment_name: str,
                                 allele_id: str, cluster_version: str) -> None:
        with self.connection() as conn:
            conn.execute(
                "UPDATE segment_kmers SET allele_id=? WHERE sequence_id=? AND segment_name=? AND cluster_version=?",
                (allele_id, sequence_id, segment_name, cluster_version),
            )

    def update_allele_assignment_conn(self, conn: Any, sequence_id: str, segment_name: str,
                                      allele_id: str, cluster_version: str) -> None:
        """Bulk-operation variant — uses a caller-supplied connection."""
        conn.execute(
            "UPDATE segment_kmers SET allele_id=? WHERE sequence_id=? AND segment_name=? AND cluster_version=?",
            (allele_id, sequence_id, segment_name, cluster_version),
        )

    # ── Cluster CRUD ───────────────────────────────────────────────────

    def insert_cluster(self, cluster_id: str, segment_name: str, subtype: str,
                       centroid_signature: bytes, member_count: int,
                       mean_diameter: float, version: str) -> None:
        now = datetime.utcnow().isoformat()
        with self.connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO clusters
                   (cluster_id, segment_name, subtype, centroid_signature,
                    member_count, mean_diameter, version, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (cluster_id, segment_name, subtype, centroid_signature,
                 member_count, mean_diameter, version, now, now),
            )

    def insert_cluster_conn(self, conn: Any, cluster_id: str, segment_name: str,
                            subtype: str, centroid_signature: bytes, member_count: int,
                            mean_diameter: float, version: str) -> None:
        """Bulk-operation variant — uses a caller-supplied connection."""
        now = datetime.utcnow().isoformat()
        conn.execute(
            """INSERT OR REPLACE INTO clusters
               (cluster_id, segment_name, subtype, centroid_signature,
                member_count, mean_diameter, version, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (cluster_id, segment_name, subtype, centroid_signature,
             member_count, mean_diameter, version, now, now),
        )

    def get_active_clusters(self, segment_name: str, subtype: str) -> List[Dict]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM clusters WHERE segment_name=? AND subtype=? AND is_active=1 ORDER BY cluster_id",
                (segment_name, subtype)).fetchall()
            return [dict(r) for r in rows]

    def get_active_clusters_by_version(self, segment_name: str, subtype: str,
                                       version: str) -> List[Dict]:
        """Load active clusters for a specific cluster_version.

        Used by the incremental pipeline to reload the reference clustering
        produced by a previous ``run()`` or ``run_recluster()`` call.
        """
        with self.connection() as conn:
            rows = conn.execute(
                """SELECT * FROM clusters
                   WHERE segment_name=? AND subtype=? AND version=? AND is_active=1
                   ORDER BY cluster_id""",
                (segment_name, subtype, version)).fetchall()
            return [dict(r) for r in rows]

    def get_latest_cluster_version(self, segment_name: str, subtype: str) -> Optional[str]:
        """Return the most recently created active cluster version for a segment/subtype.

        Used by the incremental pipeline when no explicit reference version is
        supplied — it automatically targets the most recent batch run.
        """
        with self.connection() as conn:
            row = conn.execute(
                """SELECT version FROM clusters
                   WHERE segment_name=? AND subtype=? AND is_active=1
                   ORDER BY created_at DESC LIMIT 1""",
                (segment_name, subtype)).fetchone()
            return row["version"] if row else None

    def update_cluster_member_count(self, cluster_id: str, segment_name: str,
                                    version: str, member_count: int) -> None:
        """Increment the stored member count for a cluster after incremental assignment."""
        now = datetime.utcnow().isoformat()
        with self.connection() as conn:
            conn.execute(
                """UPDATE clusters SET member_count=?, updated_at=?
                   WHERE cluster_id=? AND segment_name=? AND version=?""",
                (member_count, now, cluster_id, segment_name, version),
            )

    def get_sequence_ids_in_db(self) -> set:
        """Return the set of all sequence_ids already present in the database.

        Used by the incremental pipeline to skip sequences that have already
        been processed, avoiding duplicate inserts.
        """
        with self.connection() as conn:
            rows = conn.execute("SELECT sequence_id FROM sequences").fetchall()
            return {r["sequence_id"] for r in rows}

    # ── Allele registry CRUD ───────────────────────────────────────────

    def get_next_allele_num(self, segment_name: str, subtype_num: int) -> int:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT next_num FROM allele_counters WHERE segment_name=? AND subtype_num=?",
                (segment_name, subtype_num)).fetchone()
            if row:
                num = row[0]
                conn.execute(
                    "UPDATE allele_counters SET next_num=? WHERE segment_name=? AND subtype_num=?",
                    (num + 1, segment_name, subtype_num))
            else:
                num = 1
                conn.execute(
                    "INSERT INTO allele_counters (segment_name, subtype_num, next_num) VALUES (?, ?, ?)",
                    (segment_name, subtype_num, 2))
            return num

    def insert_allele(self, allele_name: str, segment_name: str, subtype_num: int,
                      allele_num: int, internal_cluster_id: str,
                      cluster_version: Optional[str] = None,
                      centroid_signature: Optional[bytes] = None,
                      member_count: int = 0) -> None:
        now = datetime.utcnow().isoformat()
        with self.connection() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO allele_registry
                   (allele_name, segment_name, subtype_num, allele_num,
                    internal_cluster_id, cluster_version, centroid_signature,
                    member_count, first_seen, last_seen)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (allele_name, segment_name, subtype_num, allele_num,
                 internal_cluster_id, cluster_version, centroid_signature,
                 member_count, now, now),
            )

    def get_allele(self, segment_name: str, subtype_num: int,
                   internal_cluster_id: str) -> Optional[Dict]:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM allele_registry WHERE segment_name=? AND subtype_num=? AND internal_cluster_id=?",
                (segment_name, subtype_num, internal_cluster_id)).fetchone()
            return dict(row) if row else None

    def get_allele_by_name(self, allele_name: str) -> Optional[Dict]:
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM allele_registry WHERE allele_name=?", (allele_name,)).fetchone()
            return dict(row) if row else None

    def get_alleles_for_segment(self, segment_name: str, subtype_num: Optional[int] = None) -> List[Dict]:
        with self.connection() as conn:
            if subtype_num is not None:
                rows = conn.execute(
                    "SELECT * FROM allele_registry WHERE segment_name=? AND subtype_num=? AND is_active=1 ORDER BY allele_num",
                    (segment_name, subtype_num)).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM allele_registry WHERE segment_name=? AND is_active=1 ORDER BY allele_num",
                    (segment_name,)).fetchall()
            return [dict(r) for r in rows]

    def update_allele_last_seen(self, allele_name: str, member_count: Optional[int] = None) -> None:
        now = datetime.utcnow().isoformat()
        with self.connection() as conn:
            if member_count is not None:
                conn.execute("UPDATE allele_registry SET last_seen=?, member_count=? WHERE allele_name=?",
                             (now, member_count, allele_name))
            else:
                conn.execute("UPDATE allele_registry SET last_seen=? WHERE allele_name=?", (now, allele_name))

    # ── Allele lineage ────────────────────────────────────────────────

    def record_allele_lineage(self, allele_a: str, allele_b: str,
                              similarity: float,
                              evidence: str = "cross_subtype_centroid_match") -> None:
        """Record that two allele names refer to the same biological lineage."""
        a, b = sorted([allele_a, allele_b])
        now = datetime.utcnow().isoformat()
        with self.connection() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO allele_lineage
                   (allele_a, allele_b, similarity, evidence, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (a, b, similarity, evidence, now),
            )

    def get_allele_lineage(self, allele_name: str) -> List[Dict]:
        """Return all alleles known to share a lineage with the given allele."""
        with self.connection() as conn:
            rows = conn.execute(
                """SELECT allele_b AS linked_allele, similarity, evidence, created_at
                   FROM allele_lineage WHERE allele_a = ?
                   UNION
                   SELECT allele_a AS linked_allele, similarity, evidence, created_at
                   FROM allele_lineage WHERE allele_b = ?
                   ORDER BY created_at""",
                (allele_name, allele_name),
            ).fetchall()
        return [dict(r) for r in rows]

    def retire_empty_constellations(self, containing_allele: Optional[str] = None) -> int:
        """Mark constellations with zero genotype members as inactive.

        Parameters
        ----------
        containing_allele : str, optional
            When provided, only constellations whose allele_combination
            contains this allele name are considered.  Use after a repair
            to limit the scope to constellations that were affected by the
            misnamed allele.  When None, all zero-member constellations
            are retired (use with care).

        Returns the number of constellations retired.
        """
        now = datetime.utcnow().isoformat()
        with self.connection() as conn:
            if containing_allele:
                rows = conn.execute(
                    """SELECT cr.constellation_id
                       FROM constellation_registry cr
                       WHERE cr.is_active = 1
                         AND cr.allele_combination LIKE ?
                         AND NOT EXISTS (
                             SELECT 1 FROM genotypes g
                             WHERE g.constellation_id = cr.constellation_id
                         )""",
                    (f"%{containing_allele}%",),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT cr.constellation_id
                       FROM constellation_registry cr
                       WHERE cr.is_active = 1
                         AND NOT EXISTS (
                             SELECT 1 FROM genotypes g
                             WHERE g.constellation_id = cr.constellation_id
                         )"""
                ).fetchall()
            ids = [r[0] for r in rows]
            for cid in ids:
                conn.execute(
                    "UPDATE constellation_registry SET is_active=0, last_seen=? WHERE constellation_id=?",
                    (now, cid),
                )
                logger.info(f"Constellation retired (zero members): {cid}")
            return len(ids)

    def get_all_lineage_links(self) -> List[Dict]:
        """Return every recorded allele lineage link."""
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT allele_a, allele_b, similarity, evidence, created_at "
                "FROM allele_lineage ORDER BY created_at"
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Allele repair helpers ──────────────────────────────────────────

    def get_segment_signatures_for_subtype(self, segment_name: str, subtype: str,
                                           cluster_version: Optional[str] = None) -> List[Dict]:
        """Return stored k-mer signatures for all sequences of a given subtype and segment."""
        with self.connection() as conn:
            if cluster_version:
                rows = conn.execute(
                    """SELECT sk.sequence_id, sk.kmer_signature, sk.cluster_id,
                              sk.allele_id, sk.cluster_version, sk.is_orphan
                       FROM segment_kmers sk
                       JOIN sequences s ON sk.sequence_id = s.sequence_id
                       WHERE sk.segment_name = ? AND s.subtype = ?
                         AND sk.cluster_version = ? AND sk.kmer_signature IS NOT NULL""",
                    (segment_name, subtype, cluster_version),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT sk.sequence_id, sk.kmer_signature, sk.cluster_id,
                              sk.allele_id, sk.cluster_version, sk.is_orphan
                       FROM segment_kmers sk
                       JOIN sequences s ON sk.sequence_id = s.sequence_id
                       WHERE sk.segment_name = ? AND s.subtype = ?
                         AND sk.kmer_signature IS NOT NULL""",
                    (segment_name, subtype),
                ).fetchall()
        return [dict(r) for r in rows]

    def update_genotype_allele_profile(self, sequence_id: str, cluster_version: str,
                                       allele_profile: str,
                                       constellation_id: Optional[str]) -> None:
        """Update allele_profile and constellation_id for an existing genotype row."""
        with self.connection() as conn:
            conn.execute(
                """UPDATE genotypes SET allele_profile = ?, constellation_id = ?
                   WHERE sequence_id = ? AND cluster_version = ?""",
                (allele_profile, constellation_id, sequence_id, cluster_version),
            )

    def load_allele_registry(self) -> List[Dict]:
        with self.connection() as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM allele_registry WHERE is_active=1").fetchall()]

    def load_allele_counters(self) -> Dict:
        with self.connection() as conn:
            rows = conn.execute("SELECT * FROM allele_counters").fetchall()
            return {(r["segment_name"], r["subtype_num"]): r["next_num"] for r in rows}

    # ── Constellation registry CRUD ────────────────────────────────────

    def insert_constellation(self, constellation_id: str, subtype_short: str,
                             allele_combination: str, member_count: int = 0) -> None:
        now = datetime.utcnow().isoformat()
        with self.connection() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO constellation_registry
                   (constellation_id, subtype_short, allele_combination, member_count, first_seen, last_seen)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (constellation_id, subtype_short, allele_combination, member_count, now, now),
            )

    def get_constellation(self, allele_combination: str) -> Optional[Dict]:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM constellation_registry WHERE allele_combination=?",
                (allele_combination,)).fetchone()
            return dict(row) if row else None

    def get_constellation_by_id(self, constellation_id: str) -> Optional[Dict]:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM constellation_registry WHERE constellation_id=?",
                (constellation_id,)).fetchone()
            return dict(row) if row else None

    def update_constellation_last_seen(self, constellation_id: str,
                                       member_count: Optional[int] = None) -> None:
        now = datetime.utcnow().isoformat()
        with self.connection() as conn:
            if member_count is not None:
                conn.execute("UPDATE constellation_registry SET last_seen=?, member_count=? WHERE constellation_id=?",
                             (now, member_count, constellation_id))
            else:
                conn.execute("UPDATE constellation_registry SET last_seen=? WHERE constellation_id=?",
                             (now, constellation_id))

    def load_constellation_registry(self) -> List[Dict]:
        with self.connection() as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM constellation_registry WHERE is_active=1").fetchall()]

    # ── Genotype CRUD ──────────────────────────────────────────────────

    def insert_genotype(self, sequence_id: str, genotype_profile: str,
                        cluster_version: str, allele_profile: Optional[str] = None,
                        constellation_id: Optional[str] = None,
                        reassortment_score: float = 0.0,
                        reassortment_flag: bool = False,
                        completeness: float = 1.0) -> None:
        now = datetime.utcnow().isoformat()
        with self.connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO genotypes
                   (sequence_id, genotype_profile, allele_profile, constellation_id,
                    cluster_version, reassortment_score, reassortment_flag, completeness, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (sequence_id, genotype_profile, allele_profile, constellation_id,
                 cluster_version, reassortment_score, int(reassortment_flag), completeness, now),
            )

    def insert_genotype_conn(self, conn: Any, sequence_id: str, genotype_profile: str,
                             cluster_version: str, allele_profile: Optional[str] = None,
                             constellation_id: Optional[str] = None,
                             reassortment_score: float = 0.0,
                             reassortment_flag: bool = False,
                             completeness: float = 1.0) -> None:
        """Bulk-operation variant — uses a caller-supplied connection."""
        now = datetime.utcnow().isoformat()
        conn.execute(
            """INSERT OR REPLACE INTO genotypes
               (sequence_id, genotype_profile, allele_profile, constellation_id,
                cluster_version, reassortment_score, reassortment_flag, completeness, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (sequence_id, genotype_profile, allele_profile, constellation_id,
             cluster_version, reassortment_score, int(reassortment_flag), completeness, now),
        )

    def get_genotype(self, sequence_id: str) -> Optional[Dict]:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM genotypes WHERE sequence_id=? ORDER BY created_at DESC LIMIT 1",
                (sequence_id,)).fetchone()
            return dict(row) if row else None

    def get_genotypes_by_constellation(self, constellation_id: str) -> List[Dict]:
        with self.connection() as conn:
            return [dict(r) for r in conn.execute(
                "SELECT * FROM genotypes WHERE constellation_id=?", (constellation_id,)).fetchall()]

    # ── Orphan CRUD ────────────────────────────────────────────────────

    def flag_orphan(self, sequence_id: str, segment_name: str,
                    nearest_cluster: Optional[str] = None,
                    nearest_distance: Optional[float] = None) -> None:
        now = datetime.utcnow().isoformat()
        with self.connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO orphan_sequences
                   (sequence_id, segment_name, nearest_cluster, nearest_distance, flagged_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (sequence_id, segment_name, nearest_cluster, nearest_distance, now),
            )

    def flag_orphan_conn(self, conn: Any, sequence_id: str, segment_name: str,
                         nearest_cluster: Optional[str] = None,
                         nearest_distance: Optional[float] = None) -> None:
        """Bulk-operation variant — uses a caller-supplied connection."""
        now = datetime.utcnow().isoformat()
        conn.execute(
            """INSERT OR REPLACE INTO orphan_sequences
               (sequence_id, segment_name, nearest_cluster, nearest_distance, flagged_at)
               VALUES (?, ?, ?, ?, ?)""",
            (sequence_id, segment_name, nearest_cluster, nearest_distance, now),
        )

    # ── Reassortment CRUD ──────────────────────────────────────────────

    def insert_reassortment_event(self, sequence_id: str, discordant_segments: List[str],
                                  confidence: float, description: Optional[str] = None) -> None:
        now = datetime.utcnow().isoformat()
        with self.connection() as conn:
            conn.execute(
                """INSERT INTO reassortment_events
                   (sequence_id, discordant_segments, confidence, description, detected_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (sequence_id, ",".join(discordant_segments), confidence, description, now),
            )

    def delete_reassortment_events(self, sequence_ids: List[str]) -> int:
        """Delete all reassortment events for the given sequence IDs.

        Used before re-inserting events for the same sequences in a batch
        re-run, preventing duplicate rows in reassortment_events.
        Returns the number of rows deleted.
        """
        if not sequence_ids:
            return 0
        placeholders = ",".join("?" * len(sequence_ids))
        with self.connection() as conn:
            cur = conn.execute(
                f"DELETE FROM reassortment_events WHERE sequence_id IN ({placeholders})",
                sequence_ids,
            )
            deleted = cur.rowcount
            if deleted:
                logger.debug(f"Cleared {deleted} reassortment event(s) for {len(sequence_ids)} sequence(s) before re-insert.")
            return deleted

    # ── Summary stats ──────────────────────────────────────────────────

    def get_summary_stats(self) -> Dict:
        with self.connection() as conn:
            return {
                "total_sequences": conn.execute("SELECT COUNT(*) FROM sequences").fetchone()[0],
                "by_subtype": {r[0]: r[1] for r in conn.execute(
                    "SELECT subtype, COUNT(*) FROM sequences GROUP BY subtype").fetchall()},
                "by_status": {r[0]: r[1] for r in conn.execute(
                    "SELECT status, COUNT(*) FROM sequences GROUP BY status").fetchall()},
                "total_clusters": conn.execute(
                    "SELECT COUNT(DISTINCT cluster_id || segment_name) FROM clusters WHERE is_active=1").fetchone()[0],
                "total_alleles": conn.execute(
                    "SELECT COUNT(*) FROM allele_registry WHERE is_active=1").fetchone()[0],
                "total_constellations": conn.execute(
                    "SELECT COUNT(*) FROM constellation_registry WHERE is_active=1").fetchone()[0],
                "total_genotypes": conn.execute("SELECT COUNT(*) FROM genotypes").fetchone()[0],
                "unresolved_orphans": conn.execute(
                    "SELECT COUNT(*) FROM orphan_sequences WHERE resolved=0").fetchone()[0],
                "reassortment_events": conn.execute(
                    "SELECT COUNT(*) FROM reassortment_events").fetchone()[0],
            }


    # ── Allele repair helpers ──────────────────────────────────────────

    def get_segment_signatures_for_subtype(
        self,
        segment_name: str,
        subtype: str,
        cluster_version: Optional[str] = None,
    ) -> List[Dict]:
        """Return stored k-mer signatures for all sequences of a given subtype
        and segment, optionally filtered to a specific cluster version.

        Each returned dict has keys:
            sequence_id, kmer_signature (bytes), cluster_id, allele_id,
            cluster_version, is_orphan.

        Used by ``repair_allele_subtype`` to re-run nomenclature assignment
        against existing stored signatures without re-processing raw sequences.
        """
        with self.connection() as conn:
            if cluster_version:
                rows = conn.execute(
                    """SELECT sk.sequence_id, sk.kmer_signature, sk.cluster_id,
                              sk.allele_id, sk.cluster_version, sk.is_orphan
                       FROM segment_kmers sk
                       JOIN sequences s ON sk.sequence_id = s.sequence_id
                       WHERE sk.segment_name = ?
                         AND s.subtype = ?
                         AND sk.cluster_version = ?
                         AND sk.kmer_signature IS NOT NULL""",
                    (segment_name, subtype, cluster_version),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT sk.sequence_id, sk.kmer_signature, sk.cluster_id,
                              sk.allele_id, sk.cluster_version, sk.is_orphan
                       FROM segment_kmers sk
                       JOIN sequences s ON sk.sequence_id = s.sequence_id
                       WHERE sk.segment_name = ?
                         AND s.subtype = ?
                         AND sk.kmer_signature IS NOT NULL""",
                    (segment_name, subtype),
                ).fetchall()
        return [dict(r) for r in rows]

    def update_genotype_allele_profile(
        self,
        sequence_id: str,
        cluster_version: str,
        allele_profile: str,
        constellation_id: Optional[str],
    ) -> None:
        """Update the allele_profile and constellation_id columns of an
        existing genotype row.  Used by ``repair_allele_subtype`` to correct
        genotype records in place without invalidating the genotype_profile or
        completeness values.
        """
        with self.connection() as conn:
            conn.execute(
                """UPDATE genotypes
                   SET allele_profile = ?, constellation_id = ?
                   WHERE sequence_id = ? AND cluster_version = ?""",
                (allele_profile, constellation_id, sequence_id, cluster_version),
            )
