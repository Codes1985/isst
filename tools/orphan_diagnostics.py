#!/usr/bin/env python3
"""
orphan_diagnostics.py — read-only triage of a cold-start run's orphans.

Answers four questions, in order of how much they change the interpretation:

  A. How many orphans per segment, per subtype, and at what rate?
  B. Are the SAME isolates orphaning across segments (per-isolate quality /
     rarity), or are the orphan sets disjoint per segment (per-segment fine
     structure)?
  C. Of the orphans, how many MINTED a new (provisional) allele vs. JOINED an
     existing lineage under nearest-lineage assignment?  Minted-provisional is
     the real namespace cost; joined orphans are harmless.
  D. Are the minting orphans near-misses (just past a lineage margin — likely
     real small lineages) or far outliers (likely noise/novelty)?  Plus a
     quality correlation: do orphans skew short / low-quality?

Usage:
    python3 orphan_diagnostics.py /path/to/influenza_genotyper.db [--version V]

Read-only: opens the database in immutable mode and never writes.
"""

import argparse
import json
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict


def connect_ro(path):
    # Immutable URI — guarantees no writes / locks against a live DB file.
    uri = f"file:{path}?immutable=1"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def table_columns(conn, table):
    try:
        return {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
    except sqlite3.Error:
        return set()


def latest_version(conn):
    row = conn.execute(
        "SELECT cluster_version FROM segment_kmers "
        "WHERE cluster_version IS NOT NULL "
        "ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    return row["cluster_version"] if row else None


def pct(values, p):
    if not values:
        return float("nan")
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    k = (len(values) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(values) - 1)
    frac = k - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def hr(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def section_A(conn, version):
    hr("A.  Orphan counts per segment x subtype")
    rows = conn.execute(
        """
        SELECT sk.segment_name AS seg, s.subtype AS subtype,
               SUM(sk.is_orphan) AS orphans, COUNT(*) AS total
        FROM segment_kmers sk
        JOIN sequences s ON s.sequence_id = sk.sequence_id
        WHERE sk.cluster_version = ?
        GROUP BY sk.segment_name, s.subtype
        ORDER BY s.subtype, sk.segment_name
        """,
        (version,),
    ).fetchall()
    if not rows:
        print("  (no segment_kmers rows for this version)")
        return
    print(f"  {'subtype':<12}{'seg':<6}{'orphans':>9}{'total':>8}{'rate':>8}")
    for r in rows:
        rate = (r["orphans"] / r["total"]) if r["total"] else 0
        print(f"  {r['subtype']:<12}{r['seg']:<6}{r['orphans']:>9}{r['total']:>8}{rate:>7.1%}")


def section_B(conn, version):
    hr("B.  Cross-segment overlap — are the same isolates orphaning everywhere?")
    rows = conn.execute(
        """
        SELECT sequence_id,
               SUM(is_orphan) AS seg_orphaned,
               COUNT(*)       AS seg_total
        FROM segment_kmers
        WHERE cluster_version = ?
        GROUP BY sequence_id
        """,
        (version,),
    ).fetchall()
    if not rows:
        print("  (no data)")
        return

    # Histogram of "# segments orphaned" per isolate (only isolates with >=1).
    hist = Counter()
    all_orphaned = 0          # orphaned in every segment they have
    any_orphaned = 0          # orphaned in >=1 segment
    for r in rows:
        o = r["seg_orphaned"] or 0
        if o > 0:
            hist[o] += 1
            any_orphaned += 1
            if o == r["seg_total"]:
                all_orphaned += 1

    print(f"  isolates with >=1 orphaned segment : {any_orphaned}")
    print(f"  isolates orphaned in EVERY segment : {all_orphaned}")
    print("\n  distribution of (# segments orphaned) across those isolates:")
    print(f"  {'#segments':>10}{'#isolates':>12}")
    for k in sorted(hist):
        print(f"  {k:>10}{hist[k]:>12}")
    print(
        "\n  Read: a spike at the high end (isolates orphaning in most/all "
        "segments)\n  points to per-ISOLATE causes (quality / rarity). A spread "
        "concentrated\n  at 1-2 segments points to per-SEGMENT fine structure."
    )


def section_C(conn, version, ar_cols):
    hr("C.  Orphans: minted a new (provisional) allele vs. joined a lineage")
    if "provisional" not in ar_cols:
        print("  allele_registry has no 'provisional' column — this DB predates")
        print("  the hardening schema (v4). Re-run after migrating, or inspect")
        print("  allele first_seen timestamps to separate minted vs. joined.")
        return

    # Provisional (orphan-founded) alleles minted, per segment.
    prov = conn.execute(
        "SELECT segment_name AS seg, COUNT(*) AS n "
        "FROM allele_registry WHERE provisional = 1 GROUP BY segment_name"
    ).fetchall()
    prov_map = {r["seg"]: r["n"] for r in prov}

    # For orphan rows, split by whether their assigned allele is provisional.
    rows = conn.execute(
        """
        SELECT sk.segment_name AS seg,
               SUM(CASE WHEN ar.provisional = 1 THEN 1 ELSE 0 END) AS minted,
               SUM(CASE WHEN ar.provisional = 0 OR ar.provisional IS NULL
                        THEN 1 ELSE 0 END)                          AS joined,
               COUNT(*) AS orphan_rows
        FROM segment_kmers sk
        LEFT JOIN allele_registry ar ON ar.allele_name = sk.allele_id
        WHERE sk.cluster_version = ? AND sk.is_orphan = 1
        GROUP BY sk.segment_name
        ORDER BY sk.segment_name
        """,
        (version,),
    ).fetchall()

    print(f"  {'seg':<6}{'orphans':>9}{'joined':>9}{'minted':>9}"
          f"{'prov.alleles':>14}")
    tot_orphans = tot_joined = tot_minted = 0
    for r in rows:
        seg = r["seg"]
        print(f"  {seg:<6}{r['orphan_rows']:>9}{r['joined']:>9}"
              f"{r['minted']:>9}{prov_map.get(seg, 0):>14}")
        tot_orphans += r["orphan_rows"]
        tot_joined += r["joined"] or 0
        tot_minted += r["minted"] or 0
    print(f"  {'-'*46}")
    print(f"  {'ALL':<6}{tot_orphans:>9}{tot_joined:>9}{tot_minted:>9}")
    print(
        "\n  Read: 'joined' orphans cost nothing — nearest-lineage absorbed "
        "them.\n  'minted' (= provisional alleles) is the real namespace growth; "
        "that is\n  the number to judge as real novelty vs. noise in section D."
    )


def section_D(conn, version, sk_cols):
    hr("D.  Are minting orphans near-misses or far outliers? + quality skew")

    # D1: distance-to-nearest distribution for orphans, per segment.
    dist_col = "distance_to_centroid" if "distance_to_centroid" in sk_cols else None
    if dist_col:
        rows = conn.execute(
            f"""
            SELECT segment_name AS seg, {dist_col} AS d
            FROM segment_kmers
            WHERE cluster_version = ? AND is_orphan = 1 AND {dist_col} IS NOT NULL
            """,
            (version,),
        ).fetchall()
        by_seg = defaultdict(list)
        for r in rows:
            by_seg[r["seg"]].append(r["d"])
        if by_seg:
            print("  Orphan distance-to-nearest-centroid (smaller = near-miss):")
            print(f"  {'seg':<6}{'n':>6}{'p10':>8}{'p50':>8}{'p90':>8}{'max':>8}")
            for seg in sorted(by_seg):
                ds = by_seg[seg]
                print(f"  {seg:<6}{len(ds):>6}{pct(ds,10):>8.3f}"
                      f"{pct(ds,50):>8.3f}{pct(ds,90):>8.3f}{max(ds):>8.3f}")
            print("\n  Compare p10/p50 against each segment's floor (1 - same_"
                  "threshold,\n  e.g. PB1/H3N2 = 0.07). Orphans clustered just "
                  "above the floor are\n  near-miss real lineages; a long tail to "
                  "high distance is noise/novelty.")
    else:
        print("  (no distance_to_centroid column to assess near-miss)")

    # D2: quality skew — orphan vs non-orphan sequence length per segment.
    if "sequence_length" in sk_cols:
        rows = conn.execute(
            """
            SELECT segment_name AS seg, is_orphan,
                   AVG(sequence_length) AS avg_len, COUNT(*) AS n
            FROM segment_kmers
            WHERE cluster_version = ? AND sequence_length IS NOT NULL
            GROUP BY segment_name, is_orphan
            """,
            (version,),
        ).fetchall()
        agg = defaultdict(dict)
        for r in rows:
            agg[r["seg"]][r["is_orphan"]] = (r["avg_len"], r["n"])
        if agg:
            print("\n  Mean sequence_length, orphan vs clustered (large gap = "
                  "length/quality skew):")
            print(f"  {'seg':<6}{'clustered_len':>15}{'orphan_len':>13}{'delta':>10}")
            for seg in sorted(agg):
                c = agg[seg].get(0, (None, 0))[0]
                o = agg[seg].get(1, (None, 0))[0]
                if c and o:
                    print(f"  {seg:<6}{c:>15.0f}{o:>13.0f}{(o-c):>10.0f}")

    # D3: generic numeric metadata skew (e.g. N content), if present in JSON.
    sample = conn.execute(
        "SELECT metadata_json FROM sequences WHERE metadata_json IS NOT NULL "
        "AND metadata_json != '{}' LIMIT 1"
    ).fetchone()
    if not sample:
        return
    try:
        keys = list(json.loads(sample["metadata_json"]).keys())
    except (json.JSONDecodeError, AttributeError):
        return
    print(f"\n  sequences.metadata_json keys present: {keys}")

    # Per-isolate orphan fraction across its segments.
    frac_rows = conn.execute(
        """
        SELECT sequence_id,
               CAST(SUM(is_orphan) AS REAL) / COUNT(*) AS frac
        FROM segment_kmers WHERE cluster_version = ?
        GROUP BY sequence_id
        """,
        (version,),
    ).fetchall()
    orphan_frac = {r["sequence_id"]: r["frac"] for r in frac_rows}

    meta_rows = conn.execute(
        "SELECT sequence_id, metadata_json FROM sequences"
    ).fetchall()

    # Split isolates: orphan-heavy (>= half their segments) vs the rest.
    heavy_vals = defaultdict(list)
    rest_vals = defaultdict(list)
    for r in meta_rows:
        sid = r["sequence_id"]
        if sid not in orphan_frac:
            continue
        try:
            meta = json.loads(r["metadata_json"] or "{}")
        except json.JSONDecodeError:
            continue
        bucket = heavy_vals if orphan_frac[sid] >= 0.5 else rest_vals
        for k, v in meta.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                bucket[k].append(float(v))

    numeric_keys = sorted(set(heavy_vals) | set(rest_vals))
    if numeric_keys:
        print("\n  Quality skew — mean per numeric metadata field, isolates that")
        print("  orphaned in >=50% of segments vs. the rest:")
        print(f"  {'field':<18}{'orphan-heavy':>14}{'rest':>12}{'n_heavy':>9}{'n_rest':>8}")
        for k in numeric_keys:
            h, rst = heavy_vals.get(k, []), rest_vals.get(k, [])
            hm = statistics.mean(h) if h else float("nan")
            rm = statistics.mean(rst) if rst else float("nan")
            print(f"  {k:<18}{hm:>14.3f}{rm:>12.3f}{len(h):>9}{len(rst):>8}")
        print("\n  A clear gap (e.g. higher N-fraction / lower coverage among the")
        print("  orphan-heavy group) confirms a per-isolate QUALITY cause rather")
        print("  than genuine lineage novelty.")


def section_clusters(conn, version):
    hr("Context: clusters formed (for contrast with the orphan tail)")
    rows = conn.execute(
        """
        SELECT segment_name AS seg, COUNT(*) AS n_clusters,
               SUM(member_count) AS clustered_members,
               MIN(member_count) AS min_sz, MAX(member_count) AS max_sz
        FROM clusters WHERE version = ? AND is_active = 1
        GROUP BY segment_name ORDER BY segment_name
        """,
        (version,),
    ).fetchall()
    if not rows:
        print("  (no clusters for this version)")
        return
    print(f"  {'seg':<6}{'#clusters':>10}{'members':>10}{'min':>6}{'max':>6}")
    for r in rows:
        print(f"  {r['seg']:<6}{r['n_clusters']:>10}{r['clustered_members'] or 0:>10}"
              f"{r['min_sz'] or 0:>6}{r['max_sz'] or 0:>6}")


def main():
    ap = argparse.ArgumentParser(description="Orphan triage for a genotyper DB.")
    ap.add_argument("db", help="path to influenza_genotyper.db")
    ap.add_argument("--version", help="cluster_version (default: latest)")
    args = ap.parse_args()

    try:
        conn = connect_ro(args.db)
    except sqlite3.Error as e:
        sys.exit(f"Could not open DB read-only: {e}")

    version = args.version or latest_version(conn)
    if version is None:
        sys.exit("No cluster_version found in segment_kmers — nothing to report.")
    print(f"Analysing cluster_version = {version!r}")

    sk_cols = table_columns(conn, "segment_kmers")
    ar_cols = table_columns(conn, "allele_registry")

    section_A(conn, version)
    section_clusters(conn, version)
    section_B(conn, version)
    section_C(conn, version, ar_cols)
    section_D(conn, version, sk_cols)
    print()


if __name__ == "__main__":
    main()
