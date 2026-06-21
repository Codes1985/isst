"""Orphan reporting — a read-only, derived surface over the orphan_events ledger.

Produces an informational report with no side effects and no recluster trigger:
a human reads it and decides. Two groups of panels.

Snapshot (current open episodes, optionally windowed by ``cluster_version``):
    1. category_summary  — open counts by segment and category
    2. coherence         — complete orphans grouped by sequence; several segments
                           of one isolate orphaning together is a candidate novel
                           lineage worth a closer look
    3. near_misses       — open orphans closest to joining a cluster (smallest
                           nearest_distance): threshold-review candidates
    4. partial_waiting   — partial orphans awaiting a complete example, grouped by
                           segment (a prompt to go collect a full-length sequence)

History (resolved + long-open, across all versions):
    5. cohort               — open orphans grouped by the date they entered
    6. resolution_outcomes  — resolved counts by exit door
    7. time_to_resolution   — wait-time statistics per door, in days
    8. persistent_waiters   — the oldest still-open episodes

The reporter only reads the ledger via DatabaseManager; it computes nothing that
changes state.
"""

from __future__ import annotations

import statistics
from datetime import datetime
from typing import Any, Dict, List, Optional

_DOORS = ("minted_new", "absorbed", "resolved_by_completion")


def _parse(ts: Optional[str]) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(ts)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _age_days(start_iso: Optional[str], end_iso: Optional[str]) -> Optional[float]:
    a, b = _parse(start_iso), _parse(end_iso)
    if a is None or b is None:
        return None
    return (b - a).total_seconds() / 86400.0


class OrphanReporter:
    """Builds the orphan report from the ledger. Read-only."""

    def __init__(self, db: Any):
        self.db = db

    # ── snapshot panels ──────────────────────────────────────────────────
    def category_summary(self, cluster_version: str) -> Dict[str, Any]:
        rows = self.db.count_open_orphans_by_category(cluster_version)
        by_segment: Dict[str, Dict[str, int]] = {}
        totals = {"complete": 0, "partial": 0}
        for r in rows:
            seg = by_segment.setdefault(r["segment_name"], {"complete": 0, "partial": 0})
            seg[r["category"]] = r["n"]
            totals[r["category"]] += r["n"]
        return {
            "by_segment": by_segment,
            "totals": totals,
            "total_open": totals["complete"] + totals["partial"],
        }

    def coherence(self, cluster_version: Optional[str], min_segments: int = 1) -> List[Dict[str, Any]]:
        opens = self.db.get_open_orphans(cluster_version, category="complete")
        by_seq: Dict[str, List[str]] = {}
        for o in opens:
            by_seq.setdefault(o["sequence_id"], []).append(o["segment_name"])
        groups = [
            {"sequence_id": sid, "segments": sorted(segs), "n_segments": len(segs)}
            for sid, segs in by_seq.items()
            if len(segs) >= min_segments
        ]
        groups.sort(key=lambda g: (-g["n_segments"], g["sequence_id"]))
        return groups

    def near_misses(self, cluster_version: Optional[str], limit: int = 20) -> List[Dict[str, Any]]:
        opens = self.db.get_open_orphans(cluster_version)
        scored = [o for o in opens if o["nearest_distance"] is not None]
        scored.sort(key=lambda o: (o["nearest_distance"], o["sequence_id"], o["segment_name"]))
        return [
            {
                "sequence_id": o["sequence_id"],
                "segment_name": o["segment_name"],
                "category": o["category"],
                "nearest_cluster": o["nearest_cluster"],
                "nearest_distance": o["nearest_distance"],
            }
            for o in scored[:limit]
        ]

    def partial_waiting(self, cluster_version: Optional[str]) -> Dict[str, List[Dict[str, Any]]]:
        opens = self.db.get_open_orphans(cluster_version, category="partial")
        by_segment: Dict[str, List[Dict[str, Any]]] = {}
        for o in opens:
            by_segment.setdefault(o["segment_name"], []).append(
                {
                    "sequence_id": o["sequence_id"],
                    "completeness": o["completeness"],
                    "nearest_cluster": o["nearest_cluster"],
                }
            )
        return by_segment

    # ── history panels ───────────────────────────────────────────────────
    def cohort(self, cluster_version: Optional[str] = None) -> Dict[str, int]:
        opens = self.db.get_open_orphans(cluster_version)
        by_date: Dict[str, int] = {}
        for o in opens:
            d = _parse(o["entered_at"])
            key = d.date().isoformat() if d else "unknown"
            by_date[key] = by_date.get(key, 0) + 1
        return dict(sorted(by_date.items()))

    def resolution_outcomes(
        self, cluster_version: Optional[str] = None, since: Optional[str] = None
    ) -> Dict[str, Any]:
        res = self.db.get_orphan_resolutions(cluster_version, since)
        by_door = {d: 0 for d in _DOORS}
        for r in res:
            if r["exit_reason"] in by_door:
                by_door[r["exit_reason"]] += 1
        return {"by_door": by_door, "total": sum(by_door.values())}

    def time_to_resolution(
        self, cluster_version: Optional[str] = None, since: Optional[str] = None
    ) -> Dict[str, Any]:
        res = self.db.get_orphan_resolutions(cluster_version, since)
        ages_by_door: Dict[str, List[float]] = {d: [] for d in _DOORS}
        for r in res:
            age = _age_days(r["entered_at"], r["exited_at"])
            if age is not None and r["exit_reason"] in ages_by_door:
                ages_by_door[r["exit_reason"]].append(age)

        def _stats(ages: List[float]) -> Optional[Dict[str, float]]:
            if not ages:
                return None
            return {
                "count": len(ages),
                "min_days": round(min(ages), 3),
                "median_days": round(statistics.median(ages), 3),
                "max_days": round(max(ages), 3),
                "mean_days": round(statistics.fmean(ages), 3),
            }

        all_ages = [a for ages in ages_by_door.values() for a in ages]
        return {
            "overall": _stats(all_ages),
            "by_door": {d: _stats(ages_by_door[d]) for d in _DOORS},
        }

    def persistent_waiters(
        self, cluster_version: Optional[str] = None, limit: int = 20,
        as_of: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        opens = self.db.get_open_orphans(cluster_version)
        ref = as_of or datetime.utcnow().isoformat()
        rows = []
        for o in opens:
            rows.append(
                {
                    "sequence_id": o["sequence_id"],
                    "segment_name": o["segment_name"],
                    "category": o["category"],
                    "cluster_version": o["cluster_version"],
                    "entered_at": o["entered_at"],
                    "age_days": _age_days(o["entered_at"], ref),
                }
            )
        rows.sort(key=lambda r: (r["age_days"] is None, -(r["age_days"] or 0.0)))
        return rows[:limit]

    # ── assembly ─────────────────────────────────────────────────────────
    def build(self, cluster_version: Optional[str] = None, limit: int = 20) -> Dict[str, Any]:
        """Assemble all eight panels. ``cluster_version`` windows the snapshot
        panels (None = all open episodes); history panels always span versions."""
        return {
            "cluster_version": cluster_version,
            "snapshot": {
                "category_summary": self.category_summary(cluster_version),
                "coherence": self.coherence(cluster_version),
                "near_misses": self.near_misses(cluster_version, limit=limit),
                "partial_waiting": self.partial_waiting(cluster_version),
            },
            "history": {
                "cohort": self.cohort(None),
                "resolution_outcomes": self.resolution_outcomes(None),
                "time_to_resolution": self.time_to_resolution(None),
                "persistent_waiters": self.persistent_waiters(None, limit=limit),
            },
        }

    # ── rendering ────────────────────────────────────────────────────────
    @staticmethod
    def render_text(report: Dict[str, Any]) -> str:
        """Compact human-readable rendering of a built report."""
        snap = report["snapshot"]
        hist = report["history"]
        lines: List[str] = []
        ver = report.get("cluster_version") or "all versions"
        lines.append(f"Orphan report ({ver})")

        cs = snap["category_summary"]
        lines.append(
            f"  Open: {cs['total_open']} "
            f"({cs['totals']['complete']} complete, {cs['totals']['partial']} partial)"
        )

        coh = snap["coherence"]
        multi = [g for g in coh if g["n_segments"] > 1]
        if multi:
            lines.append(f"  Multi-segment complete orphans (candidate novel lineages): {len(multi)}")
            for g in multi[:5]:
                lines.append(f"    {g['sequence_id']}: {', '.join(g['segments'])}")

        nm = snap["near_misses"]
        if nm:
            lines.append("  Nearest near-misses:")
            for o in nm[:5]:
                lines.append(
                    f"    {o['sequence_id']}/{o['segment_name']} "
                    f"→ {o['nearest_cluster']} (dist {o['nearest_distance']:.4f})"
                )

        ro = hist["resolution_outcomes"]
        lines.append(
            f"  Resolved to date: {ro['total']} "
            f"(minted_new {ro['by_door']['minted_new']}, "
            f"absorbed {ro['by_door']['absorbed']}, "
            f"by_completion {ro['by_door']['resolved_by_completion']})"
        )
        ttr = hist["time_to_resolution"]["overall"]
        if ttr:
            lines.append(
                f"  Time-to-resolution (days): median {ttr['median_days']}, "
                f"max {ttr['max_days']} (n={ttr['count']})"
            )
        pw = hist["persistent_waiters"]
        if pw and pw[0]["age_days"] is not None:
            lines.append(
                f"  Oldest waiter: {pw[0]['sequence_id']}/{pw[0]['segment_name']} "
                f"({pw[0]['age_days']:.1f} days)"
            )
        return "\n".join(lines)
