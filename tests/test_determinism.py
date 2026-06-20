"""Shuffle-twice determinism test for the portable nomenclature guarantee.

Runs a from-scratch clustering twice on the same isolates in two different
input orders (fresh DB each) and asserts that every isolate receives the same
allele profile and every allele the same radius. Internal cluster_id labels are
allowed to differ (they are ephemeral, discovery-ordered); the allele layer is
the stable identifier, so geometry is compared label-independently.

Run:  python tests/test_determinism.py
"""
import logging, random, tempfile, os, sys
from collections import Counter
logging.disable(logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from influenza_genotyper import GenotypingPipeline, GenotyperConfig
from influenza_genotyper.settings import DatabaseConfig, SEGMENT_LENGTH_RANGES

SEGNUM = {1: "PB2", 2: "PB1", 3: "PA", 4: "HA", 5: "NP", 6: "NA", 7: "M", 8: "NS"}


def _randseq(n, rng):
    return "".join(rng.choice("ACGT") for _ in range(n))


def _mutate(s, frac, rng):
    s = list(s)
    for i in rng.sample(range(len(s)), int(len(s) * frac)):
        s[i] = rng.choice("ACGT")
    return "".join(s)


def _make_records():
    g = random.Random(100)
    basesA = {k: _randseq((SEGMENT_LENGTH_RANGES[v][0] + SEGMENT_LENGTH_RANGES[v][1]) // 2, g)
              for k, v in SEGNUM.items()}
    basesB = {k: _mutate(basesA[k], 0.12, g) for k in SEGNUM}
    records = []
    for lin, bases in (("A", basesA), ("B", basesB)):
        for iso in range(12):
            records.append((f"iso{lin}{iso:02d}",
                            {k: _mutate(bases[k], 0.001, g) for k in SEGNUM}))
    return records


def _write_fasta(order, path):
    lines = []
    for sid, segs in order:
        for k, seq in segs.items():
            lines.append(f">{sid}_segment_{k}_H3N2_2024-01-05")
            lines.append(seq)
    open(path, "w").write("\n".join(lines) + "\n")


def _run_once(order, tag):
    tmp = tempfile.mkdtemp()
    fa = os.path.join(tmp, "in.fasta")
    _write_fasta(order, fa)
    cfg = GenotyperConfig.default()
    cfg.database = DatabaseConfig(db_type="sqlite", sqlite_path=os.path.join(tmp, f"{tag}.db"))
    p = GenotypingPipeline(config=cfg)
    p.db.initialize()
    p.run(fa, cluster_version="v1", detect_reassortment=False)
    with p.db.connection() as c:
        prof = {r["sequence_id"].split("_")[0]: r["allele_profile"]
                for r in c.execute("SELECT sequence_id, allele_profile FROM genotypes "
                                   "WHERE cluster_version='v1'")}
        radii = {r["allele_name"]: round(r["radius"], 6)
                 for r in c.execute("SELECT allele_name, radius FROM allele_registry")}
        crad = Counter((r["segment_name"], round(r["radius"], 6))
                       for r in c.execute("SELECT segment_name, radius FROM clusters "
                                          "WHERE version='v1'"))
    return prof, radii, crad


def main():
    records = _make_records()
    o1 = list(records); random.Random(1).shuffle(o1)
    o2 = list(records); random.Random(2).shuffle(o2)
    p1, r1, c1 = _run_once(o1, "run1")
    p2, r2, c2 = _run_once(o2, "run2")

    checks = [
        ("identical allele profile per isolate (order-independent)", p1 == p2),
        ("identical per-allele radius (order-independent)", r1 == r2),
        ("identical cluster geometry as label-independent multiset", c1 == c2),
    ]
    ok = True
    for name, cond in checks:
        print(f"[{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and cond
    print(f"\n{sum(c for _, c in checks)}/{len(checks)} checks passed")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
