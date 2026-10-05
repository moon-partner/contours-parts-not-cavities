import csv, collections, os
HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, "results.csv"))))
num = lambda r, k: (None if r[k] in ("", "None") else float(r[k]))

print("=== M1 (assembly TC) fp/fn by kind ===")
kinds = sorted({r["kind"] for r in rows})
print(f"{'kind':12s} " + " ".join(f"{c:>16s}" for c in ["C1", "C3", "C3b", "C2", "C5"]))
for k in kinds:
    cells = []
    for c in ["C1", "C3", "C3b", "C2", "C5"]:
        rs = [r for r in rows if r["kind"] == k and r["cond"] == c]
        fp = sum(num(r, "m1_fp") or 0 for r in rs)
        fn = sum(num(r, "m1_fn") or 0 for r in rs)
        cells.append(f"{fp:6.0f}/{fn:<6.0f}")
    print(f"{k:12s} " + " ".join(f"{x:>16s}" for x in cells))

print("\n=== M2 (skeleton-path TC) fp/fn by kind ===")
print(f"{'kind':12s} " + " ".join(f"{c:>16s}" for c in ["C0", "C1", "C4", "C5"]))
for k in kinds:
    cells = []
    for c in ["C0", "C1", "C4", "C5"]:
        rs = [r for r in rows if r["kind"] == k and r["cond"] == c]
        fp = sum(num(r, "m2_fp") or 0 for r in rs)
        fn = sum(num(r, "m2_fn") or 0 for r in rs)
        cells.append(f"{fp:6.0f}/{fn:<6.0f}")
    print(f"{k:12s} " + " ".join(f"{x:>16s}" for x in cells))

print("\n=== per-kind means: chamfer, M2_TC, d_cycle ===")
print(f"{'kind':12s} {'cond':5s} {'CD':>8s} {'M2':>5s} {'M1':>5s} {'dCyc':>5s} {'ncomp_d':>7s}")
for k in kinds:
    for c in ["C0", "C1", "C2", "C3", "C3b", "C4", "C5"]:
        rs = [r for r in rows if r["kind"] == k and r["cond"] == c]
        cd = sum(num(r, "chamfer") for r in rs) / len(rs)
        m2 = sum(num(r, "m2_tc") for r in rs) / len(rs)
        m1s = [num(r, "m1_tc") for r in rs if num(r, "m1_tc") is not None]
        m1 = sum(m1s) / len(m1s) if m1s else float("nan")
        dc = sum(num(r, "d_cycle") for r in rs) / len(rs)
        nc = sum(num(r, "d_ncomp") for r in rs) / len(rs)
        print(f"{k:12s} {c:5s} {cd:8.4f} {m2:5.2f} {m1:5.2f} {dc:5.2f} {nc:7.2f}")
