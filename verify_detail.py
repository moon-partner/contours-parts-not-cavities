import csv
import numpy as np
rows = list(csv.DictReader(open('prop1/results.csv')))
print("-- spring CD per condition (5 objects each) --")
for c in ['C0', 'C1', 'C2', 'C3', 'C3b', 'C4', 'C5']:
    v = [float(r['chamfer']) for r in rows if r['kind'] == 'spring' and r['cond'] == c]
    print(f"  {c:4s} {[round(x,6) for x in v]}")
print("\n-- cage_cup C4 / chair C4 / chair C2 per-object --")
for kind, cond, mets in [('cage_cup', 'C4', ['m2_fp', 'm2_fn']),
                         ('chair', 'C4', ['m2_fp', 'm2_fn']),
                         ('chair', 'C2', ['m1_fp', 'm1_fn'])]:
    rs = [r for r in rows if r['kind'] == kind and r['cond'] == cond]
    for m in mets:
        v = [float(r[m]) for r in rs if r[m] not in ('', 'None')]
        print(f"  {kind:10s} {cond:3s} {m}: total={sum(v):.0f} per-obj={np.mean(v):.1f} "
              f"values={[int(x) for x in v]}")
print("\n-- cage C2 M1 FN --")
rs = [r for r in rows if r['kind'] == 'cage_cup' and r['cond'] == 'C2']
v = [float(r['m1_fn']) for r in rs]
print(f"  total={sum(v):.0f} values={[int(x) for x in v]}")
print("\n-- bracket/chair CD (claim: bracket C4 .371 vs C1 .048; chair C2 .568 vs C1 .029) --")
for kind in ['bracket', 'chair']:
    for c in ['C1', 'C2', 'C4']:
        v = [float(r['chamfer']) for r in rows if r['kind'] == kind and r['cond'] == c]
        print(f"  {kind:8s} {c:3s} {np.mean(v):.4f}")
print("\n-- phantom ratio C4 vs C1 (claim C4 has phantoms, C1=0) --")
for c in ['C1', 'C4']:
    v = [float(r['phantom']) for r in rows if r['cond'] == c]
    print(f"  {c}: mean={np.mean(v):.4f}")
