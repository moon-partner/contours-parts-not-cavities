import gzip
import csv
import os
from collections import Counter

with gzip.open(os.path.join("real", "abo", "3dmodels.csv.gz"), "rt",
               encoding="utf-8", errors="replace", newline="") as f:
    rows = list(csv.DictReader(f))

n = len(rows)
mesh_c = Counter(int(r["meshes"]) for r in rows)
mat_c = Counter(int(r["materials"]) for r in rows)
multi = [r for r in rows if int(r["meshes"]) >= 2 or int(r["materials"]) >= 2]
print(f"total models: {n}")
print(f"meshes distribution (top): {sorted(mesh_c.items())[:8]}")
print(f"materials distribution (top): {sorted(mat_c.items())[:8]}")
print(f"models with meshes>=2 OR materials>=2 (auto-splittable candidates): "
      f"{len(multi)} = {len(multi)/n:.1%}")

# small, light, multi-part models are the best demo candidates
def faces(r):
    return int(r["faces"])

light = sorted([r for r in multi if faces(r) < 30000], key=faces)
print(f"\nafter faces<30k filter: {len(light)} models")
print("10 smallest multi-part candidates:")
for r in light[:10]:
    print(f"  {r['3dmodel_id']}  meshes={r['meshes']} materials={r['materials']} "
          f"faces={r['faces']} path={r['path']} "
          f"extent=({float(r['extent_x']):.2f},{float(r['extent_y']):.2f},{float(r['extent_z']):.2f})")
