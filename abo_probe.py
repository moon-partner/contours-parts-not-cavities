import gzip
import csv
import io
import re
import os

# --- README: how are glb files laid out?
rd = open(os.path.join("real", "abo", "README.md"), encoding="utf-8", errors="replace").read()
print("=" * 30, "README (layout rules)", "=" * 30)
for line in rd.splitlines():
    if re.search(r"(glb|original|download|s3|path|metadata)", line, re.I):
        print(line[:200])

# --- metadata CSV
print("=" * 30, "CSV schema", "=" * 30)
with gzip.open(os.path.join("real", "abo", "3dmodels.csv.gz"), "rt",
               encoding="utf-8", errors="replace", newline="") as f:
    rows = list(csv.DictReader(f))
print(f"rows = {len(rows)}")
print("columns:", list(rows[0].keys()))

# --- filter candidate categories: cups / mugs / jars / bottles
pat = re.compile(r"cups|mugs|jars|bottles|drinking", re.I)
hits = [r for r in rows if pat.search(" ".join(str(v) for v in r.values()))]
print(f"\ncandidates matching cup/mug/jar/bottle: {len(hits)}")
# show a few with whatever size-ish columns exist
size_cols = [c for c in rows[0] if re.search(r"size|mb|kb", c, re.I)]
print("size-ish columns:", size_cols)
show_cols = [c for c in ["item_id", "object_type", "product_type", "color", "scale"]
             if c in rows[0]] or list(rows[0])[:4]
for r in hits[:12]:
    print({c: str(r.get(c, ""))[:40] for c in show_cols})
