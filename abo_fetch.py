"""Batch-fetch light multi-material ABO candidates (deduped, size-capped)."""
import csv
import gzip
import os
import urllib.request

SRC = os.path.join("real", "abo", "3dmodels.csv.gz")
OUT = os.path.join("real", "abo", "models")
BASE = "https://amazon-berkeley-objects.s3.amazonaws.com/3dmodels/original/"
MAX_MB = 60
MAX_NEW = 14

with gzip.open(SRC, "rt", encoding="utf-8", errors="replace", newline="") as f:
    rows = list(csv.DictReader(f))

cands = [r for r in rows
         if (int(r["meshes"]) >= 2 or int(r["materials"]) >= 2)
         and int(r["faces"]) < 30000]
# dedupe: identical (faces, extent-rounded) = same product in different colors
seen, uniq, dropped_dup = set(), [], 0
for r in sorted(cands, key=lambda r: int(r["faces"])):
    key = (int(r["faces"]),
           round(float(r["extent_x"]), 1), round(float(r["extent_y"]), 1),
           round(float(r["extent_z"]), 1))
    if key in seen:
        dropped_dup += 1
        continue
    seen.add(key)
    uniq.append(r)
print(f"candidates={len(cands)}, unique products={len(uniq)} (dropped {dropped_dup} dup color variants)")

have = {os.path.splitext(f)[0] for f in os.listdir(OUT)}
todo = [r for r in uniq if r["3dmodel_id"] not in have][:MAX_NEW]
print(f"to download: {len(todo)}")

ok, skipped, fail = [], [], []
for r in todo:
    tid, path = r["3dmodel_id"], r["path"]
    url = BASE + path
    dest = os.path.join(OUT, f"{tid}.glb")
    try:
        req = urllib.request.Request(url, method="HEAD")
        cl = int(urllib.request.urlopen(req, timeout=30).headers.get("Content-Length", 0))
        if cl > MAX_MB * 1e6:
            skipped.append((tid, f"{cl/1e6:.0f}MB > cap"))
            continue
        urllib.request.urlretrieve(url, dest)
        got = os.path.getsize(dest)
        head = open(dest, "rb").read(4)
        tag = "OK" if head == b"glTF" else "BAD-MAGIC"
        print(f"  {tid}: {got/1e6:.1f}MB faces={r['faces']} {tag}")
        (ok if tag == "OK" else fail).append(tid)
        if tag != "OK":
            os.remove(dest)
    except Exception as e:
        fail.append(tid)
        print(f"  {tid}: FAIL {e}")

print(f"\ndownloaded OK={len(ok)} skipped_size={len(skipped)} failed={len(fail)}")
for t, why in skipped:
    print(f"  skipped {t}: {why}")
