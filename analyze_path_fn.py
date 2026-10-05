"""Quantify: paths-only FN on C0 (GT skeleton) vs other conditions.
If C0 paths-only FN ~= C4 paths-only FN, then 'chair C4 misses' are skeletonizer
contact-gap artifacts, not hull errors -> attribution in reports needs a caveat."""
import os
import sys
import numpy as np
from skimage.morphology import skeletonize

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "prop1"))
from run_experiments import load, part_masks, hull
from core import metrics

data_dir = os.path.join(ROOT, "prop1", "data")
files = [f for f in sorted(os.listdir(data_dir)) if f.endswith(".npz")]

print(f"{'kind':12s} {'obj':16s} {'C0path':>7s} {'C4path':>7s} {'C5path':>7s} {'gt':>4s}")
kind_sum = {}
for f in files:
    o = load(os.path.join(data_dir, f))
    masks = part_masks(o["label"], o["parts"])
    n = len(masks)
    gt = o["gt"]

    # C0: skeleton of GT voxels, paths only
    s0 = skeletonize(o["occ"])
    l0 = metrics.assign_labels(s0, masks)
    p0 = metrics.pair_paths(s0, l0, n, set())
    fn0 = len(gt - p0)

    # C4: skeleton of joint hull, paths only
    H = hull([o["proj"][0], o["proj"][1], o["proj"][2]])
    s4 = skeletonize(H)
    l4 = metrics.assign_labels(s4, masks)
    p4 = metrics.pair_paths(s4, l4, n, set())
    fn4 = len(gt - p4)

    # C5: union of per-part hulls, paths only
    ph = [hull([np.asarray(o["pprojs"][p["name"]][k]) for k in range(3)])
          for p in o["parts"]]
    s5 = skeletonize(np.logical_or.reduce(ph))
    l5 = metrics.assign_labels(s5, masks)
    p5 = metrics.pair_paths(s5, l5, n, set())
    fn5 = len(gt - p5)

    print(f"{o['kind']:12s} {o['name']:16s} {fn0:7d} {fn4:7d} {fn5:7d} {len(gt):4d}")
    k = o["kind"]
    a = kind_sum.setdefault(k, [0, 0, 0, 0])
    a[0] += fn0; a[1] += fn4; a[2] += fn5; a[3] += len(gt)

print("\nper-kind totals:  kind  C0path  C4path  C5path  gt")
for k, (a, b, c, g) in kind_sum.items():
    print(f"  {k:12s} {a:6d} {b:7d} {c:7d} {g:5d}")
tot = np.sum(list(kind_sum.values()), axis=0)
print(f"  {'TOTAL':12s} {tot[0]:6d} {tot[1]:7d} {tot[2]:7d} {tot[3]:5d}")
print("\nInterpretation:")
print(f"  C0 paths-only FN = {tot[0]}/{tot[3]} GT edges -> skeletonizer contact gap "
      f"(exists even on perfect voxels; masked in experiment by feeding GT edges)")
print(f"  C4 paths-only FN = {tot[1]} -> hull+skeletonizer; EXCESS over C0 = {tot[1]-tot[0]}")
print(f"  C5 paths-only FN = {tot[2]} -> per-part hull+skeletonizer; EXCESS over C0 = {tot[2]-tot[0]}")
