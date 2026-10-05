"""Qualitative figure for the real-geometry experiment (paper-ready, English).

One row per object:
  [a] colored composite render (front)
  [b] C0 GT skeleton (reference)
  [c] C1 predicted skeleton + assembly result
  [d] C4 joint-hull skeleton (baseline; lamp shows the break)
Skeleton panels: background = GT occupancy front projection (light gray),
gray dots = GT skeleton (reference behind every prediction), method skeleton
voxels colored by assigned GT part (red/blue; orange = phantom).
"""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from skimage.morphology import skeletonize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from core import metrics                          # noqa: E402
from run_real import view_grids, VIEWS            # noqa: E402

plt.rcParams.update({"font.size": 8, "figure.dpi": 200, "savefig.bbox": "tight"})
OBJS = ["B0728NW8FP", "B07HSDX2CQ", "B07DBHLX1W"]
TITLES = {"B0728NW8FP": "fabric chair", "B07HSDX2CQ": "stool",
          "B07DBHLX1W": "floor lamp"}
PART_COLORS = {1: "#d62728", 2: "#1f77b4"}
RES = {r["object"]: r for r in json.load(open(os.path.join(HERE, "real", "results.json")))}


def panel(ax, occ, entries, title, m2txt, gt_skel):
    bg = occ.any(axis=1)                       # (x, z) front projection
    ax.imshow(bg, origin="lower", cmap="Greys", alpha=0.25,
              extent=(0, bg.shape[1], 0, bg.shape[0]), interpolation="nearest")
    g = np.nonzero(gt_skel)
    ax.scatter(g[0], g[2], s=0.6, color="0.60", zorder=1)      # GT skeleton (gray)
    for sk, lb in entries:
        idx = np.nonzero(sk)
        cols = [PART_COLORS.get(int(v), "#ff7f0e") for v in lb[idx]]
        ax.scatter(idx[0], idx[2], s=1.4, c=cols, zorder=3)
    ax.set_title(f"{title}\n{m2txt}", fontsize=7.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0, bg.shape[1]); ax.set_ylim(0, bg.shape[0])


rows = []
for name in OBJS:
    gt = np.load(os.path.join(HERE, "real", "gt", f"{name}.npz"))
    occ = gt["occ"].astype(bool)
    pmasks = [gt["part_masks"][i].astype(bool) for i in range(gt["part_masks"].shape[0])]
    res = int(gt["res"])
    maxd = max(1, round(0.0625 * res))
    rd = os.path.join(HERE, "real", "abo", "rendered", name)
    n = len(pmasks)
    per_part = [view_grids(rd, name, n, res, only=[i]) for i in range(n)]
    joint = view_grids(rd, name, n, res, only=None)
    hulls = [metrics.hull([pp[VIEWS[0]], pp[VIEWS[1]], pp[VIEWS[2]]]) for pp in per_part]
    jhull = metrics.hull([joint[VIEWS[0]], joint[VIEWS[1]], joint[VIEWS[2]]])
    sk_c1 = skeletonize(np.logical_or.reduce(hulls))
    sk_c4 = skeletonize(jhull)
    rows.append(dict(name=name, occ=occ, c0=skeletonize(occ),
                     c1=(sk_c1, metrics.assign_labels(sk_c1, pmasks, maxd=maxd)),
                     c4=(sk_c4, metrics.assign_labels(sk_c4, pmasks, maxd=maxd))))

fig, axes = plt.subplots(len(rows), 4, figsize=(11.2, 8.6))

for r, (ax_row, name) in enumerate(zip(axes, OBJS)):
    row = rows[r]
    rd = os.path.join(HERE, "real", "abo", "rendered", name)
    comp = np.asarray(Image.open(os.path.join(rd, f"{name}_front.png")).convert("RGB"))
    ax_row[0].imshow(comp)
    ax_row[0].set_title(f"{name}\n{TITLES[name]}  (2 parts)", fontsize=8)
    ax_row[0].set_xticks([]); ax_row[0].set_yticks([])

    panel(ax_row[1], row["occ"], [], "C0  GT skeleton",
          "reference (mesh GT)", row["c0"])
    c1 = RES[name]["conditions"]["C1"]
    panel(ax_row[2], row["occ"], [row["c1"]], "C1  full method",
          f"M2 {c1['m2_fp']}/{c1['m2_fn']}  edges={c1['pred_edges']}", row["c0"])
    c4 = RES[name]["conditions"]["C4"]
    panel(ax_row[3], row["occ"], [row["c4"]], "C4  joint hull (baseline)",
          f"M2 {c4['m2_fp']}/{c4['m2_fn']}  edges={c4['pred_edges']}", row["c0"])

fig.suptitle("Real-geometry experiment: per-part masks -> assembly -> skeleton "
             "(red/blue = assigned part labels, orange = phantom, gray dots = GT skeleton)",
             fontsize=9, y=1.005)
out = os.path.join(HERE, "real", "abo", "real_results.png")
fig.savefig(out)
print("wrote", out)
