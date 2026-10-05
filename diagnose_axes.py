import glob
import json
import numpy as np
from PIL import Image

gt = np.load("real/gt/smoke.npz")
occ = gt["occ"]
meta = json.load(open("real/rendered/smoke/smoke_meta.json"))
R = occ.shape[0]

def union(view, nparts=2):
    acc = None
    for i in range(nparts):
        f = glob.glob(f"real/rendered/smoke/smoke_{view}_{i}_*.png")[0]
        a = np.asarray(Image.open(f))[:, :, 3] > 127
        acc = a if acc is None else (acc | a)
    return acc

def sample_grid(acc, col_axis, row_flip=True):
    """sample image at grid res: grid[a,b] -> acc[row(b), col(a)]"""
    ii = ((np.arange(R) + 0.5) / R * acc.shape[1]).astype(int)
    if row_flip:
        rr = ((np.arange(R)[::-1] + 0.5) / R * acc.shape[0]).astype(int)
    else:
        rr = ((np.arange(R) + 0.5) / R * acc.shape[0]).astype(int)
    g = acc[np.ix_(rr, ii)]
    return g if col_axis == "first" else g.T

def iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)

projs = {"side": occ.any(axis=0), "front": occ.any(axis=1), "top": occ.any(axis=2)}
for view, gt_p in projs.items():
    acc = union(view)
    print(f"{view}: image {acc.shape}, gt_proj {gt_p.shape}")
    for flip in [True, False]:
        for trans in [False, True]:
            g = sample_grid(acc, "first", row_flip=flip)
            if trans:
                g = g.T
            print(f"   row_flip={flip!s:5s} transpose={trans!s:5s} IoU={iou(g, gt_p):.3f}")
