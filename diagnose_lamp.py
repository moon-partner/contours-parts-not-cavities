"""Where does lamp GT disagree with rendered masks? Saves diff images per view."""
import sys
import os
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_abo_gt import rendered_union

name = "B07DBHLX1W"
rd = os.path.join("real", "abo", "rendered", name)
gt = np.load(os.path.join("real", "gt", f"{name}.npz"))
occ = gt["occ"].astype(bool)
res = occ.shape[0]
views = {"side": occ.any(axis=0), "front": occ.any(axis=1), "top": occ.any(axis=2)}
tiles = []
for v in ["side", "front", "top"]:
    m = rendered_union(rd, name, v, 2, res)
    g = views[v]
    only_m = m & ~g     # rendered says object, GT says empty  -> GT missing material
    only_g = g & ~m     # GT says object, rendered empty       -> GT extra material
    print(f"{v:6s} mask={m.sum():5d} gt={g.sum():5d} mask_only={only_m.sum():5d} "
          f"gt_only={only_g.sum():5d}")
    # where are mask_only pixels? show bounding rows/cols
    if only_m.sum():
        ys, xs = np.nonzero(only_m)
        print(f"       mask_only bbox: rows {ys.min()}-{ys.max()} cols {xs.min()}-{xs.max()}")
    rgb = np.zeros((res, res, 3), np.uint8)
    rgb[g] = (60, 60, 60)                 # gt gray
    rgb[m] = (40, 160, 60)                # agree green
    rgb[only_m] = (220, 40, 40)           # GT missing red
    rgb[only_g] = (40, 80, 220)           # GT extra blue
    tiles.append(np.rot90(rgb))
sheet = np.concatenate(tiles, axis=1)
Image.fromarray(sheet).save(os.path.join("real", "abo", f"{name}_diff.png"))
print("wrote diff image")
