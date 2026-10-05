import numpy as np
from PIL import Image
import os, glob, json

d = os.path.join("real", "rendered", "smoke")
meta = json.load(open(os.path.join(d, "smoke_meta.json")))
print("meta:", meta)

for view in ["front", "side", "top"]:
    ms = []
    for i, part in enumerate(meta["parts"]):
        fn = os.path.join(d, f"smoke_{view}_{i}_{part}.png")
        a = np.asarray(Image.open(fn))[:, :, 3]      # alpha channel
        vals = set(np.unique(a).tolist())
        cover = (a > 127).mean()
        ms.append(a > 127)
        print(f"{view:6s} {part:8s} cover={cover:6.2%}  alpha values={sorted(vals)[:6]}"
              f"{' ...binary OK' if vals <= {0, 255} else ' NON-BINARY!'}")
    inter = (ms[0] & ms[1]).sum()
    union = (ms[0] | ms[1]).sum()
    print(f"       part overlap IoU = {inter / max(union, 1):.4f} (want ~0; parts disjoint)")

comp = np.asarray(Image.open(os.path.join(d, "smoke_front.png")))
uniq = len(np.unique(comp.reshape(-1, 4), axis=0))
print(f"composite front: {uniq} unique RGBA colors (2 parts + bg + antialias-free edges)")
