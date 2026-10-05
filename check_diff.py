import numpy as np
from PIL import Image
im = np.asarray(Image.open("real/abo/B07DBHLX1W_diff.png").convert("RGB"))
print("shape:", im.shape)
cols, cnt = np.unique(im.reshape(-1, 3), axis=0, return_counts=True)
order = np.argsort(-cnt)
for i in order[:8]:
    print(f"  rgb={tuple(int(x) for x in cols[i])} count={int(cnt[i])}")
