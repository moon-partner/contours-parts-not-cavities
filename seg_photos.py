"""Segment real photos into part masks by color (input contract for the prop1 pipeline).

Usage
-----
    python seg_photos.py real/raw/mug --parts 2 --bg 60
    python seg_photos.py real/raw/chair --parts 4 --bg 60

Reads  {prefix}_front.jpg / _side.jpg / _top.jpg  (see SHOOTING.md)
Writes {outdir}/{prefix}_{view}_labels.png   uint8, 0=background, 1..K = parts
       {outdir}/{prefix}_{view}_preview.jpg  label colors overlaid for eyeballing

Method (kept deliberately simple and inspectable)
-------------------------------------------------
1. background = median color of the outer 3% border pixels; object = color distance > bg.
2. k-means++ (scipy) with K = expected part count, on object pixels (RGB, subsampled).
3. per-cluster morphological clean-up; warn if any part < 2% of object pixels.

Caveats (read before trusting output)
-------------------------------------
- Colors must differ in GRAYSCALE too (see SHOOTING.md); k-means in RGB cannot save
  a red-vs-green pair of similar luminance.
- Each view is cropped to its own bbox: this assumes all three photos were taken at the
  same distance with the object filling the frame similarly (approximate orthography).
- If a part is invisible in some view (single-sided sticker), its mask will be empty
  there and the per-part hull will be wrong -- shoot through-parts only.
"""
import argparse
import os
import numpy as np
from PIL import Image
from scipy.cluster.vq import kmeans2
from scipy import ndimage

VIEWS = ["front", "side", "top"]


def object_mask(img, bg_thresh):
    border = np.concatenate([
        img[: max(3, img.shape[0] // 30)].reshape(-1, 3),
        img[-max(3, img.shape[0] // 30):].reshape(-1, 3),
        img[:, : max(3, img.shape[1] // 30)].reshape(-1, 3),
        img[:, -max(3, img.shape[1] // 30):].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(img.astype(np.float32) - bg, axis=2)
    m = dist > bg_thresh
    m = ndimage.binary_closing(m, np.ones((5, 5)))
    m = ndimage.binary_fill_holes(m)
    m = ndimage.binary_opening(m, np.ones((3, 3)))
    return m, bg


def segment(img, obj, k, seed=0):
    ys, xs = np.nonzero(obj)
    if len(ys) == 0:
        raise RuntimeError("empty object mask -- check background threshold (--bg)")
    pts = img[ys, xs].astype(np.float32)
    rng = np.random.default_rng(seed)
    if len(pts) > 20000:
        pts = pts[rng.choice(len(pts), 20000, replace=False)]
    cent, _ = kmeans2(pts, k, iter=30, minit="++", seed=seed)
    d = np.linalg.norm(pts[:, None, :] - cent[None, :, :], axis=2)
    lab_pts = np.argmin(d, axis=1)
    labels = np.zeros(obj.shape, np.uint8)
    labels[ys, xs] = lab_pts.astype(np.uint8) + 1
    # per-part cleanup
    for i in range(1, k + 1):
        m = labels == i
        if m.sum() < 0.005 * obj.sum():
            print(f"  WARN part {i}: {m.sum()} px < 0.5% of object -- wrong K or bg?")
        m = ndimage.binary_opening(m, np.ones((3, 3)))
        labels[labels == i] = 0
        labels[m] = i
    return labels


def preview(img, labels, k):
    out = (img * 0.35).astype(np.uint8)
    palette = np.array([[0, 0, 0], [230, 60, 60], [60, 160, 230], [80, 200, 100],
                        [240, 200, 60], [170, 90, 210], [255, 140, 0], [90, 90, 90]],
                       np.uint8)
    for i in range(1, k + 1):
        m = labels == i
        out[m] = (0.45 * palette[i % len(palette)] + 0.55 * img[m]).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix", help="e.g. real/raw/mug  -> mug_front/side/top.jpg")
    ap.add_argument("--parts", type=int, required=True, help="expected part count K")
    ap.add_argument("--bg", type=float, default=60.0, help="background color distance")
    ap.add_argument("--maxside", type=int, default=1024)
    args = ap.parse_args()

    d, base = os.path.dirname(args.prefix), os.path.basename(args.prefix)
    outdir = d or "."
    for view in VIEWS:
        fn = f"{args.prefix}_{view}.jpg"
        if not os.path.exists(fn):
            print(f"skip {view}: {fn} not found")
            continue
        im = Image.open(fn).convert("RGB")
        if max(im.size) > args.maxside:
            im = im.resize((int(im.width * args.maxside / max(im.size)),
                            int(im.height * args.maxside / max(im.size))), Image.LANCZOS)
        img = np.asarray(im)
        obj, bg = object_mask(img, args.bg)
        print(f"{view}: bg={bg.astype(int)} object={obj.mean():.1%}")
        labels = segment(img, obj, args.parts)
        cov = [(labels == i).sum() / max(obj.sum(), 1) for i in range(1, args.parts + 1)]
        print(f"  part coverage: {[f'{c:.1%}' for c in cov]}")
        Image.fromarray(labels).save(os.path.join(outdir, f"{base}_{view}_labels.png"))
        Image.fromarray(preview(img, labels, args.parts)).save(
            os.path.join(outdir, f"{base}_{view}_preview.jpg"), quality=90)
    print(f"\nwrote label maps + previews to {outdir}/")
    print("eyeball the previews first; rerun with --bg/--parts if colors merged or leaked.")


if __name__ == "__main__":
    main()
