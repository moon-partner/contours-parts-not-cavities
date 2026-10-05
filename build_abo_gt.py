"""GLB -> shared-grid occupancy -> GT contact graph + C0 (the automatic-GT path).

    python build_abo_gt.py --glb real/smoke.glb --meta real/rendered/smoke/smoke_meta.json \
        --masks real/rendered/smoke --name smoke --res 64 --out real/gt

Coordinate contract (validated by the IoU self-check below):
  * World frame = Blender frame (the renderer's frame).
    glTF stores Y-up: p_gltf = (x, z, -y)_blender  =>  contains() queries use the inverse.
  * World cube for the pipeline = the ortho camera frame:
    center = meta.center, span = meta.ortho_scale (same for all 3 views).
  * View mappings (camera: front@-Y looking +Y, side@+X looking -X, top@+Z looking -Z):
      side  image u -> +Y, row -> -Z   => occ.any(x) indexed [y, z], z = R-1-row
      front image u -> +X, row -> -Z   => occ.any(y) indexed [x, z]
      top   image u -> +X, row -> -Y   => occ.any(z) indexed [x, y]
  * Self-check: rendered mask union vs GT occupancy projections must IoU > 0.8;
    if it fails, the axis mapping is wrong -- fix HERE, never fudge the threshold.
"""
import argparse
import json
import os
import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage

RES_DEFAULT = 64


def _norm(s):
    """MUST match render_ortho.py's object rename: strip quotes, non-alnum -> _,
    truncate to 40, strip trailing _."""
    import re
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(s).strip('"'))[:40].strip("_")


def load_parts(glb, meta_names):
    """Return list of (name, mesh) in meta part order (glTF/Y-up coords).

    Name matching is three-way because the renderer renames Blender objects to
    their MATERIAL name after separate-by-material, while trimesh exposes node /
    geometry / material names -- all three are normalized with _norm() and
    compared exactly (with startswith fallback for exporter suffixes)."""
    scene = trimesh.load(glb, force="scene")
    geoms = []  # (candidates:set[str], mesh with scene transform applied)
    for node in scene.graph.nodes_geometry:
        T, gname = scene.graph[node]
        m = scene.geometry[gname].copy()
        m.apply_transform(T)
        mat = getattr(m.visual, "material", None)
        matname = getattr(mat, "name", "") or ""
        cand = {_norm(node), _norm(gname), _norm(matname)} - {""}
        geoms.append((cand, m))
    out, used = [], set()
    for pname in meta_names:
        key = _norm(pname)
        hit = None
        for i, (cand, _) in enumerate(geoms):
            if i in used:
                continue
            if key in cand or any(c.startswith(key) or key.startswith(c) for c in cand):
                hit = i
                break
        if hit is None:
            raise RuntimeError(f"part {pname!r} not found; candidates="
                               f"{[sorted(c) for c, _ in geoms]}")
        used.add(hit)
        out.append((pname, geoms[hit][1]))
    return out


def grid_points(meta, res):
    c = np.asarray(meta["center"], float)
    s = float(meta["ortho_scale"])
    axes = []
    for k in range(3):
        axes.append(c[k] - s / 2 + (np.arange(res) + 0.5) / res * s)
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    return np.stack([X, Y, Z], -1), axes


def voxelize_parts(parts, meta, res):
    """Low-memory backend: fine-pitch surface voxelization -> bin into the shared
    ortho grid -> closing (seal pinholes) -> fill interiors.

    Replaces point-in-mesh contains(), whose ray-triangle candidate arrays blew up
    (3.19 GiB) on real meshes at 128^3.  Peak memory here is O(surface points).
    Meshes are in glTF frame; grid is in Blender frame: pb = (px, -pz, py)
    (inverse of the contains-era query pg = (xb, zb, -yb), validated by smoke IoU).
    """
    c = np.asarray(meta["center"], float)
    s = float(meta["ortho_scale"])
    origin = np.array([c[k] - s / 2 for k in range(3)])
    pitch = s / res
    fine = pitch / 3.0
    masks = []
    for name, m in parts:
        vg = m.voxelized(pitch=fine)            # surface voxels only (subdivide)
        pts = vg.points                          # glTF-frame world coords
        pb = np.column_stack([pts[:, 0], -pts[:, 2], pts[:, 1]])
        idx = np.floor((pb - origin) / pitch).astype(np.int64)
        sel = np.all((idx >= 0) & (idx < res), axis=1)
        occ = np.zeros((res, res, res), bool)
        ii = idx[sel]
        occ[ii[:, 0], ii[:, 1], ii[:, 2]] = True
        occ = ndimage.binary_closing(occ, np.ones((3, 3, 3)), iterations=1)
        occ = ndimage.binary_fill_holes(occ)
        masks.append(occ)
        print(f"  part {name:32s} voxels={int(occ.sum())}")
    return masks


def contact_graph(masks, dil=1, min_vox=2):
    edges = []
    for i in range(len(masks)):
        di = ndimage.binary_dilation(masks[i], ndimage.generate_binary_structure(3, 1),
                                     iterations=dil)
        for j in range(i + 1, len(masks)):
            if np.count_nonzero(di & masks[j]) >= min_vox:
                edges.append((i, j))
    return edges


def block_any(a, res):
    """Downsample a binary image to res x res by BLOCK COVERAGE (any pixel in the
    cell -> cell occupied).  Symmetric with GT surface binning ('any surface point
    in the voxel -> voxel occupied'); point-sampling systematically THINS features
    thinner than a block (lamp pole lost ~40% width this way)."""
    H, W = a.shape
    ry = np.linspace(0, H, res + 1).round().astype(int)
    rx = np.linspace(0, W, res + 1).round().astype(int)
    s = a.astype(np.uint32)
    return np.add.reduceat(np.add.reduceat(s, ry[:-1], axis=0), rx[:-1], axis=1) > 0


def rendered_union(masks_dir, name, view, nparts, res, only=None):
    """Union of per-part alpha masks sampled onto the res x res grid.
    only: iterable of part indices to union (default: all parts).
    Same mapping for all three views (verified by the IoU self-check):
        GT first axis  = camera u axis  -> image columns  (ii)
        GT second axis = camera v axis  -> image rows, row0 = top = max value (flip)
        => g = block_any(acc).T
    """
    import glob
    acc = None
    for i in (range(nparts) if only is None else only):
        f = glob.glob(os.path.join(masks_dir, f"{name}_{view}_{i}_*.png"))[0]
        a = np.asarray(Image.open(f))[:, :, 3] > 127
        acc = a if acc is None else (acc | a)
    # flip rows first: image row0 = TOP = max value, grid index 0 = min value
    return np.flipud(block_any(acc, res)).T


def iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--masks", required=True, help="dir of rendered per-part masks")
    ap.add_argument("--name", required=True)
    ap.add_argument("--res", type=int, default=RES_DEFAULT)
    ap.add_argument("--out", default="real/gt")
    args = ap.parse_args()

    meta = json.load(open(args.meta))
    print(f"parts (render order): {meta['parts']}")
    parts = load_parts(args.glb, meta["parts"])
    print(f"voxelizing on shared ortho grid at res={args.res} (surface+fill backend)...")
    masks = voxelize_parts(parts, meta, args.res)
    occ = np.logical_or.reduce(masks)
    # contact rule scaled to the same WORLD tolerance as prop1 (1 voxel @ res 64)
    gt_edges = contact_graph(masks, dil=max(1, round(0.015625 * args.res)))

    # first-wins disjoint label volume (eval may use masks directly)
    label = np.zeros((args.res,) * 3, np.int8)
    for i, m in enumerate(masks):
        label[m & (label == 0)] = i + 1

    # ---- IoU self-check: rendered union vs GT projections (validates axis mapping)
    print("self-check rendered union vs GT projections:")
    R = args.res
    gt_proj = {"side": occ.any(axis=0), "front": occ.any(axis=1), "top": occ.any(axis=2)}
    worst = 1.0
    for v in ["side", "front", "top"]:
        r = rendered_union(args.masks, args.name, v, len(parts), R)
        val = iou(r, gt_proj[v])
        worst = min(worst, val)
        print(f"  {v:6s} IoU = {val:.3f}  {'OK' if val > 0.8 else 'MISMATCH -> axis mapping?'}")

    os.makedirs(args.out, exist_ok=True)
    fn = os.path.join(args.out, f"{args.name}.npz")
    np.savez_compressed(fn, label=label, occ=occ,
                        part_masks=np.stack(masks).astype(np.uint8),
                        gt_edges=np.array(gt_edges, np.int64).reshape(-1, 2),
                        part_names=np.array(json.dumps(meta["parts"])),
                        glb=np.array(args.glb), res=np.array(args.res),
                        center=np.array(meta["center"]),
                        ortho_scale=np.array(meta["ortho_scale"]))
    print(f"saved {fn}: parts={len(parts)} gt_edges={gt_edges} "
          f"occ_vox={int(occ.sum())} worst_iou={worst:.3f}")
    if worst < 0.8:
        print("SELF-CHECK FAILED -- do NOT trust this GT; fix axis mapping first.")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
