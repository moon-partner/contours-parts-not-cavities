"""Real-geometry experiment: ABO per-part masks -> assembly -> compare with mesh GT.

    python run_real.py --name B0728NW8FP
    python run_real.py --all

Inputs (per object):
  real/abo/rendered/{name}/   per-part alpha masks + meta (from render_ortho.py)
  real/gt/{name}.npz          mesh-derived GT (from build_abo_gt.py): part_masks,
                              occ, gt_edges, center, ortho_scale, res=128

Conditions (same operational definitions as prop1, tolerances scaled to grid res
as FRACTIONS of the span so world tolerance matches prop1):
  C0  skeletonize(GT occ)                        reference + paths-only baseline
  C1  per-part hull -> spatial(dil~prop1 2@64) & semantic(LOO prop1 table) -> edges
  C5  union of per-part hulls, no assembly edges (+ touch-rule M1)
  C4  joint three-view hard intersection -> skeleton, paths only

Semantics: visual part-name mapping (documented inspector assignment) lets the
prop1 leave-one-out table apply zero-shot to real objects -- the "synthetic ->
real semantic transfer" claim.  The real object is NOT in the table (no leak).
"""
import argparse
import glob
import json
import os
import sys
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "prop1"))
from core import metrics, assembly                      # noqa: E402
from run_experiments import load as load_prop1          # noqa: E402
from build_abo_gt import rendered_union                 # noqa: E402

# ---- inspector-assigned semantic mapping (documented; visual judgment by us) ----
SEM_MAP = {
    "B0728NW8FP": ["seat", "leg0"],        # fabric chair: upholstery(body+back) | legs
    "B07HSDX2CQ": ["seat", "leg0"],        # stool: shell(seat+back)          | frame
    "B07DBHLX1W": ["ring_top", "post0"],   # floor lamp: shade                | pole
}
VIEWS = ["side", "front", "top"]           # -> proj (y,z), (x,z), (x,y)


def view_grids(rd, name, nparts, res, only=None):
    return {v: rendered_union(rd, name, v, nparts, res, only=only) for v in VIEWS}


def semantic_compat(key_counts, pair_names):
    """(e+1)/(n+2) WITHOUT self-subtraction: the real object is not in the table."""
    key = tuple(sorted(pair_names))
    n, e = key_counts.get(key, (0, 0))
    return (e + 1.0) / (n + 2.0)


def run(name):
    rd = os.path.join(HERE, "real", "abo", "rendered", name)
    meta = json.load(open(os.path.join(rd, f"{name}_meta.json")))
    gt = np.load(os.path.join(HERE, "real", "gt", f"{name}.npz"), allow_pickle=True)
    occ = gt["occ"].astype(bool)
    pmasks = [gt["part_masks"][i].astype(bool) for i in range(gt["part_masks"].shape[0])]
    gt_edges = {tuple(map(int, e)) for e in gt["gt_edges"].tolist()}
    res = int(gt["res"])
    span = float(gt["ortho_scale"])
    n = len(pmasks)
    pitch = span / res
    # tolerance fractions copied from prop1 (voxels/res)
    maxd = max(1, round(0.0625 * res))          # label gate: prop1 4/64
    dil_cand = max(1, round(0.03125 * res))     # candidates: prop1 2/64
    dil_touch = max(1, round(0.015625 * res))   # touch rule: prop1 1/64

    # ---- reconstruction inputs (from RENDERED masks only) ----
    per_part = [view_grids(rd, name, n, res, only=[i]) for i in range(n)]
    joint = view_grids(rd, name, n, res, only=None)
    hulls = [metrics.hull([pp[VIEWS[0]], pp[VIEWS[1]], pp[VIEWS[2]]]) for pp in per_part]
    jhull = metrics.hull([joint[VIEWS[0]], joint[VIEWS[1]], joint[VIEWS[2]]])
    union = np.logical_or.reduce(hulls)

    # ---- reference (GT side) ----
    c0 = skeletonize(occ)
    lbl0 = metrics.assign_labels(c0, pmasks, maxd=maxd)
    c0_paths = metrics.pair_paths(c0, lbl0, n, set())
    c0_fn = len(gt_edges - c0_paths)

    # ---- semantics: prop1 LOO-style table (real object absent = zero-shot) ----
    p1dir = os.path.join(HERE, "prop1", "data")
    p1objs = [load_prop1(os.path.join(p1dir, f))
              for f in sorted(os.listdir(p1dir)) if f.endswith(".npz")]
    n_tab, e_tab = assembly.build_semantic_table(p1objs)
    key_counts = {k: (n_tab.get(k, 0), e_tab.get(k, 0)) for k in set(n_tab) | set(e_tab)}
    sem_names = SEM_MAP.get(name, [f"part{i}" for i in range(n)])

    # ---- conditions ----
    cands = assembly.spatial_candidates(hulls, dil=dil_cand, min_vox=1)
    c1_edges = {p for p in cands
                if semantic_compat(key_counts, (sem_names[p[0]], sem_names[p[1]])) >= 0.5}
    touch = assembly.spatial_candidates(hulls, dil=dil_touch, min_vox=2)
    skel_c1 = skeletonize(union)
    skel_c4 = skeletonize(jhull)

    results = {"object": name, "res": res, "span": span, "parts": n,
               "part_names": meta["parts"],
               "sem_map": sem_names, "gt_edges": sorted(gt_edges),
               "c0_paths_only_fn": c0_fn, "conditions": {}}
    print(f"\n{'=' * 74}\n{name}  res={res} span={span:.3f} parts={n} "
          f"names={meta['parts']}\n  sem_map={sem_names}  gt_edges={sorted(gt_edges)}")
    print(f"  C0 paths-only baseline: {c0_fn} FN of {len(gt_edges)} GT edges")

    conds = {
        "C1": dict(skel=skel_c1, edges=c1_edges, m1=c1_edges),
        "C5": dict(skel=skel_c1, edges=set(), m1=touch),
        "C4": dict(skel=skel_c4, edges=set(), m1=None),
    }
    for cname, c in conds.items():
        lbl = metrics.assign_labels(c["skel"], pmasks, maxd=maxd)
        pred_paths = metrics.pair_paths(c["skel"], lbl, n, c["edges"])
        m2_fp, m2_fn, m2_tc = metrics.graph_tc(pred_paths, gt_edges, n)
        if c["m1"] is None:
            m1 = None
        else:
            m1 = metrics.graph_tc(c["m1"], gt_edges, n)
        # chamfer: metrics.chamfer returns mean NN in VOXEL units / 2.0 (prop1 SPAN);
        # world-normalized CD = d_vox * pitch / span = d_vox / res (dimensionless)
        cd = metrics.chamfer(c["skel"], c0, seed=7) * 2.0 / res
        entry = dict(m1=m1, m2_fp=m2_fp, m2_fn=m2_fn, m2_tc=m2_tc,
                     pred_edges=sorted(pred_paths), cd=round(cd, 4))
        results["conditions"][cname] = entry
        m1s = (f"M1={m1[2]}" if m1 is not None else "M1=n/a")
        print(f"  {cname}: {m1s}  M2 FP/FN={m2_fp}/{m2_fn}  CD={cd:.4f}  "
              f"pred_edges={sorted(pred_paths)}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    gt_files = sorted(glob.glob(os.path.join(HERE, "real", "gt", "*.npz")))
    # only objects that also have a render meta (excludes stray gt like smoke)
    names = [os.path.basename(f)[:-4] for f in gt_files
             if os.path.exists(os.path.join(HERE, "real", "abo", "rendered",
                                            os.path.basename(f)[:-4],
                                            f"{os.path.basename(f)[:-4]}_meta.json"))]
    if args.name:
        names = [args.name]
    elif not args.all:
        names = [x for x in names if x in SEM_MAP] or names
    out = [run(nm) for nm in names]
    fn = os.path.join(HERE, "real", "results.json")
    json.dump(out, open(fn, "w"), indent=1)
    print(f"\nwrote {fn}")


if __name__ == "__main__":
    main()
