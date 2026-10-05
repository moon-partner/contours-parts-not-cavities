"""
Proposition 1 -- seven-condition comparison on the synthetic dataset.

Conditions
  C0  skeletonize(GT voxels)          reference + skeletonizer noise floor
  C1  full method: per-part hull -> shared frame -> spatial(dil2) AND semantic -> skeleton + edges
  C2  no shared frame: part pose recovered by silhouette-containment search (bbox crop/re-place)
  C3  spatial adjacency only (no semantics)
  C3b spatial adjacency AND random keep p=0.5
  C4  joint three-view hard intersection (visual hull) -> skeleton
  C5  union of per-part hulls -> skeleton, no assembly edges (attribution gap)

Metrics
  M1 assembly-graph TC  = FP + FN predicted contact edges vs GT      (prop1 criterion: TED/TED-proxy)
  M2 skeleton-path TC   = FP + FN pair connectivity on skeleton       (criterion: TC)
  Chamfer(skel, C0) normalized by grid span;  d_beta1 vs C0;  component-count diff; phantom ratio.
Stats: paired Wilcoxon + Holm, median paired diff, bootstrap 95% CI.  C0 must score 0 (sanity).
"""
import os
import json
import csv
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.signal import fftconvolve
from scipy.stats import wilcoxon
from skimage.morphology import skeletonize

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
S26 = ndimage.generate_binary_structure(3, 3)
S6 = ndimage.generate_binary_structure(3, 1)
SPAN = 2.0
RNG_MASTER = np.random.default_rng(20260214)


# ------------------------------------------------------------------ helpers

def hull(views):
    """views = [proj_x(y,z), proj_y(x,z), proj_z(x,y)] -> occupancy (x,y,z)."""
    px, py, pz = [np.asarray(v, bool) for v in views]
    # px[y,z] -> (1,y,z); py[x,z] -> (x,1,z); pz[x,y] -> (x,y,1)
    return px[None, :, :] & py[:, None, :] & pz[:, :, None]


def load(fn):
    d = np.load(fn, allow_pickle=True)
    return dict(
        label=d["label"], occ=d["occ"], proj=d["proj"],
        gt={tuple(map(int, e)) for e in d["gt_edges"].tolist()},
        parts=json.loads(d["parts_json"].item()),
        pprojs=json.loads(d["part_projs_json"].item()),
        hull_iou=float(d["hull_iou"]), kind=str(d["kind"]), seed=int(d["seed"]),
        name=os.path.basename(fn),
    )


def shift_mask(M, off):
    out = np.zeros_like(M)
    src, dst = [], []
    for ax, o in enumerate(off):
        n = M.shape[ax]
        if o >= 0:
            dst.append(slice(o, n)); src.append(slice(0, n - o))
        else:
            dst.append(slice(0, n + o)); src.append(slice(-o, n))
    out[tuple(dst)] = M[tuple(src)]
    return out


def part_masks(label, parts):
    return [label == p["lid"] for p in parts]


def spatial_candidates(masks, dil=2, min_vox=1):
    cands = set()
    dils = [ndimage.binary_dilation(m, S6, iterations=dil) for m in masks]
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            if np.count_nonzero(dils[i] & masks[j]) >= min_vox:
                cands.add((i, j))
    return cands


def build_semantic_table(objs):
    n, e = {}, {}
    for o in objs:
        names = [p["name"] for p in o["parts"]]
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                key = tuple(sorted((names[i], names[j])))
                n[key] = n.get(key, 0) + 1
        for (i, j) in o["gt"]:
            key = tuple(sorted((names[i], names[j])))
            e[key] = e.get(key, 0) + 1
    return n, e


def compat(o, n_tab, e_tab, i, j):
    names = [p["name"] for p in o["parts"]]
    key = tuple(sorted((names[i], names[j])))
    n_o = 1
    e_o = 1 if (i, j) in o["gt"] or (j, i) in o["gt"] else 0
    n_loo = n_tab.get(key, 0) - n_o
    e_loo = e_tab.get(key, 0) - e_o
    return (e_loo + 1.0) / (n_loo + 2.0)


def contained_positions(crop, sil):
    """Boolean map of placements of `crop` fully inside `sil` (valid positions)."""
    h, w = crop.shape
    outside = (~sil).astype(float)
    if crop.size == 0:
        return np.zeros((0, 0), bool)
    corr = fftconvolve(outside, crop[::-1, ::-1].astype(float), mode="valid")
    return corr < 0.5


def c2_offsets(o, masks, rng):
    """Recover each part's bbox-min placement by silhouette containment in 3 views."""
    sil = [np.asarray(o["proj"][k], bool) for k in range(3)]   # (y,z),(x,z),(x,y)
    offs = []
    for m in masks:
        v = [m.any(axis=0), m.any(axis=1), m.any(axis=2)]      # (y,z),(x,z),(x,y)
        cont = []
        for k in range(3):
            mm, ss = v[k], sil[k]
            nz = np.argwhere(mm)
            i0, j0 = nz.min(axis=0)
            crop = mm[i0:nz[:, 0].max() + 1, j0:nz[:, 1].max() + 1]
            cp = contained_positions(crop, ss)
            cont.append((cp, i0, j0))
        # cont[0]: (y,z) -> Cy ; cont[1]: (x,z) -> Cx ; cont[2]: (x,y) -> Cz
        cxy, ix0, iy0 = cont[2]
        cxz, jx0, jz0 = cont[1]
        cyz, ky0, kz0 = cont[0]
        true_min = np.argwhere(m).min(axis=0)                  # (x0,y0,z0)
        pairs = np.argwhere(cxy)
        if len(pairs) > 400:
            pairs = pairs[rng.choice(len(pairs), 400, replace=False)]
        cands = []
        for (px_, py_) in pairs:
            zs = np.where(cxz[px_] & cyz[py_])[0]
            if len(zs):
                cands.append((px_, py_, int(rng.choice(zs))))
        if not cands:
            offs.append((0, 0, 0))
            continue
        c = cands[rng.integers(len(cands))]
        # new bbox mins: x from Cz/Cx pair, y from Cz, z from Cx/Cy
        new_min = np.array([c[0], c[1], c[2]])
        offs.append(tuple(int(v) for v in (new_min - true_min)))
    return offs


def assign_labels(skel, masks, maxd=4):
    """Nearest GT part within maxd voxels; 0 = phantom."""
    dstack = np.stack([ndimage.distance_transform_edt(~m) for m in masks])
    idx = np.argmin(dstack, axis=0)
    dmin = np.min(dstack, axis=0)
    lbl = np.where(skel & (dmin <= maxd), idx + 1, 0).astype(np.int16)
    return lbl


def pair_paths(skel, lbl, nparts, extra_edges):
    """M2: predicted direct connectivity per part pair."""
    pred = set()
    for i in range(nparts):
        for j in range(i + 1, nparts):
            sub = skel & ((lbl == i + 1) | (lbl == j + 1) | (lbl == 0))
            if not sub.any():
                connected = False
            else:
                cc, _ = ndimage.label(sub, S26)
                ci = np.unique(cc[lbl == i + 1])
                cj = np.unique(cc[lbl == j + 1])
                common = np.intersect1d(ci, cj)
                connected = bool(common.any() and common[0] != 0)
            if connected or ((i, j) in extra_edges):
                pred.add((i, j))
    return pred


def graph_tc(pred, gt, n):
    all_pairs = {(i, j) for i in range(n) for j in range(i + 1, n)}
    fp = len(pred - gt)
    fn = len(gt - pred)
    return fp, fn, fp + fn


def beta1(skel):
    v = int(skel.sum())
    if v == 0:
        return 0, 0, 0
    e = count_edges26(skel)
    _, c = ndimage.label(skel, S26)
    return e - v + int(c), v, int(c)


def count_edges26(skel):
    """Number of undirected adjacency pairs in the 26-connected voxel graph."""
    total = 0
    for dx in (0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if (dx, dy, dz) <= (0, 0, 0):
                    continue
                a = skel
                sx = slice(max(0, -dx), skel.shape[0] - max(0, dx))
                sy = slice(max(0, -dy), skel.shape[1] - max(0, dy))
                sz = slice(max(0, -dz), skel.shape[2] - max(0, dz))
                tx = slice(max(0, dx), skel.shape[0] - max(0, -dx))
                ty = slice(max(0, dy), skel.shape[1] - max(0, -dy))
                tz = slice(max(0, dz), skel.shape[2] - max(0, -dz))
                total += int((a[sx, sy, sz] & a[tx, ty, tz]).sum())
    return total


def chamfer(a, b, nmax=3000, seed=0):
    """Bidirectional mean nearest distance, normalized by grid span (2.0)."""
    ra, rb = np.argwhere(a), np.argwhere(b)
    if len(ra) == 0 or len(rb) == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    if len(ra) > nmax:
        ra = ra[rng.choice(len(ra), nmax, replace=False)]
    if len(rb) > nmax:
        rb = rb[rng.choice(len(rb), nmax, replace=False)]
    ta, tb = cKDTree(ra), cKDTree(rb)
    d1 = tb.query(ra, k=1)[0].mean()
    d2 = ta.query(rb, k=1)[0].mean()
    return float((d1 + d2) / 2.0 / SPAN)


def run_condition(cond, o, masks, n_tab, e_tab):
    """Returns dict(skel, extra_edges, m1_pred)."""
    gt = o["gt"]
    n = len(masks)
    cond_id = {"C1": 11, "C2": 22, "C3": 33, "C3b": 34, "C5": 55}.get(cond, 99)
    rng = np.random.default_rng(o["seed"] * 7 + cond_id)

    if cond == "C0":
        skel = skeletonize(o["occ"])
        return dict(skel=skel, edges=set(gt), m1=set(gt))

    if cond == "C4":
        skel = skeletonize(hull([o["proj"][0], o["proj"][1], o["proj"][2]]))
        return dict(skel=skel, edges=set(), m1=None)

    # part-based reconstructions
    phulls = [hull([np.asarray(o["pprojs"][p["name"]][k]) for k in range(3)])
              for p in o["parts"]]

    if cond == "C2":
        offs = c2_offsets(o, masks, rng)
        vmasks = [shift_mask(h, off) for h, off in zip(phulls, offs)]
        skel = skeletonize(np.logical_or.reduce(vmasks))
        cands = spatial_candidates(vmasks, dil=2, min_vox=1)
        edges = {p for p in cands
                 if compat(o, n_tab, e_tab, *p) >= 0.5}
        m1 = edges
    elif cond == "C5":
        vmasks = phulls
        skel = skeletonize(np.logical_or.reduce(vmasks))
        edges = set()
        m1 = spatial_candidates(vmasks, dil=1, min_vox=2)   # touch rule
    else:
        vmasks = phulls
        skel = skeletonize(np.logical_or.reduce(vmasks))
        cands = spatial_candidates(vmasks, dil=2, min_vox=1)
        if cond == "C1":
            edges = {p for p in cands if compat(o, n_tab, e_tab, *p) >= 0.5}
        elif cond == "C3":
            edges = set(cands)
        elif cond == "C3b":
            edges = {p for p in sorted(cands) if rng.random() < 0.5}
        else:
            raise ValueError(cond)
        m1 = edges
    return dict(skel=skel, edges=edges, m1=m1)


def graph_beta1(edges, n):
    """Cycle rank of the part-contact graph (E - V + C)."""
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for (i, j) in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    comps = len({find(i) for i in range(n)})
    return len(edges) - n + comps


def evaluate(res, o, masks, ref):
    skel = res["skel"]
    lbl = assign_labels(skel, masks)
    n = len(masks)
    gt = o["gt"]
    pred_paths = pair_paths(skel, lbl, n, res["edges"])
    m2_fp, m2_fn, m2_tc = graph_tc(pred_paths, gt, n)
    if res["m1"] is None:
        m1_fp = m1_fn = m1_tc = None
    else:
        m1_fp, m1_fn, m1_tc = graph_tc(res["m1"], gt, n)
    b1, v, c = beta1(skel)
    n = len(masks)
    return dict(
        m1_fp=m1_fp, m1_fn=m1_fn, m1_tc=m1_tc,
        m2_fp=m2_fp, m2_fn=m2_fn, m2_tc=m2_tc,
        chamfer=chamfer(skel, ref["skel"], seed=abs(o["seed"]) % 10000),
        d_beta1=b1 - ref["b1"],
        d_cycle=graph_beta1(pred_paths, n) - graph_beta1(gt, n),
        d_ncomp=c - ref["ncomp"],
        phantom=float(((lbl == 0) & skel).sum()) / max(int(skel.sum()), 1),
        n_skel=int(skel.sum()),
    )


def holm(ps):
    m = len(ps)
    order = np.argsort(ps)
    adj = np.zeros(m)
    prev = 0.0
    for r, idx in enumerate(order):
        val = (m - r) * ps[idx]
        prev = max(prev, val)
        adj[idx] = min(prev, 1.0)
    return adj


def paired_stats(a, b, iters=2000, seed=7):
    """a,b arrays (paired); returns w, p, median diff, CI."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    if np.allclose(d, 0):
        return 0.0, 1.0, 0.0, (0.0, 0.0)
    try:
        w, p = wilcoxon(a, b, alternative="two-sided")
    except ValueError:
        w, p = 0.0, 1.0
    rng = np.random.default_rng(seed)
    meds = []
    n = len(d)
    for _ in range(iters):
        s = d[rng.integers(0, n, n)]
        meds.append(np.median(s))
    ci = (float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5)))
    return float(w), float(p), float(np.median(d)), ci


def main():
    files = sorted(f for f in os.listdir(DATA) if f.endswith(".npz"))
    objs = [load(os.path.join(DATA, f)) for f in files]
    n_tab, e_tab = build_semantic_table(objs)
    conds = ["C0", "C1", "C2", "C3", "C3b", "C4", "C5"]
    rows = []
    for oi, o in enumerate(objs):
        masks = part_masks(o["label"], o["parts"])
        ref = dict(skel=skeletonize(o["occ"]))
        ref["b1"], _, ref["ncomp"] = beta1(ref["skel"])
        for cond in conds:
            res = run_condition(cond, o, masks, n_tab, e_tab)
            ev = evaluate(res, o, masks, ref)
            ev.update(cond=cond, obj=o["name"], kind=o["kind"], hull_iou=o["hull_iou"])
            rows.append(ev)
        print(f"[{oi+1}/{len(objs)}] {o['name']} done")

    # ---- sanity: C0 must be all zeros
    c0 = [r for r in rows if r["cond"] == "C0"]
    bad = [r for r in c0 if r["m2_tc"] != 0 or r["m1_tc"] != 0]
    print(f"\nSANITY C0 nonzero: {len(bad)} (must be 0, else metric bug)")
    for r in bad:
        print("   ", r["obj"], "m1", r["m1_tc"], "m2", r["m2_tc"])

    # ---- write CSV
    keys = ["obj", "kind", "hull_iou", "cond", "m1_fp", "m1_fn", "m1_tc",
            "m2_fp", "m2_fn", "m2_tc", "chamfer", "d_beta1", "d_cycle", "d_ncomp",
            "phantom", "n_skel"]
    with open(os.path.join(HERE, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # ---- summary table
    print("\n=== mean over 35 objects ===")
    hdr = f"{'cond':5s} {'M1_TC':>7s} {'M2_TC':>7s} {'CD':>7s} {'dB1':>6s} {'phntm':>6s}"
    print(hdr)
    summary = {}
    for cond in conds:
        rs = [r for r in rows if r["cond"] == cond]
        m1 = [r["m1_tc"] for r in rs if r["m1_tc"] is not None]
        vals = dict(
            m1=float(np.mean(m1)) if m1 else None,
            m2=float(np.mean([r["m2_tc"] for r in rs])),
            cd=float(np.mean([r["chamfer"] for r in rs])),
            db1=float(np.mean([r["d_beta1"] for r in rs])),
            ph=float(np.mean([r["phantom"] for r in rs])),
        )
        summary[cond] = vals
        print(f"{cond:5s} {vals['m1'] if vals['m1'] is not None else float('nan'):7.2f} "
              f"{vals['m2']:7.2f} {vals['cd']:7.4f} {vals['db1']:6.2f} {vals['ph']:6.3f}")

    # ---- paired tests: C1 vs others
    def col(cond, metric):
        return {r["obj"]: r[metric] for r in rows if r["cond"] == cond}

    tests = []
    for other in ["C4", "C3", "C2", "C5"]:
        for metric in ["m2_tc", "chamfer"]:
            a, b = col("C1", metric), col(other, metric)
            keys_ = sorted(set(a) & set(b))
            w, p, md, ci = paired_stats([a[k] for k in keys_], [b[k] for k in keys_])
            tests.append(dict(vs=other, metric=metric, p=p, med_diff=md, ci=ci, w=w))
    for other in ["C3", "C3b", "C2", "C5"]:
        a, b = col("C1", "m1_tc"), col(other, "m1_tc")
        keys_ = sorted(set(a) & set(b))
        if all(v is not None for v in list(a.values()) + list(b.values())):
            w, p, md, ci = paired_stats([a[k] for k in keys_], [b[k] for k in keys_])
            tests.append(dict(vs=other, metric="m1_tc", p=p, med_diff=md, ci=ci, w=w))
    pvals = [t["p"] for t in tests]
    adj = holm(pvals)
    for t, pa in zip(tests, adj):
        t["p_holm"] = float(pa)

    print("\n=== paired C1 vs X (Wilcoxon, Holm) ===")
    for t in tests:
        print(f"C1 vs {t['vs']:4s} [{t['metric']:7s}] med_diff={t['med_diff']:8.3f} "
              f"CI=({t['ci'][0]:.3f},{t['ci'][1]:.3f}) p={t['p']:.4f} holm={t['p_holm']:.4f}")

    # ---- original criteria mapping
    cd1 = summary["C1"]["cd"]; cd4 = summary["C4"]["cd"]
    tc1 = summary["C1"]["m1"]; tc3 = summary["C3"]["m1"]
    tc1b = summary["C1"]["m1"]; tc2 = summary["C2"]["m1"]
    red_cd = (cd4 - cd1) / cd4 * 100 if cd4 else 0
    red_tc = (tc3 - tc1) / tc3 * 100 if tc3 else float("nan")
    red_ted = (tc2 - tc1b) / tc2 * 100 if tc2 else float("nan")
    print(f"\ncrit1 CD(C4)->C1 reduction: {red_cd:.1f}% (need >=15%)")
    print(f"crit2 M1 TC(C3)->C1 reduction: {red_tc:.1f}% (need >=30%)")
    print(f"crit3 M1 TC(C2)->C1 reduction: {red_ted:.1f}% (need >=20%)")

    with open(os.path.join(HERE, "summary.json"), "w") as f:
        json.dump(dict(summary=summary, tests=tests,
                       crit=dict(red_cd=red_cd, red_tc=red_tc, red_ted=red_ted)), f, indent=1)
    print("\nwrote results.csv, summary.json")


if __name__ == "__main__":
    main()
