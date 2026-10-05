"""Smoke tests for the reusable core/ package -- runs in seconds, CPU only.

    python test_core.py

Each test exercises core/ on the shipped cached data so downstream users know the
importable API produces the same semantics as the reported experiments.
"""
import os
import sys
import numpy as np
import torch
from skimage.morphology import skeletonize

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from core import metrics, assembly, solid_prior

PASSED = 0


def ok(name, cond):
    global PASSED
    assert cond, f"FAILED: {name}"
    PASSED += 1
    print(f"  ok  {name}")


def main():
    sys.path.insert(0, os.path.join(ROOT, "prop1"))
    from run_experiments import load, part_masks

    # ---------- 1. hull reproduces a solid box exactly (broadcast sanity, PITFALLS A1)
    d = np.load(os.path.join(ROOT, "prop2", "data", "solid_00.npz"))
    views = [d["proj"][0], d["proj"][1], d["proj"][2]]
    H = metrics.hull(views)
    inter = (H & d["solid"]).sum()
    union = (H | d["solid"]).sum()
    ok(f"hull(solid box) IoU == 1.0 (got {inter / union:.4f})", abs(inter / union - 1) < 1e-9)

    # ---------- 2. C0 protocol: experiment semantics = paths U GT edges -> 0 FP/0 FN.
    # NOTE (PITFALLS): the FN side is masked by construction; paths-only on the GT
    # skeleton misses some thin contact necks (skeletonizer intrinsic gap, 6/120 edges
    # in this dataset, all chair). See analyze_path_fn.py for the baseline.
    fn = os.path.join(ROOT, "prop1", "data", "chair_00.npz")
    o = load(fn)
    masks = part_masks(o["label"], o["parts"])
    skel = skeletonize(o["occ"])
    lbl = metrics.assign_labels(skel, masks)
    pred = metrics.pair_paths(skel, lbl, len(masks), set())
    fp, fn_, tc = metrics.graph_tc(pred, o["gt"], len(masks))
    print(f"     (info) chair_00 paths-only: {fp} FP, {fn_} FN of {len(o['gt'])} GT edges")
    pred_exp = pred | o["gt"]        # exactly what run_experiments does for C0
    fp2, fn2, tc2 = metrics.graph_tc(pred_exp, o["gt"], len(masks))
    ok(f"C0 as computed in experiment: 0 FP / 0 FN (got {fp2}/{fn2})", tc2 == 0)
    ok("paths-only produces no FP on GT skeleton (FP side of sanity is real)", fp == 0)

    # ---------- 3. spatial candidates have full recall over GT contacts
    cands = assembly.spatial_candidates(masks, dil=2, min_vox=1)
    missing = o["gt"] - cands
    ok(f"spatial candidates cover all GT edges (missing {len(missing)})", not missing)

    # ---------- 4. semantic compat separates seen-contact from seen-noncontact pairs
    data_dir = os.path.join(ROOT, "prop1", "data")
    all_objs = [load(os.path.join(data_dir, f)) for f in sorted(os.listdir(data_dir))
                if f.endswith(".npz")]
    n_tab, e_tab = assembly.build_semantic_table(all_objs)   # LOO needs the full set
    edge_pair = sorted(o["gt"])[0]                      # e.g. (seat, leg) -> contact always
    c_edge = assembly.compat(o, n_tab, e_tab, *edge_pair)
    ok(f"compat(GT edge pair) >= 0.5 (got {c_edge:.3f})", c_edge >= 0.5)
    jar = load(os.path.join(data_dir, "jar_lid_00.npz"))
    c_no = assembly.compat(jar, n_tab, e_tab, 0, 1)     # jar body / floating lid -> never touch
    ok(f"compat(jar body-lid) < 0.5 (got {c_no:.3f})", c_no < 0.5)

    # ---------- 5. chamfer identity + graph cycle rank
    c0 = metrics.chamfer(skel, skel)
    ok(f"chamfer(X, X) == 0 (got {c0})", c0 == 0.0)
    b = metrics.graph_beta1({(0, 1), (1, 2), (0, 2)}, 3)   # triangle -> one cycle
    ok(f"graph_beta1(triangle) == 1 (got {b})", b == 1)

    # ---------- 6. solid prior: math, ramp, and empty-confidence guard
    sdf = torch.tensor([1.0, -2.0, 0.5])
    conf = torch.tensor([True, True, False])
    v_late = solid_prior.solid_prior_loss(sdf, conf, lam=1.0, it=10_000)
    v_early = solid_prior.solid_prior_loss(sdf, conf, lam=1.0, it=0)
    # relu = [1, 0, 0.5]; conf selects first two -> mean = 0.5 (unconfident point excluded)
    ok(f"solid prior: relu over confident points only (got {float(v_late):.3f})",
        abs(float(v_late) - 0.5) < 1e-6)
    ok(f"warmup ramps linearly at it=0 (got {float(v_early):.3f})",
        abs(float(v_early) - 0.5 / solid_prior.DEFAULT_WARMUP) < 1e-6)
    v_none = solid_prior.solid_prior_loss(sdf, torch.tensor([False] * 3))
    ok("empty confidence -> 0", float(v_none) == 0.0)

    # ---------- 7. holm sanity: monotone, in [0, 1]
    adj = metrics.holm([0.01, 0.04, 0.03, 0.5])
    ok(f"holm output in [0,1] & monotone (got {adj})", all(0 <= a <= 1 for a in adj)
        and adj[1] <= adj[2])

    print(f"\nALL {PASSED} core tests passed.")


if __name__ == "__main__":
    main()
