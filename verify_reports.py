"""Independent recomputation of every number quoted in ZONG_REPORT.md / prop*/report.md.
Reads only the raw result files; flags any mismatch against the reported values."""
import csv
import json
import os
import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = []
ok = fail = 0


def chk(label, got, claimed, tol=0.05):
    """tol is relative (5%); for counts compared exactly pass tol=0."""
    global ok, fail
    if claimed is None:
        verdict = "INFO"
    elif tol == 0:
        verdict = "OK" if got == claimed else "MISMATCH"
    else:
        if claimed == 0:
            verdict = "OK" if abs(got) < 1e-9 else "MISMATCH"
        else:
            verdict = "OK" if abs(got - claimed) / abs(claimed) <= tol else "MISMATCH"
    if verdict == "OK":
        ok += 1
    elif verdict == "MISMATCH":
        fail += 1
    OUT.append(f"[{verdict:8s}] {label:58s} got={got!r:>12} claimed={claimed!r}")


# ============================ PROPOSITION 1 ============================
print("=" * 100)
print("PROPOSITION 1 -- recompute from results.csv")
print("=" * 100)
rows = list(csv.DictReader(open(os.path.join(HERE, "prop1", "results.csv"))))
num = lambda r, k: (None if r[k] in ("", "None") else float(r[k]))
print(f"rows = {len(rows)} (expect 35 objects x 7 conds = 245)")
chk("P1 row count", len(rows), 245, tol=0)
objs = sorted({r["obj"] for r in rows})
chk("P1 object count", len(objs), 35, tol=0)
conds = sorted({r["cond"] for r in rows})
print(f"conditions present: {conds}")

# sanity: C0 all zero
c0bad = [r["obj"] for r in rows if r["cond"] == "C0"
         and (num(r, "m1_tc") != 0 or num(r, "m2_tc") != 0 or num(r, "chamfer") != 0)]
chk("P1 C0 sanity violations (claim 0)", len(c0bad), 0, tol=0)

# main table means
def mean(cond, metric, only=lambda r: True):
    v = [num(r, metric) for r in rows if r["cond"] == cond and only(r)
         and num(r, metric) is not None]
    return float(np.mean(v)) if v else float("nan")


print("\n-- main table (claim vs recomputed) --")
p1_main = {
    "C0": dict(m1=0.00, m2=0.00, cd=0.0000),
    "C1": dict(m1=0.00, m2=0.00, cd=0.3610),
    "C2": dict(m1=0.66, m2=0.29, cd=0.5029),
    "C3": dict(m1=0.29, m2=0.29, cd=0.3610),
    "C3b": dict(m1=1.80, m2=0.17, cd=0.3610),
    "C4": dict(m1=None, m2=0.37, cd=0.4589),
    "C5": dict(m1=0.29, m2=0.17, cd=0.3610),
}
for c, want in p1_main.items():
    print(f"  {c:4s} recomputed: M1={mean(c,'m1_tc'):.4f} M2={mean(c,'m2_tc'):.4f} "
          f"CD={mean(c,'chamfer'):.5f}")
    chk(f"P1 {c} M1_TC", round(mean(c, "m1_tc"), 2), want["m1"])
    chk(f"P1 {c} M2_TC", round(mean(c, "m2_tc"), 2), want["m2"])
    chk(f"P1 {c} Chamfer", round(mean(c, "chamfer"), 4), want["cd"])

# criteria
cd1, cd4 = mean("C1", "chamfer"), mean("C4", "chamfer")
red_cd = (cd4 - cd1) / cd4 * 100
print(f"\n-- criteria --\n  CD C4->C1: {cd4:.5f} -> {cd1:.5f} = {red_cd:.2f}%")
chk("P1 crit1 CD reduction %", round(red_cd, 1), 21.3)

tc1, tc3, tc2 = mean("C1", "m1_tc"), mean("C3", "m1_tc"), mean("C2", "m1_tc")
red_tc = (tc3 - tc1) / tc3 * 100 if tc3 else float("nan")
red_ted = (tc2 - tc1) / tc2 * 100 if tc2 else float("nan")
print(f"  M1 C3->C1: {tc3:.4f} -> {tc1:.4f} = {red_tc:.1f}%")
print(f"  M1 C2->C1: {tc2:.4f} -> {tc1:.4f} = {red_ted:.1f}%")
chk("P1 crit2 M1 reduction %", round(red_tc, 1), 100.0)
chk("P1 crit3 M1 reduction %", round(red_ted, 1), 100.0)

# paired tests recomputed
def col(cond, metric):
    return {r["obj"]: num(r, metric) for r in rows if r["cond"] == cond
            and num(r, metric) is not None}

print("\n-- paired Wilcoxon (recomputed) --")
p1_tests = {
    ("C4", "m2_tc"): 0.0114, ("C4", "chamfer"): 0.0001,
    ("C3", "m2_tc"): 0.0016, ("C2", "m2_tc"): 0.0083,
    ("C5", "m2_tc"): 0.0833, ("C3", "m1_tc"): 0.0016,
    ("C3b", "m1_tc"): 0.0000, ("C2", "m1_tc"): 0.0109, ("C5", "m1_tc"): 0.0016,
}
for (other, metric), claimed in p1_tests.items():
    a, b = col("C1", metric), col(other, metric)
    ks = sorted(set(a) & set(b))
    try:
        _, p = wilcoxon([a[k] for k in ks], [b[k] for k in ks])
    except ValueError:
        p = float("nan")
    print(f"  C1 vs {other:4s} [{metric:7s}] n={len(ks):2d} p={p:.4f} (claim {claimed})")
    chk(f"P1 C1vs{other} {metric} p", round(float(p), 4), claimed, tol=0.05)

# error-attribution counts quoted in the report
print("\n-- error attribution (per-kind totals) --")
def fpfn(kind, cond):
    rs = [r for r in rows if r["kind"] == kind and r["cond"] == cond]
    return (sum(num(r, "m2_fp") or 0 for r in rs), sum(num(r, "m2_fn") or 0 for r in rs),
            sum(num(r, "m1_fp") or 0 for r in rs), sum(num(r, "m1_fn") or 0 for r in rs))

for kind, cond, want in [("cage_cup", "C4", "6 FP TOTAL M2"), ("chair", "C4", "7 FN TOTAL M2"),
                         ("box_in_box", "C3", "5 FP M1"), ("jar_lid", "C3", "5 FP M1"),
                         ("chair", "C2", "20 FN M1")]:
    m2fp, m2fn, m1fp, m1fn = fpfn(kind, cond)
    print(f"  {kind:12s} {cond:4s}: M2 {m2fp:.0f}/{m2fn:.0f}  M1 {m1fp:.0f}/{m1fn:.0f}"
          f"   (claim {want})")

# per-object averages claimed in report (totals AND per-object means)
c4_cup = [r for r in rows if r["kind"] == "cage_cup" and r["cond"] == "C4"]
chk("P1 C4 cage_cup M2_FP TOTAL (claim 6)",
    round(sum(num(r, "m2_fp") for r in c4_cup), 1), 6.0, tol=0)
chk("P1 C4 cage_cup M2_FP per-obj (claim 1.2)",
    round(float(np.mean([num(r, "m2_fp") for r in c4_cup])), 1), 1.2, tol=0)
c4_chair = [r for r in rows if r["kind"] == "chair" and r["cond"] == "C4"]
chk("P1 C4 chair M2_FN TOTAL (claim 7)",
    round(sum(num(r, "m2_fn") for r in c4_chair), 1), 7.0, tol=0)
chk("P1 C4 chair M2_FN per-obj (claim 1.4)",
    round(float(np.mean([num(r, "m2_fn") for r in c4_chair])), 1), 1.4, tol=0)
c2_chair = [r for r in rows if r["kind"] == "chair" and r["cond"] == "C2"]
chk("P1 C2 chair M1_FN total",
    round(sum(num(r, "m1_fn") for r in c2_chair), 1), 20.0)
bd = [r for r in rows if r["kind"] == "bracket"]
chk("P1 bracket CD C4", round(np.mean([num(r, "chamfer") for r in bd if r["cond"] == "C4"]), 3), 0.371)
chk("P1 bracket CD C1", round(np.mean([num(r, "chamfer") for r in bd if r["cond"] == "C1"]), 3), 0.048)
ch = [r for r in rows if r["kind"] == "chair"]
chk("P1 chair CD C2", round(np.mean([num(r, "chamfer") for r in ch if r["cond"] == "C2"]), 3), 0.568)
chk("P1 chair CD C1", round(np.mean([num(r, "chamfer") for r in ch if r["cond"] == "C1"]), 3), 0.029)
sp = [r for r in rows if r["kind"] == "spring"]
# C1..C5 must give IDENTICAL per-condition CD vectors; mean 0.2555 (C0=0 by definition)
cond_vecs = {}
for c in ["C1", "C2", "C3", "C3b", "C4", "C5"]:
    cond_vecs[c] = [round(num(r, "chamfer"), 6) for r in sp if r["cond"] == c]
chk("P1 spring C1..C5 identical vectors (claim 1 distinct)",
    len({tuple(v) for v in cond_vecs.values()}), 1, tol=0)
chk("P1 spring C1 mean CD (claim 0.2555)",
    round(float(np.mean(cond_vecs["C1"])), 4), 0.2555)
chk("P1 spring C0 mean CD (claim 0)",
    round(float(np.mean([num(r, "chamfer") for r in sp if r["cond"] == "C0"])), 4), 0.0)
c5_chair = [r for r in rows if r["kind"] == "chair" and r["cond"] == "C5"]
chk("P1 C5 chair M2_FN total", round(sum(num(r, "m2_fn") for r in c5_chair), 1), 6.0)

# hull IoU ranges claimed per kind (CORRECTED values, post broadcast-fix regeneration)
print("\n-- hull IoU ranges (claimed, corrected) --")
man = json.load(open(os.path.join(HERE, "prop1", "data", "manifest.json")))
claims = {"cage_cup": (0.73, 0.79), "chair": (0.86, 0.91), "table": (0.89, 1.00),
          "bracket": (0.83, 0.83), "spring": (0.74, 0.79), "jar_lid": (0.45, 0.45),
          "box_in_box": (0.71, 0.84)}
for k, (clo, chi) in claims.items():
    v = [m["hull_iou"] for m in man if m["kind"] == k]
    lo, hi = min(v), max(v)
    print(f"  {k:12s} recomputed {lo:.3f}-{hi:.3f}   claim {clo}-{chi}")
    good = (clo - 0.011 <= lo <= clo + 0.011) and (chi - 0.011 <= hi <= chi + 0.011)
    if good:
        ok += 1
    else:
        fail += 1
        OUT.append(f"[MISMATCH] P1 hull_iou {k} range got=({lo:.3f},{hi:.3f}) "
                   f"claimed=({clo},{chi})")

# GT edge count consistency
print("\n-- GT edges vs declared (data integrity) --")
import numpy as np
bad = 0
for f in sorted(os.listdir(os.path.join(HERE, "prop1", "data"))):
    if not f.endswith(".npz"):
        continue
    d = np.load(os.path.join(HERE, "prop1", "data", f))
    gt = sorted(map(tuple, d["gt_edges"].tolist()))
    dec = sorted(map(tuple, d["declared_edges"].tolist()))
    if dec and gt != dec:
        bad += 1
        print(f"  MISMATCH {f}: gt={gt} declared={dec}")
chk("P1 declared==GT mismatches", bad, 0, tol=0)
gt_counts = {}
for f in sorted(os.listdir(os.path.join(HERE, "prop1", "data"))):
    if f.endswith(".npz"):
        d = np.load(os.path.join(HERE, "prop1", "data", f))
        gt_counts.setdefault(str(d["kind"]), set()).add(len(d["gt_edges"]))
print(f"  GT edge counts per kind: { {k: sorted(v) for k, v in gt_counts.items()} }")

# ============================ PROPOSITION 2 ============================
print("\n" + "=" * 100)
print("PROPOSITION 2 -- recompute from runs/*.json")
print("=" * 100)
rows2 = [json.load(open(os.path.join(HERE, "prop2", "runs", f)))
         for f in sorted(os.listdir(os.path.join(HERE, "prop2", "runs"))) if f.endswith(".json")]
n_npz = len([f for f in sorted(os.listdir(os.path.join(HERE, "prop2", "data")))
             if f.endswith(".npz")])
print(f"\nruns = {len(rows2)} (npz objects x 4 conds = {n_npz * 4})")
# 主分析口径与 prop2 stats 一致：只用四条件齐全的物体做均值/计数；
# 配对检验（crit1/2/3）按各自指标的交集配对，允许含碎片 run。
_cs = {}
for r in rows2:
    _cs.setdefault(r["obj"], set()).add(r["cond"])
complete = {o for o, s in _cs.items() if {"E1", "E2", "E3", "E4"} <= s}
rows2m = [r for r in rows2 if r["obj"] in complete]
print(f"complete-4-cond objects = {len(complete)} (means/explosion table basis); "
      f"fragment runs = {len(rows2) - 4 * len(complete)}")
chk("P2 complete objects >= 16", len(complete) >= 16, True, tol=0)
chk("P2 original 60 runs all present", len(rows2) >= 60, True, tol=0)


def g(kind, cond, metric):
    v = [r[metric] for r in rows2m if r["kind"] == kind and r["cond"] == cond
         and metric in r]
    return float(np.mean(v)) if v else float("nan")

print("\n-- main table --")
p2_main = [
    ("solid", "E2", "false_cavity", 0.478), ("solid", "E1", "false_cavity", 0.138),
    ("solid", "E3", "false_cavity", 0.650), ("solid", "E4", "false_cavity", 0.099),
    ("solid", "E1", "dilation", 3.073), ("solid", "E2", "dilation", 8.654),
    ("solid", "E1", "sil_iou", 0.789), ("solid", "E2", "sil_iou", 0.669),
    ("hole", "E1", "false_cavity", 0.171), ("hole", "E2", "false_cavity", 0.680),
    ("hole", "E1", "hole_retention", 0.962), ("hole", "E2", "hole_retention", 0.981),
    ("hole", "E4", "hole_retention", 0.756), ("hole", "E1", "sil_iou", 0.910),
    ("hole", "E4", "dilation", 5.29),
    ("cavity", "E1", "cavity_fill", 0.999), ("cavity", "E2", "cavity_fill", 0.599),
    ("cavity", "E3", "cavity_fill", 0.587), ("cavity", "E4", "cavity_fill", 1.000),
    ("cavity", "E1", "false_cavity", 0.278), ("cavity", "E2", "false_cavity", 0.793),
    ("cavity", "E1", "sil_iou", 0.881),
]
for kind, cond, metric, want in p2_main:
    got = g(kind, cond, metric)
    print(f"  {kind:7s} {cond:3s} {metric:15s} got={got:.4f} claim={want}")
    chk(f"P2 {kind} {cond} {metric}", round(got, 3), want, tol=0.02)

print("\n-- criteria --")
# pairwise (intersection on object id) so partial/expanded data never desyncs
e1 = {r["obj"]: r["false_cavity"] for r in rows2
      if r["kind"] == "solid" and r["cond"] == "E1"}
e2 = {r["obj"]: r["false_cavity"] for r in rows2
      if r["kind"] == "solid" and r["cond"] == "E2"}
ks = sorted(set(e1) & set(e2))
a1, a2 = [e1[k] for k in ks], [e2[k] for k in ks]
red = (np.mean(a2) - np.mean(a1)) / np.mean(a2) * 100
_, p1 = wilcoxon(a1, a2)
print(f"  crit1 false_cavity E2->E1: {np.mean(a2):.4f} -> {np.mean(a1):.4f} = {red:.1f}%  p={p1:.4f} n={len(ks)}")
chk("P2 crit1 reduction %", round(red, 1), 75.5)
chk("P2 crit1 p", round(float(p1), 4), 0.0391, tol=0.05)

# SECONDARY pooled analysis (quoted in report)
pa, pb = {}, {}
for gk in ["solid", "hole", "cavity"]:
    pa.update({r["obj"]: r["false_cavity"] for r in rows2
               if r["kind"] == gk and r["cond"] == "E1"})
    pb.update({r["obj"]: r["false_cavity"] for r in rows2
               if r["kind"] == gk and r["cond"] == "E2"})
pk = sorted(set(pa) & set(pb))
aa = np.array([pa[k] for k in pk]); bb = np.array([pb[k] for k in pk])
red_p = (bb.mean() - aa.mean()) / bb.mean() * 100
_, p_p = wilcoxon(aa, bb)
print(f"  pooled crit1: reduction={red_p:.1f}% p={p_p:.5f} n={len(pk)} "
      f"fwd={int((aa < bb).sum())}/{len(pk)}")
chk("P2 pooled crit1 reduction %", round(red_p, 1), 72.9, tol=0.02)
chk("P2 pooled crit1 p", round(float(p_p), 5), 0.00008, tol=0.15)
chk("P2 pooled crit1 fwd", int((aa < bb).sum()), 15, tol=0)

d1, d4 = {}, {}
for k in ["solid", "hole", "cavity"]:
    d1.update({r["obj"]: r["dilation"] for r in rows2 if r["kind"] == k and r["cond"] == "E1"})
    d4.update({r["obj"]: r["dilation"] for r in rows2 if r["kind"] == k and r["cond"] == "E4"})
ks = sorted(set(d1) & set(d4))
red4 = (np.mean([d4[k] for k in ks]) - np.mean([d1[k] for k in ks])) / np.mean([d4[k] for k in ks]) * 100
_, p4 = wilcoxon([d1[k] for k in ks], [d4[k] for k in ks])
print(f"  crit2 dilation E4->E1: {np.mean([d4[k] for k in ks]):.4f} -> {np.mean([d1[k] for k in ks]):.4f} = {red4:.1f}%  p={p4:.4f} n={len(ks)}")
chk("P2 crit2 reduction %", round(red4, 1), 81.6)
chk("P2 crit2 p", round(float(p4), 4), 0.0977, tol=0.05)
_, p4_1s = wilcoxon([d1[k] for k in ks], [d4[k] for k in ks], alternative="less")
print(f"  crit2 one-sided (pre-registered direction) p={p4_1s:.4f}")
chk("P2 crit2 one-sided p", round(float(p4_1s), 4), 0.0488, tol=0.05)

s1, s2 = {}, {}
for k in ["solid", "hole", "cavity"]:
    s1.update({r["obj"]: r["sil_iou"] for r in rows2 if r["kind"] == k and r["cond"] == "E1"})
    s2.update({r["obj"]: r["sil_iou"] for r in rows2 if r["kind"] == k and r["cond"] == "E2"})
ks = sorted(set(s1) & set(s2))
diff = abs(np.mean([s1[k] for k in ks]) - np.mean([s2[k] for k in ks])) * 100
_, p3 = wilcoxon([s1[k] for k in ks], [s2[k] for k in ks])
print(f"  crit3 sil_iou: E1={np.mean([s1[k] for k in ks]):.4f} E2={np.mean([s2[k] for k in ks]):.4f} diff={diff:.1f}%  p={p3:.4f} n={len(ks)}")
chk("P2 crit3 diff %", round(diff, 1), 1.8)
chk("P2 crit3 p", round(float(p3), 4), 0.0385, tol=0.05)

print("\n-- explosion counts, COMPLETE objects only (report basis) --")
for c, want in [("E3", 0), ("E2", 2), ("E1", 1), ("E4", 4)]:
    rs = [r for r in rows2m if r["cond"] == c]
    n_ex = sum(1 for r in rs if r["dilation"] > 1.0)
    print(f"  {c}: {n_ex}/{len(rs)} exploded (dilation>1)")
    chk(f"P2 {c} explosion count (complete)", n_ex, want, tol=0)
frag_expl = [r["obj"] + "_" + r["cond"] for r in rows2
             if r["obj"] not in complete and r["dilation"] > 1.0]
print(f"  fragment-run explosions (excluded from main tables, disclosed): {frag_expl}")

print("\n-- post-hoc: excluding exploded runs, solid false_cavity --")
for c in ["E1", "E2", "E3", "E4"]:
    v = [r["false_cavity"] for r in rows2
         if r["kind"] == "solid" and r["cond"] == c and r["dilation"] <= 1.0]
    print(f"  {c}: n={len(v)} mean={np.mean(v):.4f}")
v1 = [r["false_cavity"] for r in rows2 if r["kind"] == "solid" and r["cond"] == "E1" and r["dilation"] <= 1]
v2 = [r["false_cavity"] for r in rows2 if r["kind"] == "solid" and r["cond"] == "E2" and r["dilation"] <= 1]
if v1 and v2:
    r_ph = (np.mean(v2) - np.mean(v1)) / np.mean(v2) * 100
    print(f"  post-hoc reduction = {r_ph:.1f}% (claim 75%)")
    chk("P2 post-hoc reduction %", round(r_ph, 1), 75.0, tol=0.05)
    # post-hoc wilcoxon on the 5 clean pairs
    a = {r["obj"]: r["false_cavity"] for r in rows2 if r["kind"] == "solid" and r["cond"] == "E1" and r["dilation"] <= 1}
    b = {r["obj"]: r["false_cavity"] for r in rows2 if r["kind"] == "solid" and r["cond"] == "E2" and r["dilation"] <= 1}
    ks = sorted(set(a) & set(b))
    if len(ks) >= 2:
        _, pp = wilcoxon([a[k] for k in ks], [b[k] for k in ks])
        print(f"  post-hoc paired Wilcoxon n={len(ks)} p={pp:.4f}")

print("\n-- hull IoU of prop2 data (claim solid/hole 1.0, cavity 0.615-0.657) --")
ct = {}
for f in sorted(os.listdir(os.path.join(HERE, "prop2", "data"))):
    if not f.endswith(".npz"):
        continue
    d = np.load(os.path.join(HERE, "prop2", "data", f))
    solid, hull, kind = d["solid"], d["hull"], str(d["kind"])
    iou = (hull & solid).sum() / (hull | solid).sum()
    ct.setdefault(kind, []).append(round(float(iou), 3))
for k, v in ct.items():
    print(f"  {k:7s} n={len(v)} hull_iou={sorted(set(v))}")
chk("P2 solid hull_iou all 1.0",
    all(abs(x - 1.0) < 1e-9 for x in ct.get("solid", [])), True, tol=0)
chk("P2 hole hull_iou all 1.0",
    all(abs(x - 1.0) < 1e-9 for x in ct.get("hole", [])), True, tol=0)
cav = ct.get("cavity", [])
chk("P2 cavity hull_iou in 0.615-0.657",
    (round(min(cav), 3), round(max(cav), 3)), (0.615, 0.657), tol=0)

# ============================ SUMMARY ============================
print("\n" + "=" * 100)
print(f"VERIFICATION SUMMARY: {ok} checks OK, {fail} MISMATCH")
print("=" * 100)
if fail:
    print("\nMISMATCHES:")
    for line in OUT:
        if "MISMATCH" in line:
            print(" ", line)
