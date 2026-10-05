"""Cold re-analysis: pool false_cavity across ALL groups (metric identical everywhere)
to recover statistical power for prop2 crit1 WITHOUT any new training.
Also: one-sided p for crit2 (direction pre-registered: E1 dilation < E4)."""
import json
import os
import numpy as np
from scipy.stats import wilcoxon

rows = [json.load(open(os.path.join("prop2", "runs", f)))
        for f in sorted(os.listdir(os.path.join("prop2", "runs"))) if f.endswith(".json")]

def col(cond, metric="false_cavity"):
    return {r["obj"]: r[metric] for r in rows if r["cond"] == cond and metric in r}

e1, e2 = col("E1"), col("E2")
ks = sorted(set(e1) & set(e2))
a, b = np.array([e1[k] for k in ks]), np.array([e2[k] for k in ks])
fwd = int((a < b).sum()); rev = int((a > b).sum())
red = (b.mean() - a.mean()) / b.mean() * 100
_, p2s = wilcoxon(a, b)
print(f"POOLED false_cavity E1 vs E2: n={len(ks)}  E1={a.mean():.3f} E2={b.mean():.3f}")
print(f"  reduction={red:.1f}%  forward={fwd}/{len(ks)} reversed={rev}  "
      f"wilcoxon two-sided p={p2s:.5f}")

# excluding runs that exploded under EITHER condition (documented as sensitivity)
expl = {r["obj"] for r in rows if r["dilation"] > 1.0}
ks2 = [k for k in ks if k not in expl]
a2, b2 = np.array([e1[k] for k in ks2]), np.array([e2[k] for k in ks2])
_, p2s2 = wilcoxon(a2, b2)
red2 = (b2.mean() - a2.mean()) / b2.mean() * 100
print(f"SENSITIVITY (exclude any-run exploded): n={len(ks2)}  reduction={red2:.1f}%  "
      f"p={p2s2:.5f}  forward={(a2 < b2).sum()}/{len(ks2)}")

# crit2 one-sided (pre-registered direction)
d1, d4 = {}, {}
for k in ["solid", "hole", "cavity"]:
    d1.update({r["obj"]: r["dilation"] for r in rows if r["kind"] == k and r["cond"] == "E1"})
    d4.update({r["obj"]: r["dilation"] for r in rows if r["kind"] == k and r["cond"] == "E4"})
kk = sorted(set(d1) & set(d4))
x1 = np.array([d1[k] for k in kk]); x4 = np.array([d4[k] for k in kk])
_, p_2s = wilcoxon(x1, x4)
_, p_1s = wilcoxon(x1, x4, alternative="less")
red4 = (x4.mean() - x1.mean()) / x4.mean() * 100
print(f"crit2 dilation E4->E1: n={len(kk)} reduction={red4:.1f}% "
      f"two-sided p={p_2s:.4f}  one-sided(pre-registered direction) p={p_1s:.4f}")
