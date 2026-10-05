"""Generate all paper figures from raw results (English labels, ready for slides/paper)."""
import csv
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.bbox": "tight"})

# ---------------------------------------------------------------- prop1
rows = list(csv.DictReader(open(os.path.join(HERE, "prop1", "results.csv"))))
num = lambda r, k: (None if r[k] in ("", "None") else float(r[k]))
CONDS1 = ["C0", "C1", "C2", "C3", "C3b", "C4", "C5"]
LABEL1 = {"C0": "C0\nref", "C1": "C1\nfull", "C2": "C2\nno-\nshared",
          "C3": "C3\nno-\nsem", "C3b": "C3b\nrand", "C4": "C4\nhull",
          "C5": "C5\nno-\nassm"}
KINDS1 = ["cage_cup", "chair", "table", "bracket", "spring", "jar_lid", "box_in_box"]


def m1(cond):
    v = [num(r, "m1_tc") for r in rows if r["cond"] == cond and num(r, "m1_tc") is not None]
    return np.mean(v) if v else np.nan


def agg(cond, metric):
    return np.mean([num(r, metric) for r in rows if r["cond"] == cond])


# --- fig 1: main table
fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.6))
metrics = [("m1_tc", "Assembly-graph TC\n(lower better)", "tab:blue"),
           ("m2_tc", "Skeleton-path TC\n(lower better)", "tab:orange"),
           ("chamfer", "Chamfer to C0 ref\n(lower better)", "tab:green")]
for ax, (met, title, c) in zip(axes, metrics):
    vals = []
    for cond in CONDS1:
        v = [num(r, met) for r in rows if r["cond"] == cond and num(r, met) is not None]
        vals.append(np.mean(v) if v else 0.0)
    bars = ax.bar(range(len(CONDS1)), vals, color=c, alpha=.85, width=.68)
    ax.set_xticks(range(len(CONDS1)))
    ax.set_xticklabels([LABEL1[c2] for c2 in CONDS1], fontsize=6.2)
    ax.set_title(title, fontsize=8.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center",
                va="bottom", fontsize=6.5)
fig.suptitle("Proposition 1: seven-condition comparison (35 objects)", y=1.04, fontsize=10)
fig.savefig(os.path.join(FIG, "p1_main.png"))
plt.close(fig)

# --- fig 2: error attribution by kind (grouped bars, color=condition, hatch=FN)
fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.0))
for ax, cond_list, met, ttl in [
        (axes[0], ["C1", "C2", "C3", "C3b", "C5"], "m1", "Assembly-graph errors by kind (M1)"),
        (axes[1], ["C0", "C1", "C4", "C5"], "m2", "Skeleton-path errors by kind (M2)")]:
    w = 0.16
    x = np.arange(len(KINDS1))
    colors = plt.cm.tab10(np.linspace(0, .9, 8))
    for i, cond in enumerate(cond_list):
        fp = [sum(num(r, f"{met}_fp") or 0 for r in rows if r["kind"] == k and r["cond"] == cond)
              for k in KINDS1]
        fn = [sum(num(r, f"{met}_fn") or 0 for r in rows if r["kind"] == k and r["cond"] == cond)
              for k in KINDS1]
        off = (i - (len(cond_list) - 1) / 2) * w
        c = colors[i]
        ax.bar(x + off, fp, w, color=c, label=f"{cond} FP", zorder=3)
        ax.bar(x + off, fn, w, bottom=fp, color=c, hatch="///", edgecolor="white",
               linewidth=.4, label=f"{cond} FN", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(KINDS1, rotation=25, ha="right", fontsize=6.5)
    ax.set_title(ttl, fontsize=8.5)
    ax.legend(fontsize=5.8, ncol=5, frameon=False, columnspacing=.8, handlelength=1.2)
fig.suptitle("Error attribution: closed cavities -> C3/C5 FP (solid); phantoms -> C4 FP; "
             "symmetry -> C2 FN; chair FN ~ skeletonizer baseline (see analyze_path_fn)",
             y=1.08, fontsize=8.5)
fig.savefig(os.path.join(FIG, "p1_attribution.png"))
plt.close(fig)

# --- fig 3: paired dot plots for the three criteria
fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.7))
pairs = [("C4", "chamfer", "Chamfer  C1 vs C4"),
         ("C3", "m1_tc", "Assembly TC  C1 vs C3"),
         ("C2", "m1_tc", "Assembly TC  C1 vs C2")]
for ax, (other, met, ttl) in zip(axes, pairs):
    a = {r["obj"]: num(r, met) for r in rows if r["cond"] == "C1" and num(r, met) is not None}
    b = {r["obj"]: num(r, met) for r in rows if r["cond"] == other and num(r, met) is not None}
    ks = sorted(set(a) & set(b))
    for k in ks:
        ax.plot([0, 1], [b[k], a[k]], color="silver", lw=.6, zorder=1)
    ax.scatter([0] * len(ks), [b[k] for k in ks], s=9, color="tab:orange", zorder=2)
    ax.scatter([1] * len(ks), [a[k] for k in ks], s=9, color="tab:blue", zorder=2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([other, "C1"])
    ax.set_title(ttl, fontsize=8.5)
fig.suptitle("Paired per-object comparison (connected lines = pairs)", y=1.05, fontsize=9.5)
fig.savefig(os.path.join(FIG, "p1_paired.png"))
plt.close(fig)

# ---------------------------------------------------------------- prop2
rows2 = [json.load(open(os.path.join(HERE, "prop2", "runs", f)))
         for f in sorted(os.listdir(os.path.join(HERE, "prop2", "runs"))) if f.endswith(".json")]
# same basis as stats: only objects with all four conditions (fragment runs excluded)
_cs = {}
for r in rows2:
    _cs.setdefault(r["obj"], set()).add(r["cond"])
_complete = {o for o, s in _cs.items() if {"E1", "E2", "E3", "E4"} <= s}
rows2 = [r for r in rows2 if r["obj"] in _complete]
CONDS2 = ["E3", "E2", "E1", "E4"]
LABEL2 = {"E3": "E3\ncontour only", "E2": "E2\n+stroke",
          "E1": "E1\n+prior (full)", "E4": "E4\nprior x5"}
GROUPS = ["solid", "hole", "cavity"]


def gvals(kind, cond, metric):
    return [r[metric] for r in rows2 if r["kind"] == kind and r["cond"] == cond
            and metric in r]


# --- fig 4: four metrics by group x condition
fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.8), constrained_layout=True)
specs = [("solid", "false_cavity", "False-cavity rate (solid group)", "tab:red"),
         ("hole", "hole_retention", "Visible-hole retention (hole group)", "tab:green"),
         ("cavity", "cavity_fill", "Hidden-cavity fill (cavity group)", "tab:purple"),
         None]
for ax, spec in zip(axes.flat, specs):
    if spec is None:
        # dilation with explosion dots across all groups
        ax.set_title("Dilation / over-fill (all objects, log-ish)", fontsize=9)
        for i, cond in enumerate(CONDS2):
            v = [r["dilation"] for r in rows2 if r["cond"] == cond]
            ax.scatter([i] * len(v), np.maximum(v, 1e-4), s=12, color="tab:orange",
                       alpha=.7, zorder=2)
            ax.plot([i, i], [0, np.mean(v)], color="k", lw=1.2, zorder=1)
            ax.plot(i, np.mean(v), "kD", ms=4, zorder=3)
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_xticks(range(4))
        ax.set_xticklabels([LABEL2[c] for c in CONDS2], fontsize=6.5)
        ax.set_ylabel("dilation (x GT box volume)")
        continue
    kind, metric, ttl, c = spec
    for i, cond in enumerate(CONDS2):
        v = gvals(kind, cond, metric)
        if not v:
            continue
        ax.scatter([i] * len(v), v, s=14, color=c, alpha=.55, zorder=2)
        ax.plot([i - .2, i + .2], [np.mean(v)] * 2, color="k", lw=1.6, zorder=3)
    ax.set_xticks(range(4))
    ax.set_xticklabels([LABEL2[c] for c in CONDS2], fontsize=6.5)
    ax.set_title(ttl, fontsize=9)
    ax.set_ylim(-0.05, 1.05)
fig.suptitle("Proposition 2: metrics by condition (black bar = mean, dots = objects)",
             fontsize=10)
fig.savefig(os.path.join(FIG, "p2_main.png"))
plt.close(fig)

# --- fig 5: explosion dose-response
fig, ax = plt.subplots(figsize=(4.2, 2.6))
counts = []
for c in CONDS2:
    counts.append(sum(1 for r in rows2 if r["cond"] == c and r["dilation"] > 1.0))
n_tot = {c: sum(1 for r in rows2 if r["cond"] == c) for c in CONDS2}
bars = ax.bar(range(4), counts, color=["tab:green", "tab:orange", "tab:blue", "tab:red"],
              alpha=.85, width=.6)
for i, c in enumerate(CONDS2):
    ax.text(i, counts[i] + .1, f"{counts[i]}/{n_tot[c]}", ha="center", fontsize=8)
ax.set_xticks(range(4))
ax.set_xticklabels([LABEL2[c] for c in CONDS2], fontsize=6.5)
ax.set_ylabel("exploded runs (dilation > 1)")
ax.set_title("All-solid attractor: dose-response", fontsize=9.5)
fig.savefig(os.path.join(FIG, "p2_explosion.png"))
plt.close(fig)

# --- fig 6: pilot trajectory (from pilot logs)
fig, ax = plt.subplots(figsize=(4.2, 2.6))
iters = [400, 800, 1200]
ax.plot(iters, [0.675, 0.679, 0.677], "o-", color="tab:red", label="E2 (contour only)")
ax.plot(iters, [0.079, 0.036, 0.073], "o-", color="tab:blue", label="E1 (+ solid prior)")
ax.set_xlabel("training iteration")
ax.set_ylabel("false-cavity rate")
ax.set_title("Premise check: shell pathology is stable", fontsize=9.5)
ax.legend(frameon=False, fontsize=8)
fig.savefig(os.path.join(FIG, "p2_trajectory.png"))
plt.close(fig)

print("figures written:", sorted(os.listdir(FIG)))
