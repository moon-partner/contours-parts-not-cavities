"""Build the merged workshop paper as a Word document.

    python paper/build_docx.py   ->  paper/workshop_paper.docx

Content source: paper/draft.md (v0.1) rendered with figures and tables inline.
Word count target ~4000 + 5 figures + 4 tables (~8 pages at default styles).
"""
import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
import sys
OUT = (sys.argv[1] if len(sys.argv) > 1
       else os.path.join(HERE, "workshop_paper.docx"))


def fig(doc, path, caption, width=6.2):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(os.path.join(ROOT, path), width=Inches(width))
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = c.add_run(caption)
    r.italic = True
    r.font.size = Pt(8.5)


def table(doc, rows, caption=None, widths=None):
    if caption:
        c = doc.add_paragraph()
        r = c.add_run(caption)
        r.italic = True
        r.font.size = Pt(8.5)
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = t.cell(i, j)
            cell.text = str(val)
            for par in cell.paragraphs:
                for run in par.runs:
                    run.font.size = Pt(8.5)
                    if i == 0:
                        run.bold = True
    doc.add_paragraph()
    return t


def h(doc, text, level):
    doc.add_heading(text, level=level)


def para(doc, text, bold_prefix=None, size=10):
    p = doc.add_paragraph()
    if bold_prefix:
        r = p.add_run(bold_prefix)
        r.bold = True
        r.font.size = Pt(size)
    r = p.add_run(text)
    r.font.size = Pt(size)
    return p


def bullet(doc, text):
    p = doc.add_paragraph(style="List Bullet")
    p.add_run(text).font.size = Pt(10)


doc = Document()
st = doc.styles["Normal"]
st.font.name = "Calibri"
st.font.size = Pt(10)
doc.core_properties.title = ("Contours Tell You Parts, Not Cavities: an Identifiability "
                             "Boundary for Orthographic Three-View Reconstruction, with "
                             "Semantic Part Assembly and Confidence-Weighted Solid Priors")
doc.core_properties.author = "TODO(author)"

# ---------------------------------------------------------------- title
tp = doc.add_heading("", level=0)
tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = tp.add_run("Contours Tell You Parts, Not Cavities: an Identifiability Boundary "
                 "for Orthographic Three-View Reconstruction, with Semantic Part "
                 "Assembly and Confidence-Weighted Solid Priors")
run.font.size = Pt(16)
sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
sr = sub.add_run("TODO(author, affiliation, contact)  ·  Submitted to a non-archival "
                 "3D reconstruction/generation workshop (venue TBD)")
sr.italic = True
sr.font.size = Pt(9)
sr.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

# ---------------------------------------------------------------- abstract
h(doc, "Abstract", 1)
para(doc,
     "Reconstructing 3D shape from three orthographic silhouettes is a deceptively "
     "under-constrained problem: what is observable, and what is fundamentally not? We "
     "make the boundary explicit. (1) We prove that for any object whose exterior is "
     "axis-visible, the visual-hull error set equals exactly the union of enclosed "
     "cavities — closed cavities carry zero information in contours, for every method, "
     "per-part or joint. (2) Within what is observable, we study two mechanisms. "
     "Semantic part assembly: per-part silhouettes in a shared tri-plane frame, joined "
     "by spatial adjacency gated through a leave-one-out semantic compatibility table, "
     "induce the part-contact graph and hence a skeleton topology (35 synthetic objects "
     "x 7 conditions: exact contact recovery; -100% graph errors vs. no-semantics and "
     "no-shared-frame ablations, family-wise p<=0.011). Confidence-weighted solid "
     "priors: a hull-gated solid-volume regularizer with linear warmup fixes the "
     "false-cavity pathology of contour-only SDF fitting (-75.5%, p=0.039 primary; "
     "-72.9%, p=8e-5 pooled) without harming visible holes or silhouettes, while "
     "revealing an all-solid attractor explosion as a measurable dose-dependent failure "
     "mode. (3) Both findings transfer zero-shot to real geometry: three ABO product "
     "models, split and voxelized fully automatically, yield exact contact recovery and "
     "1-2% world-normalized Chamfer, while the joint-hull baseline fails on thin "
     "structures. All code, data, per-run records, and a 98-assertion independent "
     "verification suite are released for direct reuse.")

# ---------------------------------------------------------------- 1 intro
h(doc, "1. Introduction", 1)
para(doc,
     "Three orthographic silhouettes are the cheapest 3D signal there is — and one of "
     "the most misunderstood. Two communities meet at this input: contour-driven "
     "implicit reconstruction (solid/inflation priors guiding an SDF) and part-aware "
     "reconstruction (assemble semantic pieces, then skeletonize). Both implicitly "
     "assume the input can carry their claims. This paper states exactly where that "
     "assumption dies, and builds both methods on the surviving side of it:")
bullet(doc, "C1 — Boundary: Theorem 1. Under axis-visibility of the exterior, "
            "visual-hull error = enclosed cavities. Closed cavities are "
            "information-theoretically invisible to all contour methods — including "
            "per-part contours and priors.")
bullet(doc, "C2 — Method A (assembly): shared-frame per-part reconstruction + "
            "semantics-gated assembly recovers contact topology exactly, with a "
            "five-way ablation attributing each ingredient.")
bullet(doc, "C3 — Method B (solid prior): a confidence-weighted solid prior repairs "
            "the false-cavity drift of contour-only SDF fitting, with a documented, "
            "dose-dependent failure mode and honest boundary behavior.")
bullet(doc, "C4 — Reproducibility: a self-contained package in which a 98-assertion "
            "verification suite re-derives every quoted number from raw per-run "
            "records.")

# ---------------------------------------------------------------- 2 related
h(doc, "2. Related work", 1)
para(doc,
     "SketchFormer3D (Expert Systems with Applications, 2025) injects pretrained SDF "
     "networks as geometric priors into sketch-to-3D diffusion; it supports the "
     "'priors help' direction, while our contribution is the confidence-weighted "
     "mechanism, the dose-response failure analysis, and the identifiability bound. "
     "REVIVE 3D (CVPR 2026) builds an inflation prior from foreground silhouettes — "
     "the closest relative of our solid prior; our over-fill/dilation metrics and "
     "explosion observations transfer directly to that setting. REVIVE3D (ICCVW, "
     "retrieval-based volumetric infusion) is a non-inflation alternative. "
     "TODO(author): complete author lists and page numbers.",
     bold_prefix="SDF / diffusion priors. ")
para(doc,
     "Implicit reconstruction classics: IGR / SIREN / NeuS-style contour- or "
     "distance-supervised SDF fitting. Visual hull: Laurentini (PAMI 1994), the object "
     "Theorem 1 is about. Part-level 3D: PartNet / PartNet-Mobility part graphs and "
     "neural assembly pipelines — our contact-graph target, and the line along which "
     "an open-set semantic component would matter (Sec. 8). TODO(author): exact "
     "citations (7-10 entries).")

# ---------------------------------------------------------------- 3 problem
h(doc, "3. Problem statement and input contract", 1)
para(doc,
     "Input: three orthographic silhouettes along the X/Y/Z axes at known camera "
     "geometry; optionally part-resolved silhouettes (one mask per part — from colored "
     "rendering in synthesis, from automatic material-split rendering in ABO). Output: "
     "(Task A) a part-contact graph and skeleton curve graph; (Task B) a watertight "
     "SDF. We assume the exterior is axis-visible: every point of the exterior "
     "component of the complement has an axis ray to infinity within free space (true "
     "for all objects in our datasets; verified per-object by the released checker).")

# ---------------------------------------------------------------- 4 theorem
h(doc, "4. The identifiability boundary", 1)
para(doc,
     "For view axis a, the silhouette says 'occupied' on the line L_a(x) through x "
     "iff L_a(x) intersects O. The hull is H = {x : every axis line through x "
     "intersects O}.", bold_prefix="Definition. ")
tp = doc.add_paragraph()
r = tp.add_run("Theorem 1 (boundary). ")
r.bold = True
tp.add_run("If the exterior of O is axis-visible, then H ∖ O equals the union of the "
           "bounded components of the complement of O — i.e. the hull error is exactly "
           "the union of enclosed cavities.")
para(doc,
     "Proof (sketch). (⊇) A point in a bounded component has every axis ray blocked "
     "by O (the ray must exit the bounded component), so it satisfies H while not "
     "being in O. (⊆) If x in H ∖ O were in the exterior component, axis-visibility "
     "gives an axis ray from x to infinity inside free space — a ray disjoint from O, "
     "contradicting x in H. ∎")
para(doc,
     "(i) Enclosed cavities are invisible to every contour-based method: per-part "
     "contours are still contours; priors impose an assumption, not information; no "
     "method recovers cavity geometry without side information. (ii) For our objects, "
     "the hull error equals the cavity set — verified per-object on all 35 synthetic "
     "shapes (released).", bold_prefix="Corollaries. ")
para(doc,
     "Empirical cross-validation. The boundary appears independently in both tasks: "
     "synthetic cavity objects (floating-lid jar, box-in-box) defeat all conditions "
     "including per-part hulls (CD=1.0, structures merge), and in Task B the solid "
     "prior fills hidden cavities 99.9% while being unable to avoid it (Sec. 6). We "
     "report these as boundary confirmations, not failures.")

# ---------------------------------------------------------------- 5 task A
h(doc, "5. Task A — semantic part assembly in a shared frame", 1)
h(doc, "5.1 Method", 2)
para(doc,
     "Per-part silhouettes → per-part hulls in one shared frame → candidate contacts "
     "by dilated spatial adjacency → keep pairs whose semantic compatibility passes: "
     "a leave-one-out table P(contact | part-name pair) with Laplace smoothing, "
     "trained only on other objects → predicted contact graph G-hat; skeleton = "
     "Lee-thinning of the per-part hull union, augmented with G-hat's edges.")
h(doc, "5.2 Metrics (formal)", 2)
para(doc,
     "Let O* be ground truth with labeled parts and contact graph G*. "
     "M1 (assembly-graph TC) = symmetric difference between predicted and GT edge "
     "sets = insertions + deletions; on labeled same-cardinality nodes this equals "
     "the normalized graph edit distance (relabel cost 0), i.e. our TED proxy, "
     "reported as FP/FN. "
     "M2 (skeleton-path TC): label each skeleton voxel by its nearest GT part within "
     "tau = 0.0625·span (else ⊥ = phantom); pair (i,j) is path-connected iff the "
     "skeleton contains a path between an i-labeled and a j-labeled voxel using only "
     "labels {i, j, ⊥}; M2 = symmetric difference of (paths ∪ G-hat) vs G*. Third-part "
     "labels block paths (anti-shortcut); ⊥ allows them, so M2 must always be read "
     "with the phantom ratio and Chamfer. "
     "C0 protocol: the reference skeleton is the same thinning algorithm run on GT "
     "voxels; C0's false positives are structurally zero. Disclosure: on perfect "
     "voxels the path rule still misses 6/120 thin contact necks — the skeletonizer "
     "contact-gap baseline, against which every method's misses must be read (it "
     "absorbed 6 of the joint-hull baseline's 7 chair misses). Supporting metrics: "
     "Chamfer to C0 in world units (d_vox·pitch/span), graph cycle-rank difference, "
     "component-count difference, phantom ratio.")

h(doc, "5.3 Synthetic experiment (35 objects × 7 conditions)", 2)
fig(doc, "figures/p1_main.png",
    "Figure 1: Seven-condition comparison on 35 synthetic objects (lower is better). "
    "C0 reference is zero by construction; C1 is exact on both graph metrics.")
fig(doc, "figures/p1_attribution.png",
    "Figure 2: Error attribution by object kind. Solid bars = false positives, "
    "hatched = false negatives. No-semantics false links land exactly on the "
    "enclosed-cavity candidates; no-shared-frame errors concentrate on symmetric "
    "parts; chair-level FN additionally carries the skeletonizer baseline.")
table(doc, [
    ["Criterion", "Measured", "Threshold", "Family-wise p", "Verdict"],
    ["Chamfer vs joint hull (C4)", "-21.3%", ">=15%", "0.0004", "pass"],
    ["M1 vs no-semantics (C3)", "-100%", ">=30%", "0.0048", "pass"],
    ["M1 vs no-shared-frame (C2)", "-100%", ">=20%", "0.0109", "pass"],
], caption="Table 1: Pre-registered criteria for Task A (paired Wilcoxon, "
           "Holm by metric family).")
para(doc,
     "Attribution: all no-semantics false links land exactly on enclosed-cavity "
     "candidates (10/10) — semantics contributes negative knowledge (which pairs "
     "never touch); no-shared-frame errors concentrate on symmetric parts (chair 20 "
     "misses, cage 3); random assembly is significantly worse throughout "
     "(family-wise p=2e-4); single-part objects are identical across conditions "
     "(implementation control). Disclosed: Holm by metric family (per-family 4; "
     "correction across all 12 makes the C2 comparison p=0.0545); n=35, 7 categories "
     "seen by the leave-one-out table; part masks follow the input contract "
     "(Sec. 8).")

# ---------------------------------------------------------------- 6 task B
h(doc, "6. Task B — confidence-weighted solid priors for contour SDFs", 1)
h(doc, "6.1 Method", 2)
para(doc,
     "SIREN 3×128 with orthographic ray rendering (64 samples): loss = "
     "stroke-weighted silhouette BCE + 0.1·Eikonal + λ·Ramp(t)·mean(ReLU(sdf) | conf), "
     "where conf = membership in the three-view hull and Ramp is a linear warmup over "
     "400 steps. Conditions: E1 (full, λ=1), E2 (no prior), E3 (no stroke, no prior), "
     "E4 (λ=5).")
h(doc, "6.2 Synthetic experiment (16 complete objects)", 2)
fig(doc, "figures/p2_main.png",
    "Figure 3: Proposition 2 metrics by condition (black bar = mean, dots = objects). "
    "The solid prior removes false cavities (top-left) while preserving visible holes "
    "(top-right); hidden cavities are filled by construction (bottom-left).")
fig(doc, "figures/p2_explosion.png",
    "Figure 4: The all-solid attractor: exploded runs per condition (complete "
    "objects). Dose-dependent tail risk; even the no-prior baseline exploded once "
    "among fragment runs.")
table(doc, [
    ["Criterion", "Measured", "Threshold", "p", "Verdict"],
    ["False-cavity E2→E1 (primary, n=8)", "-75.5%", ">=20%", "0.0391", "pass"],
    ["Same, pooled all groups (n=17)", "-72.9% (15/17 same sign)", ">=20%",
     "8e-5", "pass (secondary)"],
    ["Over-fill E4→E1 (n=16)", "-81.6%", ">=40%",
     "0.0977 two-sided / 0.0488 pre-registered one-sided", "pass"],
    ["Silhouette IoU |E1−E2| (n=17)", "1.8% (E1 better)", "<=5%", "0.0385",
     "pass (pairing-sensitive: 7.1% at n=15; disclosed)"],
], caption="Table 2: Pre-registered criteria for Task B. All three pass; analysis "
           "choices (pooled secondary, one-sided pre-registered direction, "
           "pairing-set sensitivity) disclosed.")
para(doc,
     "Premise: contour-only fitting develops a stable shell pathology — 65% "
     "false-cavity rate, flat across training (trajectory released). Failure mode: "
     "the all-solid attractor appears as a dose-dependent tail — 0/16 (E3), 2/16 "
     "(E2), 1/16 (E1), 4/16 (E4) on complete objects; even E3 exploded once among "
     "fragment runs, so the tail is not exclusive to any term; warmup is necessary "
     "but not sufficient. Boundary behavior: hidden cavities are filled 99.9% by the "
     "prior while the no-prior baseline merely drifts (~59%) — neither is correct "
     "because the input contains zero cavity information (Theorem 1). Visible holes "
     "are preserved (0.962 vs 0.981 without the prior).")

# ---------------------------------------------------------------- 7 real
h(doc, "7. Transfer to real geometry (ABO, fully automatic)", 1)
para(doc,
     "Pipeline: public single-file ABO download → automatic material-split "
     "orthographic rendering (headless Blender) → surface-voxelized mesh GT with a "
     "three-view IoU self-check (0.89–0.995) → the same M1/M2/CD harness at 128³.")
fig(doc, "real/abo/real_results.png",
    "Figure 5: Real-geometry results. Left to right: colored composite render, C0 "
    "mesh-GT skeleton, C1 prediction (exact on all three), C4 baseline — the floor "
    "lamp's joint hull breaks at the thin pole (edges=[], M2 miss).", width=6.4)
table(doc, [
    ["Object", "C1 M1", "C1 M2", "C1 Chamfer", "C4 M2", "C4 Chamfer"],
    ["Fabric chair", "0", "0/0", "0.0182", "0/0", "0.0177"],
    ["Stool", "0", "0/0", "0.0125", "0/0", "0.0147"],
    ["Floor lamp", "0", "0/0", "0.0104", "0/1 miss", "0.0274"],
], caption="Table 3: Real ABO results (world-normalized Chamfer).")
para(doc,
     "The semantic table is learned leave-one-out from synthetic data only; real "
     "objects participate in nothing. The joint-hull baseline's real failure mode "
     "(thin-structure break) differs from its synthetic one (phantoms/cavities) — "
     "same root cause, complementary evidence. Honest disclosures: n=3, two parts per "
     "object; on thick-contact real objects the no-assembly baseline ties C1 — the "
     "value density of assembly edges depends on contact geometry (thin necks in "
     "synthesis: 6/50 edges rescued; thick contacts: paths suffice).")

# ---------------------------------------------------------------- 8 limitations
h(doc, "8. Limitations", 1)
bullet(doc, "Segmentation is part of the input contract, not solved: synthesis uses "
            "rendered part colors, ABO uses material splits; mask-noise robustness "
            "is untested.")
bullet(doc, "Closed-set semantics: the compatibility table falls back to 0.5 on "
            "unseen name pairs; an embedding variant is future work.")
bullet(doc, "Pilot-scale samples (35 / 16 / 3); Task B's primary analysis is n=8 "
            "with one-sided and pairing-sensitivity disclosures (Sec. 6.2); single "
            "seeds; box-and-bar geometry dominates synthesis.")
bullet(doc, "Theorem 1 assumes axis-visibility; a released checker flags objects "
            "that violate it.")
bullet(doc, "Prior strength λ and warmup are untuned beyond λ ∈ {1, 5}.")

# ---------------------------------------------------------------- 9 repro
h(doc, "9. Reproducibility", 1)
para(doc,
     "One command regenerates every synthetic table from cached records in about "
     "three minutes; a verification script re-derives all quoted numbers from raw "
     "per-run JSON/CSV via 98 independent assertions (0 mismatches); 12 API smoke "
     "tests cover the reusable core; a PITFALLS document records eleven measured "
     "implementation traps (broadcast axis bugs, unit conventions, memory blowups, "
     "sampling asymmetries). Licenses: MIT (code); ABO models CC-BY 4.0. "
     "TODO(author): attach anonymized repository link upon submission.")

# ---------------------------------------------------------------- 10 conclusion
h(doc, "10. Conclusion", 1)
para(doc,
     "Contours can tell you how parts connect — reliably, once semantics and shared "
     "coordinates are in place — but they can never tell you what is sealed inside. "
     "We proved the boundary, built both methods on the observable side of it, "
     "measured their failure modes as carefully as their successes, and released the "
     "machinery to check every number.")

# ---------------------------------------------------------------- refs
h(doc, "References", 1)
refs = [
    "[1] SketchFormer3D: Generating 3D shapes from sketches with implicit SDF priors "
    "via diffusion models. Expert Systems with Applications, 2025. "
    "TODO(author): authors, volume, pages.",
    "[2] REVIVE 3D: Refinement via Encoded Voluminous Inflated prior for Volume "
    "Enhancement. CVPR, 2026. TODO(author): authors, pages.",
    "[3] REVIVE3D: REtrieval-based Volumetric Infusion via Visual Editing. ICCV "
    "Workshop (submitted). TODO(author): confirm status.",
    "[4] Laurentini, A. The visual hull concept for silhouette-based image "
    "understanding. IEEE TPAMI, 1994. TODO(author): pages.",
    "[5] SIT / IGR / SIREN / NeuS / PartNet / PartNet-Mobility: TODO(author) — add "
    "exact citations (7–10 entries) before submission.",
]
for rr in refs:
    p = doc.add_paragraph(rr)
    p.paragraph_format.space_after = Pt(2)
    for run in p.runs:
        run.font.size = Pt(9)

doc.save(OUT)
print("wrote", OUT)
