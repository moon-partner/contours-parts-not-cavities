# Contours Tell You Parts, Not Cavities: an Identifiability Boundary for Orthographic Three-View Reconstruction, with Semantic Part Assembly and Confidence-Weighted Solid Priors

> Workshop draft v0.1 (2026-10-05; date stamp corrected from a mistaken 2026-02-14)
> Target: non-archival workshop ladder AAAI-27 (~Nov–Dec 2026) → ICLR-27 → CVPR-27.
> NeurIPS 2026 and 3DV 2027 submission deadlines have already passed (Sep 5 / Aug 28, 2026) as of writing.
> TODO(author): name, affiliation, contact
> Figure files are relative to repo root; regenerate with `python make_figures.py && python make_real_figures.py`.

## Abstract

Reconstructing 3D shape from three orthographic silhouettes is a deceptively under-constrained problem: what is observable, and what is fundamentally not? We make the boundary explicit. **(1)** We prove that for any object whose exterior is axis-visible, the visual-hull error set equals exactly the union of enclosed cavities — closed cavities carry *zero* information in contours, for every method, per-part or joint. **(2)** Within what *is* observable, we study two mechanisms. *Semantic part assembly*: per-part silhouettes in a shared tri-plane frame, joined by spatial adjacency gated through a leave-one-out semantic compatibility table, induce the part-contact graph and hence a skeleton topology (35 synthetic objects × 7 conditions: exact contact recovery, −100% graph errors vs. no-semantics and no-shared-frame ablations, p≤0.011 family-wise). *Confidence-weighted solid priors*: a hull-gated solid-volume regularizer with linear warmup fixes the false-cavity pathology of contour-only SDF fitting (−75.5%, p=0.039 primary; −72.9%, p=8e-5 pooled) without harming visible holes or silhouettes, while revealing an "all-solid attractor" explosion as a measurable dose-dependent failure mode. **(3)** Both findings transfer zero-shot to real geometry: three ABO product models, split and voxelized fully automatically, yield exact contact recovery and 1–2% world-normalized Chamfer, while the joint-hull baseline fails on thin structures. All code, data, per-run records, and a 98-assertion independent verification suite are released for direct reuse.

## 1. Introduction

Three orthographic silhouettes are the cheapest 3D signal there is — and one of the most misunderstood. Two communities meet at this input: contour-driven implicit reconstruction (solid/inflation priors guiding an SDF) and part-aware reconstruction (assemble semantic pieces, then skeletonize). Both implicitly assume the input can carry their claims. This paper's first contribution is to state exactly where that assumption dies:

- **Theorem 1 (boundary).** Under axis-visibility of the exterior, visual-hull error = enclosed cavities. Closed cavities are information-theoretically invisible to *all* contour methods — including per-part contours, priors, and learned embeddings.
- **Method A (assembly, observable regime).** Within the observable regime we show that shared-frame per-part reconstruction + semantics-gated assembly recovers contact topology exactly, and we attribute each ingredient with a five-way ablation.
- **Method B (solid prior, observable regime).** A confidence-weighted solid prior repairs the false-cavity drift of contour-only SDF fitting, with a documented, dose-dependent failure mode.

Contributions: (C1) identifiability theorem + proof + cross-validated empirical confirmation on both tasks; (C2) assembly method with exact attribution (M1/M2 metrics, C0 reference protocol); (C3) solid-prior loss with warmup, explosion dose-response, and boundary-honest behavior; (C4) a fully reproducible package (98-assertion verification over raw records).

## 2. Related work

- **SDF/diffusion priors.** *SketchFormer3D: Generating 3D shapes from sketches with implicit SDF priors via diffusion models*, Expert Systems with Applications, 2025 — pretrained SDF networks as geometric prior; supports the "priors help" direction; our contribution is the confidence-weighted *mechanism*, the dose-response failure analysis, and the identifiability bound.
- **Inflation/volume priors.** *REVIVE 3D: Refinement via Encoded Voluminous Inflated prior for Volume Enhancement*, CVPR 2026 (accepted) — silhouette inflation as volume prior, the closest relative to our solid prior; our over-fill/dilation metrics and explosion observations transfer directly to their setting. Also *REVIVE3D: REtrieval-based Volumetric Infusion via Visual Editing*, ICCVW (retrieval-based alternative).
  TODO(author): full author lists, page numbers (verified existence 2026-10-05; format per venue).
- **Implicit reconstruction classics.** IGR / SIREN / NeuS-style contour-or-distance-supervised SDF fitting — cite exact works TODO(author).
- **Visual hull.** Laurentini 1994 and the silhouette-carving line — the object Theorem 1 is about.
- **Part-level 3D.** PartNet/PartNet-Mobility-style part graphs and neural assembly pipelines — our assembly target (contact graph) and where an open-set semantic component is expected to matter (see §7).

## 3. Problem statement and input contract

Input: three orthographic silhouettes along the X/Y/Z axes at known camera geometry; optionally *part-resolved* silhouettes (one mask per part — from colored rendering in synthesis, from material-split rendering in ABO). Output: (Task A) a part-contact graph and skeleton curve graph; (Task B) a watertight SDF. We assume the exterior is **axis-visible**: every point of the exterior component of the complement has an axis ray to infinity within free space (true for all objects in our datasets; verified per-object in our released checker).

## 4. The identifiability boundary

**Definition.** For view axis a, silhouette S_a says "occupied" on the line L_a(x) through x iff L_a(x) ∩ O ≠ ∅. The hull is H = {x : ∀a, L_a(x) ∩ O ≠ ∅}.

**Theorem 1.** If the exterior of O is axis-visible, then H \ O = ⋃ (bounded components of R³ \ O), i.e. the hull error is exactly the union of enclosed cavities.

*Proof (sketch).* (⊇) A point in a bounded component has every axis ray blocked by O (the ray must exit the bounded component), so it satisfies H's definition while not being in O. (⊆) If x ∈ H \ O were in the exterior component, axis-visibility gives an axis ray from x to infinity inside free space — a ray disjoint from O, contradicting x ∈ H. ∎

**Corollaries.** (i) Enclosed cavities are invisible to every contour-based method: per-part contours are contours; priors impose an *assumption*, not information; no method can recover cavity geometry without side information. (ii) For axis-aligned exterior-visible objects, hull error is exactly the cavity set — verified empirically on all 35 synthetic objects (per-object hull IoU released).

**Empirical cross-validation.** The boundary shows up independently in both tasks: synthetic cavity objects (floating-lid jar, box-in-box) defeat *all* conditions including per-part hulls (CD=1.0, structure merged), and in Task B the solid prior fills hidden cavities 99.9% while being unable to avoid it (§6.2). We report these as *boundary confirmations*, not failures.

## 5. Task A — semantic part assembly in a shared frame

### 5.1 Method

Per-part silhouettes → per-part hulls in one shared tri-plane/world frame → candidate contacts by dilated spatial adjacency → keep pairs whose **semantic compatibility** passes: a leave-one-out table P(contact | name-pair) with Laplace smoothing, trained only on *other* objects → predicted contact graph Ĝ; skeleton = Lee-thinning of the per-part hull union, augmented with Ĝ's edges.

### 5.2 Metrics (formal)

Let O\* = ground truth with labeled parts {P_i}, contact graph G\* = (V, E\*).

- **M1 (assembly-graph TC)** = |Ê Δ E\*| = insertions + deletions. On labeled same-cardinality nodes this *equals* the normalized graph edit distance (relabel cost 0), i.e. our TED-proxy; reported as FP/FN separately.
- **M2 (skeleton-path TC).** For skeleton voxel set S, label each voxel by nearest P_i within τ = 0.0625·span (else ⊥ = phantom). Pair (i,j) is *path-connected* iff S contains a path between an i-labeled and a j-labeled voxel using only labels {i, j, ⊥}. M2 = |(paths ∪ Ĝ) Δ E\*|. Third-party labels block paths (anti-shortcut); ⊥ allows them (phantoms can bridge — so M2 must always be read with the phantom ratio and CD).
- **C0 protocol.** Reference skeleton = the same thinning algorithm run on ground-truth voxels; C0's TC is structurally 0 on the FP side. Disclosure: on perfect voxels the path rule still misses 6/120 thin contact necks (all chairs) — the *skeletonizer contact-gap baseline*; every method's FN count must be read against it (this baseline absorbed 6 of baseline-C4's 7 chair misses on synthetic data).
- Supporting: Chamfer to C0 in world units (d_vox · pitch / span), graph cycle-rank difference Δβ₁ (the RCS analog), component-count difference, phantom ratio.

### 5.3 Synthetic experiment (35 objects × 7 conditions)

![p1 main](figures/p1_main.png)
![p1 attribution](figures/p1_attribution.png)
![p1 paired](figures/p1_paired.png)

| criterion | measured | threshold | family-wise p | verdict |
|---|---|---|---|---|
| CD vs. joint hull (C4) | −21.3% | ≥15% | 0.0004 | pass |
| M1 vs. no-semantics (C3) | −100% | ≥30% | 0.0048 | pass |
| M1 vs. no-shared-frame (C2) | −100% | ≥20% | 0.0109 | pass |

Attribution (Fig. 2): all no-semantics false links land exactly on the enclosed-cavity candidates (10/10) — semantics contributes *negative knowledge* ("which pairs never touch"); no-shared-frame errors concentrate on symmetric parts (chair 20 FN, cage 3 FN); random assembly is significantly worse throughout (Holm p=2e-4); single-part objects are identical across conditions (implementation control). Disclosed: Holm by metric family (per-family4; all-12 correction makes C2's comparison 0.0545); n=35, 7 categories seen by the LOO table; oracle part masks (input contract, see §7).

## 6. Task B — confidence-weighted solid priors for contour SDFs

### 6.1 Method

SIREN 3×128, orthographic ray rendering (64 samples): L = stroke-weighted silhouette BCE + 0.1·Eikonal + λ·Ramp(t)·mean(ReLU(sdf) | conf), where conf = membership in the three-view hull and Ramp is a linear warmup over 400 steps. Conditions: E1 (full, λ=1), E2 (no prior), E3 (no stroke, no prior), E4 (λ=5).

### 6.2 Synthetic experiment (16 complete objects; paired tests via metric-wise intersections)

![p2 main](figures/p2_main.png)
![p2 explosion](figures/p2_explosion.png)
![p2 trajectory](figures/p2_trajectory.png)

| criterion | measured | threshold | p | verdict |
|---|---|---|---|---|
| false-cavity E2→E1 (primary, solid pairs n=8) | −75.5% | ≥20% | 0.0391 | pass |
| same, pooled all groups (n=17, 15/17 same sign) | −72.9% | ≥20% | 8e-5 | pass (secondary) |
| over-fill E4→E1 (n=16) | −81.6% | ≥40% | 0.0977 two-sided / **0.0488 pre-registered one-sided** | pass |
| silhouette IoU \|E1−E2\| (n=17) | 1.8% (E1 better) | ≤5% | 0.0385 | pass (pairing-set sensitive: 7.1% at n=15 — disclosed) |

Premise: contour-only fitting develops a *stable* shell pathology — 65% false-cavity rate flat across training (Fig. 6). **Failure mode:** the all-solid attractor (explosion) appears as a dose-dependent tail: 0/16 (E3), 2/16 (E2), 1/16 (E1), 4/16 (E4) on complete objects; even E3 exploded once among fragment runs — tail risk, not exclusive to any term; warmup is necessary but not sufficient. **Boundary behavior:** hidden cavities are filled 99.9% by the prior (E1) while the no-prior baseline merely drifts (~59%) — neither is correct because the input contains zero cavity information (Theorem 1). Visible holes are preserved (0.962 vs 0.981 without prior).

## 7. Transfer to real geometry (ABO, fully automatic)

Pipeline: public single-file ABO download → material-split orthographic rendering (headless) → surface-voxelized mesh GT with three-view IoU self-check (0.89–0.995) → same M1/M2/CD harness at 128³.

![real results](real/abo/real_results.png)

| object | C1 M1 | C1 M2 | C1 CD | C4 M2 | C4 CD |
|---|---|---|---|---|---|
| fabric chair | 0 | 0/0 | 0.0182 | 0/0 | 0.0177 |
| stool | 0 | 0/0 | 0.0125 | 0/0 | 0.0147 |
| floor lamp | 0 | 0/0 | **0.0104** | **0/1 miss** | 0.0274 |

The semantic table is learned leave-one-out from *synthetic* data only; real objects participate in nothing. C4's real failure mode differs from synthetic (thin-structure break vs. phantom/cavity) — same root cause (contour information loss), complementary evidence. **Honest disclosures:** n=3, 2 parts per object; on thick-contact real objects the no-assembly baseline ties C1 — the value density of assembly edges depends on contact geometry (thin necks in synthesis: 6/50 rescued; thick contacts: paths suffice).

## 8. Limitations

1. **Segmentation is part of the input contract**, not solved: synthesis uses rendered part colors, ABO uses material splits. Noise robustness (mask erosion/merging) is untested.
2. **Closed-set semantics**: the compatibility table falls back to 0.5 on unseen name pairs; an embedding variant is future work (an open-set assembly component is expected to matter, cf. §2 part-level line).
3. Sample sizes are pilot-scale (35 / 16 / 3); prop2's primary analysis n=8, one-sided and pairing-sensitivity disclosures in §6.2; single seeds; box-and-bar geometry dominates synthesis.
4. Theorem 1 assumes axis-visibility; objects violating it (axis-shadowed open pockets) may have hull error beyond cavities — checker released.
5. Prior strength λ and warmup are untuned beyond λ∈{1,5}.

## 9. Reproducibility

`run_all.py` regenerates every synthetic table from cached records in ~3 minutes; `verify_reports.py` re-derives all quoted numbers from raw per-run JSON/CSV via **98 independent assertions (0 mismatches)**; `test_core.py` provides 12 API smoke tests; PITFALLS.md documents eleven measured implementation traps (broadcast axis bugs, unit conventions, memory blowups, sampling asymmetries). Licenses: MIT (code); ABO models CC-BY 4.0.

## 10. Conclusion

Contours can tell you how parts connect — reliably, once semantics and shared coordinates are in place — but they can never tell you what is sealed inside. We proved the boundary, built both methods on the observable side of it, measured their failure modes as carefully as their successes, and released the machinery to check every number.

## References

1. SketchFormer3D: Generating 3D shapes from sketches with implicit SDF priors via diffusion models. *Expert Systems with Applications*, 2025. TODO(author): authors, volume, pages.
2. REVIVE 3D: Refinement via Encoded Voluminous Inflated prior for Volume Enhancement. *CVPR*, 2026. TODO(author): authors, pages.
3. REVIVE3D: REtrieval-based Volumetric Infusion via Visual Editing. *ICCV Workshop* (submitted). TODO(author): confirm final status.
4. Laurentini, A. The visual hull concept for silhouette-based image understanding. *PAMI*, 1994. TODO(page numbers)
5. SIREN / IGR / NeuS / PartNet / PartNet-Mobility: TODO(author) — add exact citations (7–10 entries).
