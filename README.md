# Contours Tell You Parts, Not Cavities

![verification](https://img.shields.io/badge/verification-98%2F98%20assertions%20passing-brightgreen)
![core tests](https://img.shields.io/badge/core%20tests-12%2F12%20passing-brightgreen)
![license](https://img.shields.io/badge/license-MIT-blue)
![python](https://img.shields.io/badge/python-3.14-orange)
![compute](https://img.shields.io/badge/compute-CPU%20only%20%7C%20no%20GPU-lightgrey)

Artifacts for two validated propositions on reconstructing 3D shape from three
orthographic silhouettes, plus an identifiability boundary theorem:

1. **Boundary (Theorem 1).** Under axis-visibility of the exterior, visual-hull
   error equals exactly the union of enclosed cavities — closed cavities are
   invisible to *every* contour method. Cross-validated on both tasks below.
2. **Semantic part assembly (Proposition 1 — validated).** Per-part silhouettes
   in a shared frame + spatial adjacency gated by a leave-one-out semantic
   compatibility table recover the part-contact graph exactly (35 synthetic
   objects × 7 conditions; all three pre-registered criteria pass).
3. **Confidence-weighted solid prior (Proposition 2 — validated, with
   disclosures).** A hull-gated solid-volume regularizer with warmup fixes the
   false-cavity pathology of contour-only SDF fitting (−75.5%, p=0.039 primary;
   −72.9%, p=8e-5 pooled) while preserving visible holes and silhouettes; the
   "all-solid attractor" explosion is reported as a dose-dependent failure mode.
4. **Real-geometry transfer.** Three ABO product models, split and voxelized
   fully automatically: exact contact recovery, 1–2% world-normalized Chamfer,
   zero-shot semantic transfer from the synthetic leave-one-out table.

## Reproduce everything (~3 minutes, cached results)

```bash
pip install -r requirements.txt
python run_all.py        # all synthetic tables + verification + figures
python test_core.py      # 12 API smoke tests
python verify_reports.py # 98 assertions re-deriving every quoted number
```

Fresh training (optional, CPU hours): `python run_all.py --fresh`.

## Real-geometry pipeline (ABO; requires Blender for rendering only)

```bash
python abo_fetch.py                                          # re-download candidates
blender --background --python render_ortho.py -- --glb <model.glb> \
        --out real/abo/rendered --name <id>                  # 3 views + per-part masks
python build_abo_gt.py --glb <model.glb> --meta <meta.json> \
        --masks <dir> --name <id> --res 128                  # mesh GT + IoU self-check
python run_real.py --all                                     # C1/C5/C4 vs GT
python make_real_figures.py
```

Rendered masks for the three reported models are shipped in `real/abo/rendered/`,
so the quantitative step (`run_real.py`) runs **without Blender**. Raw `.glb`
models (CC-BY 4.0, 416 MB) are *not* shipped; `abo_fetch.py` re-downloads them
from the public ABO bucket using `real/abo/3dmodels.csv.gz`.

## Repository layout

| Path | Contents |
|---|---|
| `core/` | reusable metrics, assembly, solid-prior loss (importable) |
| `prop1/`, `prop2/` | experiments, reports, per-run records (CSV/JSON) |
| `real/` | ABO pipeline outputs: GT, results.json, masks, figures |
| `figures/` | paper figures (regenerate with `make_figures.py`) |
| `paper/` | workshop draft (markdown + docx) |
| `PITFALLS.md` | 11 measured implementation traps — read before modifying |
| `ZONG_REPORT.md` | combined Chinese report incl. the data-verification log |
| `zenodo/` | deposit metadata + submission steps |

## Cite

See `CITATION.cff`. DOI: [10.6084/m9.figshare.34068864](https://doi.org/10.6084/m9.figshare.34068864)

## License

MIT (code). ABO models: CC-BY 4.0 (Amazon Berkeley Objects).

