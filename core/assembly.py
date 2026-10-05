"""core.assembly -- part-contact graph construction (Proposition 1 pipeline).

Canonical implementations live in prop1/run_experiments.py; re-exported here.

  spatial_candidates    dilate-and-overlap spatial adjacency (dil=2, min_vox=1 for
                        recall-oriented assembly candidates; dil=1, min_vox=2 = GT touch rule)
  build_semantic_table  global counts of part-name-pair co-occurrence (n, e)
  compat                leave-one-out Laplace-smoothed P(contact | name-pair);
                        pairs never seen in training fall back to 0.5 (open-set limitation,
                        see PITFALLS B5 -- embedding upgrade pending)
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_P1 = os.path.join(_ROOT, "prop1")
if _P1 not in sys.path:
    sys.path.insert(0, _P1)

from run_experiments import spatial_candidates, build_semantic_table, compat  # noqa: E402

__all__ = ["spatial_candidates", "build_semantic_table", "compat"]
