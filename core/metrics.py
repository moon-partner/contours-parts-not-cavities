"""core.metrics -- canonical metric implementations for contour-based 3D reconstruction.

Re-exported from the experiment scripts (single source of truth) so downstream code
never drifts from the numbers reported in the papers:

  hull              three-view hard intersection (broadcast-checked axes; see PITFALLS A1)
  assign_labels     skeleton voxels -> nearest GT part (<= 4 voxels), 0 = phantom
  pair_paths        M2: predicted direct connectivity per part pair (label-restricted paths)
  graph_tc          FP/FN/TC between predicted and GT edge sets
  graph_beta1       cycle rank of a part-contact graph (E - V + C) -- the RCS metric
  beta1/count_edges  voxel-level cycle rank (descriptive only; dominated by thickness)
  chamfer           bidirectional mean nearest distance, normalized by grid span
  holm              Holm-Bonferroni correction (per family)
  paired_stats      paired Wilcoxon + median diff + bootstrap 95% CI
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_P1 = os.path.join(_ROOT, "prop1")
if _P1 not in sys.path:
    sys.path.insert(0, _P1)

from run_experiments import (  # noqa: E402
    hull,
    assign_labels,
    pair_paths,
    graph_tc,
    graph_beta1,
    beta1,
    count_edges26,
    chamfer,
    holm,
    paired_stats,
)

__all__ = [
    "hull", "assign_labels", "pair_paths", "graph_tc", "graph_beta1",
    "beta1", "count_edges26", "chamfer", "holm", "paired_stats",
]
