"""core.solid_prior -- confidence-weighted solid-volume prior for SDF fitting.

Standalone, framework-light (operates on already-computed tensors) so it can be dropped
into any contour-based SDF / NeRF / occupancy training loop:

    from core.solid_prior import solid_prior_loss
    loss = loss_contour + 0.1 * loss_eikonal + solid_prior_loss(sdf_vals, conf, lam=1.0, it=it)

What it does
------------
Penalizes *positive* SDF (empty space) only at sample points whose confidence is high,
where confidence here = membership in the three-view visual hull (prop2's implementation);
a probabilistic tri-plane confidence map is a drop-in replacement and is the intended
upgrade (see report: binary confidence is the simplification that was actually measured).

Defaults that matter (measured, do not "simplify" them away)
-------------------------------------------------------------
- lam=1.0, warmup=400 steps (linear ramp). WITHOUT warmup the training collapses into an
  "all-solid attractor" within ~300 steps (dilation 16x, silhouette IoU 0.14; pilot log in
  prop2/report.md).
- Even with warmup the explosion risk is real and dose-dependent: measured 0/15 (no stroke,
  no prior), 2/15 (+stroke), 1/15 (+prior), 4/15 (prior x5). See PITFALLS B2.
- The prior can only *fix false cavities* (-70% measured); it CANNOT recover hidden
  cavities -- it fills them (99.9% measured). Do not claim otherwise (PITFALLS B3).

Math (identical to prop2/run_prop2.py train loop)
-------------------------------------------------
    ramp  = min(1, (it + 1) / warmup)
    loss  = lam * ramp * mean( relu(sdf) [conf] )
"""
import torch
import torch.nn.functional as F

DEFAULT_LAM = 1.0
DEFAULT_WARMUP = 400


def solid_prior_loss(sdf_vals, conf_mask, lam=DEFAULT_LAM, it=0,
                     warmup=DEFAULT_WARMUP):
    """Confidence-weighted ReLU(sdf) penalty with linear warmup.

    Args:
        sdf_vals: 1-D tensor of SDF values at sample points (positive = empty).
        conf_mask: bool tensor, same shape -- True where the solid assumption is trusted.
        lam: prior strength (1.0 measured; 5.0 explodes ~4/15 without other controls).
        it: current training iteration (for the warmup ramp).
        warmup: linear ramp length in steps.

    Returns:
        scalar loss (0.0 as float-compatible tensor if no confident points).
    """
    if lam <= 0:
        return torch.zeros((), dtype=sdf_vals.dtype, device=sdf_vals.device)
    conf_mask = conf_mask.to(torch.bool)
    if not bool(conf_mask.any()):
        return torch.zeros((), dtype=sdf_vals.dtype, device=sdf_vals.device)
    ramp = min(1.0, (it + 1) / warmup)
    return F.relu(sdf_vals)[conf_mask].mean() * lam * ramp
