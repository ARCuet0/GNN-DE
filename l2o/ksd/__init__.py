"""KSD-loss package — Kernelised Stein Discrepancy as meta-objective for TersQ.

See `/.claude/ksd_loss_implementation_brief.md` for the design and
`/.claude/plans/en-claude-he-introducido-unified-floyd.md` for the plan.
"""
from l2o.ksd.score import compute_score
from l2o.ksd.kernel import multi_scale_kernel, bandwidth_with_ema
from l2o.ksd.loss import ksd_loss
from l2o.ksd.svgd import svgd_phi_analytic

__all__ = [
    'compute_score',
    'multi_scale_kernel',
    'bandwidth_with_ema',
    'ksd_loss',
    'svgd_phi_analytic',
]
