"""Early-stop predicate for training, extracted from train_distributed.py.

Background: the prior inline implementation (train_distributed.py:1083-1088)
hardcoded drift_threshold=1e-4, two orders of magnitude below the real
Adam-with-lr=3e-5 drift regime (~1e-3). The gate never fired on plateau,
so training ran to --steps even though best_eval.pth was already saved.
Discovered 2026-04-26 during E11 ablation analysis.
"""
from typing import Sequence


DEFAULT_DRIFT_THRESHOLD = 1e-3
"""Matches Adam-with-lr=3e-5 plateau regime. Old hardcoded value (1e-4)
was 2 OOM below the real regime — gate never fired."""


def should_early_stop(
    patience_steps: int,
    eval_every_steps: int,
    evals_without_improvement: int,
    drift_history: Sequence[float],
    drift_threshold: float = DEFAULT_DRIFT_THRESHOLD,
) -> bool:
    """Decide whether to early-stop based on patience + parameter drift.

    Args:
        patience_steps: caller-set patience in steps. ``<= 0`` disables stop.
        eval_every_steps: steps between evals; converts patience to eval count.
        evals_without_improvement: consecutive evals without best_eval improvement.
        drift_history: recent ``‖θ_now − θ_prev‖ / ‖θ_prev‖`` per eval.
        drift_threshold: strict-less-than threshold on drift samples.

    Returns:
        ``True`` iff: patience is enabled, the no-improvement window has been
        reached, at least 3 drift samples are available, and **all of the last
        3** drift samples are strictly below ``drift_threshold``.
    """
    if patience_steps <= 0:
        return False
    eval_patience = max(1, patience_steps // max(eval_every_steps, 1))
    if evals_without_improvement < eval_patience:
        return False
    if len(drift_history) < 3:
        return False
    return all(d < drift_threshold for d in drift_history[-3:])
