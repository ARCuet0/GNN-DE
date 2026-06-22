"""
adaptive_sigma.py — Automatic step-size (σ) adaptation for ES training.

Uses gradient Signal-to-Noise Ratio (SNR) as the primary signal:
  SNR = ||mean(ε * f_shaped)|| / mean(std(ε * f_shaped))

When SNR > target, σ can grow (clear gradient signal, room to explore).
When SNR < target, σ must shrink (gradient too noisy).

Updates are in log-space with EMA smoothing and clamped rate.
A secondary 1/5th-rule safety net prevents pathological σ values.
"""

import math
from typing import Optional

import torch


class AdaptiveSigma:
    """Drop-in replacement for fixed σ in ES training loops.

    Usage::

        adaptive = AdaptiveSigma(sigma_init=0.02)
        for step in range(n_steps):
            eps = torch.randn(M, n_params)
            fitness = rollout(theta + sigma * eps)
            g = (eps * fitness.unsqueeze(1)).mean(0) / sigma
            # ... adam update ...
            sigma = adaptive.observe(eps, fitness_shaped, improvement_frac)
    """

    def __init__(
        self,
        sigma_init: float = 0.02,
        sigma_min: float = 1e-4,
        sigma_max: float = 0.1,
        snr_target: float = 5.0,
        ema_decay: float = 0.95,
        update_rate: float = 0.05,
        warmup_steps: int = 20,
    ):
        self.sigma = sigma_init
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.snr_target = snr_target
        self.ema_decay = ema_decay
        self.update_rate = update_rate
        self.warmup_steps = warmup_steps

        # EMA accumulators (None until first observation)
        self._snr_ema: Optional[float] = None
        self._improv_ema: Optional[float] = None
        self._step = 0

    def observe(
        self,
        epsilons: torch.Tensor,
        fitness_shaped: torch.Tensor,
        improvement_frac: float = 0.2,
    ) -> float:
        """Compute SNR from this step's data and return updated σ.

        Args:
            epsilons: (M, n_params) — perturbation directions used this step.
            fitness_shaped: (M,) — rank-normalized or raw fitness differences.
            improvement_frac: fraction of perturbations that improved fitness.

        Returns:
            Updated sigma value (also stored in self.sigma).
        """
        self._step += 1

        # ── SNR computation ──
        with torch.no_grad():
            # Per-epsilon gradient contributions: (M, n_params)
            contrib = epsilons * fitness_shaped.unsqueeze(1)
            grad_mean = contrib.mean(dim=0)          # (n_params,)
            grad_std = contrib.std(dim=0).mean()     # scalar
            snr = (grad_mean.norm() / (grad_std + 1e-10)).item()

        # ── EMA update ──
        d = self.ema_decay
        if self._snr_ema is None:
            self._snr_ema = snr
            self._improv_ema = improvement_frac
        else:
            self._snr_ema = d * self._snr_ema + (1 - d) * snr
            self._improv_ema = d * self._improv_ema + (1 - d) * improvement_frac

        # ── Warmup: don't adapt yet ──
        if self._step <= self.warmup_steps:
            return self.sigma

        # ── Primary: SNR-based log-space update ──
        ratio = self._snr_ema / (self.snr_target + 1e-10)
        # Positive log_ratio → SNR above target → increase σ
        # Negative log_ratio → SNR below target → decrease σ
        log_ratio = math.log(max(ratio, 1e-10))
        delta = max(-self.update_rate, min(self.update_rate, self.update_rate * log_ratio))

        # ── Secondary: 1/5th rule safety net ──
        if self._improv_ema < 0.05 and self._snr_ema < self.snr_target * 0.5:
            # Almost nothing improves AND SNR is very low → force reduce
            delta = min(delta, -self.update_rate)
        elif self._improv_ema > 0.4:
            # Almost everything improves → safe to be bolder
            delta = max(delta, self.update_rate * 0.5)

        # ── Apply ──
        self.sigma = self.sigma * math.exp(delta)
        self.sigma = max(self.sigma_min, min(self.sigma_max, self.sigma))

        return self.sigma

    def reset(self, sigma_init: Optional[float] = None):
        """Reset EMA accumulators. Call when parameter space changes."""
        self._snr_ema = None
        self._improv_ema = None
        self._step = 0
        if sigma_init is not None:
            self.sigma = sigma_init

    def state_dict(self) -> dict:
        return {
            'sigma': self.sigma,
            'sigma_min': self.sigma_min,
            'sigma_max': self.sigma_max,
            'snr_target': self.snr_target,
            'ema_decay': self.ema_decay,
            'update_rate': self.update_rate,
            'warmup_steps': self.warmup_steps,
            'snr_ema': self._snr_ema,
            'improv_ema': self._improv_ema,
            'step': self._step,
        }

    def load_state_dict(self, d: dict):
        self.sigma = d['sigma']
        self.sigma_min = d.get('sigma_min', self.sigma_min)
        self.sigma_max = d.get('sigma_max', self.sigma_max)
        self.snr_target = d.get('snr_target', self.snr_target)
        self.ema_decay = d.get('ema_decay', self.ema_decay)
        self.update_rate = d.get('update_rate', self.update_rate)
        self.warmup_steps = d.get('warmup_steps', self.warmup_steps)
        self._snr_ema = d.get('snr_ema')
        self._improv_ema = d.get('improv_ema')
        self._step = d.get('step', 0)

    def __repr__(self):
        return (f"AdaptiveSigma(σ={self.sigma:.5f}, snr_ema={self._snr_ema:.1f}, "
                f"improv_ema={self._improv_ema:.2f}, step={self._step})"
                if self._snr_ema is not None
                else f"AdaptiveSigma(σ={self.sigma:.5f}, warmup)")
