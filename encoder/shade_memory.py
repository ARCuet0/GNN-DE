"""
shade_memory.py — GPU-batched L-SHADE success-history circular buffer.

Maintains per-population circular buffers of successful (F, CR) pairs.
Updated via deferred reporting: record_trials() before eval,
report_child_fitness() after eval.

Reference: Tanabe & Fukunaga 2013 (SHADE).
"""
import math
import torch


class BatchedSHADEMemory:
    """SHADE success-history memory for B independent populations on GPU.

    All B memories are stored as (B, H) tensors for fully vectorized ops.
    """

    def __init__(self, B, H=10, device='cpu'):
        self.B = B
        self.H = H
        self.device = device
        self.M_F = torch.full((B, H), 0.5, device=device, dtype=torch.float64)
        self.M_CR = torch.full((B, H), 0.5, device=device, dtype=torch.float64)
        self.k = torch.zeros(B, dtype=torch.long, device=device)
        self._pending_F = None
        self._pending_CR = None
        self._pending_parent_fit = None

    def sample(self, N):
        """Sample F, CR for B populations x N individuals.

        F ~ Cauchy(M_F[r], 0.1) clamped to [0.01, 1.0]
        CR ~ Normal(M_CR[r], 0.1) clamped to [0.0, 1.0]

        Returns:
            F_vals: (B, N) tensor
            CR_vals: (B, N) tensor
        """
        # Random memory index per individual
        r = torch.randint(0, self.H, (self.B, N), device=self.device)
        M_F_sel = self.M_F.gather(1, r)   # (B, N)
        M_CR_sel = self.M_CR.gather(1, r)  # (B, N)

        # Cauchy noise for F via tan(pi*(u-0.5))
        u = torch.rand(self.B, N, device=self.device, dtype=torch.float64)
        F_vals = (M_F_sel + 0.1 * torch.tan(math.pi * (u - 0.5))).clamp(0.01, 1.0)

        # Normal noise for CR
        CR_vals = (M_CR_sel + 0.1 * torch.randn(
            self.B, N, device=self.device, dtype=torch.float64)).clamp(0.0, 1.0)

        return F_vals.detach(), CR_vals.detach()

    def record_trials(self, F_vals, CR_vals, parent_fitness):
        """Record F, CR, parent fitness for deferred update. All (B, N) tensors."""
        self._pending_F = F_vals.detach().to(dtype=torch.float64)
        self._pending_CR = CR_vals.detach().to(dtype=torch.float64)
        self._pending_parent_fit = parent_fitness.detach().to(dtype=torch.float64)

    def report_child_fitness(self, child_fitness):
        """Update memory with child fitness. (B, N) tensor. Fully vectorized."""
        if self._pending_F is None:
            return
        child_fitness = child_fitness.detach().to(dtype=torch.float64)
        success = child_fitness < self._pending_parent_fit  # (B, N)
        has_success = success.any(dim=1)  # (B,)

        if not has_success.any():
            self._pending_F = None
            self._pending_CR = None
            self._pending_parent_fit = None
            return

        # Masked weighted Lehmer mean — vectorized over B
        s_mask = success.float()  # (B, N)
        delta = ((self._pending_parent_fit - child_fitness).abs() * s_mask)  # (B, N)
        w = delta / (delta.sum(dim=1, keepdim=True) + 1e-15)  # (B, N)

        # Weighted Lehmer mean for F: sum(w * F^2) / sum(w * F)
        F_masked = self._pending_F * s_mask
        new_M_F = ((w * F_masked ** 2).sum(dim=1)
                   / ((w * F_masked).sum(dim=1) + 1e-15))  # (B,)

        # Weighted arithmetic mean for CR
        CR_masked = self._pending_CR * s_mask
        new_M_CR = (w * CR_masked).sum(dim=1)  # (B,)

        # Scatter into memory at position k, only for populations with success
        active_b = has_success.nonzero(as_tuple=True)[0]  # (n_active,)
        k_active = self.k[active_b]  # (n_active,)
        self.M_F[active_b, k_active] = new_M_F[active_b]
        self.M_CR[active_b, k_active] = new_M_CR[active_b]
        self.k[active_b] = (k_active + 1) % self.H

        self._pending_F = None
        self._pending_CR = None
        self._pending_parent_fit = None
