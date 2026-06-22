"""L-SHADE F/CR memory teacher (Tanabe & Fukunaga 2014).

Per-batch circular buffer of (M_F, M_CR), each (B, H=6), init 0.5. Sampling
uses Cauchy(M_F[r], 0.1) for F (resample if F<=0, clamp F>1) and
Normal(M_CR[r], 0.1) for CR (clamp [0, 1]).

Updates use weighted Lehmer mean for F and weighted arithmetic mean for CR,
weighted by Δfitness of successful trials.

Used as the teacher in online distillation (E13): drives F/CR sampling so
the trajectory follows L-SHADE, while the GNN's μ_F_pred/μ_CR_pred are
supervised against the realized values.
"""
import math

import torch


class LShadeMemory:
    """Per-batch L-SHADE F/CR memory.

    Attributes:
        M_F:  (B, H) float32 tensor, init 0.5. NOT a Parameter (no gradient).
        M_CR: (B, H) float32 tensor, init 0.5.
        k:    (B,)   long tensor, current write index per batch.
        H:    int, memory size (default 6 per L-SHADE paper).
    """

    def __init__(self, B: int, H: int = 6, device='cpu'):
        self.B = int(B)
        self.H = int(H)
        self.device = device
        self.M_F = torch.full((self.B, self.H), 0.5, device=device,
                              dtype=torch.float32)
        self.M_CR = torch.full((self.B, self.H), 0.5, device=device,
                               dtype=torch.float32)
        self.k = torch.zeros(self.B, dtype=torch.long, device=device)

    def to(self, device):
        self.device = device
        self.M_F = self.M_F.to(device)
        self.M_CR = self.M_CR.to(device)
        self.k = self.k.to(device)
        return self

    def reset(self):
        """Reset memory to initial state (used at start of each training step)."""
        self.M_F.fill_(0.5)
        self.M_CR.fill_(0.5)
        self.k.zero_()

    def resize(self, B_new: int):
        """Defensive: re-init when batch size changes (e.g., between train/eval)."""
        if B_new == self.B:
            return self
        self.B = int(B_new)
        self.M_F = torch.full((self.B, self.H), 0.5, device=self.device,
                              dtype=torch.float32)
        self.M_CR = torch.full((self.B, self.H), 0.5, device=self.device,
                               dtype=torch.float32)
        self.k = torch.zeros(self.B, dtype=torch.long, device=self.device)
        return self

    def sample(self, N: int, M: int = 1):
        """Sample F, CR per (M, B, N).

        F ~ Cauchy(M_F[r_i], 0.1); resample if F<=0; clamp F>1 to 1.
        CR ~ Normal(M_CR[r_i], 0.1); clamp to [0, 1].

        Returns:
            F:  (M, B, N) float32, in (0, 1].
            CR: (M, B, N) float32, in [0, 1].
            r:  (M, B, N) long, memory index used per (m, b, i).
        """
        device = self.M_F.device
        # r per-(M, B, N) ~ U(0, H).
        r = torch.randint(0, self.H, (M, self.B, N), device=device)
        # Gather mu_F[b, r] and mu_CR[b, r] per (m, b, n).
        # M_F shape (B, H); r shape (M, B, N) → expand to gather dim=1 of M_F.
        # Easier: index M_F via flat: idx = b * H + r → flatten then gather.
        b_idx = torch.arange(self.B, device=device).view(1, -1, 1).expand(M, -1, N)
        flat = b_idx * self.H + r  # (M, B, N)
        mu_F = self.M_F.view(-1)[flat]
        mu_CR = self.M_CR.view(-1)[flat]

        # Cauchy F: F = mu_F + 0.1 * tan(pi * (u - 0.5))
        # Resample if F<=0; clamp F=1 if F>1. Per L-SHADE paper, max ~10 iters.
        F = mu_F + 0.1 * torch.tan(math.pi * (torch.rand_like(mu_F) - 0.5))
        for _ in range(10):
            bad = F <= 0.0
            if not bad.any():
                break
            new = mu_F + 0.1 * torch.tan(math.pi * (torch.rand_like(mu_F) - 0.5))
            F = torch.where(bad, new, F)
        # Fallback for any remaining F<=0 (extreme cases): clamp to 0.05.
        F = F.clamp(min=0.05)
        F = F.clamp(max=1.0)

        # Normal CR: clamp to [0, 1].
        CR = (mu_CR + 0.1 * torch.randn_like(mu_CR)).clamp(0.0, 1.0)

        return F, CR, r

    @torch.no_grad()
    def update(self, F_succ: torch.Tensor, CR_succ: torch.Tensor,
               delta: torch.Tensor, success_mask: torch.Tensor):
        """Update memory using successful trials.

        Args:
            F_succ:       (B, N) realized F per parent (only entries where
                          success_mask[b, i] are used).
            CR_succ:      (B, N) realized CR per parent.
            delta:        (B, N) Δfitness (positive = improvement) per parent.
            success_mask: (B, N) bool, True where the trial improved.

        Per-batch update:
            * If any successes in batch b: compute weighted means (Lehmer for F,
              arithmetic for CR), write to slot k[b], advance k[b].
            * If no successes: leave M_F[b], M_CR[b], k[b] unchanged.
        """
        for b in range(self.B):
            mask_b = success_mask[b]
            if not mask_b.any():
                continue
            F_b = F_succ[b][mask_b].float()
            CR_b = CR_succ[b][mask_b].float()
            d_b = delta[b][mask_b].float().clamp(min=0.0)
            w_sum = d_b.sum() + 1e-30
            w = d_b / w_sum
            # Weighted Lehmer mean for F: Σ(w·F²) / Σ(w·F)
            num = (w * F_b * F_b).sum()
            den = (w * F_b).sum() + 1e-30
            new_F = (num / den).clamp(0.05, 1.0)
            # Weighted arithmetic mean for CR.
            new_CR = (w * CR_b).sum().clamp(0.0, 1.0)
            slot = int(self.k[b].item())
            self.M_F[b, slot] = new_F
            self.M_CR[b, slot] = new_CR
            self.k[b] = (self.k[b] + 1) % self.H
