"""Genealogy credit diagnostic — binary ancestry with γ-decay.

Tracks which operator was applied to the population's best individual
at each generation, then computes γ-weighted credit per operator head.
All bookkeeping is detached (no autograd). Zero .item() on the hot path.
"""
import torch


class GenealogyTracker:
    """Lightweight per-episode genealogy credit tracker.

    With greedy 1:1 selection, individual i is always at index i.
    We track: which index holds the best, which operator was applied
    to it, and whether it improved that generation.
    """

    def __init__(self, K: int, gamma: float = 0.9):
        self.K = K
        self.gamma = gamma
        self._best_idx: list[torch.Tensor] = []
        self._winner_best: list[torch.Tensor] = []
        self._improved_best: list[torch.Tensor] = []

    def reset(self, B: int, device: torch.device):
        """Call at episode start."""
        self._best_idx.clear()
        self._winner_best.clear()
        self._improved_best.clear()
        self._B = B
        self._device = device

    def record(self, fitness: torch.Tensor, winner: torch.Tensor,
               improved: torch.Tensor):
        """Record one generation's state. All tensors must be detached.

        Args:
            fitness: (B, N) post-selection fitness.
            winner: (M, B, N) or (B, N) hard operator index per individual.
            improved: (B, N) bool mask of which individuals improved.
        """
        if winner.dim() == 3:
            winner = winner[0]
        best_idx = fitness.argmin(dim=-1)  # (B,)
        b_range = torch.arange(fitness.shape[0], device=fitness.device)
        self._best_idx.append(best_idx)
        self._winner_best.append(winner[b_range, best_idx])
        self._improved_best.append(improved[b_range, best_idx])

    def compute_credit(self) -> dict:
        """Compute γ-weighted credit per operator. Call at episode end.

        Returns dict with keys:
            genealogy/credit_k{i}: fraction of total credit for operator i
            genealogy/n_gens: number of generations tracked
            genealogy/best_changed_frac: fraction of gen transitions where
                the best individual's index changed
        """
        G = len(self._best_idx)
        if G == 0:
            result = {f'genealogy/credit_k{k}': 0.0 for k in range(self.K)}
            result['genealogy/n_gens'] = 0
            result['genealogy/best_changed_frac'] = 0.0
            return result

        winner_best = torch.stack(self._winner_best)      # (G, B)
        improved_best = torch.stack(self._improved_best)   # (G, B)
        best_idx = torch.stack(self._best_idx)             # (G, B)

        # γ^(G-1-g): more recent gens get higher weight
        weights = self.gamma ** torch.arange(
            G - 1, -1, -1, device=winner_best.device, dtype=torch.float32)

        credit = torch.zeros(self.K, device=winner_best.device)
        for k in range(self.K):
            mask = (winner_best == k) & improved_best  # (G, B)
            credit[k] = (weights.unsqueeze(1) * mask.float()).sum()

        total = credit.sum().clamp(min=1e-8)
        frac = credit / total

        if G > 1:
            changed = (best_idx[1:] != best_idx[:-1]).float().mean()
        else:
            changed = torch.tensor(0.0)

        result = {f'genealogy/credit_k{k}': round(frac[k].item(), 4)
                  for k in range(self.K)}
        result['genealogy/n_gens'] = G
        result['genealogy/best_changed_frac'] = round(changed.item(), 4)
        return result
