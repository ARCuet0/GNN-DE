"""Pairwise surrogate: scores augmented population (parents + K proposals).

Replaces router+gate with a single ranking mechanism. Linear scorer on
backbone embeddings of proposed positions (h_proposal). Selects top-M
candidates for real fitness evaluation.

Trained with pairwise BCE ranking loss using eval_all labels.
"""
import torch
import torch.nn as nn

from encoder.variants.activity_gate import topk_mask


class PairwiseSurrogate(nn.Module):
    """Scorer on h_proposal + h_global for conditional candidate ranking.

    h_proposal carries per-candidate quality signal.
    h_global carries population-level context (function identity, convergence state).
    Concatenating both lets the scorer learn per-function policies.
    """

    def __init__(self, backbone_dim: int = 128, conditional: bool = True):
        super().__init__()
        self.conditional = conditional
        scorer_in = backbone_dim * 2 if conditional else backbone_dim
        # MLP scorer: scorer_in → 128 → 64 → 16 → 1. Replaces the original
        # Linear(256, 1) head. F5 finding (session_2026_04_23) showed the
        # linear head was statistically indistinguishable from random
        # selection at inference (Bonferroni p=0.07 ns) — adding capacity
        # to test whether selection is genuinely under-parameterised.
        self.scorer = nn.Sequential(
            nn.Linear(scorer_in, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, h_augmented: torch.Tensor,
                h_global: torch.Tensor = None) -> torch.Tensor:
        """Score augmented population.

        Args:
            h_augmented: (B, N_aug, backbone_dim) backbone embeddings
            h_global: (B, backbone_dim) population-level context. Required
                      when conditional=True.
        Returns:
            scores: (B, N_aug) raw ranking scores
        """
        if self.conditional and h_global is not None:
            N_aug = h_augmented.shape[1]
            h_ctx = h_global.unsqueeze(1).expand(-1, N_aug, -1)
            h_in = torch.cat([h_augmented, h_ctx], dim=-1)
        else:
            h_in = h_augmented
        return self.scorer(h_in).squeeze(-1)

    @staticmethod
    def select_topM(scores: torch.Tensor, M: int):
        """Select top-M candidates by score.

        Returns:
            top_idx: (B, M) indices into augmented population
            selection_mask: (B, N_aug) binary {0, 1}
        """
        return topk_mask(scores, M)
