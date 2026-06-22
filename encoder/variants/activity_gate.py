"""Activity gate hierarchy: per-individual binary decision (operate or preserve).

Uses adaptive-threshold sigmoid gating: BCE learns the ranking (which
individuals to activate), and a budget-aware quantile threshold controls
how many. The threshold adapts with FES fraction: more selective as budget
depletes. Straight-through gradient flows through sigmoid.

Three variants: Linear, MLP, Attention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def topk_mask(scores, k):
    """Select top-k by score, return (top_idx, binary_mask). Shared utility."""
    k = min(max(1, k), scores.shape[-1])
    _, top_idx = scores.topk(k, dim=-1)
    mask = torch.zeros_like(scores)
    mask.scatter_(-1, top_idx, 1.0)
    return top_idx, mask


def _adaptive_threshold_gate(logit, target_frac, training):
    """Adaptive-threshold sigmoid gate with straight-through gradient.

    BCE learns the ranking (logit order). The threshold is set per-batch
    as the quantile that activates target_frac of individuals. This
    decouples "which" (learned) from "how many" (scheduled).

    Args:
        logit: (B, N) raw gate logits
        target_frac: float in (0, 1], fraction of individuals to activate
        training: bool — adds noise in training for exploration

    Returns:
        (gate, soft):
            gate: (B, N) binary {0,1} via ST estimator
            soft: (B, N) sigmoid(logit) in [0, 1]
    """
    soft = torch.sigmoid(logit)
    # Quantile threshold: activate top target_frac by logit value
    k = max(1, int(logit.shape[-1] * target_frac))
    # topk gives the k largest logits; the k-th largest is the threshold
    threshold = logit.topk(k, dim=-1).values[..., -1:]  # (B, 1)
    hard = (logit >= threshold).float()
    return hard - soft.detach() + soft, soft


class _ActivityGateBase(nn.Module):
    """Base class for all gate variants (ActivityGate, RankerGate).

    Provides _prepare_input(), get_soft(), contrafactual_bce_loss(),
    compute_target_frac(). Subclasses must implement get_logits(h).
    """
    use_global: bool = False
    node_feat_dim: int = 0

    def get_soft(self, h, **kwargs):
        """Return soft gate values (sigmoid) for monitoring. In [0, 1]."""
        logit = self.get_logits(h, **kwargs)
        return torch.sigmoid(logit)

    def contrafactual_bce_loss(self, h, parent_fit, off_fit,
                               scale_ema=None, **kwargs):
        """BCE loss with magnitude weighting for W=1 contrafactual labels.

        Args:
            h: (B, N, in_dim) backbone embeddings
            parent_fit: (B, N) parent fitness
            off_fit: (B, N) or (M, B, N) offspring fitness (from eval_all)
            scale_ema: EMA scale for normalization
            **kwargs: passed to get_logits (e.g. h_global, node_feat)
        Returns:
            Scalar loss with gradient to gate parameters.
        """
        logit = self.get_logits(h, **kwargs)  # (B, N)

        # Reduce M dimension: take best offspring across M samples
        if off_fit.dim() == 3:
            off_fit = off_fit.min(dim=0).values  # (B, N)

        improvement = parent_fit - off_fit  # positive = improved
        improved = (improvement > 0).float()  # (B, N)

        # Magnitude weights for both improvers and non-improvers
        abs_change = improvement.abs()
        if scale_ema is not None:
            scale = scale_ema
        else:
            scale = abs_change.median().clamp(min=1e-6)
        weights = torch.log1p(abs_change / scale) + 1e-3

        return F.binary_cross_entropy_with_logits(
            logit, improved, weight=weights)

    @staticmethod
    def compute_target_frac(fes_frac):
        """Budget-aware target activation fraction.

        Starts at 80% (explore broadly), decays to 30% (exploit selectively).
        fes_frac: fraction of FES budget consumed, in [0, 1].
        """
        return max(0.3, 0.8 - 0.5 * fes_frac)

    def _prepare_input(self, h, h_global=None, node_feat=None):
        parts = [h]
        if self.use_global:
            if h_global is not None:
                parts.append(h_global.unsqueeze(1).expand_as(h))
            else:
                parts.append(torch.zeros_like(h))
        if self.node_feat_dim > 0:
            if node_feat is not None:
                parts.append(node_feat)
            else:
                B, N = h.shape[:2]
                parts.append(h.new_zeros(B, N, self.node_feat_dim))
        return torch.cat(parts, dim=-1)


class ActivityGate(_ActivityGateBase):
    """Per-individual binary gate: Linear(d->1) + adaptive threshold.

    BCE learns which individuals to activate (ranking). The threshold
    is set by compute_target_frac(fes_frac) to control how many.
    No sigmoid saturation problem: the threshold moves with the logit
    distribution, not against it.

    With use_global=True, concatenates h_global (B, d) broadcast to
    (B, N, d) with h (B, N, d) before the linear layer.

    With node_feat_dim>0, concatenates raw node features (B, N, nfd) to
    the gate input.
    """

    def __init__(self, in_dim: int, use_global: bool = False,
                 node_feat_dim: int = 0):
        super().__init__()
        self.use_global = use_global
        self.node_feat_dim = node_feat_dim
        linear_in = in_dim * 2 if use_global else in_dim
        linear_in += node_feat_dim
        self.linear = nn.Linear(linear_in, 1)
        nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, h: torch.Tensor, h_global=None, node_feat=None,
                fes_frac=0.0, **_ignored):
        h_in = self._prepare_input(h, h_global, node_feat)
        logit = self.linear(h_in).squeeze(-1)  # (B, N)
        target_frac = self.compute_target_frac(fes_frac)
        mask, _ = _adaptive_threshold_gate(logit, target_frac, self.training)
        return mask

    def get_logits(self, h, h_global=None, node_feat=None):
        h_in = self._prepare_input(h, h_global, node_feat)
        return self.linear(h_in).squeeze(-1)


class ActivityGateMLP(_ActivityGateBase):
    """MLP(d->32->1) gate — richer than Linear(d->1)."""

    def __init__(self, in_dim: int, hidden: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h, fes_frac=0.0, **_ignored):
        logit = self.mlp(h).squeeze(-1)
        target_frac = self.compute_target_frac(fes_frac)
        mask, _ = _adaptive_threshold_gate(logit, target_frac, self.training)
        return mask

    def get_logits(self, h):
        return self.mlp(h).squeeze(-1)


class RankerGate(_ActivityGateBase):
    """Pairwise ranking gate: scores individuals by expected improvement.

    Produces raw scores (no sigmoid, no threshold in forward). Selection
    is done externally via select_topk(). Trained with pairwise BCE loss
    on (score_i - score_j) vs (imp_i > imp_j).

    Inherits _prepare_input and compute_target_frac from _ActivityGateBase.
    """

    def __init__(self, in_dim: int, use_global: bool = False,
                 node_feat_dim: int = 0):
        super().__init__()
        self.use_global = use_global
        self.node_feat_dim = node_feat_dim
        linear_in = in_dim * 2 if use_global else in_dim
        linear_in += node_feat_dim
        self.linear = nn.Linear(linear_in, 1)
        nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, h: torch.Tensor, h_global=None, node_feat=None,
                **_ignored) -> torch.Tensor:
        """Return raw scores (B, N). No sigmoid, no threshold."""
        h_in = self._prepare_input(h, h_global, node_feat)
        return self.linear(h_in).squeeze(-1)

    def get_logits(self, h, h_global=None, node_feat=None):
        """Alias for forward (backward compat with AUC diagnostics)."""
        return self.forward(h, h_global=h_global, node_feat=node_feat)

    @staticmethod
    def select_topk(scores: torch.Tensor, target_frac: float) -> torch.Tensor:
        """Select top-k individuals by score. Returns binary mask (B, N)."""
        k = max(1, int(scores.shape[-1] * target_frac))
        _, mask = topk_mask(scores, k)
        return mask


class ActivityGateAttn(_ActivityGateBase):
    """Attention-head gate: Q/K attention over population -> gate logit."""

    def __init__(self, in_dim: int, d_k: int = 16):
        super().__init__()
        self.q_proj = nn.Linear(in_dim, d_k)
        self.k_proj = nn.Linear(in_dim, d_k)
        self.gate_proj = nn.Linear(in_dim + d_k, 1)
        self.d_k = d_k

    def forward(self, h, fes_frac=0.0, **_ignored):
        Q = self.q_proj(h)
        K = self.k_proj(h)
        A = torch.bmm(Q, K.transpose(1, 2)) / (self.d_k ** 0.5)
        A = torch.softmax(A, dim=-1)
        context = torch.bmm(A, h[:, :, :self.d_k])
        logit = self.gate_proj(
            torch.cat([h, context], dim=-1)).squeeze(-1)
        target_frac = self.compute_target_frac(fes_frac)
        mask, _ = _adaptive_threshold_gate(logit, target_frac, self.training)
        return mask

    def get_logits(self, h):
        Q = self.q_proj(h)
        K = self.k_proj(h)
        A = torch.softmax(
            torch.bmm(Q, K.transpose(1, 2)) / (self.d_k ** 0.5), dim=-1)
        context = torch.bmm(A, h[:, :, :self.d_k])
        return self.gate_proj(torch.cat([h, context], dim=-1)).squeeze(-1)
