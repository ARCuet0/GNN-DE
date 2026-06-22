"""Coordinate-wise local search operator heads."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder.operators._base import _ParamMLP, _make_proj


class BatchedDiffCoordLS(nn.Module):
    """Differentiable coordinate-wise local search (MTS-LS1 spirit).

    Random +/-1 probing per dimension. Network controls step size and
    sparsity (how many dims to probe). Dimension-agnostic: no Linear(*, D).
    """

    def __init__(self, embed_dim=16, head_idx=1, backbone_dim=0):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_idx = head_idx
        self.backbone_dim = backbone_dim
        self.proj, self.proj_norm = _make_proj(backbone_dim, embed_dim)

        full_dim = embed_dim + backbone_dim
        self.step_mlp = _ParamMLP(full_dim, 1, hidden=64)
        self.sparsity_mlp = _ParamMLP(full_dim, 1, hidden=64)
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def get_embedding(self, h_backbone):
        if self.proj is not None:
            return self.proj_norm(self.proj(h_backbone))
        return h_backbone[..., :self.embed_dim]

    def compute_params(self, h_out, coords, fitness, adj=None,
                       route_probs=None, bounds_span=200.0, h_backbone=None, **_kwargs):
        if h_backbone is not None and self.proj is not None:
            h_out = self.get_embedding(h_backbone)
        h_in = torch.cat([h_out, h_backbone], dim=-1) if h_backbone is not None else h_out
        step_mu = self.step_mlp(h_in).squeeze(-1)
        sparsity_logit = self.sparsity_mlp(h_in).squeeze(-1)
        return {'step_mu': step_mu, 'sparsity_logit': sparsity_logit}

    def sample_batch(self, params_dict, coords, bounds_span, M):
        B, N, D = coords.shape
        dtype, device = coords.dtype, coords.device

        noise = torch.randn(M, B, N, device=device)
        step = F.softplus(
            params_dict['step_mu'].unsqueeze(0) + noise * 0.5
        ) * self.log_scale.exp()

        sparsity = torch.sigmoid(params_dict['sparsity_logit'])
        dim_scores = torch.randn(M, B, N, D, device=device, dtype=dtype)
        shift = torch.erfinv(2 * (1 - sparsity).clamp(0.01, 0.99) - 1) * 1.414
        dim_mask = torch.sigmoid(10.0 * (dim_scores - shift.unsqueeze(0).unsqueeze(-1)))

        sign = (torch.rand(M, B, N, D, device=device, dtype=dtype) < 0.5).to(dtype) * 2 - 1
        delta = sign * step.unsqueeze(-1).to(dtype) * dim_mask
        return delta


# Keep old name importable for existing tests
BatchedDiffMTSLS1 = BatchedDiffCoordLS


class NeuralCoordLS(nn.Module):
    """Neural-directed coordinate-wise local search.

    Uses per-dimension coordinate features scored by a shared linear layer
    to bias which dimensions are probed. Fitness-coordinate correlation
    informs the probe direction (sign).

    Dimension-agnostic: features are per-dim marginals, scorer is shared.
    """
    N_COORD_FEATURES = 4

    def __init__(self, embed_dim=16, head_idx=1, backbone_dim=0):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_idx = head_idx
        self.backbone_dim = backbone_dim
        self.proj, self.proj_norm = _make_proj(backbone_dim, embed_dim)

        full_dim = embed_dim + backbone_dim
        self.step_mlp = _ParamMLP(full_dim, 1, hidden=64)
        self.sparsity_mlp = _ParamMLP(full_dim, 1, hidden=64)
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)))

        scorer_in = embed_dim + self.N_COORD_FEATURES
        self.dim_scorer = nn.Linear(scorer_in, 1)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.sign_threshold = 0.15

    def _coord_features(self, coords, fitness):
        """Build per-dim features: (B, N, D, N_COORD_FEATURES). All marginal."""
        B, N, D = coords.shape
        c_float = coords.float()
        c_mean = c_float.mean(dim=1, keepdim=True)
        c_std = c_float.std(dim=1, keepdim=True).clamp(min=1e-8)
        c_norm = (c_float - c_mean) / c_std

        pop_std = c_std.expand(B, N, D)
        pop_std_norm = pop_std / pop_std.max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        ranks = c_float.argsort(dim=1).argsort(dim=1).float() / max(N - 1, 1)

        fit_float = fitness.float()
        fit_centered = fit_float - fit_float.mean(dim=1, keepdim=True)
        coord_centered = c_float - c_mean
        cov_fc = (fit_centered.unsqueeze(-1) * coord_centered).mean(dim=1)
        fit_std_scalar = fit_centered.std(dim=1, correction=0).clamp(min=1e-8)
        c_std_biased = coord_centered.std(dim=1, correction=0).clamp(min=1e-8)
        pearson = (cov_fc / (fit_std_scalar.unsqueeze(-1) * c_std_biased)).clamp(-1, 1)

        abs_pearson = pearson.abs().unsqueeze(1).expand(B, N, D)
        features = torch.stack([c_norm, pop_std_norm, ranks, abs_pearson], dim=-1)
        return features, pearson

    def get_embedding(self, h_backbone):
        if self.proj is not None:
            return self.proj_norm(self.proj(h_backbone))
        return h_backbone[..., :self.embed_dim]

    def compute_params(self, h_out, coords, fitness, adj=None,
                       route_probs=None, bounds_span=200.0, h_backbone=None, **_kwargs):
        if h_backbone is not None and self.proj is not None:
            h_out = self.get_embedding(h_backbone)
        B, N, D = coords.shape

        h_in = torch.cat([h_out, h_backbone], dim=-1) if h_backbone is not None else h_out
        step_mu = self.step_mlp(h_in).squeeze(-1)
        sparsity_logit = self.sparsity_mlp(h_in).squeeze(-1)

        coord_feats, pearson = self._coord_features(coords, fitness)
        h_exp_bcast = h_out.unsqueeze(2).expand(B, N, D, -1)
        scorer_in = torch.cat([h_exp_bcast, coord_feats.float()], dim=-1)
        dim_bias = self.dim_scorer(scorer_in).squeeze(-1)

        confident = pearson.abs() > self.sign_threshold
        sign_dir = -pearson.sign()
        sign_bias = torch.where(confident, sign_dir, torch.zeros_like(sign_dir))
        sign_bias = sign_bias.unsqueeze(1).expand(B, N, D)

        sensitivity_target = pearson.abs().unsqueeze(1).expand(B, N, D).detach()

        return {
            'step_mu': step_mu,
            'sparsity_logit': sparsity_logit,
            'dim_bias': dim_bias,
            'sign_bias': sign_bias,
            '_sensitivity_target': sensitivity_target,
        }

    def sample_batch(self, params_dict, coords, bounds_span, M):
        B, N, D = coords.shape
        dtype, device = coords.dtype, coords.device

        noise = torch.randn(M, B, N, device=device)
        step = F.softplus(
            params_dict['step_mu'].unsqueeze(0) + noise * 0.5
        ) * self.log_scale.exp()

        sparsity = torch.sigmoid(params_dict['sparsity_logit'])
        dim_scores_rand = torch.randn(M, B, N, D, device=device, dtype=dtype)
        alpha = torch.sigmoid(self.alpha)
        dim_scores = dim_scores_rand + alpha * params_dict['dim_bias'].unsqueeze(0).to(dtype)

        shift = torch.erfinv(2 * (1 - sparsity).clamp(0.01, 0.99) - 1) * 1.414
        dim_mask = torch.sigmoid(10.0 * (dim_scores - shift.unsqueeze(0).unsqueeze(-1)))

        sign_bias = params_dict['sign_bias'].unsqueeze(0).to(dtype)
        random_sign = (torch.rand(M, B, N, D, device=device, dtype=dtype) < 0.5).to(dtype) * 2 - 1
        has_bias = (sign_bias.abs() > 0.5)
        sign = torch.where(has_bias, sign_bias.expand_as(random_sign), random_sign)

        delta = sign * step.unsqueeze(-1).to(dtype) * dim_mask
        return delta
