"""CMA-ES operator heads: BatchedDiffCMAES and NeuralCMAES."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder.operators._base import _ParamMLP, _make_proj


class BatchedDiffCMAES(nn.Module):
    """Batched CMA-ES: batched Cholesky, no per-graph loops."""

    def __init__(self, embed_dim=16, head_idx=2, backbone_dim=0):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_idx = head_idx
        self.backbone_dim = backbone_dim
        self.proj, self.proj_norm = _make_proj(backbone_dim, embed_dim)

        full_dim = embed_dim + backbone_dim
        self.param_mlp = _ParamMLP(full_dim, 2, hidden=64)
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.cov_temp = nn.Parameter(torch.tensor(1.0))

    def get_embedding(self, h_backbone):
        if self.proj is not None:
            return self.proj_norm(self.proj(h_backbone))
        return h_backbone[..., :self.embed_dim]

    def compute_params(self, h_out, coords, fitness, adj=None,
                       route_probs=None, bounds_span=200.0, h_backbone=None, **_kwargs):
        if h_backbone is not None and self.proj is not None:
            h_out = self.get_embedding(h_backbone)
        B, N, D = coords.shape
        dtype = coords.dtype

        h_in = torch.cat([h_out, h_backbone], dim=-1) if h_backbone is not None else h_out
        params_raw = self.param_mlp(h_in)
        sigma_mu = params_raw[..., 0]
        sigma_logsig = params_raw[..., 1]

        fit_std = fitness.std(dim=-1, keepdim=True).clamp(min=1e-8)
        w = F.softmax(-fitness / (fit_std * self.cov_temp.abs().clamp(min=0.1)), dim=-1)

        # Detach coords in Cholesky path: nonlinear amplifier, unbounded Jacobian
        mu = (w.unsqueeze(-1).to(dtype) * coords.detach()).sum(dim=1, keepdim=True)
        centered = coords.detach() - mu
        weighted_centered = centered * w.unsqueeze(-1).to(dtype)
        cov = torch.bmm(weighted_centered.transpose(1, 2), centered)
        cov = cov + 1e-4 * torch.eye(D, device=coords.device, dtype=dtype).unsqueeze(0)

        L, info = torch.linalg.cholesky_ex(cov)
        bad = (info > 0) | ~torch.isfinite(L).all(dim=-1).all(dim=-1)
        diag = torch.diag_embed(cov.diagonal(dim1=-2, dim2=-1).clamp(min=1e-6).sqrt())
        L = torch.where(bad.unsqueeze(-1).unsqueeze(-1), diag, L)

        return {'sigma_mu': sigma_mu, 'sigma_logsig': sigma_logsig, 'L': L}

    def sample_batch(self, params_dict, coords, bounds_span, M):
        B, N, D = coords.shape
        dtype, device = coords.dtype, coords.device

        noise_s = torch.randn(M, B, N, device=device)
        sigma = F.softplus(
            params_dict['sigma_mu'] + F.softplus(params_dict['sigma_logsig']) * noise_s
        ) * self.log_scale.exp()

        z = torch.randn(M, B, N, D, device=device, dtype=dtype)
        L = params_dict['L']
        shaped = torch.einsum('bde,mbne->mbnd', L, z)
        delta = sigma.unsqueeze(-1).to(dtype) * shaped
        return delta


class NeuralCMAES(nn.Module):
    """CMA-ES with per-individual mean shift via embedding attention.

    Extends BatchedDiffCMAES: shared Cholesky L (unchanged), but each
    individual gets a mutation center offset from attention-weighted
    population differences.
    """

    def __init__(self, embed_dim=16, head_idx=2, backbone_dim=0):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_idx = head_idx
        self.backbone_dim = backbone_dim
        self.proj, self.proj_norm = _make_proj(backbone_dim, embed_dim)

        # Smaller MLP than BatchedDiffCMAES (16 vs 64 hidden, embed-only input):
        # NeuralCMAES delegates per-individual adaptation to shift attention,
        # so param_mlp only needs to predict batch-level sigma.
        self.param_mlp = _ParamMLP(embed_dim, 2, hidden=16)
        with torch.no_grad():
            self.param_mlp.mlp[-1].weight.zero_()
            self.param_mlp.mlp[-1].bias.zero_()

        self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.cov_temp = nn.Parameter(torch.tensor(1.0))

        attn_dim = 8
        self.shift_query = nn.Linear(embed_dim, attn_dim, bias=False)
        self.shift_key = nn.Linear(embed_dim, attn_dim, bias=False)
        self.shift_scale = nn.Parameter(torch.tensor(0.0))

    def get_embedding(self, h_backbone):
        if self.proj is not None:
            return self.proj_norm(self.proj(h_backbone))
        return h_backbone[..., :self.embed_dim]

    def compute_params(self, h_out, coords, fitness, adj=None,
                       route_probs=None, bounds_span=200.0, h_backbone=None, **_kwargs):
        if h_backbone is not None and self.proj is not None:
            h_out = self.get_embedding(h_backbone)
        B, N, D = coords.shape
        dtype = coords.dtype

        params_raw = self.param_mlp(h_out)
        sigma_mu = params_raw[..., 0]
        sigma_logsig = params_raw[..., 1]

        # Shared Cholesky
        fit_std = fitness.std(dim=-1, keepdim=True).clamp(min=1e-8)
        w = F.softmax(-fitness / (fit_std * self.cov_temp.abs().clamp(min=0.1)), dim=-1)
        mu = (w.unsqueeze(-1).to(dtype) * coords.detach()).sum(dim=1, keepdim=True)
        centered = coords.detach() - mu
        weighted_centered = centered * w.unsqueeze(-1).to(dtype)
        cov = torch.bmm(weighted_centered.transpose(1, 2), centered)
        cov = cov + 1e-4 * torch.eye(D, device=coords.device, dtype=dtype).unsqueeze(0)
        L, info = torch.linalg.cholesky_ex(cov)
        bad = (info > 0) | ~torch.isfinite(L).all(dim=-1).all(dim=-1)
        diag = torch.diag_embed(cov.diagonal(dim1=-2, dim2=-1).clamp(min=1e-6).sqrt())
        L = torch.where(bad.unsqueeze(-1).unsqueeze(-1), diag, L)

        # Per-individual mean shift via attention
        q = self.shift_query(h_out)
        k = self.shift_key(h_out)
        attn_logits = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(q.shape[-1])
        diag_mask = torch.eye(N, device=coords.device, dtype=torch.bool).unsqueeze(0)
        attn_logits = attn_logits.masked_fill(diag_mask, -1e9)
        # Z-score fit_bias so it has magnitude O(1), comparable to Q·K.
        # Previous `-fitness/fit_std` was softmax-safe (translation-invariant),
        # BUT its absolute magnitude (~1e5 on CEC2017) dominated Q·K (O(1)),
        # reducing attention to pure fitness-rank and ignoring learned Q·K.
        fit_mean = fitness.mean(dim=-1, keepdim=True)
        fit_bias = (-(fitness - fit_mean) / fit_std).clamp(min=-5, max=5).unsqueeze(1)
        attn_logits = attn_logits + 0.5 * fit_bias
        attn_weights = F.softmax(attn_logits, dim=-1)

        x_donor = torch.bmm(attn_weights.to(dtype), coords)
        raw_offset = x_donor - coords

        pop_scale = coords.float().std(dim=1).median(dim=-1).values
        pop_scale = pop_scale.clamp(min=1e-4).unsqueeze(-1).unsqueeze(-1)
        scale = torch.sigmoid(self.shift_scale)
        mu_offset = torch.tanh(raw_offset / pop_scale.to(dtype)) * scale * pop_scale.to(dtype)

        return {
            'sigma_mu': sigma_mu,
            'sigma_logsig': sigma_logsig,
            'L': L,
            'mu_offset': mu_offset,
            '_attn_logits': attn_logits,
        }

    def sample_batch(self, params_dict, coords, bounds_span, M):
        B, N, D = coords.shape
        dtype, device = coords.dtype, coords.device

        noise_s = torch.randn(M, B, N, device=device)
        sigma = F.softplus(
            params_dict['sigma_mu'] + F.softplus(params_dict['sigma_logsig']) * noise_s
        ) * self.log_scale.exp()

        z = torch.randn(M, B, N, D, device=device, dtype=dtype)
        L = params_dict['L']
        shaped = torch.einsum('bde,mbne->mbnd', L, z)

        mu_offset = params_dict['mu_offset'].unsqueeze(0)
        delta = sigma.unsqueeze(-1).to(dtype) * shaped + mu_offset
        return delta
