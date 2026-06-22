"""Disentangle loss: 2D structured supervision (q_explor, q_exploit) + HSIC orthogonality.

Per user proposal 2026-05-04 (memory feedback_disentangle_explor_exploit_design_2026_05_04):

  q_explor(c)  = -‖t_c - x*_global‖_2     (oracle, training-only)
  q_exploit(c) = -‖t_c - x*_local(i)‖_2   (oracle, training-only)

The user requested disentanglement enforced by HSIC. Two regression heads on top of
h_aug predict each axis; HSIC penalty between predictions forces orthogonality.

Hypothesis (ii) UNIVERSAL (5 falsified workstreams) targets scalar/categorical CE.
This is 2D regression with orthogonality constraint — different mathematical object.

x*_local strategy:
  F1-F19  : pop-conditional (top-K=5 lowest-f members; basin proxies)
  F20-F28 : Voronoi over fn.shift_mat[k] composition centers (oracle)

CEC17 native autograd unreliable on multimodal landscapes (per user feedback);
RBF surrogate was attempted in disentangle_probe but extrapolates poorly. The
hybrid pop-conditional + shift_mat strategy is the production-grade alternative.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# HSIC (Hilbert-Schmidt Independence Criterion) — differentiable estimator
# ============================================================================

def gaussian_kernel(x, y, sigma):
    """K(x, y) = exp(-(x-y)^2 / (2σ²)) for vectors of scalars (1D each).

    x, y: (n,) vectors → returns (n, n) kernel matrix.
    """
    x = x.reshape(-1, 1)  # (n, 1)
    y = y.reshape(1, -1)  # (1, n)
    sqd = (x - y) ** 2
    return torch.exp(-sqd / (2.0 * sigma * sigma + 1e-12))


def hsic(x, y, sigma_x=None, sigma_y=None):
    """Differentiable HSIC estimator with Gaussian kernels.

    Args:
        x, y: (n,) prediction vectors (scalars per sample).
        sigma_x, sigma_y: kernel bandwidth. Default: median heuristic via batch std.
    Returns:
        scalar HSIC value (≥ 0). Higher = more dependent.
    """
    n = x.shape[0]
    if sigma_x is None:
        sigma_x = max(x.detach().std().item(), 1e-3)
    if sigma_y is None:
        sigma_y = max(y.detach().std().item(), 1e-3)

    Kx = gaussian_kernel(x, x, sigma_x)
    Ky = gaussian_kernel(y, y, sigma_y)
    H = torch.eye(n, device=x.device, dtype=x.dtype) - 1.0 / n
    return torch.trace(Kx @ H @ Ky @ H) / max((n - 1) ** 2, 1)


# ============================================================================
# q_explor / q_exploit oracle target computation
# ============================================================================

def _get_fid_int(fn, fid_hint=None):
    """Robust fid lookup. Prefer the explicit fid_hint (passed by train loop);
    fall back to fn.func_id (CEC2017Torch); finally try fn.fn.func_id (AugmentedFunction wraps a CEC2017Torch).
    """
    if fid_hint is not None:
        return int(fid_hint)
    if hasattr(fn, 'func_id'):
        return int(fn.func_id)
    inner = getattr(fn, 'fn', None)
    if inner is not None and hasattr(inner, 'func_id'):
        return int(inner.func_id)
    raise AttributeError("Could not determine fid: pass fid_hint explicitly")


def _get_shift_attr(fn, attr):
    """Pick fn.shift or fn.shift_mat; AugmentedFunction wraps a base CEC2017Torch."""
    if hasattr(fn, attr):
        return getattr(fn, attr)
    inner = getattr(fn, 'fn', None)
    if inner is not None and hasattr(inner, attr):
        return getattr(inner, attr)
    raise AttributeError(f"Could not access {attr} on fn (or fn.fn)")


def get_x_global(fn, ndim, fid_hint=None):
    """Global optimum location for CEC2017 fid.

    F1-F19: fn.shift (D,)
    F20-F28: fn.shift_mat[0] (D,) — primary basin per documented convention

    Handles both CEC2017Torch and AugmentedFunction (which wraps it).
    """
    fid = _get_fid_int(fn, fid_hint)
    if fid <= 19:
        return _get_shift_attr(fn, 'shift')
    sm = _get_shift_attr(fn, 'shift_mat')
    if sm.dim() == 2:
        return sm[0]
    return sm[0, :ndim]


def get_basin_proxies_per_batch(coords, fitness, fid, fn, ndim, k_pop=5):
    """Hybrid basin discovery, batched over (B,).

    Args:
      coords:  (B, N, D)
      fitness: (B, N)
      fid:     int (single fid per batch in train loop)
      fn:      CEC2017Torch instance
      ndim:    D
    Returns:
      basin_proxies: (B, K_basins, D)
        K_basins = k_pop for fid≤19, K_basins = shift_mat.shape[0] otherwise
    """
    if fid <= 19:
        # Pop-conditional: top-K=5 lowest-f members per batch
        k = min(k_pop, fitness.shape[1])
        topk_idx = torch.topk(fitness, k, largest=False).indices  # (B, k)
        basin_proxies = torch.gather(
            coords, 1,
            topk_idx.unsqueeze(-1).expand(-1, -1, ndim))
        return basin_proxies  # (B, k, D)

    # Composition: shift_mat[k] for k basins. shift_mat is shared across batch.
    # Cast to coords dtype to avoid Double-vs-Float cdist mismatch.
    sm = _get_shift_attr(fn, 'shift_mat').to(coords.dtype).to(coords.device)
    if sm.dim() == 2:
        shift_mat_d = sm  # (K, D)
    else:
        shift_mat_d = sm[:, :ndim]  # (K, D)
    B = coords.shape[0]
    return shift_mat_d.unsqueeze(0).expand(B, -1, -1).contiguous()


def compute_q_targets(coords_aug, parents, fitness, x_global, ndim,
                       N, M_proposals, k_pop=5):
    """Compute z-scored (q_explor, q_exploit) per proposal in augmented pop.

    Always uses pop-conditional basin proxies (top-K=k_pop lowest-fitness pop
    members) to avoid CEC2017 attribute access (handles vanilla + augmented).

    Args:
      coords_aug: (B, N + M_proposals, D) — augmented pop (parents + proposals)
                  Layout per opt_variant._run_surrogate: prop[m, n, k] sits at
                  index N + m*N*K + n*K + k. K=1 in deployed config.
      parents:    (B, N, D) — parent coords (for basin proxies)
      fitness:    (B, N) — pop fitness (for top-K basin proxies)
      x_global:   (D,) or (1, D) or (B, D) — global optimum location.
                  Use train_distributed._get_x_star which handles aug + composition.
      ndim:       D
      N:          parent count
      M_proposals: M_var × K (proposals per (m, parent, head))
    Returns:
      q_explor_z:  (B, N + M_proposals) z-scored per batch over all positions
      q_exploit_z: (B, N + M_proposals) z-scored per batch
    """
    B, N_aug, D = coords_aug.shape
    device = coords_aug.device
    dtype = coords_aug.dtype

    # Normalize x_global shape to (D,)
    xg = x_global.to(device).to(dtype)
    if xg.dim() == 2 and xg.shape[0] == 1:
        xg = xg[0]  # (1, D) → (D,)
    elif xg.dim() == 2 and xg.shape[0] == B:
        # Per-batch x_global (rare). Reshape for broadcast
        diff_g = coords_aug - xg.view(B, 1, D)
        q_explor = -torch.norm(diff_g, dim=-1)
        xg = None
    else:
        xg = xg.reshape(D)

    if xg is not None:
        diff_g = coords_aug - xg.view(1, 1, D)
        q_explor = -torch.norm(diff_g, dim=-1)  # (B, N_aug)

    # parents must match coords_aug dtype for cdist
    parents = parents.to(dtype)

    # Pop-conditional basin proxies always (avoids fn attribute access; works
    # for vanilla + augmented + composition).
    k = min(k_pop, fitness.shape[1])
    topk_idx = torch.topk(fitness, k, largest=False).indices  # (B, k)
    basin_proxies = torch.gather(
        parents, 1,
        topk_idx.unsqueeze(-1).expand(-1, -1, D))  # (B, k, D)
    K_basins = basin_proxies.shape[1]

    # For each parent in (B, N), find closest basin proxy
    d_parent_to_basin = torch.cdist(parents, basin_proxies)  # (B, N, K_basins)
    closest_basin_per_parent = d_parent_to_basin.argmin(dim=-1)  # (B, N)

    # Map each augmented-pop position to its parent index n.
    # Layout: positions 0..N-1 are parents (own self); N + m*N + n is proposal
    # m of parent n (assuming K=1).
    parent_for_each_pos = torch.cat([
        torch.arange(N, device=device),                          # parents
        torch.arange(N, device=device).repeat(M_proposals // N), # proposals
    ])  # (N_aug,)
    # Correct for case where M_proposals is not a multiple of N (defensive)
    if parent_for_each_pos.shape[0] < N_aug:
        # Pad with parent 0
        pad = N_aug - parent_for_each_pos.shape[0]
        parent_for_each_pos = torch.cat([parent_for_each_pos,
                                          torch.zeros(pad, device=device, dtype=torch.long)])
    parent_for_each_pos = parent_for_each_pos[:N_aug]

    # closest basin for each augmented-pop position (per batch)
    closest_basin_per_pos = closest_basin_per_parent[:, parent_for_each_pos]  # (B, N_aug)

    # Gather x*_local per (B, N_aug) position
    x_local_per_pos = torch.gather(
        basin_proxies, 1,
        closest_basin_per_pos.unsqueeze(-1).expand(-1, -1, D))  # (B, N_aug, D)

    diff_l = coords_aug - x_local_per_pos
    q_exploit = -torch.norm(diff_l, dim=-1)  # (B, N_aug)

    # Z-score per batch (across all N_aug positions)
    q_e_mean = q_explor.mean(dim=1, keepdim=True)
    q_e_std = q_explor.std(dim=1, keepdim=True).clamp(min=1e-9)
    q_explor_z = (q_explor - q_e_mean) / q_e_std

    q_x_mean = q_exploit.mean(dim=1, keepdim=True)
    q_x_std = q_exploit.std(dim=1, keepdim=True).clamp(min=1e-9)
    q_exploit_z = (q_exploit - q_x_mean) / q_x_std

    return q_explor_z, q_exploit_z


# ============================================================================
# Heads
# ============================================================================

class DisentangleHeads(nn.Module):
    """Two small MLPs reading h_aug → predicts (q_explor, q_exploit).

    Per-proposal application: input is h_aug (B, N_aug, hidden_dim).
    Hidden dim 128 (matches E7d backbone). Each head: Linear → ReLU → Linear → scalar.
    """
    def __init__(self, hidden_dim=128, mid_dim=128):
        super().__init__()
        self.h_explor = nn.Sequential(
            nn.Linear(hidden_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, 1),
        )
        self.h_exploit = nn.Sequential(
            nn.Linear(hidden_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, 1),
        )

    def forward(self, h_aug):
        """h_aug: (B, N_aug, 128) → (q_explor_pred, q_exploit_pred), each (B, N_aug)."""
        e = self.h_explor(h_aug).squeeze(-1)
        x = self.h_exploit(h_aug).squeeze(-1)
        return e, x


# ============================================================================
# Composed loss
# ============================================================================

def compute_disentangle_loss(h_aug, coords_aug, parents, fitness, x_global, ndim,
                              N, M_proposals, heads,
                              lambda_e=0.5, lambda_x=0.5, lambda_h=0.5,
                              k_pop=5, hsic_max_n=512,
                              random_target=False):
    """Compose disentangle loss: λ_e MSE_explor + λ_x MSE_exploit + λ_h HSIC.

    Args:
      h_aug:       (B, N_aug, 128) backbone output, augmented pop
      coords_aug:  (B, N_aug, D)
      parents:     (B, N, D)
      fitness:     (B, N)
      x_global:    (D,) or (1, D) or (B, D) — pre-computed x*_global location
                    (use train_distributed._get_x_star which handles aug + composition)
      ndim:        D
      N:           parent count
      M_proposals: M_var × K proposals per generation (excl parents)
      heads:       DisentangleHeads instance
      lambda_*:    loss weights
      k_pop:       top-K pop members for basin proxy (always pop-conditional)
      hsic_max_n:  cap HSIC samples per batch for kernel cost (memory)
    Returns:
      (loss_total, diag_dict)
    """
    # Compute targets (z-scored). Cast inputs to h_aug dtype to keep backward
    # graph dtype-consistent (h_aug may be bfloat16/float32; coords usually float64).
    h_dtype = h_aug.dtype
    coords_aug_h = coords_aug.to(h_dtype)
    parents_h = parents.to(h_dtype)
    fitness_h = fitness.to(h_dtype)
    q_e_oracle, q_x_oracle = compute_q_targets(
        coords_aug_h, parents_h, fitness_h, x_global, ndim, N, M_proposals, k_pop=k_pop)
    # (B, N_aug)
    q_e_oracle = q_e_oracle.to(h_dtype).detach()
    q_x_oracle = q_x_oracle.to(h_dtype).detach()

    if random_target:
        # Arm C ablation: replace oracle targets with per-step Gaussian noise of
        # same shape. Z-scored statistics match (mean=0, std=1) so MSE magnitude
        # is comparable to the oracle case. Tests whether disen's effect comes
        # from semantic content of targets vs. extra gradient flow alone.
        q_e_target = torch.randn_like(q_e_oracle).detach()
        q_x_target = torch.randn_like(q_x_oracle).detach()
    else:
        q_e_target = q_e_oracle
        q_x_target = q_x_oracle

    # Predict via heads
    q_e_pred, q_x_pred = heads(h_aug)  # each (B, N_aug)

    # MSE losses
    L_e = F.mse_loss(q_e_pred, q_e_target)
    L_x = F.mse_loss(q_x_pred, q_x_target)

    # HSIC on flat predictions (subsample for kernel cost if too large)
    flat_e = q_e_pred.reshape(-1)
    flat_x = q_x_pred.reshape(-1)
    if flat_e.shape[0] > hsic_max_n:
        idx = torch.randperm(flat_e.shape[0], device=flat_e.device)[:hsic_max_n]
        flat_e = flat_e[idx]
        flat_x = flat_x[idx]
    L_h = hsic(flat_e, flat_x)

    L_total = lambda_e * L_e + lambda_x * L_x + lambda_h * L_h

    # Diagnostics
    with torch.no_grad():
        # R² approximation per batch
        ss_res_e = ((q_e_pred - q_e_target) ** 2).sum()
        ss_tot_e = ((q_e_target - q_e_target.mean()) ** 2).sum().clamp(min=1e-9)
        r2_e = 1.0 - ss_res_e / ss_tot_e
        ss_res_x = ((q_x_pred - q_x_target) ** 2).sum()
        ss_tot_x = ((q_x_target - q_x_target.mean()) ** 2).sum().clamp(min=1e-9)
        r2_x = 1.0 - ss_res_x / ss_tot_x

        # Anti-collapse diagnostic: std of the heads' predictions. If the model
        # learns to output a constant under random_target, HSIC trivially → 0
        # without anti-leak detecting it (M1 collapse). std(pred) → 0 would
        # signal that.
        predstd_e = q_e_pred.detach().reshape(-1).float().std()
        predstd_x = q_x_pred.detach().reshape(-1).float().std()

        diag = {
            'disentangle_L_e': float(L_e.detach().item()),
            'disentangle_L_x': float(L_x.detach().item()),
            'disentangle_L_hsic': float(L_h.detach().item()),
            'disentangle_R2_explor': float(r2_e.detach().item()),
            'disentangle_R2_exploit': float(r2_x.detach().item()),
            'disentangle_total': float(L_total.detach().item()),
            'disentangle_predstd_explor': float(predstd_e.item()),
            'disentangle_predstd_exploit': float(predstd_x.item()),
        }

        if random_target:
            def _safe_corr(a, b, eps=1e-9):
                # Guard against constant tensors (degenerate populations) which
                # would make corrcoef return NaN. Returns 0.0 in that case.
                a = a.reshape(-1).float()
                b = b.reshape(-1).float()
                if a.std() < eps or b.std() < eps:
                    return torch.zeros((), device=a.device)
                return torch.corrcoef(torch.stack([a, b]))[0, 1]
            cor_e = _safe_corr(q_e_target, q_e_oracle)
            cor_x = _safe_corr(q_x_target, q_x_oracle)
            diag['antileak_cor_explor'] = float(cor_e.detach().item())
            diag['antileak_cor_exploit'] = float(cor_x.detach().item())

    return L_total, diag
