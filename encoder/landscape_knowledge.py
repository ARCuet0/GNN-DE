"""Nadaraya-Watson landscape knowledge exploration loss (binary classification).

Predicts P(f(z) < best_f) at unseen test points using NW kernel regression
over binary labels derived from accumulated population observations.
Loss = BCE(predicted_prob, actual_label).

The NW estimator with binary labels y_i ∈ {0, 1}:
    P̂(z) = sum_i K(z, x_i) · y_i / sum_i K(z, x_i)
    K(z, x) = exp(-||z - x||² / (2σ²))

Implemented via softmax for numerical stability.

Gradient flows through x_obs (current gen coords) → backward through cdist
→ to GNN parameters. Labels and historical coords are detached.
"""

import torch
import torch.nn.functional as F


def nadaraya_watson_predict(
    x_obs: torch.Tensor,                    # (B, N_obs, D)
    y_obs: torch.Tensor,                    # (B, N_obs) — binary labels or values
    x_test: torch.Tensor,                   # (B, M_test, D)
    sigma: float | torch.Tensor,            # scalar or (B,) per-batch
) -> torch.Tensor:                           # (B, M_test)
    """Predict at test points using NW kernel regression.

    With binary y_obs ∈ {0,1}, output is P̂(improvement) ∈ [0,1].
    sigma can be a scalar (shared) or (B,) tensor (per-batch bandwidth).
    """
    dists_sq = torch.cdist(x_test, x_obs).pow(2)  # (B, M_test, N_obs)
    if isinstance(sigma, torch.Tensor):
        denom = 2.0 * (sigma * sigma).unsqueeze(-1).unsqueeze(-1)
    else:
        denom = 2.0 * sigma * sigma
    weights = F.softmax(-dists_sq / denom, dim=-1)
    return (weights * y_obs.unsqueeze(1)).sum(dim=-1)


def make_binary_labels(
    fitness: torch.Tensor,    # (B, N) or (B, M_test)
    threshold: torch.Tensor,  # (B,) — threshold per batch
) -> torch.Tensor:             # (B, N) float {0, 1}
    """Binary labels: 1 where f < threshold, 0 otherwise."""
    return (fitness < threshold.unsqueeze(-1)).float()


def landscape_knowledge_loss(
    x_obs: torch.Tensor,        # (B, N_obs, D)
    y_obs: torch.Tensor,        # (B, N_obs) — binary labels {0, 1}
    x_test: torch.Tensor,       # (B, M_test, D)
    y_test: torch.Tensor,       # (B, M_test) — binary labels {0, 1}
    sigma: float | torch.Tensor,  # scalar or (B,) per-batch
) -> torch.Tensor:               # scalar
    """BCE between NW-predicted improvement probability and actual labels."""
    p_pred = nadaraya_watson_predict(x_obs, y_obs, x_test, sigma)
    # Clamp predictions away from 0/1 for numerical stability in BCE
    p_pred = p_pred.clamp(1e-6, 1.0 - 1e-6)
    return F.binary_cross_entropy(p_pred, y_test)


def build_observation_set(
    coords: torch.Tensor,        # (B, N, D) — current gen, may have grad
    fitness: torch.Tensor,        # (B, N)
    coords_ring: torch.Tensor,   # (B, W, N, D) — ring buffer, detached
    fitness_ring: torch.Tensor,   # (B, W, N)
    n_valid: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge current-gen coords (with grad) + ring buffer history (detached).

    Returns (x_obs, f_obs) where x_obs[:, :N] = current gen (gradient flows),
    and x_obs[:, N:] = flattened ring buffer history (no gradient).
    fitness is always detached to prevent gradient shortcuts through fitness values.
    """
    B, N, D = coords.shape
    fitness = fitness.detach()
    if n_valid == 0:
        return coords, fitness
    ring_coords = coords_ring[:, :n_valid].reshape(B, n_valid * N, D)
    ring_fitness = fitness_ring[:, :n_valid].reshape(B, n_valid * N)
    x_obs = torch.cat([coords, ring_coords], dim=1)
    f_obs = torch.cat([fitness, ring_fitness], dim=1)
    return x_obs, f_obs


def adaptive_sigma(x_obs: torch.Tensor, D: int) -> torch.Tensor:
    """Compute per-batch kernel bandwidth from median pairwise distance.

    Returns (B,) tensor: sigma_b = median_pairwise_distance_b / sqrt(D).
    Each batch element gets its own bandwidth based on its population spread.
    """
    B, N_obs, _ = x_obs.shape
    n_sub = min(N_obs, 200)
    # Per-batch independent random subsamples
    idx = torch.stack([torch.randperm(N_obs, device=x_obs.device)[:n_sub]
                       for _ in range(B)])  # (B, n_sub)
    sub = x_obs.gather(1, idx.unsqueeze(-1).expand(-1, -1, x_obs.shape[-1]))  # (B, n_sub, D)
    dists = torch.cdist(sub, sub)  # (B, n_sub, n_sub)
    mask = ~torch.eye(n_sub, dtype=torch.bool, device=dists.device)
    medians = dists[:, mask].reshape(B, -1).median(dim=1).values
    return (medians / (D ** 0.5)).clamp(min=1e-6)


def sample_test_points(
    B: int, M_test: int, D: int, device: str | torch.device,
) -> torch.Tensor:
    """Sample M_test points uniformly in [-100, 100]^D."""
    return torch.rand(B, M_test, D, dtype=torch.float64, device=device) * 200.0 - 100.0
