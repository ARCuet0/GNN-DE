"""BatchedNoOp: zero-delta operator (LPSR-equivalent preservation)."""
import torch
import torch.nn as nn


class BatchedNoOp(nn.Module):
    """No-op: preserve individual unchanged. Zero compute, zero params.

    Equivalent to L-SHADE's LPSR — the router learns which individuals
    should NOT be perturbed (elite preservation, stagnation detection,
    effective population reduction).
    """

    def __init__(self, embed_dim=16, head_idx=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_idx = head_idx
        self.proj = None
        self.proj_norm = None

    def get_embedding(self, h_backbone):
        """NoOp returns zeros — no projection needed."""
        B, N = h_backbone.shape[:2]
        return torch.zeros(B, N, self.embed_dim,
                           device=h_backbone.device, dtype=h_backbone.dtype)

    def compute_params(self, h_out, coords, fitness, adj=None,
                       route_probs=None, bounds_span=200.0, h_backbone=None, **_kwargs):
        return {}

    def sample_batch(self, params_dict, coords, bounds_span, M):
        B, N, D = coords.shape
        return torch.zeros(M, B, N, D, dtype=coords.dtype, device=coords.device)


# Keep old SBX importable for backward compat of existing tests
BatchedDiffSBX = BatchedNoOp
