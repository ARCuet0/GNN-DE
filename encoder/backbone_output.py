"""
BackboneOutput — NamedTuple return type for the sparse GATv2 backbone family.

Fields:
    h            — node embeddings (B, N, gatv2_hidden)
    e            — edge features (B, N, k, gatv2_hidden)  sparse
    h_per_head   — per-head view (B, N, n_heads, head_dim)
    h_global     — population summary (B, global_out_dim)
    donor_logits — all-to-all donor selection logits (B, N, N, n_roles) or None
    h_pooled     — temporal pooled features (B, N, d_temporal) or None

The two Optional fields default to None so base `SparseGATv2Backbone` and
`TemporalSparseGATv2Backbone` can compose: base populates donor_logits, the
temporal wrapper appends h_pooled via `_replace(h_pooled=...)`.

NamedTuple is compatible with positional unpacking and `torch.func.functional_call`.
"""
from typing import NamedTuple, Optional

import torch


class BackboneOutput(NamedTuple):
    h:            torch.Tensor
    e:            torch.Tensor
    h_per_head:   torch.Tensor
    h_global:     torch.Tensor
    donor_logits:  Optional[torch.Tensor] = None
    h_pooled:      Optional[torch.Tensor] = None
    # D1000 line: when the kNN-restricted DonorSelectionKNN head is in use,
    # logits live on a (B, N, k_donor, R) sparse axis and donor_cand_idx
    # (B, N, k_donor) maps each local slot to its global donor index.
    # None when the legacy all-to-all DonorSelectionGATv2 head is in use.
    donor_cand_idx: Optional[torch.Tensor] = None
