# Unified Encoder Architecture

## Data Flow

```
Population: x(N,D), fitness(N,)
         │
    ┌────┴──────────────────────────────────┐
    ▼                                       ▼
 graph_features.py                   augment_node_features_lineage()
 compute_shared_intermediates()      op_onehot, disp_mag, impr_ratio,
 coords_norm, fit_rank, knn_idx,    parent_rank → scaled to [-1,1]
 gradient_consistency, nbc, etc.
    │                                       │
    ├── NODE(9 base) + 7 lineage ───────────┘
    ├── EDGE(9)
    └── GLOBAL(16)
         │
╔════════╧═══════════════════════════════════════════════════╗
║  PNA ENCODER (pna_model.py)                               ║
║  4 layers, hidden=64, out=32, ~594K params                ║
║  Aggs: [mean, max, min, std]                              ║
║  Scalers: [identity, amplify, attenuate]                  ║
║  FiLM: global(16) → γ,β per layer (node + edge)          ║
║  GraphNorm + residual + dropout                           ║
║                                                           ║
║  RETURNS:                                                 ║
║    h_nodes(N,32), h_edges(E,32),                          ║
║    h_global(B,32), h_node_edge(N,32)                      ║
╚════════╤═══════════════════════════════════════════════════╝
         │
╔════════╧═══════════════════════════════════════════════════╗
║  PNAGATv2Backbone.encode() (backbone.py)                  ║
║                                                           ║
║  Bridges:                                                 ║
║    h = node_bridge(cat(h_nodes, h_node_edge))  → (N, 64) ║
║    e = edge_bridge(h_edges)                    → (E, 64) ║
║    h_g = global_bridge(h_global)               → (B, 64) ║
║    h += h_g[v_indices]                                    ║
║                                                           ║
║  GATv2 × L layers (n_heads parametrizable):               ║
║    for i, layer in enumerate(gatv2_layers):                ║
║        h, e = GATv2ConcatLayer(h, edge_index, e)          ║
║        if i < L-1:                                        ║
║            h += h_g[v_indices]    ← re-injection          ║
║                                                           ║
║  Edge readout:                                            ║
║    h += edge_readout(scatter_mean(e, dst))                ║
║                                                           ║
║  RETURNS 4-tuple:                                         ║
║    h          (N, 64)              concat of all heads     ║
║    e          (E, 64)              edge embeddings         ║
║    h_per_head (N, n_heads, 64//n_heads)  raw attn heads   ║
║    h_global   (B, 32)              PNA graph readout       ║
╚════════╤═══════════════════════════════════════════════════╝
         │
    x_dec = cat([h, h_global[v_idx]])  → (N, 96)
         │
╔════════╧═══════════════════════════════════════════════════╗
║  DECISION HEADS (per-variant, each imports backbone)      ║
║                                                           ║
║  GNN_MOS_Classic (K=2, n_heads=2):                        ║
║    ls1_head(x_dec) → sigmoid → per-node LS1 budget        ║
║                                                           ║
║  NEURAL_META_K4 (K=4, n_heads=4):                         ║
║    HeadAlignedRouting(h_per_head) → K=4 logits            ║
║    DiffOperator heads → displacements                     ║
║                                                           ║
║  HyperOPT (K=6, n_heads=4):                               ║
║    CrossDimAttn + 6 kernel heads → deltas                 ║
║    KernelRouter(h) → K=6 weights                          ║
║                                                           ║
║  NEURAL_ELA_MOS (K=4, n_heads=4):                         ║
║    backbone.encode() + DimTemporal merge                  ║
║    Same K4 heads on enriched h                            ║
║                                                           ║
║  NEURAL_META_K2 (K=2, n_heads=2):                         ║
║    SwitchHead(x_dec) → per-node binary SHADE/LS1          ║
║                                                           ║
║  ENSEMBLE_K4 (K=4, n_heads=4):                            ║
║    PerNodeAllocationHead(x_dec) → per-node K=4 categorical║
╚════════════════════════════════════════════════════════════╝
```

## Parameter Summary

| Component | Params | Location |
|-----------|--------|----------|
| PNA encoder | ~594K | encoder/pna_model.py |
| Bridges (node+edge+global) | ~12K | encoder/backbone.py |
| GATv2 × 2 layers | ~50K | encoder/backbone.py via gatv2_layer.py |
| Edge readout | ~4K | encoder/backbone.py |
| **Backbone total** | **~660K** | |
| Decision heads | 1-35K | Per-variant |

## Key Invariants

1. `encode()` always returns `(h, e, h_per_head, h_global)` — 4-tuple
2. `h_global` is re-injected between GATv2 layers (not just once before)
3. `n_heads` is a backbone param; `K` is a variant param — decoupled
4. No scatter_mean pooling for decisions — all heads operate per-node
5. `x_dec = cat([h(64), h_global(32)]) = (N, 96)` is the standard head input
