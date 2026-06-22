# Per-method wall-clock — MetaBox Protein-Docking @ 2000 FES (12D)

Single optimization run (one instance, full 2000 FES), averaged over 3 seeds,
1 CPU thread, SAME hardware (Magerit tersq.sif container). Warm-up run discarded.
Measured 2026-06-21.

| Method        | s / run | × vs fastest | × vs GNN-DE |
|---------------|--------:|-------------:|------------:|
| GNN-DE        |   9.756 |        9.1×  |       1.00× |
| RLDE-AFL      |   2.310 |        2.1×  |       0.24× |
| MadDE         |   1.221 |        1.1×  |       0.13× |
| NL-SHADE-LBC  |   1.075 |        1.0×  |       0.11× |
| Random_search |   1.106 |        1.0×  |       0.11× |

GNN-DE is ~8-9× slower per run than the DE/PSO classics and ~4× slower than the
other learned method (RLDE-AFL). Cause: the O(N^2) all-to-all donor head plus the
augmented-population second backbone forward (see CLAUDE.md complexity note).

GPU (local RTX 4070 Super, GNN-DE only — classics have no GPU path):
GNN-DE drops to 1.34 s/run (CPU-1thread on the SAME local box = 5.61 s, so ~4.2×
faster on GPU). On GPU vs CPU classics the gap narrows to ~2×.

Full 280-instance x 51-seed sweep, single core (s/run x 14280):
GNN-DE ~38.7 h, RLDE-AFL ~9.2 h, MadDE ~4.8 h, RS ~4.4 h, NL-SHADE-LBC ~4.3 h.
This is why the run is sharded across Magerit + local.
