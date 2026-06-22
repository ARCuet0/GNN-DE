"""CLI argument parser for train_distributed.py."""
import argparse


def _lpsr_n_min_type(s):
    """Validate --lpsr-N-min: BatchedDiffAttDE asserts N >= 3 at runtime."""
    v = int(s)
    if v < 3:
        raise argparse.ArgumentTypeError(
            f'--lpsr-N-min must be >= 3 (BatchedDiffAttDE asserts N >= 3 '
            f'in compute_params), got {v}')
    return v


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description='Function-parallel distributed L2O training')

    a = p.add_argument_group('architecture')
    a.add_argument('--topology', choices=['embedding_knn', 'learned_scorer',
                   'coordinate_knn', 'torch_knn'], default='embedding_knn',
                   help='Graph topology. torch_knn (D1000 line) uses '
                        'strict-O(N*k) NN-Descent approximate kNN; falls back '
                        'to exact cdist for N <= --knn-fallback-n.')
    a.add_argument('--knn-n-iters', type=int, default=3,
                   help='[D1000] NN-Descent refinement iterations '
                        '(only used with --topology torch_knn).')
    a.add_argument('--knn-fallback-n', type=int, default=64,
                   help='[D1000] If N <= this, torch_knn falls back to exact '
                        'cdist (small-N parity guarantee).')
    a.add_argument('--knn-seed', type=int, default=0,
                   help='[D1000] Seed for the random init of NN-Descent.')
    a.add_argument('--donor-kind', choices=['all2all', 'knn'],
                   default='all2all',
                   help='[D1000] all2all = legacy DonorSelectionGATv2 '
                        '(O(N^2)). knn = DonorSelectionKNN (kNN-restricted '
                        'donor pool, strict O(N*k_donor)). Required for the '
                        'D1000 lema.')
    a.add_argument('--donor-pbest-frac', type=float, default=0.1,
                   help='[D1000] Fraction of population in pbest_pool for '
                        'kNN-restricted donor (only with --donor-kind knn).')
    a.add_argument('--donor-chunk-size', type=int, default=0,
                   help='[D1000] When > 0 with --donor-kind all2all, '
                        'chunked all-to-all donor: splits N_q into chunks of '
                        'this size, capping peak memory at O(C*N*R*d). '
                        'Bit-exact to monolithic. Compute total stays '
                        'O(N^2) (does not honor the strict-O(N*k) lema, but '
                        'preserves architecture and ckpt semantics). 0 = '
                        'monolithic (default). Ignored under --donor-kind '
                        'knn.')
    a.add_argument('--surrogate-augment', choices=['rebuild', 'delta'],
                   default='rebuild',
                   help='[D1000] How to build the augmented (parents+proposals) '
                        'graph in _run_surrogate. rebuild = call _build_graph_cache '
                        'on the full augmented coords (still triggers cdist). '
                        'delta = inherit parent kNN per proposal '
                        '(strict-O(N_aug*k)). Required for the D1000 lema.')
    a.add_argument('--d-rnn', type=int, default=64)
    a.add_argument('--temporal-layers', type=int, default=2)
    a.add_argument('--gatv2-hidden', type=int, default=128)
    a.add_argument('--gatv2-layers', type=int, default=3)
    a.add_argument('--n-heads', type=int, default=8)
    a.add_argument('--k-neighbors', type=int, default=8)
    a.add_argument('--pooler', choices=['induced', 'mean'], default='induced')
    a.add_argument('--n-induced', type=int, default=8)
    a.add_argument('--pool-dim', type=int, default=0)
    a.add_argument('--gru-window', type=int, default=16)
    a.add_argument('--dropout', type=float, default=0.1)
    a.add_argument('--operators', choices=['gated', 'classic', 'direct',
                   'neural', 'neural_att', 'k5', 'k5_att', 'k2', 'k1'],
                   default='gated')
    a.add_argument('--backbone-type',
                   choices=['sparse_gatv2', 'set_attention',
                            'set_attention_edge', 'identity'],
                   default='sparse_gatv2',
                   help='Inner message-passing backbone. sparse_gatv2 '
                        '(default, deployed) = O(N*k) sparse GATv2 over the '
                        'k-NN graph. set_attention = graph-vs-set ablation '
                        '(no edges, no topology). set_attention_edge = B2 '
                        'arm of the 2026-05-29 topology/edges ablation '
                        '(all-to-all self-attn with 3-d dense edge bias). '
                        'identity = C arm of the same ablation (relational '
                        'floor: per-node Linear, no relational layers; '
                        'donor head retained).')

    t = p.add_argument_group('training')
    t.add_argument('--steps', type=int, default=999999)
    t.add_argument('--patience', type=int, default=0)
    t.add_argument('--b-per-gpu', type=int, default=0,
                   help='Populations per GPU. 0 = auto-detect from VRAM')
    t.add_argument('--m-samples', type=int, default=20)
    t.add_argument('--D', type=int, nargs='+', default=[10])
    t.add_argument('--N', type=int, default=50)
    t.add_argument('--budget-mult', type=int, default=1000)
    t.add_argument('--min-fes', type=int, default=500)
    t.add_argument('--gumbel-tau', type=float, default=1.0)
    t.add_argument('--lr-backbone', type=float, default=3e-5)
    t.add_argument('--lr-variant', type=float, default=3e-4)
    t.add_argument('--fit-damp', type=float, default=0.95)
    t.add_argument('--bptt-w-init', type=int, default=20)
    t.add_argument('--bptt-w-min', type=int, default=20)
    t.add_argument('--bptt-w-max', type=int, default=5000)
    t.add_argument('--bptt-chunk', type=int, default=16)
    t.add_argument('--grad-clip', type=float, default=10.0)
    t.add_argument('--weight-decay', type=float, default=0.0)
    t.add_argument('--fn-downweight', type=str, default='',
                   help='Comma-sep fid:mult pairs, e.g. "13:0.5,19:0.5"')
    t.add_argument('--hitting-bb-scale', type=float, default=0.0)
    t.add_argument('--same-function', action='store_true')
    t.add_argument('--single-fn', type=int, default=0)
    t.add_argument('--benchmark', choices=['cec2017', 'augmented'],
                   default='cec2017')
    t.add_argument('--env', choices=['cec2017', 'linked-flame', 'bbob'], default='cec2017',
                   help='Training optimization environment. cec2017 (default) '
                        'uses CEC2017 with random affine augmentation. '
                        'linked-flame uses the 5-Level RBF curriculum '
                        '(see encoder/linked_flame.py). bbob uses the 24 BBOB '
                        '(COCO-2009) functions, differentiable torch port '
                        '(encoder/bbob_torch.py), variety via native instances; '
                        'domain [-5,5] — pass --lb -5 --ub 5. Eval pipeline is '
                        'unaffected — held-out evaluation always uses CEC2017.')
    t.add_argument('--lf-levels', type=str, default='1,2,3,4,5',
                   help='Comma-separated linked-flame Levels to enable '
                        '(subset of {1,2,3,4,5}). Ignored unless '
                        '--env linked-flame.')

    l = p.add_argument_group('loss')
    l.add_argument('--loss', choices=['log1p_linear', 'log_gap',
                   'adaptive_log1p'], default='log1p_linear')
    l.add_argument('--loss-knee', type=float, default=5.0)
    l.add_argument('--geo-weight', type=float, default=0.0)
    l.add_argument('--aux-weight', type=float, default=0.0)
    l.add_argument('--lupi-grad', action='store_true')
    l.add_argument('--no-curriculum', action='store_true')
    l.add_argument('--task-mix-vanilla-prob', type=float, default=None,
                   help='Probe mode: replace curriculum + pool weighting with '
                        'a per-step Bernoulli(vanilla_prob) over a random fid. '
                        'Implies --no-curriculum semantics on the selection '
                        'side. Default None = use task pool.')
    l.add_argument('--n-targets', type=int, default=15)
    l.add_argument('--start-target', type=int, default=0)
    l.add_argument('--improvement-weight', type=float, default=0.3)
    l.add_argument('--contrafactual', default=True,
                   action=argparse.BooleanOptionalAction)
    l.add_argument('--contra-weight', type=float, default=0.02)
    l.add_argument('--cf-loss-weight', type=float, default=0.0,
                   help='Per-slot cf_improvement loss weight (Arm 1, '
                        'discussion 2026-05-02). 0.0 = disabled (deployed). '
                        '0.3 = consensus discussion arm. Adds '
                        'compute_cf_improvement_loss(parent_fit, off_fit) '
                        'to the chunk loss.')
    l.add_argument('--cf-loss-normalize', default='rank_ste',
                   choices=['rank_ste', 'sigma_batch'],
                   help='Heavy-tail control inside compute_cf_improvement_loss.')
    l.add_argument('--gate-bce-weight', type=float, default=0.1,
                   help='Multiplier for legacy ActivityGate BCE / RankerGate '
                        'pairwise loss. NOT used by the surrogate path — see '
                        '--surrogate-loss-weight. Inert under '
                        '--gate-type surrogate.')
    l.add_argument('--surrogate-loss-weight', type=float, default=0.0,
                   help='Multiplier for the PairwiseSurrogate ranking loss '
                        '(only path that trains the proposal scorer under '
                        '--gate-type surrogate). Default 0.0: surrogate is '
                        'left at init. Set to 0.1 to enable training.')
    l.add_argument('--fcr-beta-weight', type=float, default=0.0,
                   help='Weight for Beta F/CR supervised loss (0=off)')
    l.add_argument('--fcr-oracle-mode', choices=['grid', 'from_m'], default='from_m',
                   help='F/CR supervision source: grid (legacy; spends extra FES '
                        'on a separate F/CR grid evaluation) or from_m (uses the '
                        'realized F/CR of best-m among the M proposals; FES-free). '
                        'Default: from_m.')
    l.add_argument('--disentangle-lambda-e', type=float, default=0.0,
                   help='[2026-05-04 disentangle] Weight for q_explor MSE loss '
                        '(2D structured supervision: distance to x*_global). '
                        '0.0 = disabled. Recommended initial: 0.5. Requires '
                        '--gate-type surrogate (uses h_aug from augmented pop forward).')
    l.add_argument('--disentangle-lambda-x', type=float, default=0.0,
                   help='[2026-05-04 disentangle] Weight for q_exploit MSE loss '
                        '(distance to x*_local). 0.0 = disabled. Recommended initial: 0.5.')
    l.add_argument('--disentangle-lambda-h', type=float, default=0.0,
                   help='[2026-05-04 disentangle] Weight for HSIC orthogonality '
                        'penalty between q_explor and q_exploit predictions. '
                        '0.0 = no disentanglement enforced (axes free to entangle). '
                        'Recommended initial: 0.5.')
    l.add_argument('--disentangle-k-pop', type=int, default=5,
                   help='[2026-05-04 disentangle] Top-K lowest-fitness pop members '
                        'used as basin proxies for x*_local (F1-F19 only; F20+ uses '
                        'shift_mat). Default 5.')
    l.add_argument('--disen-random-target', action='store_true', default=False,
                   help='[2026-05-06 ablation arm C] Replace oracle q_explor/q_exploit '
                        'targets with per-step resampled Gaussian tensors of same shape. '
                        'Tests M5 hypothesis: is the disen ~8% effect from extra '
                        'gradient flow alone, or from semantic content of the targets? '
                        'Requires --disentangle-lambda-{e,x} > 0 to have effect.')
    l.add_argument('--disen-grad-clip', type=float, default=0.0,
                   help='[2026-05-08 A.1c mitigation] Per-component gradient clip on '
                        'disen aux loss. When >0, the chunk backward path splits '
                        '(hit+other_geo+cf) from (disen) and backprops each separately, '
                        'clipping the disen gradient to this max_norm BEFORE summing '
                        'into the accumulated p.grad. Mitigates A.0 finding: disen aux '
                        'amplifies explosion rate 2.4x vs null. Default 0=off (current '
                        'behavior, single backward + post-allreduce clip).')
    l.add_argument('--donor-oracle-weight', type=float, default=0.0,
                   help='CE weight for donor-attention oracle loss across M proposals. '
                        'Pushes A_pbest/A_r1/A_r2 logits toward the pbest/r1/r2 indices '
                        'that best-m used. Requires --per-m-donors. 0=off (default).')
    l.add_argument('--donor-w-pbest', type=float, default=1.0,
                   help='Per-component weight on A_pbest CE loss inside '
                        'compute_donor_oracle_loss. Default 1.0.')
    l.add_argument('--donor-w-r1', type=float, default=1.0,
                   help='Per-component weight on A_r1 CE loss. Default 1.0.')
    l.add_argument('--donor-w-r2', type=float, default=1.0,
                   help='Per-component weight on A_r2 loss (CE or soft, per '
                        '--donor-r2-mode). Default 1.0.')
    l.add_argument('--donor-r2-mode', choices=['ce', 'soft', 'off'], default='ce',
                   help='A_r2 supervision mode. ce: CE against best-m r2 index '
                        '(may fight the bad-region inductive bias). soft: CE '
                        'against uniform distribution over bottom-K (by fitness) '
                        'parents excluding self — aligned with r2 semantics. '
                        'off: no r2 gradient (overrides w_r2=0).')
    l.add_argument('--donor-r2-soft-frac', type=float, default=0.3,
                   help='Fraction of population in bottom-K soft target when '
                        '--donor-r2-mode=soft. Default 0.3.')
    # ── KL distill (Etapa A) — donor_selector → L-SHADE atomic soft target ──
    l.add_argument('--kl-distill-weight', type=float, default=0.0,
                   help='Weight for KL distillation of donor_selector logits '
                        'against L-SHADE atomic soft target (FES-aware adaptive '
                        'p). 0=off (default). Etapa A canonical: 0.3.')
    l.add_argument('--kl-p-max', type=float, default=0.2,
                   help='pbest top-p%% upper bound at fes_progress=0. Default 0.2 '
                        '(L-SHADE canonical).')
    l.add_argument('--kl-p-min', type=float, default=0.05,
                   help='pbest top-p%% lower bound at fes_progress=1. Default '
                        '0.05 (L-SHADE canonical).')
    l.add_argument('--kl-w-pbest', type=float, default=1.0,
                   help='Per-component weight on KL pbest loss. Default 1.0.')
    l.add_argument('--kl-w-r1', type=float, default=1.0,
                   help='Per-component weight on KL r1 loss. Default 1.0.')
    l.add_argument('--kl-w-r2', type=float, default=1.0,
                   help='Per-component weight on KL r2 loss. Default 1.0.')
    l.add_argument('--gate-type', choices=['adaptive', 'ranker', 'none', 'surrogate'],
                   default='adaptive',
                   help='Gate type: adaptive=threshold sigmoid, ranker=pairwise ranking, none=disabled, surrogate=pairwise surrogate.')
    l.add_argument('--gate-target-frac', type=float, default=0.5,
                   help='Fixed target_frac for ranker gate (fraction of N to activate).')
    l.add_argument('--gate-n-pairs', type=int, default=500,
                   help='Number of random pairs per batch for pairwise ranking loss.')
    l.add_argument('--threshold-quantile', type=float, default=0.0,
                   help='Adaptive quantile threshold for pairwise loss. '
                        '0=fixed threshold (default), 0.25=exclude bottom 25%% of pairs.')
    l.add_argument('--surrogate-M', type=int, default=0,
                   help='Candidates to evaluate in surrogate mode. 0=N. '
                        'When --surrogate-m-final > 0 this is the INITIAL '
                        'value at fes_frac=0 (linearly decays to that final).')
    l.add_argument('--surrogate-m-final', type=int, default=0,
                   help='LPSR-like decreasing population: linearly decay '
                        'surrogate_M from --surrogate-M (init) at fes_frac=0 '
                        'down to this value at fes_frac=1. 0=constant (no decay).')
    l.add_argument('--archive-capacity', type=int, default=0,
                   help='External per-batch FIFO archive of discarded parents '
                        '(L-SHADE style). 0 disables (default). When >0, '
                        'archive nodes participate in the donor pool of the '
                        'donor_selector via cand-mask-aware asymmetric '
                        'attention. See archive_design.md.')
    l.add_argument('--archive-evict', choices=['fifo', 'random'], default='fifo',
                   help='Archive eviction policy when full. fifo (default, E9) '
                        'overwrites oldest entry; random matches L-SHADE canonical.')
    l.add_argument('--donor-mode',
                   choices=['neural', 'lshade', 'lshade_masked'],
                   default='neural',
                   help='Donor (pbest, r1, r2) selection mode. neural (default) '
                        'uses backbone donor_selector via Gumbel-softmax. lshade '
                        'overrides logits with hand-crafted L-SHADE rules (uniform '
                        'within top-p_i pool, etc.) — used as E13 distillation '
                        'teacher. lshade_masked masks neural logits to the same '
                        'L-SHADE pool but PRESERVES neural preferences within '
                        '(inductive bias for E12).')
    l.add_argument('--lshade-pbest-max', type=float, default=0.11,
                   help='p_max for per-individual pbest sampling under '
                        '--donor-mode lshade. Each individual i samples '
                        'p_i ~ U(2/N, p_max) per generation, then takes top-'
                        'round(p_i*N) by fitness as the pbest pool. Default 0.11.')
    l.add_argument('--fcr-mode', choices=['beta', 'lshade', 'cauchy_neural'],
                   default='beta',
                   help='F/CR sampling mode. beta (default) uses AdaptiveFCRBeta head. '
                        'lshade uses AdaptiveFCRCauchy + LShadeMemory teacher (driver=teacher; '
                        'distillation training mode). cauchy_neural uses AdaptiveFCRCauchy with '
                        'Cauchy(μ_F_pred, 0.1) sampling (no teacher; inference post-distill).')
    l.add_argument('--fcr-distill-weight', type=float, default=0.0,
                   help='Online distillation MSE weight for AdaptiveFCRCauchy head: '
                        '|μ_F_pred - mean_M(F_realized)|² + |μ_CR_pred - mean_M(CR_realized)|². '
                        'Active under --fcr-mode lshade. Default 0.0 (off).')
    l.add_argument('--fcr-distill-mode', choices=['mse', 'cauchy_nll'],
                   default='mse',
                   help='Distillation loss form for the F head. mse (default) '
                        'matches mean_M(F_realized) — collapses heavy-tail to '
                        'mode (E13 mode-collapse fault). cauchy_nll matches '
                        'realized F samples under Cauchy(μ_F_pred, σ_F_pred); '
                        'requires --fcr-learn-sigma so the head emits σ_F. '
                        'Used by arm C of the 2026-04-28 falsification '
                        '(docs/falsification_2026_04_28.md).')
    l.add_argument('--fcr-learn-sigma', action='store_true',
                   help='Make AdaptiveFCRCauchy emit σ_F per-individual via a '
                        '3rd MLP channel (default off → σ_F fixed at 0.1). '
                        'Required for --fcr-distill-mode cauchy_nll.')
    l.add_argument('--legacy-tau-clamp', action='store_true',
                   help='Use the legacy tau.clamp(0.1, 5.0) on the BatchedDiffAttDE '
                        'Gumbel temperature instead of the gradient-preserving '
                        '0.1 + softplus(tau). Sets head.tau_mode="clamp". Used '
                        'by the control arm (A) of the falsification experiment.')
    l.add_argument('--lshade-memory-H', type=int, default=6,
                   help='Size of L-SHADE F/CR circular memory (per-batch). Default 6 (paper).')
    l.add_argument('--lpsr-N', dest='lpsr_N', action='store_true',
                   help='Linear Population Size Reduction over N. Population '
                        'shrinks from --N to --lpsr-N-min linearly with '
                        'cumulative_fes / step_fes. Reindexes all per-individual '
                        'state (coords_ring, fitness_ring, stagnation, etc.). '
                        'Required for L-SHADE-faithful training.')
    l.add_argument('--lpsr-N-min', dest='lpsr_N_min',
                   type=_lpsr_n_min_type, default=4,
                   help='Final population size at fes_frac=1 under --lpsr-N. '
                        'Default 4 (L-SHADE canonical). Must be >= 3 — '
                        'BatchedDiffAttDE.compute_params asserts this.')
    l.add_argument('--gate-node-feat', type=int, default=0,
                   help='Concat raw node_feat (8-dim) to gate input. 0=off, 8=on.')
    l.add_argument('--train-selection', type=str, default='topk',
                   help='Selection spec during training. Values: topk | uniform | '
                        'exp:LAM | weibull:K:LAM | power:ALPHA | random_1pp | '
                        'oracle_1pp | oracle_kpp:K | top1_1pp. Default topk '
                        'reproduces the surrogate-M LPSR behavior.')
    l.add_argument('--per-m-donors', action='store_true',
                   help='Option A: resample donor triple (pbest/r1/r2) per M sample '
                        'inside BatchedDiffAttDE. Default False (backward compat).')
    l.add_argument('--bias-only-pbest', action='store_true',
                   help='Architectural probe: A_pbest = alpha*fit_bias only '
                        '(drops h @ h^T similarity term). Tests whether the '
                        'h-based term is expressive or structurally saturated. '
                        'Default False.')
    l.add_argument('--donor-diag-always', action='store_true',
                   help='Always compute donor-oracle metrics (entropy, '
                        'agreement, CE loss trace) when --per-m-donors is on, '
                        'even if --donor-oracle-weight=0. For probe runs.')
    l.add_argument('--gate-l0-weight', type=float, default=0.0,
                   help='L0 regularization weight for Hard Concrete gate. 0=off.')
    l.add_argument('--oracle-router-weight', type=float, default=0.0,
                   help='Oracle CE loss weight for router. 0=off.')

    d = p.add_argument_group('search domain')
    d.add_argument('--lb', type=float, default=-100.0,
                   help='Search-domain lower bound. CEC2017: -100. BBOB: -5. '
                        'Forwarded to GenerationStep clamps. Required '
                        'explicit since 2026-05-10 BBOB clampfix.')
    d.add_argument('--ub', type=float, default=100.0,
                   help='Search-domain upper bound. CEC2017: 100. BBOB: 5.')

    tp = p.add_argument_group('task pool')
    tp.add_argument('--pool-size', type=int, default=48)
    tp.add_argument('--ceiling-window', type=int, default=200)
    tp.add_argument('--eval-every', type=int, default=1000)
    tp.add_argument('--min-task-age', type=int, default=30)

    ws = p.add_argument_group('warm start')
    ws.add_argument('--warm-start-prob', type=float, default=0.0,
                    help='Probability of warm-starting from pool (0=disabled)')
    ws.add_argument('--warm-start-dir', type=str, default='warm_start_pool',
                    help='Root dir for warm start pool (subdir D{D}/ expected)')

    i = p.add_argument_group('infrastructure')
    i.add_argument('--save-dir', type=str, default='checkpoints/l2o')
    i.add_argument('--seed', type=int, default=None,
                   help='Global RNG seed. Applied to torch/numpy/random and '
                        'CUDA at startup. None (default) = stochastic (legacy '
                        'behavior). Required for the topology/edges ablation '
                        'reps so that 3 seeds × 4 variants are reproducible.')
    i.add_argument('--backbone-ckpt', type=str,
                   default='checkpoints/backbone/step_15000.pth')
    i.add_argument('--resume', type=str, default='')
    i.add_argument('--reset-step-counter', action='store_true',
                   help='When resuming, ignore the checkpoint step and start '
                        'from 0. Useful for smoke tests that want warm weights '
                        'but a short run (--steps 10).')
    i.add_argument('--ckpt-every', type=int, default=500)
    i.add_argument('--diag-every', type=int, default=10)

    return p.parse_args(argv)
