"""Run the RLDE-AFL paper's algorithms (+ GNN-DE) on MetaBox BBOB, difficult-D10.

Every algorithm runs through the SAME MetaBox machinery on the SAME 16 held-out
difficult functions at instance_seed=3849 (MetaBox's default = the seed our GNN-DE
BBOB eval already used), N native per method, maxFEs=20000 (= 2000*D, the RLDE-AFL
paper's budget), 51 seeds. For each (algorithm, function) it dumps the per-seed
final gap-to-optimum and the best-so-far cost curve (n_logpoint+1 points), so the
SAME run feeds both the convergence figure (curves) and the Wilcoxon W/L/T (finals).

Three dispatch patterns:
  * classic BBO  (DE, MADDE, NLSHADELBC, JDE21, Random_search): autonomous
    Basic_Optimizer with run_episode(problem) -> {'cost', 'fes'}.
  * learned MetaBBO (DEDDQN, DEDQN, LDE, RLHPSDE, GLEET, RLDAS, RLDEAFL): load the
    .pkl agent, build its XXX_Optimizer, PBO_Env(problem, optimizer),
    agent.rollout_episode(env, seed) -> {'cost', 'fes'}.
  * GNN-DE: our eval_metabox.gnnde_optimizer.GNN_DE Basic_Optimizer wrapper.

W/L/T is scored separately (Mann-Whitney U, unpaired) by score_bbob_wilcoxon.py.

Usage (one algorithm per array task):
    python run_metabox_bbob.py --algo DEDDQN --seeds 51 --out out/DEDDQN.json
"""
import argparse, json, os, sys, time
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import _mbx_load as M

# algo -> (kind, module, class_or_optimizer)
CLASSIC = {
    'DE':            ('baseline.bbo.de', 'DE'),
    'MADDE':         ('baseline.bbo.madde', 'MADDE'),
    'NLSHADELBC':    ('baseline.bbo.nlshadelbc', 'NLSHADELBC'),
    'JDE21':         ('baseline.bbo.jde21', 'JDE21'),
    'Random_search': ('baseline.bbo.random_search', 'Random_search'),
}
# learned: ckpt filename stem -> optimizer module/class
LEARNED = {
    'DEDDQN':  ('environment.optimizer.deddqn_optimizer', 'DEDDQN_Optimizer'),
    'DEDQN':   ('environment.optimizer.dedqn_optimizer', 'DEDQN_Optimizer'),
    'LDE':     ('environment.optimizer.lde_optimizer', 'LDE_Optimizer'),
    'RLHPSDE': ('environment.optimizer.rlhpsde_optimizer', 'RLHPSDE_Optimizer'),
    'GLEET':   ('environment.optimizer.gleet_optimizer', 'GLEET_Optimizer'),
    'RLDAS':   ('environment.optimizer.rldas_optimizer', 'RLDAS_Optimizer'),
    'RLDEAFL': ('environment.optimizer.rldeafl_optimizer', 'RLDEAFL_Optimizer'),
}
CKPT_DIR = os.path.join(_HERE, '..', 'literature', 'rl_metabbo', 'MetaBox',
                        'src', 'model', 'bbob-10D', 'difficult')


def build_config(dim=10, device='cpu'):
    cfgmod = M.imp('config')
    cfg = cfgmod.get_config([
        '--train_problem', 'bbob-10D', '--test_problem', 'bbob-10D',
        '--upperbound', '5', '--train_difficulty', 'difficult',
        '--full_meta_data', '',
    ])
    cfg = cfgmod.init_config(cfg)
    cfg.dim = dim
    cfg.device = device
    return cfg


def get_problems(instance_seed=3849, version='numpy'):
    """Realize the 16 held-out difficult BBOB functions at a given instance.

    instance_seed seeds the (shift, rotation) generation in BBOB_Dataset, so
    distinct seeds yield distinct landscape realizations of the SAME functions
    (the COCO multi-instance protocol). Default 3849 = the published single
    instance, kept as instance #1 for backward compatibility.
    """
    bd = M.imp('environment.problem.SOO.COCO_BBOB.bbob_dataset')
    _, test = bd.BBOB_Dataset.get_datasets(
        suit='bbob10D', upperbound=5, difficulty='difficult',
        instance_seed=instance_seed, version=version)
    return test.data  # 16 problems, each with .dim/.lb/.ub/.optimum/.eval/.__str__


def _to_double(agent):
    import torch.nn as nn
    torch.set_default_dtype(torch.float64)
    for k in list(vars(agent)):
        v = getattr(agent, k)
        if isinstance(v, nn.Module):
            setattr(agent, k, v.double())
    for attr in ('model',):
        if hasattr(agent, attr) and isinstance(getattr(agent, attr), nn.Module):
            setattr(agent, attr, getattr(agent, attr).double())


def run_classic(algo, cfg, problems, seeds):
    mod_name, cls_name = CLASSIC[algo]
    cls = getattr(M.imp(mod_name), cls_name)
    out = {}
    for prob in problems:
        finals, fes_l, curves = [], [], []
        for sd in seeds:
            np.random.seed(sd); torch.manual_seed(sd)
            opt = cls(cfg)
            if hasattr(opt, 'seed'):
                try: opt.seed(sd)
                except Exception: pass
            prob.reset()
            r = opt.run_episode(prob)
            cost = list(map(float, r['cost']))           # MetaBox cost already = gap-to-optimum
            finals.append(cost[-1])
            curves.append(cost)
            fes_l.append(int(r.get('fes', cfg.maxFEs)))
        out[prob.__str__()] = {'finals': finals, 'fes': fes_l, 'curves': curves}
    return out


def run_learned(algo, cfg, problems, seeds):
    opt_mod, opt_cls = LEARNED[algo]
    OptCls = getattr(M.imp(opt_mod), opt_cls)
    ckpt = os.path.join(CKPT_DIR, f'{algo}.pkl')
    out = {}
    for prob in problems:
        finals, fes_l, curves = [], [], []
        for sd in seeds:
            np.random.seed(sd); torch.manual_seed(sd)
            agent = M.load_ckpt(ckpt)
            if hasattr(agent, '_agent__device'): agent._agent__device = 'cpu'
            if hasattr(agent, '_agent__config'):
                try: agent._agent__config.device = 'cpu'
                except Exception: pass
            _to_double(agent)
            optimizer = OptCls(cfg)
            env = M.imp('environment.basic_environment').PBO_Env(prob, optimizer)
            prob.reset()
            res = agent.rollout_episode(env, seed=sd, required_info={})
            cost = list(map(float, res['cost']))         # already gap-to-optimum
            finals.append(cost[-1])
            curves.append(cost)
            fes_l.append(int(res.get('fes', cfg.maxFEs)))
        out[prob.__str__()] = {'finals': finals, 'fes': fes_l, 'curves': curves}
    return out


def run_gnnde(cfg, problems, seeds, out_path=None, meta=None):
    """GNN-DE on GPU/CPU per cfg.device. Saves incrementally after each function
    and resumes from a partial JSON (skips already-done functions), so a wall-clock
    timeout never loses completed work."""
    from gnnde_optimizer import GNN_DE
    out = {}
    if out_path and os.path.isfile(out_path):
        try:
            out = json.load(open(out_path)).get('data', {})
            if out:
                print(f"[GNN_DE] resuming, {len(out)} funcs already done", flush=True)
        except Exception:
            out = {}

    def _flush():
        if not out_path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        json.dump({'meta': meta or {}, 'data': out}, open(out_path, 'w'))

    for pi, prob in enumerate(problems):
        key = prob.__str__()
        if key in out and len(out[key].get('finals', [])) == len(seeds):
            print(f"[GNN_DE] skip {key} (done)", flush=True)
            continue
        t0 = time.time()
        finals, fes_l, curves = [], [], []
        for sd in seeds:
            o = GNN_DE(cfg)
            o.seed(sd)
            prob.reset()
            r = o.run_episode(prob)
            cost = list(map(float, r['cost']))
            opt_val = float(getattr(prob, 'optimum', 0.0) or 0.0)
            finals.append(cost[-1] - opt_val)
            curves.append([c - opt_val for c in cost])
            fes_l.append(int(r.get('fes', cfg.maxFEs)))
        out[key] = {'finals': finals, 'fes': fes_l, 'curves': curves}
        _flush()
        med = float(np.median(finals))
        print(f"[GNN_DE] {pi+1}/{len(problems)} {key}: median gap={med:.3e} "
              f"({time.time()-t0:.0f}s, {(time.time()-t0)/len(seeds):.1f}s/seed)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--algo', required=True)
    ap.add_argument('--seeds', type=int, default=51)
    ap.add_argument('--dim', type=int, default=10)
    ap.add_argument('--device', default='cpu',
                    help="cuda for GNN_DE on GPU; classic/learned stay cpu (metaevobox).")
    ap.add_argument('--func-indices', default=None,
                    help="comma-separated indices into the 16-problem difficult list; "
                         "run only this subset (for array-parallel fan-out). Default = all.")
    ap.add_argument('--instance-seed', type=int, default=3849,
                    help="BBOB instance seed (shift+rotation realization). Default 3849 "
                         "= the published single instance. Vary it for the COCO "
                         "multi-instance robustness check.")
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    M.prep()
    # GNN-DE honours --device (GPU-capable); metaevobox classic/learned stay on cpu.
    dev = args.device if args.algo == 'GNN_DE' else 'cpu'
    cfg = build_config(args.dim, device=dev)
    problems = get_problems(instance_seed=args.instance_seed)
    if args.func_indices:
        idxs = sorted(int(x) for x in args.func_indices.split(','))
        problems = [problems[i] for i in idxs]
        print(f"[{args.algo}] func subset {idxs} -> {[p.__str__() for p in problems]}", flush=True)
    seeds = list(range(1, args.seeds + 1))
    print(f"[{args.algo}] {len(problems)} funcs x {args.seeds} seeds, "
          f"maxFEs={cfg.maxFEs}, D={args.dim}, device={dev}", flush=True)

    meta = {'algo': args.algo, 'seeds': args.seeds, 'dim': args.dim,
            'maxFEs': int(cfg.maxFEs), 'difficulty': 'difficult',
            'instance_seed': args.instance_seed, 'metric': 'gap_to_optimum',
            'device': dev}

    t0 = time.time()
    if args.algo == 'GNN_DE':
        data = run_gnnde(cfg, problems, seeds, out_path=args.out, meta=meta)
    elif args.algo in CLASSIC:
        data = run_classic(args.algo, cfg, problems, seeds)
    elif args.algo in LEARNED:
        data = run_learned(args.algo, cfg, problems, seeds)
    else:
        sys.exit(f"unknown algo {args.algo}")

    payload = {'meta': meta, 'data': data}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(payload, open(args.out, 'w'))
    print(f"[{args.algo}] saved -> {args.out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == '__main__':
    main()
