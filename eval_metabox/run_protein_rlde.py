"""Magerit-side RAW runner for RLDE-AFL on MetaBox Protein-Docking.

Companion to run_protein_raw.py (which runs GNN-DE + the classic BBO baselines).
RLDE-AFL is a learned MetaBBO method with a different call contract (load a .pkl
agent, build its optimizer, wrap the problem in PBO_Env, call rollout_episode), so
it lives in its own runner / its own process. That also keeps RLDE-AFL's
to-double / float64 default-dtype switch from leaking into the GNN-DE process.

RLDE-AFL has NO protein-trained checkpoint in MetaBox; it runs ZERO-SHOT from its
bbob-10D/difficult RLDEAFL.pkl. The rollout needs the container's vendored gym/dill
(launch with PYTHONPATH=vendor_lib). Output schema matches run_protein_raw so
eval_metabox/score_aei.py merges both dirs by instance id (pid).

Granular storage (user pref): every seed's final cost, fes, and full curve.

Usage (one SLURM array task):
    PYTHONPATH=vendor_lib python eval_metabox/run_protein_rlde.py \
        --shard-id $SLURM_ARRAY_TASK_ID --n-shards 20 --seeds 51 --maxFEs 2000 \
        --out eval_metabox/results/rlde_protein_2000fes/shard_${SLURM_ARRAY_TASK_ID}.json
"""
import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_TERSQ_ROOT = os.path.dirname(_HERE)
if _TERSQ_ROOT not in sys.path:
    sys.path.insert(0, _TERSQ_ROOT)

import eval_metabox._mbx_load as M  # noqa: E402
from eval_metabox._metabox_compat import load_protein_testset  # noqa: E402

AGENT_NAME = "RLDE_AFL"
DEFAULT_RLDE_CKPT = os.path.join(
    _HERE, "..", "literature", "rl_metabbo", "MetaBox", "src",
    "model", "bbob-10D", "difficult", "RLDEAFL.pkl",
)
_RLDE_OPT = ("environment.optimizer.rldeafl_optimizer", "RLDEAFL_Optimizer")


def make_cfg(maxFEs, device="cpu"):
    return SimpleNamespace(
        maxFEs=maxFEs, n_logpoint=50, log_interval=max(1, maxFEs // 50),
        device=device, full_meta_data=False, test_problem="protein")


def shard_instances(n_total, n_shards, shard_id):
    """Identical split to run_protein_raw.py so the two runners cover the same
    instances per shard (np.array_split over arange)."""
    return np.array_split(np.arange(n_total), n_shards)[shard_id].tolist()


def _to_double(agent):
    import torch
    import torch.nn as nn
    torch.set_default_dtype(torch.float64)
    for k in list(vars(agent)):
        v = getattr(agent, k)
        if isinstance(v, nn.Module):
            setattr(agent, k, v.double())
    for attr in ("model",):
        if hasattr(agent, attr) and isinstance(getattr(agent, attr), nn.Module):
            setattr(agent, attr, getattr(agent, attr).double())


def build_rlde(cfg, ckpt):
    """Load the RLDE-AFL agent once and resolve the optimizer/env classes.
    Returns a bundle reused across seeds (the agent itself is reloaded per seed
    inside run_episode_rlde to avoid cross-episode state, mirroring the BBOB
    runner)."""
    OptCls = getattr(M.imp(_RLDE_OPT[0]), _RLDE_OPT[1])
    PBO_Env = M.imp("environment.basic_environment").PBO_Env
    return {"ckpt": ckpt, "OptCls": OptCls, "PBO_Env": PBO_Env, "cfg": cfg}


def run_episode_rlde(bundle, prob, seed, cfg):
    """One zero-shot RLDE-AFL rollout. Reloads the agent per call (fresh state),
    restores the torch default dtype afterwards so a co-resident float32 model is
    never contaminated by the to-double switch."""
    import torch
    prev_dtype = torch.get_default_dtype()
    try:
        agent = M.load_ckpt(bundle["ckpt"])
        if hasattr(agent, "_agent__device"):
            agent._agent__device = "cpu"
        if hasattr(agent, "_agent__config"):
            try:
                agent._agent__config.device = "cpu"
            except Exception:
                pass
        _to_double(agent)
        optimizer = bundle["OptCls"](cfg)
        env = bundle["PBO_Env"](prob, optimizer)
        prob.reset()
        res = agent.rollout_episode(env, seed=seed, required_info={})
        cost = [float(x) for x in res["cost"]]
        return {"cost": cost, "fes": int(res.get("fes", cfg.maxFEs))}
    finally:
        torch.set_default_dtype(prev_dtype)


def build_payload(shard_id, n_shards, instance_ids, seeds, maxFEs, data):
    return {
        "meta": {
            "shard_id": shard_id, "n_shards": n_shards,
            "instance_ids": list(instance_ids),
            "seeds": seeds, "maxFEs": maxFEs, "agents": [AGENT_NAME],
            "device": "cpu",
        },
        "data": data,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-id", type=int, required=True)
    ap.add_argument("--n-shards", type=int, required=True)
    ap.add_argument("--seeds", type=int, default=51)
    ap.add_argument("--maxFEs", type=int, default=2000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ckpt", default=DEFAULT_RLDE_CKPT)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = make_cfg(args.maxFEs, args.device)
    all_instances = load_protein_testset("numpy", "all")
    my_idx = shard_instances(len(all_instances), args.n_shards, args.shard_id)
    my_instances = [all_instances[i] for i in my_idx]
    print(f"[rlde shard {args.shard_id}/{args.n_shards}] {len(my_instances)} instances "
          f"x {args.seeds} seeds, maxFEs={args.maxFEs}, ckpt={os.path.basename(args.ckpt)}",
          flush=True)

    bundle = build_rlde(cfg, args.ckpt)
    data = {}
    t0 = time.perf_counter()
    for j, prob in enumerate(my_instances):
        pid = str(prob)
        finals, fess, curves = [], [], []
        for s in range(args.seeds):
            res = run_episode_rlde(bundle, prob, s, cfg)
            curve = res["cost"]
            finals.append(float(curve[-1]))
            fess.append(int(res["fes"]))
            curves.append(curve)
        data[pid] = {AGENT_NAME: {"finals": finals, "fes": fess, "curves": curves}}
        el = time.perf_counter() - t0
        print(f"[rlde shard {args.shard_id}] {j + 1}/{len(my_instances)} {pid} done ({el:.0f}s)",
              flush=True)

    payload = build_payload(args.shard_id, args.n_shards, [str(p) for p in my_instances],
                            args.seeds, args.maxFEs, data)
    payload["meta"]["wall_seconds"] = time.perf_counter() - t0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f)
    print(f"[rlde shard {args.shard_id}] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
