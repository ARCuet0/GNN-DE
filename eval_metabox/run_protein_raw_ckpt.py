"""RAW protein-docking runner with incremental, resumable checkpointing.

Companion to run_protein_raw.py. The original buffers all 2040 episodes
(instances x agents x seeds) in RAM and writes the JSON only after the FINAL
instance, so a crash, OOM, or reboot at instance N loses the whole shard
(~2.5 h of compute). This runner adds the two robustness properties the original
lacks:

  * Incremental ATOMIC write after EVERY instance (write to ``.tmp`` then
    ``os.replace``), so an interruption loses at most the in-progress instance.
  * ``--resume``: on restart, instances already present in the out file are
    skipped and the shard continues where it left off.

The output is the exact same ``{meta, data}`` contract that score_aei.py and the
other consumers read (a partial file is just a valid file with fewer instances),
plus a ``meta.completed_instances`` counter.

This is a SEPARATE file on purpose: it does not modify run_protein_raw.py, so
any in-flight shards keep running the original code unchanged.

Usage (resumable):
    python eval_metabox/run_protein_raw_ckpt.py --shard-id 17 --n-shards 28 \
        --seeds 51 --maxFEs 2000 --device cpu \
        --agents GNN_DE Random_search MadDE NLSHADELBC \
        --ckpt checkpoint/published_checkpoint.pth \
        --out eval_metabox/results/protein_real_51s_2000fes/shard_17.json --resume
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_TERSQ_ROOT = os.path.dirname(_HERE)
if _TERSQ_ROOT not in sys.path:
    sys.path.insert(0, _TERSQ_ROOT)

from types import SimpleNamespace


def _atomic_write_json(obj, out_path):
    """Write JSON to a temp file and atomically rename, so a kill mid-write
    never leaves a half-written (corrupt) out file. Also fsync the containing
    directory so the rename survives a power loss / reboot (best effort)."""
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out_path)
    try:
        dfd = os.open(os.path.dirname(out_path) or ".", os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except (OSError, AttributeError):
        pass  # O_DIRECTORY unsupported on this platform; rename already durable


def _read_prev(out_path):
    """Return the parsed ``{meta, data}`` dict already on disk, or None if
    absent. If the file exists but is unparseable, move it aside to
    ``.corrupt.<ts>`` and abort rather than silently discarding (and later
    overwriting) a damaged but possibly-recoverable shard."""
    if not os.path.exists(out_path):
        return None
    try:
        with open(out_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError, OSError) as e:
        aside = f"{out_path}.corrupt.{int(time.time())}"
        try:
            os.replace(out_path, aside)
        except OSError:
            aside = out_path
        raise SystemExit(
            f"[resume] {out_path} exists but is unreadable ({e}); moved to "
            f"{aside}. Inspect or remove it, then rerun.")


def run_shard(instances, agents, seeds, episode_fn, out_path, meta_base,
              resume=False, log=print):
    """Run ``agents`` x ``seeds`` on each instance, checkpointing after each one.

    ``episode_fn(agent_name, instance, seed) -> (final, fes, curve)``. Injectable
    so the loop is testable without the MetaBox stack. With ``resume=True``,
    instances already present in ``out_path`` are skipped. An instance is
    committed to ``data`` only after its full agent x seed sweep succeeds, so a
    crash mid-instance leaves the on-disk file at the last fully-completed state.
    """
    if int(seeds) < 1:
        raise SystemExit(f"--seeds must be >= 1 (got {seeds})")
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    data, prior_wall = {}, 0.0
    n_total = len(instances)
    if resume:
        prev = _read_prev(out_path)
        if prev is not None:
            pm = prev.get("meta", {}) or {}
            # Refuse to merge a partial computed under a different configuration:
            # mixing agent sets / seed counts / budgets corrupts downstream scoring.
            if pm.get("agents") is not None and list(pm["agents"]) != list(agents):
                raise SystemExit(f"[resume] agents mismatch: file={pm.get('agents')} args={list(agents)}")
            if pm.get("seeds") is not None and int(pm["seeds"]) != int(seeds):
                raise SystemExit(f"[resume] seeds mismatch: file={pm.get('seeds')} args={seeds}")
            if (pm.get("maxFEs") is not None and meta_base.get("maxFEs") is not None
                    and int(pm["maxFEs"]) != int(meta_base["maxFEs"])):
                raise SystemExit(f"[resume] maxFEs mismatch: file={pm.get('maxFEs')} args={meta_base.get('maxFEs')}")
            if (pm.get("shard_id") is not None and meta_base.get("shard_id") is not None
                    and int(pm["shard_id"]) != int(meta_base["shard_id"])):
                raise SystemExit(f"[resume] shard_id mismatch: file={pm.get('shard_id')} args={meta_base.get('shard_id')}")
            data = prev.get("data", {}) or {}
            stray = set(data.keys()) - {str(p) for p in instances}
            if stray:
                raise SystemExit(f"[resume] file holds {len(stray)} instances not in this shard "
                                 f"(wrong --shard-id/--n-shards?): {sorted(stray)[:5]}")
            prior_wall = float(pm.get("wall_seconds", 0.0) or 0.0)
    done = set(data.keys())
    if resume and done:
        log(f"[resume] {len(done)}/{n_total} instances already done; skipping them",
            flush=True)

    def _snapshot(elapsed):
        return {
            "meta": {**meta_base, "agents": list(agents), "seeds": int(seeds),
                     "instance_ids": [str(p) for p in instances],
                     "completed_instances": len(data),
                     "wall_seconds": prior_wall + elapsed},
            "data": data,
        }

    t0 = time.perf_counter()
    sid = meta_base.get("shard_id", "?")
    for j, prob in enumerate(instances):
        pid = str(prob)
        if pid in done:
            log(f"[skip] {j + 1}/{n_total} {pid} (already done)", flush=True)
            continue
        per_agent = {}
        for agent_name in agents:
            finals, fess, curves = [], [], []
            for s in range(seeds):
                final, fes, curve = episode_fn(agent_name, prob, s)
                finals.append(float(final))
                fess.append(int(fes))
                curves.append([float(x) for x in curve])
            per_agent[agent_name] = {"finals": finals, "fes": fess, "curves": curves}
        # Commit only after the whole instance succeeded, then checkpoint.
        data[pid] = per_agent
        el = time.perf_counter() - t0
        _atomic_write_json(_snapshot(el), out_path)
        log(f"[shard {sid}] {j + 1}/{n_total} {pid} done ({el:.0f}s) [checkpointed]",
            flush=True)
    return out_path


def make_cfg(maxFEs, device):
    return SimpleNamespace(
        maxFEs=maxFEs, n_logpoint=50, log_interval=max(1, maxFEs // 50),
        device=device, full_meta_data=False, test_problem='protein')


def _build_episode_fn(cfg, ckpt):
    """Real episode runner reusing the original agent factory (fresh agent per
    episode, exactly as run_protein_raw.py does)."""
    from eval_metabox.run_protein_raw import build_agent

    def episode_fn(agent_name, prob, seed):
        opt = build_agent(agent_name, cfg, ckpt)
        opt.seed(seed)
        res = opt.run_episode(prob)
        curve = [float(x) for x in res['cost']]
        return curve[-1], int(res['fes']), curve

    return episode_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard-id', type=int, required=True)
    ap.add_argument('--n-shards', type=int, required=True)
    ap.add_argument('--seeds', type=int, default=51)
    ap.add_argument('--maxFEs', type=int, default=500)
    ap.add_argument('--agents', nargs='+',
                    default=['GNN_DE', 'Random_search', 'MadDE', 'NLSHADELBC'])
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--ckpt', default=None, help='override deployed ckpt path')
    ap.add_argument('--out', required=True)
    ap.add_argument('--resume', action='store_true',
                    help='skip instances already present in --out')
    args = ap.parse_args()

    from eval_metabox._metabox_compat import (
        get_basic_optimizer, load_protein_testset)
    get_basic_optimizer()  # pre-load skeleton packages once
    cfg = make_cfg(args.maxFEs, args.device)

    all_instances = load_protein_testset('numpy', 'all')
    shards = np.array_split(np.arange(len(all_instances)), args.n_shards)
    my_instances = [all_instances[i] for i in shards[args.shard_id]]
    print(f"[shard {args.shard_id}/{args.n_shards}] {len(my_instances)} instances "
          f"x {args.seeds} seeds, agents={args.agents}, maxFEs={args.maxFEs}, "
          f"resume={args.resume}", flush=True)

    meta_base = {'shard_id': args.shard_id, 'n_shards': args.n_shards,
                 'maxFEs': args.maxFEs, 'device': args.device}
    run_shard(my_instances, args.agents, args.seeds,
              _build_episode_fn(cfg, args.ckpt), args.out, meta_base,
              resume=args.resume)
    print(f"[shard {args.shard_id}] wrote {args.out}", flush=True)


if __name__ == '__main__':
    main()
