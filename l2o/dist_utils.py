"""Distributed training utilities for function-parallel L2O."""
import os

import torch
import torch.distributed as dist


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_distributed() else 0


def get_world_size():
    return dist.get_world_size() if is_distributed() else 1


def is_main():
    return get_rank() == 0


def setup_distributed():
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', rank))
        dist.init_process_group('nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return local_rank
    return 0


def cleanup_distributed():
    if is_distributed():
        dist.destroy_process_group()


def broadcast_params(model):
    if not is_distributed():
        return
    for p in model.parameters():
        dist.broadcast(p.data, src=0)


_BUCKET_SIZE = 25 * 1024 * 1024 // 4  # ~25 MB in float32 elements


def allreduce_grads(params, world_size):
    """All-reduce gradients with bucketed communication (no flat cat peak)."""
    if not is_distributed():
        return
    graded = [p for p in params if p.grad is not None]
    if not graded:
        return
    # Bucketed allreduce: avoids duplicating all gradient memory at once.
    bucket = []
    bucket_size = 0
    for p in graded:
        bucket.append(p)
        bucket_size += p.grad.numel()
        if bucket_size >= _BUCKET_SIZE:
            _allreduce_bucket(bucket, world_size)
            bucket = []
            bucket_size = 0
    if bucket:
        _allreduce_bucket(bucket, world_size)


def _allreduce_bucket(params, world_size):
    """Allreduce a single bucket of parameters."""
    flat = torch.cat([p.grad.flatten() for p in params])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat /= world_size
    offset = 0
    for p in params:
        numel = p.grad.numel()
        p.grad.copy_(flat[offset:offset + numel].view_as(p.grad))
        offset += numel


def check_nan_any_rank(grad_norm, device):
    """Returns Python bool: True if any rank has non-finite grad_norm."""
    if not is_distributed():
        return bool(not torch.isfinite(grad_norm))
    has_nan = torch.tensor(
        0.0 if torch.isfinite(grad_norm) else 1.0, device=device)
    dist.all_reduce(has_nan, op=dist.ReduceOp.SUM)
    return has_nan.item() > 0


def get_grad_norm(params):
    norms = [p.grad.float().norm().square() for p in params if p.grad is not None]
    if not norms:
        return torch.tensor(0.0, device='cpu')
    return torch.stack(norms).sum().sqrt()


def compute_clip_metrics(grad_norm, max_norm):
    """Per-step clip-activation metrics for diagnostics_rank{N}.jsonl.

    Combined with the existing `fn_id` field per step, this produces the
    composition + severity profile of `--grad-clip` activations:
      - clip_activated: True if pre-clip norm exceeded threshold
      - clip_ratio: pre_norm / max_norm (severity — distinguishes 1.1 from 165)
      - clip_pre_norm: raw value pre-clip
      - clip_max_norm: configured threshold

    Use case: detect (a) WHICH fids dominate clip activations, (b) HOW
    SEVERELY. Without ratio, all activations look equal in the log; with it,
    'F23 just barely tripped' (1.1) is distinguishable from 'F23 brutal' (165).

    Args:
        grad_norm: scalar (Python float or 0-dim torch.Tensor) — pre-clip total
            norm as returned by torch.nn.utils.clip_grad_norm_.
        max_norm: scalar — the configured max_norm threshold.
    """
    if torch.is_tensor(grad_norm):
        norm_val = float(grad_norm.item())
    else:
        norm_val = float(grad_norm)
    max_val = float(max_norm)
    return {
        'clip_pre_norm': norm_val,
        'clip_max_norm': max_val,
        'clip_ratio': norm_val / max_val if max_val > 0 else 0.0,
        'clip_activated': norm_val > max_val,
    }


def build_named_groups(backbone, variant, gen_step):
    """Per-component reporting groups for grad-norm diagnostics.

    `backbone.donor_selector` lives inside `backbone.backbone` (inner
    SparseGATv2Backbone) and therefore was lumped under `bb.gat` historically.
    Under K=1 + lshade_masked the LUPI contrafactual loss routes large
    gradients through donor_selector via the Gumbel-ST hard backward; without
    its own group those spikes are invisible in per-component diagnostics
    (E12 spikes 1e6–5e7 reported as `bb.gat` only after global clip).
    """
    named_groups = {}
    if hasattr(backbone, 'temporal'):
        named_groups['bb.temporal'] = list(backbone.temporal.parameters())
    if hasattr(backbone, 'pooler'):
        named_groups['bb.pooler'] = list(backbone.pooler.parameters())

    inner_bb = getattr(backbone, 'backbone', None)
    donor_sel = getattr(inner_bb, 'donor_selector', None) if inner_bb is not None else None
    donor_sel_ids = {id(p) for p in donor_sel.parameters()} if donor_sel is not None else set()

    if inner_bb is not None:
        named_groups['bb.gat'] = [p for p in inner_bb.parameters()
                                   if id(p) not in donor_sel_ids]
    if donor_sel is not None:
        named_groups['bb.donor_selector'] = list(donor_sel.parameters())

    for k_idx, head in enumerate(variant.heads):
        named_groups[f'var.h{k_idx}'] = list(head.parameters())
    named_groups['var.router'] = (list(variant.scorer.parameters()) +
                                  list(variant.score_norm.parameters()))
    if variant.activity_gate is not None:
        named_groups['var.gate'] = list(variant.activity_gate.parameters())
    if gen_step.surrogate is not None:
        named_groups['var.surrogate'] = list(gen_step.surrogate.parameters())
    return named_groups


def broadcast_es_stopped(es_stopped: bool, device, src: int = 0) -> bool:
    """Broadcast the early-stop flag from ``src`` to all ranks.

    The eval block runs ONLY on rank 0 and may flip ``_es_stopped`` to True.
    Without this broadcast, the next iteration's ``if _es_stopped: break``
    fires only on rank 0; other ranks proceed into the BPTT loop and hang
    at the next allreduce. Call right after the eval block, outside any
    ``is_main()`` guard, so the boolean reaches every rank before the loop
    head re-checks it.
    """
    if not is_distributed():
        return es_stopped
    flag = torch.tensor(1.0 if es_stopped else 0.0, device=device)
    dist.broadcast(flag, src=src)
    return flag.item() > 0.5
