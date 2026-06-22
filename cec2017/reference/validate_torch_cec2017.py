"""
Validate the new differentiable torch_cec2017.CEC2017Official against the
official C oracle, all 30 official functions, D in {10,30,50,100}.

Compares bias-subtracted g = f - fid*100 on shared random points + the optimum.
Run from repo root: python cec2017/reference/validate_torch_cec2017.py
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from cec2017.reference.torch_cec2017 import CEC2017Official, _load_shift_vec  # noqa: E402
from cec2017.reference.oracle import official_eval                            # noqa: E402

DIMS = [10, 30, 50, 100]
N_RAND = 50
RTOL = 1e-6
SEED = 20260518


def main():
    print(f"{'fid':>3} {'D':>4} {'max_rel':>11} {'max_abs':>11} {'naN':>4}  status")
    print("-" * 60)
    bad = []
    for fid in range(1, 31):
        for D in DIMS:
            rng = np.random.default_rng(SEED + fid * 13 + D)
            try:
                opt = _load_shift_vec(fid, D)
            except Exception:
                opt = rng.uniform(-100, 100, D)
            X = np.vstack([opt, rng.uniform(-100, 100, (N_RAND, D))])
            try:
                fn = CEC2017Official(fid, D, "cpu")
                with torch.no_grad():
                    ft = fn(torch.tensor(X, dtype=torch.float64)).numpy()
                gt = ft - fid * 100.0
                go = official_eval(fid, D, X) - fid * 100.0
            except Exception as e:
                print(f"{fid:>3} {D:>4}  ERROR: {e}")
                bad.append((fid, D))
                continue
            fin = np.isfinite(gt) & np.isfinite(go)
            nan = int((~fin).sum())
            rel = np.abs(gt[fin] - go[fin]) / (np.abs(go[fin]) + 1.0)
            ab = np.abs(gt[fin] - go[fin])
            mr = float(rel.max()) if rel.size else float("inf")
            ma = float(ab.max()) if ab.size else float("inf")
            ok = mr < RTOL and nan == 0
            if not ok:
                bad.append((fid, D))
            print(f"{fid:>3} {D:>4} {mr:>11.2e} {ma:>11.2e} {nan:>4}  "
                  f"{'ok' if ok else '*** MISMATCH ***'}")
    print("-" * 60)
    print(f"{120 - len(bad)}/120 cells pass   |   failures: {sorted(bad)}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
