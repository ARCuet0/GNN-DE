"""
Differentiable PyTorch CEC2017 — faithful reimplementation of the official C
reference (cec17_test_func.cpp, Awad/Suganthan).

This is a NEW module. The legacy `cec2017/benchmark.py` + `base_functions.py`
are left untouched. Validated against the C oracle by
`cec2017/reference/validate_torch_cec2017.py` (target: all 29 funcs match to
rtol 1e-6 on the bias-subtracted value).

OFFICIAL numbering 1..30 (F2 = Sum of Different Power is present). Reads the
official shift/rotate/shuffle data from `cec2017/reference/input_data/`, so the
module has NO opfunu dependency.

Faithful-to-C quirks deliberately reproduced (the C *is* the de-facto standard
the literature competes against):
  * F8 step_rastrigin == plain rastrigin: the non-continuous rounding in the C
    is applied to a stale buffer and overwritten by sr_func -> dead code.
  * schaffer_F7 ignores the rotation matrix: it reads the pre-rotation buffer
    `y`. As a hybrid sub-function it reads the *first* m entries of the shuffled
    vector (global-buffer aliasing in the C).
  * hybrid partition sizes use ceil(Gp*D), not floor.
"""
import math
import os
import numpy as np
import torch

_REF_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_REF_DIR, "input_data")
PI = math.pi
E = math.e

# --- per-base-function domain scale (sh_rate in the official sr_func calls) ---
RATE = {
    "sphere": 1.0, "ellips": 1.0, "sum_diff_pow": 1.0, "zakharov": 1.0,
    "levy": 1.0, "bent_cigar": 1.0, "discus": 1.0, "rosenbrock": 2.048 / 100.0,
    "schaffer": 1.0, "ackley": 1.0, "weierstrass": 0.5 / 100.0,
    "griewank": 600.0 / 100.0, "rastrigin": 5.12 / 100.0, "schwefel": 1000.0 / 100.0,
    "katsuura": 5.0 / 100.0, "grie_rosen": 5.0 / 100.0, "escaffer6": 1.0,
    "happycat": 5.0 / 100.0, "hgbat": 5.0 / 100.0,
}


# --------------------------------------------------------------------------
# data loading (official files)
# --------------------------------------------------------------------------
def _load_shift_vec(fid, D):
    a = np.genfromtxt(os.path.join(_DATA_DIR, f"shift_data_{fid}.txt"), dtype=float)
    return a.ravel()[:D]


def _load_shift_mat(fid, D):
    a = np.genfromtxt(os.path.join(_DATA_DIR, f"shift_data_{fid}.txt"), dtype=float)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    return a[:10, :D]


def _load_M(fid, D, n=1):
    a = np.genfromtxt(os.path.join(_DATA_DIR, f"M_{fid}_D{D}.txt"), dtype=float)
    return a.reshape(n, D, D) if n > 1 else a.reshape(D, D)


def _load_shuffle(fid, D, n=1):
    a = np.genfromtxt(os.path.join(_DATA_DIR, f"shuffle_data_{fid}_D{D}.txt"), dtype=int)
    a = (a - 1).ravel()
    return a.reshape(n, D) if n > 1 else a[:D]


def _round_ste(v):
    """floor(v + 0.5) (round-half-up, matches C) with straight-through grad."""
    r = torch.floor(v + 0.5)
    return v + (r - v).detach()


# --------------------------------------------------------------------------
# base functions  — input z is (B, m), already shifted/scaled/rotated as needed
# --------------------------------------------------------------------------
def _f_sphere(z):
    return torch.sum(z * z, dim=1)


def _f_ellips(z):
    D = z.shape[1]
    i = torch.arange(D, device=z.device, dtype=z.dtype)
    coeff = 10.0 ** (6.0 * i / max(D - 1, 1))
    return torch.sum(coeff * z * z, dim=1)


def _f_sum_diff_pow(z):
    D = z.shape[1]
    i = torch.arange(D, device=z.device, dtype=z.dtype)
    return torch.sum(torch.abs(z) ** (i + 1.0), dim=1)


def _f_bent_cigar(z):
    return z[:, 0] ** 2 + 1e6 * torch.sum(z[:, 1:] ** 2, dim=1)


def _f_discus(z):
    return 1e6 * z[:, 0] ** 2 + torch.sum(z[:, 1:] ** 2, dim=1)


def _f_zakharov(z):
    D = z.shape[1]
    i = torch.arange(1, D + 1, device=z.device, dtype=z.dtype)
    sum1 = torch.sum(z * z, dim=1)
    sum2 = torch.sum(0.5 * i * z, dim=1)
    return sum1 + sum2 ** 2 + sum2 ** 4


def _f_rosenbrock(z):
    z = z + 1.0  # shift to origin
    t1 = z[:, :-1] ** 2 - z[:, 1:]
    t2 = z[:, :-1] - 1.0
    return torch.sum(100.0 * t1 * t1 + t2 * t2, dim=1)


def _f_rastrigin(z):
    return torch.sum(z * z - 10.0 * torch.cos(2.0 * PI * z) + 10.0, dim=1)


def _f_levy(z):
    w = 1.0 + (z - 1.0) / 4.0
    term1 = torch.sin(PI * w[:, 0]) ** 2
    term3 = (w[:, -1] - 1.0) ** 2 * (1.0 + torch.sin(2.0 * PI * w[:, -1]) ** 2)
    wm = w[:, :-1]
    s = torch.sum((wm - 1.0) ** 2 * (1.0 + 10.0 * torch.sin(PI * wm + 1.0) ** 2), dim=1)
    return term1 + s + term3


def _f_ackley(z):
    D = z.shape[1]
    s1 = torch.sum(z * z, dim=1)
    s2 = torch.sum(torch.cos(2.0 * PI * z), dim=1)
    return E - 20.0 * torch.exp(-0.2 * torch.sqrt(s1 / D)) - torch.exp(s2 / D) + 20.0


def _f_griewank(z):
    D = z.shape[1]
    i = torch.arange(D, device=z.device, dtype=z.dtype)
    s = torch.sum(z * z, dim=1) / 4000.0
    p = torch.prod(torch.cos(z / torch.sqrt(1.0 + i)), dim=1)
    return 1.0 + s - p


def _f_schwefel(z):
    D = z.shape[1]
    z = z + 4.209687462275036e+002
    az = torch.abs(z)
    fmod = torch.fmod(az, 500.0)
    gt = z > 500.0
    lt = z < -500.0
    base = -z * torch.sin(torch.sqrt(az))
    g = -(500.0 - fmod) * torch.sin(torch.sqrt(torch.abs(500.0 - fmod))) \
        + ((z - 500.0) / 100.0) ** 2 / D
    l = -(-500.0 + fmod) * torch.sin(torch.sqrt(torch.abs(500.0 - fmod))) \
        + ((z + 500.0) / 100.0) ** 2 / D
    fx = torch.where(gt, g, torch.where(lt, l, base))
    return torch.sum(fx, dim=1) + 4.189828872724338e+002 * D


def _f_katsuura(z):
    B, D = z.shape
    tmp3 = (1.0 * D) ** 1.2
    j = torch.arange(1, 33, device=z.device, dtype=z.dtype)
    tj = (2.0 ** j).view(1, 1, 32)
    v = tj * z.unsqueeze(2)
    temp = torch.sum(torch.abs(v - _round_ste(v)) / tj, dim=2)  # (B, D)
    i = torch.arange(1, D + 1, device=z.device, dtype=z.dtype)
    factors = (1.0 + i * temp) ** (10.0 / tmp3)
    prod = torch.exp(torch.sum(torch.log(factors), dim=1))
    c = 10.0 / D / D
    return prod * c - c


def _f_weierstrass(z):
    D = z.shape[1]
    a, b, kmax = 0.5, 3.0, 20
    k = torch.arange(0, kmax + 1, device=z.device, dtype=z.dtype)
    ak = a ** k
    bk = b ** k
    cos_vals = ak * torch.cos(2.0 * PI * bk * (z.unsqueeze(2) + 0.5))  # (B,D,k)
    f = torch.sum(cos_vals, dim=(1, 2))
    base = torch.sum(ak * torch.cos(2.0 * PI * bk * 0.5))
    return f - D * base


def _f_grie_rosen(z):
    z = z + 1.0
    z1 = z
    z2 = torch.roll(z, -1, dims=1)
    t1 = z1 ** 2 - z2
    t2 = z1 - 1.0
    temp = 100.0 * t1 * t1 + t2 * t2
    return torch.sum(temp * temp / 4000.0 - torch.cos(temp) + 1.0, dim=1)


def _f_escaffer6(z):
    z2 = torch.roll(z, -1, dims=1)
    ss = z ** 2 + z2 ** 2
    t1 = torch.sin(torch.sqrt(ss)) ** 2
    t2 = (1.0 + 0.001 * ss) ** 2
    return torch.sum(0.5 + (t1 - 0.5) / t2, dim=1)


def _f_happycat(z):
    D = z.shape[1]
    z = z - 1.0
    r2 = torch.sum(z * z, dim=1)
    sz = torch.sum(z, dim=1)
    return torch.abs(r2 - D) ** (2.0 / 8.0) + (0.5 * r2 + sz) / D + 0.5


def _f_hgbat(z):
    D = z.shape[1]
    z = z - 1.0
    r2 = torch.sum(z * z, dim=1)
    sz = torch.sum(z, dim=1)
    return torch.abs(r2 ** 2 - sz ** 2) ** (2.0 / 4.0) + (0.5 * r2 + sz) / D + 0.5


def _f_schaffer(y):
    """Schaffer F7 on the pre-rotation buffer y (B, m). No wrap-around."""
    D = y.shape[1]
    s = torch.sqrt(y[:, :-1] ** 2 + y[:, 1:] ** 2)
    tmp = torch.sin(50.0 * s ** 0.2)
    f = torch.sum(s ** 0.5 + s ** 0.5 * tmp * tmp, dim=1)
    return f * f / (D - 1) / (D - 1)


# core dispatch: name -> (core_fn, rate). schaffer/bi_rastrigin handled separately.
_CORE = {
    "sphere": (_f_sphere, RATE["sphere"]),
    "ellips": (_f_ellips, RATE["ellips"]),
    "sum_diff_pow": (_f_sum_diff_pow, RATE["sum_diff_pow"]),
    "bent_cigar": (_f_bent_cigar, RATE["bent_cigar"]),
    "discus": (_f_discus, RATE["discus"]),
    "zakharov": (_f_zakharov, RATE["zakharov"]),
    "rosenbrock": (_f_rosenbrock, RATE["rosenbrock"]),
    "rastrigin": (_f_rastrigin, RATE["rastrigin"]),
    "levy": (_f_levy, RATE["levy"]),
    "ackley": (_f_ackley, RATE["ackley"]),
    "griewank": (_f_griewank, RATE["griewank"]),
    "schwefel": (_f_schwefel, RATE["schwefel"]),
    "katsuura": (_f_katsuura, RATE["katsuura"]),
    "weierstrass": (_f_weierstrass, RATE["weierstrass"]),
    "grie_rosen": (_f_grie_rosen, RATE["grie_rosen"]),
    "escaffer6": (_f_escaffer6, RATE["escaffer6"]),
    "happycat": (_f_happycat, RATE["happycat"]),
    "hgbat": (_f_hgbat, RATE["hgbat"]),
}


def _rotate(v, M):
    """v (B, D), M (D, D) -> v @ M.T  (matches C rotatefunc xrot_i = sum_j v_j M_ij)."""
    return v @ M.T


def _bi_rastrigin(x, Os, M, s_flag, r_flag):
    """Lunacek bi-Rastrigin. x (B,m), Os (m,), M (m,m)."""
    B, m = x.shape
    mu0, d = 2.5, 1.0
    s = 1.0 - 1.0 / (2.0 * math.sqrt(m + 20.0) - 8.2)
    mu1 = -math.sqrt((mu0 * mu0 - d) / s)
    y = (x - Os) if s_flag else x
    y = y * (10.0 / 100.0)
    tmpx = 2.0 * y
    tmpx = torch.where(Os < 0.0, -tmpx, tmpx)        # sign flip per Os
    tmp1 = torch.sum(tmpx ** 2, dim=1)
    tmp2 = s * torch.sum((tmpx + mu0 - mu1) ** 2, dim=1) + d * m
    zc = _rotate(tmpx, M) if r_flag else tmpx
    tcos = torch.sum(torch.cos(2.0 * PI * zc), dim=1)
    return torch.minimum(tmp1, tmp2) + 10.0 * (m - tcos)


# hybrid sub-function order (official hf01..hf10), proportions Gp
_HYBRID = {
    11: ([0.2, 0.4, 0.4], ["zakharov", "rosenbrock", "rastrigin"]),
    12: ([0.3, 0.3, 0.4], ["ellips", "schwefel", "bent_cigar"]),
    13: ([0.3, 0.3, 0.4], ["bent_cigar", "rosenbrock", "bi_rastrigin"]),
    14: ([0.2, 0.2, 0.2, 0.4], ["ellips", "ackley", "schaffer", "rastrigin"]),
    15: ([0.2, 0.2, 0.3, 0.3], ["bent_cigar", "hgbat", "rastrigin", "rosenbrock"]),
    16: ([0.2, 0.2, 0.3, 0.3], ["escaffer6", "hgbat", "rosenbrock", "schwefel"]),
    17: ([0.1, 0.2, 0.2, 0.2, 0.3], ["katsuura", "ackley", "grie_rosen", "schwefel", "rastrigin"]),
    18: ([0.2, 0.2, 0.2, 0.2, 0.2], ["ellips", "ackley", "rastrigin", "hgbat", "discus"]),
    19: ([0.2, 0.2, 0.2, 0.2, 0.2], ["bent_cigar", "rastrigin", "grie_rosen", "weierstrass", "escaffer6"]),
    20: ([0.1, 0.1, 0.2, 0.2, 0.2, 0.2], ["hgbat", "katsuura", "ackley", "rastrigin", "schwefel", "schaffer"]),
}

# composition: (delta, [sub-fn names], [rescale factor or None per sub])
_COMP = {
    21: ([10, 20, 30], ["rosenbrock", "ellips", "rastrigin"], [None, 1e4 / 1e10, None]),
    22: ([10, 20, 30], ["rastrigin", "griewank", "schwefel"], [None, 1e3 / 1e2, None]),
    23: ([10, 20, 30, 40], ["rosenbrock", "ackley", "schwefel", "rastrigin"],
         [None, 1e3 / 1e2, None, None]),
    24: ([10, 20, 30, 40], ["ackley", "ellips", "griewank", "rastrigin"],
         [1e3 / 1e2, 1e4 / 1e10, 1e3 / 1e2, None]),
    25: ([10, 20, 30, 40, 50], ["rastrigin", "happycat", "ackley", "discus", "rosenbrock"],
         [1e4 / 1e3, 1e3 / 1e3, 1e3 / 1e2, 1e4 / 1e10, None]),
    26: ([10, 20, 20, 30, 40], ["escaffer6", "schwefel", "griewank", "rosenbrock", "rastrigin"],
         [1e4 / 2e7, None, 1e3 / 1e2, None, 1e4 / 1e3]),
    27: ([10, 20, 30, 40, 50, 60],
         ["hgbat", "rastrigin", "schwefel", "bent_cigar", "ellips", "escaffer6"],
         [1e4 / 1e3, 1e4 / 1e3, 1e4 / 4e3, 1e4 / 1e30, 1e4 / 1e10, 1e4 / 2e7]),
    28: ([10, 20, 30, 40, 50, 60],
         ["ackley", "griewank", "discus", "rosenbrock", "happycat", "escaffer6"],
         [1e3 / 1e2, 1e3 / 1e2, 1e4 / 1e10, None, 1e3 / 1e3, 1e4 / 2e7]),
    29: ([10, 30, 50], [15, 16, 17], [None, None, None]),   # cf09: hf05,hf06,hf07
    30: ([10, 30, 50], [15, 18, 19], [None, None, None]),   # cf10: hf05,hf08,hf09
}


def _partition(D, Gp):
    """Hybrid partition sizes: ceil(Gp*D) for all but last, remainder for last."""
    sizes = [math.ceil(p * D) for p in Gp[:-1]]
    sizes.append(D - sum(sizes))
    return sizes


class CEC2017Official:
    """Differentiable batched CEC2017, OFFICIAL numbering 1..30.

    fn = CEC2017Official(5, 30, 'cuda'); fn(x)  # x (B,30) -> (B,) raw f (incl bias)
    """

    def __init__(self, fid, D=30, device="cpu"):
        if not 1 <= fid <= 30:
            raise ValueError(f"fid must be 1..30, got {fid}")
        self.fid = fid
        self.func_id = fid          # alias (official numbering)
        self.D = D
        self.ndim = D               # alias
        self.device = device
        self.f_bias = fid * 100.0
        self.f_optimal = self.f_bias
        self.name = f"F{fid}"
        self.category = ("Unimodal" if fid <= 3 else "Multimodal" if fid <= 10
                         else "Hybrid" if fid <= 20 else "Composition")
        self.lb = torch.full((D,), -100.0, device=device, dtype=torch.float64)
        self.ub = torch.full((D,), 100.0, device=device, dtype=torch.float64)

        def T(a):
            return torch.tensor(a, device=device, dtype=torch.float64)

        if fid <= 20:
            self.shift = T(_load_shift_vec(fid, D))
            self.M = T(_load_M(fid, D))
            if 11 <= fid <= 20:
                self.shuffle = torch.tensor(_load_shuffle(fid, D), device=device,
                                            dtype=torch.long)
                self.Gp, self.sub = _HYBRID[fid]
                self.sizes = _partition(D, self.Gp)
        else:
            sm = _load_shift_mat(fid, D)
            self.delta, self.sub, self.rescale = _COMP[fid]
            n = len(self.delta)
            self.shift_mat = T(sm[:n])
            # the C always stores 10 rotation matrices per composition file
            self.M = T(_load_M(fid, D, 10))[:n]
            self.bias = T([100.0 * i for i in range(n)])
            if fid in (29, 30):
                shuf = _load_shuffle(fid, D, 10)[:n]
                self.shuffle_mat = torch.tensor(shuf, device=device, dtype=torch.long)

    # -- standalone simple function evaluation (also used for composition subs) --
    @staticmethod
    def _simple(name, x, shift, M):
        """x (B,m). Apply official shift+scale+rotate then base core."""
        if name == "schaffer":          # ignores rotation: uses y = shifted-scaled
            return _f_schaffer((x - shift) * RATE["schaffer"])
        if name == "bi_rastrigin":
            return _bi_rastrigin(x, shift, M, 1, 1)
        core, rate = _CORE[name]
        z = _rotate((x - shift) * rate, M)
        return core(z)

    def _eval_hybrid(self, x, shift, M):
        """Hybrid: shift+rotate, shuffle, partition, sum sub-functions."""
        z = _rotate((x - shift), M)              # hybrid sh_rate is 1.0
        shuffled = z[:, self.shuffle]
        out = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        start = 0
        for name, m in zip(self.sub, self.sizes):
            block = shuffled[:, start:start + m]
            if name == "schaffer":
                # C bug: schaffer reads global y[:m] == shuffled[:, :m]
                out = out + _f_schaffer(shuffled[:, :m])
            elif name == "bi_rastrigin":
                # sub-call s_flag=0,r_flag=0; Os = hybrid shift[:m]
                out = out + _bi_rastrigin(block, shift[:m], None, 0, 0)
            else:
                core, rate = _CORE[name]
                out = out + core(block * rate)
            start += m
        return out

    def __call__(self, x):
        if x.shape[1] != self.D:
            raise ValueError(f"expected D={self.D}, got {x.shape[1]}")
        fid = self.fid

        if fid == 8:                              # step_rastrigin == rastrigin (C dead code)
            z = _rotate((x - self.shift) * RATE["rastrigin"], self.M)
            return _f_rastrigin(z) + self.f_bias
        if fid <= 10:
            name = {1: "bent_cigar", 2: "sum_diff_pow", 3: "zakharov", 4: "rosenbrock",
                    5: "rastrigin", 6: "schaffer", 7: "bi_rastrigin", 9: "levy",
                    10: "schwefel"}[fid]
            return self._simple(name, x, self.shift, self.M) + self.f_bias
        if fid <= 20:
            return self._eval_hybrid(x, self.shift, self.M) + self.f_bias

        # composition (21..30)
        B = x.shape[0]
        n = len(self.delta)
        fits, ws = [], []
        for i in range(n):
            Os_i, M_i = self.shift_mat[i], self.M[i]
            if fid in (29, 30):
                sub_fid = self.sub[i]
                sub = CEC2017Official.__new__(CEC2017Official)
                sub.D = self.D
                sub.shuffle = self.shuffle_mat[i]
                sub.Gp, sub.sub = _HYBRID[sub_fid]
                sub.sizes = _partition(self.D, sub.Gp)
                fit = sub._eval_hybrid(x, Os_i, M_i)
            else:
                fit = self._simple(self.sub[i], x, Os_i, M_i)
                if self.rescale[i] is not None:
                    fit = fit * self.rescale[i]
            fit = fit + self.bias[i]
            w = torch.sum((x - Os_i) ** 2, dim=1)
            wi = torch.where(w != 0,
                             (1.0 / torch.clamp(w, min=1e-300)) ** 0.5
                             * torch.exp(-w / (2.0 * self.D * self.delta[i] ** 2)),
                             torch.full_like(w, 1e99))
            fits.append(fit)
            ws.append(wi)
        fits = torch.stack(fits, dim=1)           # (B, n)
        ws = torch.stack(ws, dim=1)
        w_max = ws.max(dim=1, keepdim=True).values
        allzero = (w_max == 0)
        ws = torch.where(allzero, torch.ones_like(ws), ws)
        ws = ws / torch.sum(ws, dim=1, keepdim=True)
        return torch.sum(ws * fits, dim=1) + self.f_bias


# ---------------------------------------------------------------------------
# CEC2017Corrected — drop-in replacement for the legacy cec2017.CEC2017Torch
# ---------------------------------------------------------------------------
# repo func_id (cec2017.FUNCTIONS, 1..29) names; the repo drops official F2.
_REPO_NAME = {
    1: "Bent Cigar", 2: "Zakharov", 3: "Rosenbrock", 4: "Rastrigin",
    5: "Expanded Schaffer F6", 6: "Lunacek Bi-Rastrigin",
    7: "Non-Continuous Rastrigin", 8: "Levy", 9: "Schwefel",
    **{k: f"Hybrid {k - 9}" for k in range(10, 20)},
    **{k: f"Composition {k - 19}" for k in range(20, 30)},
}


class CEC2017Corrected:
    """Drop-in replacement for legacy ``cec2017.CEC2017Torch`` backed by the
    validated official benchmark. Uses the repo numbering 1..29 (official F2,
    Sum of Different Power, is dropped); repo k -> official k+1 for k>=2.

    Same public surface as the legacy class: func_id, ndim, device, f_bias,
    f_optimal, lb, ub, name, category, shift / shift_mat, __call__.
    """

    def __init__(self, func_id, ndim=30, device="cpu"):
        if not 1 <= func_id <= 29:
            raise ValueError(f"func_id must be 1..29, got {func_id}")
        official = 1 if func_id == 1 else func_id + 1
        self._fn = CEC2017Official(official, ndim, device)
        self.func_id = func_id
        self.official_fid = official
        self.ndim = ndim
        self.device = device
        # f returns the official raw value (bias = official*100); gap = f - f_optimal
        self.f_optimal = self._fn.f_optimal
        self.f_bias = self._fn.f_bias
        self.lb = self._fn.lb
        self.ub = self._fn.ub
        self.name = _REPO_NAME[func_id]
        self.category = ("Unimodal" if func_id == 1 else
                         "Multimodal" if func_id <= 9 else
                         "Hybrid" if func_id <= 19 else "Composition")
        if hasattr(self._fn, "shift"):
            self.shift = self._fn.shift
        if hasattr(self._fn, "shift_mat"):
            self.shift_mat = self._fn.shift_mat

    def __call__(self, x):
        return self._fn(x)


def make_benchmark_fn(func_id, ndim, device, benchmark=None):
    """Factory: legacy cec2017.CEC2017Torch vs the corrected official benchmark.

    benchmark: 'legacy' | 'official' | None. None -> env var TERSQ_BENCHMARK
    (default 'legacy'). Lets eval scripts switch benchmark without arg plumbing
    (env vars propagate to ProcessPool workers).
    """
    if benchmark is None:
        benchmark = os.environ.get("TERSQ_BENCHMARK", "legacy")
    if benchmark == "official":
        return CEC2017Corrected(func_id, ndim, device)
    if benchmark == "legacy":
        from encoder.cec2017_torch import CEC2017Torch
        return CEC2017Torch(func_id, ndim, device)
    raise ValueError(f"benchmark must be 'legacy' or 'official', got {benchmark!r}")
