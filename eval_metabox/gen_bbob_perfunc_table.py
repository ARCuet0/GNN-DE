"""BBOB per-function median (IQR) table, instance 3849, 51 seeds, paper style.
Replaces the W/L/T tab:bbob. median in cell, IQR in scriptsize parens, best per row bold.
DE-DDQN is a genuine competitor (net -1.5 vs GNN-DE on the 14-instance robustness run, see
finding_bbob_15inst_audit_percategory_2026_06_18) so it is now a COLUMN. Only the truly
budget-crippled DE-DQN / RL-HPSDE (net +16 sweeps) stay in the Supplementary table.
"""
import json
import math
import os

import numpy as np

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--dir", default="eval_metabox/results/bbob_metabox")
_ap.add_argument("--dim", type=int, default=10)
_ap.add_argument("--out", default="paper/mdpi_submission/tab_bbob_perfunc.tex")
_ap.add_argument("--label", default="tab:bbob")
_args = _ap.parse_args()
DIR = _args.dir   # instance_seed=3849
NAME = {"GNN_DE": "GNN-DE", "DE": "DE", "MADDE": "MadDE", "NLSHADELBC": "NL-SHADE-LBC",
        "JDE21": "jDE21", "LDE": "LDE", "GLEET": "GLEET", "RLDAS": "RL-DAS",
        "RLDEAFL": "RLDE-AFL", "DEDDQN": "DE-DDQN", "Random_search": "RS"}
METHODS = ["GNN_DE", "RLDEAFL", "DEDDQN", "NLSHADELBC", "MADDE", "JDE21", "Random_search"]

CATS = [
    ("Separable", [("Buche_Rastrigin", "Bueche-Rastrigin")]),
    ("Low/moderate conditioning (unimodal)", [
        ("Attractive_Sector", "Attractive Sector"), ("Step_Ellipsoidal", "Step Ellipsoidal"),
        ("Rosenbrock_original", "Rosenbrock"), ("Rosenbrock_rotated", "Rosenbrock rot.")]),
    ("High conditioning (unimodal)", [
        ("Ellipsoidal_high_cond", "Ellipsoidal"), ("Discus", "Discus"),
        ("Bent_Cigar", "Bent Cigar"), ("Sharp_Ridge", "Sharp Ridge"),
        ("Different_Powers", "Different Powers")]),
    ("Multimodal, adequate structure", [
        ("Schaffers_high_cond", "Schaffers"), ("Composite_Grie_rosen", "Composite G-R")]),
    ("Multimodal, weak structure", [
        ("Schwefel", "Schwefel"), ("Gallagher_21Peaks", "Gallagher 21"),
        ("Katsuura", "Katsuura"), ("Lunacek_bi_Rastrigin", "Lunacek")]),
]

_cache = {}
def load(m):
    if m not in _cache:
        _cache[m] = json.load(open(os.path.join(DIR, f"{m}.json")))["data"]
    return _cache[m]

def stat(m, fn):
    v = np.asarray(load(m)[fn]["finals"], float)
    return float(np.median(v)), float(np.percentile(v, 75) - np.percentile(v, 25))

def fnum(x):
    if x < 1e-10:
        return r"<\!10^{-10}"
    if 1e-3 <= abs(x) < 1e4:
        return f"{x:.2f}" if abs(x) >= 1 else f"{x:.2e}".replace("e-0", r"\times10^{-").replace("e-", r"\times10^{-") + "}"
    e = int(math.floor(math.log10(abs(x))))
    return f"{x/10**e:.2f}\\times10^{{{e}}}"

def cell(med, iqr, best):
    m = fnum(med)
    m = (r"{\boldmath $" + m + r"$}") if best else (f"${m}$")
    return m + r" {\scriptsize$(" + fnum(iqr) + r")$}"

_capextra = ("" if _args.dim == 10 else
             " At $D=20$ every method is out of distribution on dimension, including the "
             "learned baselines, whose checkpoints were trained on BBOB-10D.")
rows = ["\\begin{table}[H]", "\\centering\\footnotesize",
        "\\caption{Zero-shot final gap to the optimum on the 16 held-out BBOB \\emph{difficult} "
        f"functions ($D={_args.dim}$, instance\\_seed 3849, 51 seeds, $\\mathrm{{maxFEs}}=20000$). "
        "Each cell is the median with the interquartile range (IQR) in parentheses; lowest "
        "median per row in \\textbf{bold}. Lower is better. RL-DAS, GLEET and LDE and the "
        "budget-crippled DE-DQN and RL-HPSDE are omitted for space; the full twelve-method "
        f"table is in the Supplementary Material.{_capextra}\\label{{{_args.label}}}}}",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l" + "c" * len(METHODS) + "}", "\\toprule",
        "\\textbf{Function} & " + " & ".join("\\textbf{" + NAME[m] + "}" for m in METHODS) + " \\\\",
        "\\midrule"]
for cat, fns in CATS:
    rows.append(f"\\multicolumn{{{len(METHODS)+1}}}{{l}}{{\\textit{{{cat}}}}} \\\\")
    for key, disp in fns:
        st = {m: stat(m, key) for m in METHODS}
        best = min(s[0] for s in st.values())
        cells = [cell(st[m][0], st[m][1], st[m][0] <= best * 1.0000001) for m in METHODS]
        rows.append(f"{disp} & " + " & ".join(cells) + " \\\\")
    rows.append("\\midrule")
rows[-1] = "\\bottomrule"
rows += ["\\end{tabular}}", "\\end{table}"]
out = "\n".join(rows)
open(_args.out, "w").write(out)
print(out)
