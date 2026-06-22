"""15-instance COCO robustness, D=20: median(IQR) LaTeX table + boxplot, 8-method roster.
Pools 14 new COCO instances (bbob_15inst_d20/inst_1..14) + instance 3849 (bbob_metabox_d20),
deduping per (instance, function) across GNN-DE's two part-naming schemes (f* and g*of8) and
the learned func-quad parts. Mirrors gen_bbob_perfunc_table.py formatting.
"""
import json, math, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

B = "eval_metabox/results/bbob_15inst_d20"; LEG = "eval_metabox/results/bbob_metabox_d20"
METH = ["GNN_DE", "RLDEAFL", "DEDDQN", "DE", "MADDE", "NLSHADELBC", "JDE21", "Random_search"]
NAME = {"GNN_DE": "GNN-DE", "RLDEAFL": "RLDE-AFL", "DEDDQN": "DE-DDQN", "DE": "DE",
        "MADDE": "MadDE", "NLSHADELBC": "NL-SHADE-LBC", "JDE21": "jDE21", "Random_search": "RS"}
QUAD = {"RLDEAFL", "DEDDQN"}
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


def load(p):
    try:
        return json.load(open(p))["data"]
    except Exception:
        return {}


def inst_funcs(m, i):
    fd = {}
    if m == "GNN_DE":
        for g in range(8):
            for v in (load(f"{B}/inst_{i}/parts/GNN_DE_g{g}of8.json"),
                      load(f"{B}/inst_{i}/parts/GNN_DE_f{g}.json")):
                for fn, val in v.items():
                    if fn not in fd and len(val["finals"]) == 51:
                        fd[fn] = val["finals"]
    elif m in QUAD:
        for g in range(4):
            for fn, val in load(f"{B}/inst_{i}/parts/{m}_g{g}of4.json").items():
                if fn not in fd:
                    fd[fn] = val["finals"]
    else:
        for fn, val in load(f"{B}/inst_{i}/{m}.json").items():
            fd[fn] = val["finals"]
    return fd


def collect(m):
    s = {}
    for i in range(1, 15):
        for fn, fin in inst_funcs(m, i).items():
            s.setdefault(fn, []).extend(fin)
    for fn, val in load(f"{LEG}/{m}.json").items():
        s.setdefault(fn, []).extend(val["finals"])
    return s


DATA = {m: collect(m) for m in METH}


def stat(m, fn):
    v = np.asarray(DATA[m][fn], float)
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


rows = ["\\begin{table}[H]", "\\centering\\footnotesize",
        "\\caption{Zero-shot final gap to the optimum on the 16 held-out BBOB \\emph{difficult} "
        "functions at $D=20$, aggregated over the 15-instance COCO protocol (14 held-out instance "
        "realizations plus instance\\_seed 3849, 51 seeds each, $\\mathrm{maxFEs}=20000$). Each cell "
        "is the median with the interquartile range (IQR) in parentheses over the pooled runs; lowest "
        "median per row in \\textbf{bold}. Lower is better. At $D=20$ every method is out of "
        "distribution on dimension, the learned baselines included (checkpoints trained on BBOB-10D)."
        "\\label{tab:bbob15d20}}",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l" + "c" * len(METH) + "}", "\\toprule",
        "\\textbf{Function} & " + " & ".join("\\textbf{" + NAME[m] + "}" for m in METH) + " \\\\",
        "\\midrule"]
for cat, fns in CATS:
    rows.append(f"\\multicolumn{{{len(METH)+1}}}{{l}}{{\\textit{{{cat}}}}} \\\\")
    for key, disp in fns:
        st = {m: stat(m, key) for m in METH}
        best = min(s[0] for s in st.values())
        cells = [cell(st[m][0], st[m][1], st[m][0] <= best * 1.0000001) for m in METH]
        rows.append(f"{disp} & " + " & ".join(cells) + " \\\\")
    rows.append("\\midrule")
rows[-1] = "\\bottomrule"
rows += ["\\end{tabular}}", "\\end{table}"]
open("paper/mdpi_submission/tab_bbob_perfunc_15inst_d20.tex", "w").write("\n".join(rows))
print("wrote paper/mdpi_submission/tab_bbob_perfunc_15inst_d20.tex")

# boxplot (drop RS for legibility)
BM = METH[:-1]; BS = [NAME[m] for m in BM]
colors = ["#d62728"] + ["#4c78a8"] * (len(BM) - 1); FLOOR = 1e-12
fig, axes = plt.subplots(4, 4, figsize=(24, 16)); axes = axes.ravel()
ALLF = [(k, d) for _, fns in CATS for k, d in fns]
for ax, (k, disp) in zip(axes, ALLF):
    data = [np.clip(np.asarray(DATA[m][k], float), FLOOR, None) for m in BM]
    meds = [float(np.median(d)) for d in data]; wi = int(np.argmin(meds))
    bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col); patch.set_alpha(0.6)
    for mm in bp["medians"]:
        mm.set_color("black"); mm.set_linewidth(1.6)
    xs = list(range(1, len(BM) + 1))
    ax.scatter(xs, meds, marker="D", s=30, facecolor="white", edgecolor="black", zorder=5)
    ax.scatter([wi + 1], [meds[wi]], marker="*", s=300, color="gold", edgecolor="black", zorder=6)
    ax.set_yscale("log"); ax.set_title(disp, fontsize=12, fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels(BS, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
fig.suptitle("BBOB difficult $D=20$, 15-instance COCO robustness (up to 765 runs, log scale; RS omitted). "
             "White diamond = median, gold star = lowest median.", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.985])
for ext in ("png", "pdf"):
    fig.savefig(f"paper/mdpi_submission/figures/fig_bbob_box_15inst_d20.{ext}",
                dpi=150 if ext == "png" else None, bbox_inches="tight")
print("wrote paper/mdpi_submission/figures/fig_bbob_box_15inst_d20.{png,pdf}")
