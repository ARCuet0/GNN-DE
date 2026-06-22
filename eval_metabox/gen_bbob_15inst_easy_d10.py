"""Training-ground (difficulty=easy = the 8 difficult-TRAIN funcs) 15-instance COCO, D=10:
median(IQR) markdown + LaTeX table + boxplot, 8-method roster. Pools 15 instances
(inst_1..14 + inst_3849), learned parts named <ALGO>_g{0,1}of2.json (func-quad).
This is the shared in-distribution home turf of every MetaBox learned method; GNN-DE is
zero-shot here. Mirrors gen_bbob_15inst_d20.py.
"""
import json, math, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

B = "eval_metabox/results/bbob_15inst_easy_d10"
INSTS = [str(i) for i in range(1, 15)] + ["3849"]
METH = ["GNN_DE", "RLDEAFL", "DEDDQN", "DE", "MADDE", "NLSHADELBC", "JDE21", "Random_search"]
NAME = {"GNN_DE": "GNN-DE", "RLDEAFL": "RLDE-AFL", "DEDDQN": "DE-DDQN", "DE": "DE",
        "MADDE": "MadDE", "NLSHADELBC": "NL-SHADE-LBC", "JDE21": "jDE21", "Random_search": "RS"}
HEAVY = {"GNN_DE", "RLDEAFL", "DEDDQN"}
CATS = [
    ("Separable", [("Sphere", "Sphere"), ("Ellipsoidal", "Ellipsoidal (sep)"),
                   ("Rastrigin", "Rastrigin (sep)"), ("Linear_Slope", "Linear Slope")]),
    ("Multimodal, adequate structure", [
        ("Rastrigin_F15", "Rastrigin"), ("Weierstrass", "Weierstrass"), ("Schaffers", "Schaffers F7")]),
    ("Multimodal, weak structure", [("Gallagher_101Peaks", "Gallagher 101")]),
]
ALLF = [(k, d) for _, fns in CATS for k, d in fns]


def load(p):
    try:
        return json.load(open(p))["data"]
    except Exception:
        return {}


def inst_funcs(m, i):
    fd = {}
    if m in HEAVY:
        for g in range(2):
            for fn, val in load(f"{B}/inst_{i}/parts/{m}_g{g}of2.json").items():
                if fn not in fd:
                    fd[fn] = val["finals"]
    else:
        for fn, val in load(f"{B}/inst_{i}/{m}.json").items():
            fd[fn] = val["finals"]
    return fd


def collect(m):
    s = {}
    for i in INSTS:
        for fn, fin in inst_funcs(m, i).items():
            s.setdefault(fn, []).extend(fin)
    return s


DATA = {m: collect(m) for m in METH}
for m in METH:
    cs = [len(DATA[m].get(k, [])) for k, _ in ALLF]
    print(f"{m:<14} {min(cs)}-{max(cs)} seeds (~{min(cs)//51}-{max(cs)//51} inst)")


def fmt(x):
    if abs(x) < 1e-10:
        return "<1e-10"
    if 1e-3 <= abs(x) < 1e4:
        return f"{x:.2f}" if abs(x) >= 1 else f"{x:.1e}"
    e = int(math.floor(math.log10(abs(x))))
    return f"{x/10**e:.1f}e{e}"


# markdown
print("\n| Function | " + " | ".join(NAME[m] for m in METH) + " |")
print("|" + "---|" * (len(METH) + 1))
gw = 0
for cat, fns in CATS:
    print(f"| **_{cat}_** |" + " |" * len(METH))
    for k, disp in fns:
        meds = {m: float(np.median(DATA[m][k])) for m in METH}
        best = min(meds.values())
        if meds["GNN_DE"] <= best * 1.0000001:
            gw += 1
        cells = []
        for m in METH:
            a = np.asarray(DATA[m][k], float)
            md, iq = float(np.median(a)), float(np.percentile(a, 75) - np.percentile(a, 25))
            cells.append(f"**{fmt(md)} ({fmt(iq)})**" if md <= best * 1.0000001 else f"{fmt(md)} ({fmt(iq)})")
        print(f"| {disp} | " + " | ".join(cells) + " |")
print(f"\nGNN-DE lowest median (of 8) on the training ground D=10: {gw}/8")


# LaTeX
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
        "\\caption{In-distribution comparison on the learned methods' training ground: the 8 BBOB "
        "\\emph{difficult-train} functions ($D=10$), the shared training set of every MetaBox learned "
        "baseline. Aggregated over the 15-instance COCO protocol (14 held-out instance realizations "
        "plus instance\\_seed 3849, 51 seeds each, $\\mathrm{maxFEs}=20000$). Each cell is the median "
        "with the interquartile range (IQR); lowest median per row in \\textbf{bold}. Lower is better. "
        "The learned baselines (DE-DDQN, RLDE-AFL) are \\emph{in distribution} here, GNN-DE is "
        "zero-shot.\\label{tab:bbobtrain}}",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l" + "c" * len(METH) + "}", "\\toprule",
        "\\textbf{Function} & " + " & ".join("\\textbf{" + NAME[m] + "}" for m in METH) + " \\\\",
        "\\midrule"]
for cat, fns in CATS:
    rows.append(f"\\multicolumn{{{len(METH)+1}}}{{l}}{{\\textit{{{cat}}}}} \\\\")
    for key, disp in fns:
        st = {m: (float(np.median(DATA[m][key])),
                  float(np.percentile(DATA[m][key], 75) - np.percentile(DATA[m][key], 25))) for m in METH}
        best = min(s[0] for s in st.values())
        rows.append(f"{disp} & " + " & ".join(cell(st[m][0], st[m][1], st[m][0] <= best * 1.0000001) for m in METH) + " \\\\")
    rows.append("\\midrule")
rows[-1] = "\\bottomrule"
rows += ["\\end{tabular}}", "\\end{table}"]
open("paper/mdpi_submission/tab_bbob_train_d10.tex", "w").write("\n".join(rows))
print("\nwrote paper/mdpi_submission/tab_bbob_train_d10.tex")

# boxplot (drop RS)
BM = METH[:-1]; BS = [NAME[m] for m in BM]
colors = ["#d62728"] + ["#4c78a8"] * (len(BM) - 1); FLOOR = 1e-12
fig, axes = plt.subplots(2, 4, figsize=(22, 9)); axes = axes.ravel()
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
    ax.set_xticks(xs); ax.set_xticklabels(BS, rotation=45, ha="right", fontsize=8); ax.grid(axis="y", alpha=0.3)
for ax in axes[len(ALLF):]:
    ax.axis("off")
fig.suptitle("BBOB training ground (8 difficult-train funcs), $D=10$, 15-instance COCO (765 runs, log; "
             "RS omitted). Learned baselines in distribution, GNN-DE zero-shot. "
             "White diamond = median, gold star = lowest median.", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
for ext in ("png", "pdf"):
    fig.savefig(f"paper/mdpi_submission/figures/fig_bbob_box_train_d10.{ext}",
                dpi=150 if ext == "png" else None, bbox_inches="tight")
print("wrote paper/mdpi_submission/figures/fig_bbob_box_train_d10.{png,pdf}")
