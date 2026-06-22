"""15-instance COCO robustness, D=10 (difficult): boxplot of the held-out roster.

Pools 14 COCO instances (bbob_15inst/inst_1..14) + instance 3849 (bbob_metabox
legacy), reading one JSON per method (no part-splitting at D=10). Mirrors the
styling of gen_bbob_15inst_d20.py (gold star = lowest median, white diamond =
median, RS dropped from the boxplot for legibility) so the D=10 and D=20 held-out
figures are directly comparable, including DE-DDQN whose ill-conditioned strength
at D=10 collapses at D=20.
"""
import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

B = "eval_metabox/results/bbob_15inst"
LEG = "eval_metabox/results/bbob_metabox"
METH = ["GNN_DE", "RLDEAFL", "DEDDQN", "MADDE", "NLSHADELBC", "JDE21", "Random_search"]
NAME = {"GNN_DE": "GNN-DE", "RLDEAFL": "RLDE-AFL", "DEDDQN": "DE-DDQN",
        "MADDE": "MadDE", "NLSHADELBC": "NL-SHADE-LBC", "JDE21": "jDE21",
        "Random_search": "RS"}
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


def _load(p):
    try:
        return json.load(open(p))["data"]
    except Exception:
        return {}


def collect(m):
    """Pool a method's per-function finals over the 14 COCO instances + legacy."""
    s = {}
    for d in sorted(glob.glob(f"{B}/inst_*")):
        for fn, val in _load(f"{d}/{m}.json").items():
            s.setdefault(fn, []).extend(val["finals"])
    for fn, val in _load(f"{LEG}/{m}.json").items():
        s.setdefault(fn, []).extend(val["finals"])
    return s


def main():
    DATA = {m: collect(m) for m in METH}
    BM = METH[:-1]  # drop RS for legibility
    BS = [NAME[m] for m in BM]
    colors = ["#d62728"] + ["#4c78a8"] * (len(BM) - 1)
    FLOOR = 1e-12
    fig, axes = plt.subplots(4, 4, figsize=(24, 16))
    axes = axes.ravel()
    ALLF = [(k, d) for _, fns in CATS for k, d in fns]
    for ax, (k, disp) in zip(axes, ALLF):
        data = [np.clip(np.asarray(DATA[m][k], float), FLOOR, None) for m in BM]
        meds = [float(np.median(d)) for d in data]
        wi = int(np.argmin(meds))
        bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.6)
        for mm in bp["medians"]:
            mm.set_color("black")
            mm.set_linewidth(1.6)
        xs = list(range(1, len(BM) + 1))
        ax.scatter(xs, meds, marker="D", s=30, facecolor="white", edgecolor="black", zorder=5)
        ax.scatter([wi + 1], [meds[wi]], marker="*", s=300, color="gold", edgecolor="black", zorder=6)
        ax.set_yscale("log")
        ax.set_title(disp, fontsize=12, fontweight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels(BS, rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("BBOB difficult $D=10$, 15-instance COCO robustness (765 runs, log scale; RS omitted). "
                 "White diamond = median, gold star = lowest median.", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    for ext in ("png", "pdf"):
        fig.savefig(f"paper/mdpi_submission/figures/fig_bbob_box_15inst_d10.{ext}",
                    dpi=150 if ext == "png" else None, bbox_inches="tight")
    print("wrote paper/mdpi_submission/figures/fig_bbob_box_15inst_d10.{png,pdf}")


if __name__ == "__main__":
    main()
