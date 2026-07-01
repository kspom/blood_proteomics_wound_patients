"""Figure 4 -- per-protein and combined-score boxplots for Complement-5.

Six panels: one boxplot per protein in the Complement-5 panel
(CFB, F9, IGHV3-64, MINPP1, PI16) plus the combined Protein Healing
Score from a 5-protein LDA fit on all 71 samples. Healers and non-
healers are shown side-by-side with jittered dots. The PHS panel makes
the discriminative gain over single proteins visible at a glance.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from lib.data_utils import load_longitudinal
from lib.panels import PANELS, POS, NEG

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def plot_box(ax, values, labels, title, ylabel):
    v0 = values[labels == 0]; v1 = values[labels == 1]
    bp = ax.boxplot([v0, v1], positions=[1, 2], widths=0.55,
                    showfliers=False, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    for patch, col in zip(bp["boxes"], [NEG, POS]):
        patch.set_facecolor(col); patch.set_alpha(0.35)
        patch.set_edgecolor("black"); patch.set_linewidth(1.2)
    for el in bp["whiskers"] + bp["caps"]:
        el.set_color("black"); el.set_linewidth(1.2)
    rng = np.random.default_rng(0)
    for vals, col, pos in [(v0, NEG, 1), (v1, POS, 2)]:
        jit = rng.uniform(-0.17, 0.17, size=len(vals))
        ax.scatter(pos + jit, vals, s=55, color=col, edgecolor="black",
                   linewidth=0.5, alpha=0.85, zorder=3)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"non-healer\n(n={len(v0)})",
                        f"healer\n(n={len(v1)})"], fontsize=22)
    ax.set_ylabel(ylabel, fontsize=22)
    ax.set_title(title, fontsize=23, loc="left", fontweight="bold")
    ax.tick_params(axis="y", labelsize=19)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)


def main():
    m, y, g, vis, wk, names, info = load_longitudinal(
        apply_per_protein_qc=False, apply_per_sample_qc=True)
    name_to_idx = {n: i for i, n in enumerate(names)}
    proteins = PANELS["Complement-5"]
    idxs = [name_to_idx[p] for p in proteins]
    X = m[idxs].T
    clf = LinearDiscriminantAnalysis().fit(X, y)
    phs = clf.decision_function(X)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for ax, prot in zip(axes.flat, proteins):
        plot_box(ax, m[name_to_idx[prot]], y, prot, f"{prot} intensity")
    plot_box(axes.flat[5], phs, y, "5-protein score",
             "PHS  f(x)")
    axes.flat[5].axhline(0, color="#333", linestyle="--", linewidth=1.6)

    fig.tight_layout()
    out = FIG / "fig3_pairwise1_boxplots.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
