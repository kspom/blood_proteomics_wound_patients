"""Figure 5 -- per-patient PHS trajectories for the five comparison panels.

For each panel, fit LDA on all 71 QC-passed samples, score every sample
with f(x) = w . x + b, plot one line per patient against week of sample
collection, coloured by class. Layout is 2 x 3 with the 6th cell hidden
or used for a legend.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from lib.data_utils import load_longitudinal
from lib.panels import PANELS, POS, NEG

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def plot_traj(ax, scores, classes, patients, weeks, title):
    ax.axhline(0.0, color="#333", linestyle="--", linewidth=1.8, zorder=1)
    for p in sorted(np.unique(patients)):
        mask = patients == p; cls = int(classes[mask][0])
        col = POS if cls == 1 else NEG
        order = np.argsort(weeks[mask])
        ax.plot(weeks[mask][order], scores[mask][order], "-o", color=col,
                linewidth=1.6, markersize=6.5, alpha=0.85,
                markerfacecolor=col, markeredgecolor="white",
                markeredgewidth=0.6)
    ax.set_xlabel("Week", fontsize=24)
    ax.set_ylabel("Protein Healing Score", fontsize=24)
    ax.set_title(title, fontsize=29, fontweight="bold", loc="left")
    ax.tick_params(axis="both", labelsize=20)
    ax.grid(True, linestyle="--", alpha=0.3)


def main():
    m, y, g, vis, wk, names, info = load_longitudinal(
        apply_per_protein_qc=False, apply_per_sample_qc=True)
    name_to_idx = {n: i for i, n in enumerate(names)}

    fig = plt.figure(figsize=(22, 12))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.85],
                          hspace=0.30, wspace=0.25)
    panel_axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
                  fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    for ax, (pname, proteins) in zip(panel_axes, PANELS.items()):
        idxs = [name_to_idx[p] for p in proteins]
        X = m[idxs].T
        clf = LinearDiscriminantAnalysis().fit(X, y)
        plot_traj(ax, clf.decision_function(X), y, g, wk,
                  f"{pname}  (size {len(proteins)})")
    legend_ax = fig.add_subplot(gs[:, 2])
    legend_ax.axis("off")
    legend_h = [
        Line2D([0], [0], marker="o", color=POS, markerfacecolor=POS,
               markeredgecolor="white", markersize=14, linewidth=2.0,
               label="healer patient"),
        Line2D([0], [0], marker="o", color=NEG, markerfacecolor=NEG,
               markeredgecolor="white", markersize=14, linewidth=2.0,
               label="non-healer patient"),
        Line2D([0], [0], color="#333", linestyle="--", linewidth=2.0,
               label="classification\nboundary  PHS = 0"),
    ]
    legend_ax.legend(handles=legend_h, loc="center", fontsize=24,
                     framealpha=0.95, handlelength=2.5,
                     borderpad=1.2, labelspacing=1.5)
    out = FIG / "fig5_phs_trajectories.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
