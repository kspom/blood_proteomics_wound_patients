"""Figure S7 -- per-patient PHS range and class summaries.

For each panel, fit LDA on the 13 baseline samples and apply to all 71.
Two parts per subplot:
  (top)    one row per patient, dot at mean PHS, bar from min to max
           across visits; sorted within class by mean PHS.
  (bottom) two summary rows -- class mean PHS as a dot, with a bar from
           min to max across all samples in the class.
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


def baseline_phs(m, y, vis, name_to_idx, proteins):
    idxs = [name_to_idx[p] for p in proteins]
    X = m[idxs].T
    clf = LinearDiscriminantAnalysis().fit(X[vis == 1], y[vis == 1])
    return clf.decision_function(X)


def per_patient_stats(I, y, g):
    rows = []
    for p in sorted(np.unique(g)):
        mask = g == p
        rows.append({"patient_id": int(p), "class": int(y[mask][0]),
                     "mean_I": float(I[mask].mean()),
                     "min_I": float(I[mask].min()),
                     "max_I": float(I[mask].max())})
    import pandas as pd
    return pd.DataFrame(rows).sort_values(["class", "mean_I"]).reset_index(drop=True)


def plot_panel(ax, pp, title, I, y):
    n = len(pp); ys = np.arange(n)
    cols = [POS if c == 1 else NEG for c in pp["class"]]
    for i, (_, r) in enumerate(pp.iterrows()):
        ax.plot([r["min_I"], r["max_I"]], [ys[i]] * 2, color=cols[i],
                linewidth=2.2, alpha=0.6, zorder=1)
    ax.scatter(pp["mean_I"], ys, c=cols, s=55, zorder=3,
               edgecolors="black", linewidths=0.5)
    ax.axvline(0, color="#444", linestyle="--", linewidth=1.0)

    I_neg = I[y == 0]; I_pos = I[y == 1]
    y_neg, y_pos, sep = -2.0, -3.0, -0.7
    ax.axhline(sep, color="#bbb", linewidth=0.8, linestyle=":")
    if len(I_neg):
        ax.plot([I_neg.min(), I_neg.max()], [y_neg] * 2, color=NEG,
                linewidth=3.0, alpha=0.6, zorder=1)
        ax.scatter([I_neg.mean()], [y_neg], c=NEG, s=80, zorder=3,
                   edgecolors="black", linewidths=0.6)
    if len(I_pos):
        ax.plot([I_pos.min(), I_pos.max()], [y_pos] * 2, color=POS,
                linewidth=3.0, alpha=0.6, zorder=1)
        ax.scatter([I_pos.mean()], [y_pos], c=POS, s=80, zorder=3,
                   edgecolors="black", linewidths=0.6)

    n_neg = int((pp["class"] == 0).sum()); n_pos = int((pp["class"] == 1).sum())
    ax.set_yticks(list(ys) + [y_neg, y_pos])
    ax.set_yticklabels(
        [f"P{int(p)} {'h' if c == 1 else 'nh'}"
         for p, c in zip(pp["patient_id"], pp["class"])]
        + [f"non-healers  (n={n_neg})", f"healers  (n={n_pos})"], fontsize=18)
    ax.set_ylim(y_pos - 0.7, n - 0.3)
    ax.set_xlabel("Protein Healing Score", fontsize=20)
    ax.set_title(title, fontsize=26, fontweight="bold", loc="left")
    ax.tick_params(axis="x", labelsize=16)
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)


def main():
    m, y, g, vis, wk, names, info = load_longitudinal(
        apply_per_protein_qc=False, apply_per_sample_qc=True)
    name_to_idx = {n: i for i, n in enumerate(names)}

    fig = plt.figure(figsize=(24, 14))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.55],
                          hspace=0.35, wspace=0.35)
    panel_axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
                  fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    for ax, (pname, proteins) in zip(panel_axes, PANELS.items()):
        I = baseline_phs(m, y, vis, name_to_idx, proteins)
        pp = per_patient_stats(I, y, g)
        plot_panel(ax, pp, f"{pname}  (size {len(proteins)})", I, y)
    legend_ax = fig.add_subplot(gs[:, 2])
    legend_ax.axis("off")
    legend_h = [
        Line2D([0], [0], marker="o", color=POS, markerfacecolor=POS,
               markeredgecolor="black", markersize=16, linewidth=2.8,
               label="healer"),
        Line2D([0], [0], marker="o", color=NEG, markerfacecolor=NEG,
               markeredgecolor="black", markersize=16, linewidth=2.8,
               label="non-healer"),
        Line2D([0], [0], color="#444", linestyle="--", linewidth=1.4,
               label="classification\nboundary  PHS = 0"),
    ]
    legend_ax.legend(handles=legend_h, loc="center", fontsize=22,
                     framealpha=0.95, handlelength=2.5,
                     borderpad=1.2, labelspacing=1.8)
    out = FIG / "figS7_per_patient_range.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
