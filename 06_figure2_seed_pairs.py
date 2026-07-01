"""Figure 2 -- representative top-ranked seed pairs.

For each of the two top-ranked pairs by LDA training error
    1.  (CFB, PI16)
    2.  (ICOSLG, CFB)
produce three panels: scatter in the two-protein plane with LDA boundary
and misclassified samples marked, plus per-protein box plots showing
that neither protein alone separates the two classes.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from lib.data_utils import load_longitudinal
from lib.panels import POS, NEG

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)

PAIRS = [("CFB", "PI16"), ("ICOSLG", "CFB")]


def plot_scatter(ax, x, y, labels, px, py, title, wrap_labels=True,
                 legend_loc="upper left"):
    clf = LinearDiscriminantAnalysis()
    X = np.column_stack([x, y])
    clf.fit(X, labels)
    pred = clf.predict(X); wrong = pred != labels; n_err = int(wrong.sum())

    for cls, col, lbl in [(0, NEG, "non-healer"), (1, POS, "healer")]:
        mk = labels == cls
        ax.scatter(x[mk], y[mk], c=col, s=120, alpha=0.9,
                   edgecolors="black", linewidths=0.8, label=lbl, zorder=3)
    if wrong.any():
        mis_lbl = (f"misclassi-\nfied ({n_err})" if wrap_labels
                   else f"misclassified ({n_err})")
        ax.scatter(x[wrong], y[wrong], facecolors="none", edgecolors="black",
                   s=280, linewidths=2.0, label=mis_lbl, zorder=4)
    w = clf.coef_.ravel(); b = float(clf.intercept_.ravel()[0])
    xs = np.array([x.min() * 0.95, x.max() * 1.05])
    bnd_lbl = "classification\nboundary" if wrap_labels else "classification boundary"
    if abs(w[1]) > 1e-12:
        ys = -(w[0] * xs + b) / w[1]
        ax.plot(xs, ys, "--", color="#333", linewidth=1.8, label=bnd_lbl, zorder=2)
    ax.set_xlabel(px, fontsize=36, fontweight="bold")
    ax.set_ylabel(py, fontsize=36, fontweight="bold")
    ax.set_title(title, fontsize=28, loc="left")
    ax.tick_params(axis="both", labelsize=26)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=20, loc=legend_loc, framealpha=0.95)
    ax.set_xlim(min(x.min() * 0.95, 0), x.max() * 1.08)
    ax.set_ylim(min(y.min() * 0.95, 0), y.max() * 1.08)


def plot_box(ax, values, labels, pname):
    v0 = values[labels == 0]; v1 = values[labels == 1]
    bp = ax.boxplot([v0, v1], positions=[1, 2], widths=0.55,
                    showfliers=False, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    for patch, col in zip(bp["boxes"], [NEG, POS]):
        patch.set_facecolor(col); patch.set_alpha(0.35)
        patch.set_edgecolor("black"); patch.set_linewidth(1.2)
    for whisker in bp["whiskers"]:
        whisker.set_color("black"); whisker.set_linewidth(1.2)
    for cap in bp["caps"]:
        cap.set_color("black"); cap.set_linewidth(1.2)
    rng = np.random.default_rng(0)
    for vals, col, pos in [(v0, NEG, 1), (v1, POS, 2)]:
        jitter = rng.uniform(-0.17, 0.17, size=len(vals))
        ax.scatter(pos + jitter, vals, s=60, color=col, edgecolor="black",
                   linewidth=0.6, alpha=0.85, zorder=3)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"non-healer\n(n={len(v0)})", f"healer\n(n={len(v1)})"],
                       fontsize=30)
    ax.set_ylabel(f"{pname} intensity", fontsize=31)
    ax.set_title(f"Single-protein view: {pname}", fontsize=28, loc="left")
    ax.tick_params(axis="y", labelsize=26)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)


def main():
    m, y, g, vis, wk, names, info = load_longitudinal(
        apply_per_protein_qc=True, apply_per_sample_qc=True)
    name_to_idx = {n: i for i, n in enumerate(names)}

    fig, axes = plt.subplots(2, 3, figsize=(28, 17))
    drawn = set()
    for row, (pa, pb) in enumerate(PAIRS):
        ia, ib = name_to_idx[pa], name_to_idx[pb]
        xa, xb = m[ia], m[ib]
        is_first = row == 0
        plot_scatter(axes[row, 0], xa, xb, y, pa, pb,
                     f"Seed pair #{row + 1}: {pa} + {pb}",
                     wrap_labels=is_first,
                     legend_loc="upper left" if is_first else "lower left")
        col = 1
        for prot, vals in [(pa, xa), (pb, xb)]:
            if prot in drawn:
                continue
            plot_box(axes[row, col], vals, y, prot)
            drawn.add(prot); col += 1
        for c in range(col, 3):
            axes[row, c].axis("off")

    fig.tight_layout()
    out = FIG / "fig2_seed_pairs.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
