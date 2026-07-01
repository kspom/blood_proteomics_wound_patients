"""Figure S5 -- top-20 seed pairs by training error, in protein-protein
planes.

A 4 x 5 grid, one cell per pair. Each cell is a scatter of samples in
the two-protein plane, coloured by class, with the LDA decision boundary
drawn. The number on top of each plane is the LDA training error.
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


def main():
    m, y, g, vis, wk, names, info = load_longitudinal(
        apply_per_protein_qc=True, apply_per_sample_qc=True)
    n = m.shape[0]

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            X = m[[i, j]].T
            clf = LinearDiscriminantAnalysis().fit(X, y)
            err = int((clf.predict(X) != y).sum())
            pairs.append((err, i, j, clf))
    pairs.sort(key=lambda r: r[0])
    top = pairs[:20]

    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    for ax, (err, i, j, clf) in zip(axes.flat, top):
        xa, xb = m[i], m[j]
        for cls, col in [(0, NEG), (1, POS)]:
            mask = y == cls
            ax.scatter(xa[mask], xb[mask], c=col, s=120, alpha=0.9,
                       edgecolors="none", zorder=3)
        w = clf.coef_.ravel(); b = float(clf.intercept_.ravel()[0])
        xs = np.array([xa.min() * 0.95, xa.max() * 1.05])
        if abs(w[1]) > 1e-12:
            ys = -(w[0] * xs + b) / w[1]
            ax.plot(xs, ys, "--", color="#333", linewidth=1.5, zorder=2)
        ax.set_xlabel(names[i], fontsize=22, fontweight="bold")
        ax.set_ylabel(names[j], fontsize=22, fontweight="bold")
        ax.set_title(f"err = {err}", fontsize=24)
        ax.tick_params(axis="both", labelsize=16)
        ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    out = FIG / "figS5_top20_seed_pairs.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
