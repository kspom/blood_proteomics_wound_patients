"""Figure S4 -- distribution of seed-pair classification errors.

Histogram of LDA training error across all 100,576 protein pairs in the
449-protein filtered pool. The vertical line marks the cutoff used to
define the seed pairs (train_error <= 5) that enter the greedy-forward
branching procedure.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from lib.data_utils import load_longitudinal

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)
OUT = HERE / "output"
OUT.mkdir(parents=True, exist_ok=True)

SEED_CUTOFF_ERROR = 5


def main():
    m, y, g, vis, wk, names, info = load_longitudinal(
        apply_per_protein_qc=True, apply_per_sample_qc=True)
    n = m.shape[0]

    errors = []
    for i in range(n):
        for j in range(i + 1, n):
            X = m[[i, j]].T
            clf = LinearDiscriminantAnalysis().fit(X, y)
            errors.append(int((clf.predict(X) != y).sum()))
    errors = np.array(errors)
    print(f"scanned {len(errors)} pairs, min train_error = {errors.min()}")
    n_seeds = int((errors <= SEED_CUTOFF_ERROR).sum())
    print(f"seed pairs at train_error <= {SEED_CUTOFF_ERROR}: {n_seeds}")

    levels = np.arange(errors.min(), errors.max() + 1)
    counts = np.array([(errors == e).sum() for e in levels])
    colors = ["#b2182b" if e <= SEED_CUTOFF_ERROR else "#5577aa"
              for e in levels]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(levels, counts, color=colors, edgecolor="black", linewidth=0.4)
    for e, c in zip(levels, counts):
        if e <= SEED_CUTOFF_ERROR and c > 0:
            ax.text(e, c * 1.25, str(int(c)), ha="center", va="bottom",
                    fontsize=11, color="#b2182b", fontweight="bold")
    ax.axvline(SEED_CUTOFF_ERROR + 0.5, color="#b2182b", linestyle="--",
               linewidth=1.6,
               label=f"seed cutoff: train_error $\\leq$ {SEED_CUTOFF_ERROR} "
                     f"({n_seeds} pairs kept)")
    ax.set_xticks(levels[::2] if len(levels) > 20 else levels)
    ax.set_xlabel("LDA training error (number of misclassified samples)",
                  fontsize=13)
    ax.set_ylabel("Number of protein pairs", fontsize=13)
    ax.set_yscale("log")
    ax.legend(fontsize=12, loc="upper left")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    out = FIG / "figS4_seed_pair_error_histogram.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
