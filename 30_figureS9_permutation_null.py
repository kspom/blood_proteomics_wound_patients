"""Figure S9 -- nested validation error under permuted labels.

Histogram of the nested leave-one-patient-out sample error (out of 68)
over the label permutations of 29_tableS23_permutation_test.py, with the
error observed for the true labels (Table S22) marked by a vertical line.
Reads output only; run 28 and 29 first.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)
OUT = HERE / "output"


def main():
    runs = pd.read_csv(OUT / "tableS23_permutation_test.csv")
    s22 = pd.read_csv(OUT / "tableS22_nested_lopo.csv")
    observed = int(s22["sample_errors"].sum())
    n_samples = int(s22["n_samples"].sum())
    null = runs["sample_errors"].to_numpy()
    N = len(null)
    p = (1 + int((null <= observed).sum())) / (1 + N)

    levels = np.arange(0, n_samples + 1)
    counts = np.array([(null == e).sum() for e in levels])
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(levels, counts, color="#5577aa", edgecolor="black", linewidth=0.4,
           label=f"permuted labels ({N} permutations)")
    ax.axvline(observed, color="#b2182b", linestyle="--", linewidth=1.8,
               label=f"true labels: {observed}/{n_samples} errors "
                     f"(p = {p:.2g})")
    ax.set_xlim(-0.5, n_samples + 0.5)
    ax.set_xlabel("Nested leave-one-patient-out error "
                  f"(misclassified samples of {n_samples})", fontsize=13)
    ax.set_ylabel("Number of permutations", fontsize=13)
    ax.legend(fontsize=12, loc="upper left")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    out = FIG / "figS9_permutation_null.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out}  (observed {observed}, p = {p:.4f}, N = {N})")


if __name__ == "__main__":
    main()
