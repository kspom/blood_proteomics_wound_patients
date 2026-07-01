"""Figure S2 -- zero-fraction distributions for the two QC filters.

(a) per-protein zero-fraction across all 1326 proteins; the 449 proteins
    with the lowest zero-fraction (<= 25%) are kept and form the
    "filtered pool" used for the search and the DE analysis.
(b) per-sample zero-fraction across all 77 samples, coloured by class;
    the [0.10, 0.80] band marks the QC-pass region.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from lib.data_utils import (load_longitudinal, FILTERED_POOL_FILE,
                            PER_SAMPLE_ZERO_LOW, PER_SAMPLE_ZERO_HIGH)
from lib.panels import POS, NEG

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def main():
    m, y_all, g_all, v_all, w_all, names, info = load_longitudinal(
        apply_per_protein_qc=False, apply_per_sample_qc=False)
    per_protein_zf = info["per_protein_zf_all"]
    per_sample_zf = info["per_sample_zf_all"]

    with open(FILTERED_POOL_FILE, encoding="utf-8-sig") as f:
        pool = {ln.strip() for ln in f if ln.strip()}
    in_pool = np.array([n in pool for n in names])

    order = np.argsort(per_protein_zf)
    zf_sorted = per_protein_zf[order]
    in_pool_sorted = in_pool[order]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    xs = np.arange(len(zf_sorted))
    colors = ["#1f77b4" if k else "#cccccc" for k in in_pool_sorted]
    ax.bar(xs, zf_sorted, color=colors, width=1.0, edgecolor="none")
    n_pool = int(in_pool.sum())
    ax.axvline(n_pool, color="black", linestyle="--", linewidth=1.0)
    ax.text(n_pool + 5, 0.05,
            f"{n_pool} proteins kept (filtered pool)",
            fontsize=10)
    ax.set_xlabel("Protein index (sorted by per-protein zero-fraction)",
                  fontsize=11)
    ax.set_ylabel("Per-protein zero-fraction", fontsize=11)
    ax.set_title("(a) Per-protein zero-fraction across 1326 proteins;  "
                 "blue = kept, grey = dropped", fontsize=12, loc="left")
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    ax = axes[1]
    n_neg = int((y_all == 0).sum()); n_pos = int((y_all == 1).sum())
    ax.hist([per_sample_zf[y_all == 0], per_sample_zf[y_all == 1]],
            bins=20, color=[NEG, POS], stacked=True,
            edgecolor="white", linewidth=0.4,
            label=[f"non-healer (n={n_neg})", f"healer (n={n_pos})"])
    ax.axvline(PER_SAMPLE_ZERO_LOW, color="#555", linestyle="--", linewidth=1)
    ax.axvline(PER_SAMPLE_ZERO_HIGH, color="#555", linestyle="--", linewidth=1)
    ax.set_xlabel("Per-sample zero-fraction", fontsize=11)
    ax.set_ylabel("Number of samples", fontsize=11)
    ax.set_title("(b) Per-sample zero-fraction across 77 samples;  "
                 f"QC band = [{PER_SAMPLE_ZERO_LOW:.2f}, "
                 f"{PER_SAMPLE_ZERO_HIGH:.2f}]", fontsize=12, loc="left")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    out = FIG / "figS2_qc_distributions.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
