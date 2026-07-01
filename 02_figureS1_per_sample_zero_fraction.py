"""Figure S1 -- per-sample zero-fraction vs week of collection.

Each dot is one of the 77 samples. Dashed lines mark the [0.10, 0.80]
QC band; samples outside the band are labelled with their patient and
visit indices and are dropped from all downstream analyses.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from lib.data_utils import (load_longitudinal,
                            PER_SAMPLE_ZERO_LOW, PER_SAMPLE_ZERO_HIGH)
from lib.panels import POS, NEG

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def main():
    m, y_all, g_all, v_all, w_all, _, info = load_longitudinal(
        apply_per_protein_qc=False, apply_per_sample_qc=False)
    zf = info["per_sample_zf_all"]
    keep = ((zf >= PER_SAMPLE_ZERO_LOW) & (zf <= PER_SAMPLE_ZERO_HIGH))

    fig, ax = plt.subplots(figsize=(9, 5))
    for cls, col, lbl in [(0, NEG, "non-healer"), (1, POS, "healer")]:
        mask = y_all == cls
        ax.scatter(w_all[mask], zf[mask], c=col, s=40, alpha=0.85,
                   edgecolors="black", linewidths=0.4, label=lbl)
    ax.axhline(PER_SAMPLE_ZERO_LOW, color="#555", linestyle="--", linewidth=1,
               label=f"zf = {PER_SAMPLE_ZERO_LOW:.2f}")
    ax.axhline(PER_SAMPLE_ZERO_HIGH, color="#555", linestyle=":", linewidth=1,
               label=f"zf = {PER_SAMPLE_ZERO_HIGH:.2f}")
    for i in range(len(y_all)):
        if not keep[i]:
            ax.annotate(f"P{int(g_all[i])} v{int(v_all[i])}",
                        (w_all[i], zf[i]), fontsize=7, color="black",
                        xytext=(5, 0), textcoords="offset points", va="center")
    ax.set_xlabel("Week of sample collection", fontsize=10)
    ax.set_ylabel("Per-sample zero-fraction", fontsize=10)
    ax.set_title("Figure S1. Per-sample zero-fraction vs week.  "
                 "QC-dropped samples labelled.", fontsize=11, loc="left")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    out = FIG / "figS1_per_sample_zero_fraction.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
