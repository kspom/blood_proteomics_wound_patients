"""Figure 4 -- per-patient trajectories of six panel-contributing proteins,
selected to cover all four comparison panels.

A 2 x 3 grid, one subplot per protein. Each subplot shows the protein
intensity (log2(x + 1)) on the y-axis vs week of sample collection, one
line per patient, coloured by class. The figure illustrates that single
panel-contributing proteins are individually noisy -- substantial
within-patient variability and considerable between-class overlap -- in
contrast to the stable composite Protein Healing Score trajectories
(Figure 5).

The six proteins cover all four comparison panels:
    CFB       Complement-5, Core-3, Clinical-3 (only DE-significant member)
    PI16      Complement-5, Core-3 (ECM axis)
    IGHV3-64  Complement-5, Core-3, Metabolic-4 (B-cell clonotype)
    ADIPOQ    Metabolic-4 (anti-inflammatory adipokine, alternative axis)
    F2        Clinical-3 (prothrombin, clinically measurable)
    MINPP1    Complement-5 (multivariate-only contributor)
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from lib.data_utils import load_longitudinal
from lib.panels import POS, NEG

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)

PROTEINS = ["CFB", "PI16", "IGHV3-64", "ADIPOQ", "F2", "MINPP1"]


def plot_protein_traj(ax, values, classes, patients, weeks, title):
    ax.axhline(np.median(values), color="#666", linestyle=":",
               linewidth=0.8, zorder=1, alpha=0.6)
    for p in sorted(np.unique(patients)):
        mask = patients == p
        cls = int(classes[mask][0])
        col = POS if cls == 1 else NEG
        order = np.argsort(weeks[mask])
        ax.plot(weeks[mask][order], values[mask][order], "-o", color=col,
                linewidth=1.4, markersize=5.5, alpha=0.85,
                markerfacecolor=col, markeredgecolor="white",
                markeredgewidth=0.5)
    ax.set_xlabel("Week", fontsize=18)
    ax.set_ylabel("log2(intensity + 1)", fontsize=18)
    ax.set_title(title, fontsize=22, fontweight="bold", loc="left")
    ax.tick_params(axis="both", labelsize=15)
    ax.grid(True, linestyle="--", alpha=0.3)


def main():
    m, y, g, vis, wk, names, info = load_longitudinal(
        apply_per_protein_qc=False, apply_per_sample_qc=True)
    name_to_idx = {n: i for i, n in enumerate(names)}
    missing = [p for p in PROTEINS if p not in name_to_idx]
    if missing:
        raise SystemExit(f"Proteins not found: {missing}")

    fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharex=True)
    for ax, prot in zip(axes.flat, PROTEINS):
        idx = name_to_idx[prot]
        vals = np.log2(m[idx] + 1.0)
        plot_protein_traj(ax, vals, y, g, wk, prot)

    legend_h = [
        Line2D([0], [0], marker="o", color=POS, markerfacecolor=POS,
               markeredgecolor="white", markersize=10, linewidth=1.6,
               label="healer"),
        Line2D([0], [0], marker="o", color=NEG, markerfacecolor=NEG,
               markeredgecolor="white", markersize=10, linewidth=1.6,
               label="non-healer"),
    ]
    adipoq_idx = PROTEINS.index("ADIPOQ")
    adipoq_ax = axes.flat[adipoq_idx]
    adipoq_ax.legend(handles=legend_h, loc="upper right", fontsize=14,
                     framealpha=0.95)

    fig.tight_layout()
    out = FIG / "fig4_panel_protein_trajectories.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
