"""Figure 1 -- dataset overview and quality control.

Reproduces the four-panel figure that opens the Results section:
  (a) patient-by-week sampling grid: shows all 18 enrolled patients with
      three visual categories per sample:
        * analytical (passed QC and inclusion):    solid coloured circle
        * QC-dropped (failed per-sample QC):       X marker, full colour
        * SAE-excluded (patients 13 and 14):       square marker, faded
      Rows for SAE-excluded patients have a grey background band and the
      label suffix "(SAE)".
  (b) samples per visit, by class,
  (c) samples per week bin, by class.

Pre-QC and pre-inclusion samples are shown so the reader can see the
complete enrollment-to-analysis flow at a glance.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from lib.data_utils import (load_longitudinal,
                            PER_SAMPLE_ZERO_LOW, PER_SAMPLE_ZERO_HIGH)
from lib.panels import POS, NEG

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)
OUT = HERE / "output"
OUT.mkdir(parents=True, exist_ok=True)

SAE_EXCLUDED_PATIENTS = {13, 14}


def main():
    # load with ALL filters off so we see the full 77-sample design
    m, y_all, g_all, v_all, w_all, names, info = load_longitudinal(
        apply_per_protein_qc=False,
        apply_per_sample_qc=False,
        apply_inclusion_mask=False)
    keep = ((info["per_sample_zf_all"] >= PER_SAMPLE_ZERO_LOW)
            & (info["per_sample_zf_all"] <= PER_SAMPLE_ZERO_HIGH))
    sae = np.array([int(p) in SAE_EXCLUDED_PATIENTS for p in g_all])
    analytical = keep & ~sae
    zf = info["per_sample_zf_all"]
    n_total = len(y_all)
    n_analytical = int(analytical.sum())
    n_sae = int(sae.sum())
    n_qc_drop = int((~keep & ~sae).sum())

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1.0], hspace=0.35,
                          wspace=0.22)

    # ---- (a) patient x week grid ----
    ax = fig.add_subplot(gs[0, :])
    patients = sorted({int(p) for p in g_all})
    pat_to_y = {p: i for i, p in enumerate(patients)}
    pat_class = {int(p): int(y_all[g_all == p][0]) for p in patients}

    # shade SAE-excluded patient rows
    for p in patients:
        if p in SAE_EXCLUDED_PATIENTS:
            ax.axhspan(pat_to_y[p] - 0.45, pat_to_y[p] + 0.45,
                       color="#cccccc", alpha=0.45, zorder=0)

    for i in range(n_total):
        p = int(g_all[i]); cls = int(y_all[i])
        col = POS if cls == 1 else NEG
        if sae[i]:
            marker = "s"; ms = 240; alpha = 0.5
        elif keep[i]:
            marker = "o"; ms = 260; alpha = 1.0
        else:
            marker = "X"; ms = 220; alpha = 1.0
        ax.scatter(w_all[i], pat_to_y[p], c=[col], s=ms, marker=marker,
                   edgecolors="black", linewidths=1.0, alpha=alpha, zorder=3)
        ax.annotate(str(int(v_all[i])), (w_all[i], pat_to_y[p]),
                    fontsize=13, color="white", ha="center", va="center",
                    zorder=4, fontweight="bold")
    for p in patients:
        mask = g_all == p
        order = np.argsort(w_all[mask])
        line_alpha = 0.3 if p in SAE_EXCLUDED_PATIENTS else 0.6
        ax.plot(w_all[mask][order], [pat_to_y[p]] * mask.sum(),
                color="#888", linewidth=0.8, alpha=line_alpha, zorder=1)

    def _row_label(p):
        cls = pat_class[p]
        tag = "(SAE)" if p in SAE_EXCLUDED_PATIENTS else (
            "(h)" if cls == 1 else "(nh)")
        return f"P{p:<2d}  {tag}"
    ax.set_yticks(range(len(patients)))
    ax.set_yticklabels([_row_label(p) for p in patients],
                       fontsize=17, fontfamily="monospace")
    ax.tick_params(axis="x", labelsize=17)
    ax.set_xlabel("Week of sample collection", fontsize=20)
    ax.set_title(
        f"(a) Sampling design: {n_total} blood samples.  "
        f"Analyzed cohort = {n_analytical} samples "
        f"({n_sae} SAE-excluded; {n_qc_drop} QC-dropped).",
        fontsize=20, loc="left")
    ax.set_xlim(-1.5, 34)
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    ax.invert_yaxis()
    legend_h = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=NEG,
               markeredgecolor="black", markersize=16,
               label="non-healer"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=POS,
               markeredgecolor="black", markersize=16,
               label="healer"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor=NEG,
               markeredgecolor="black", markersize=16,
               label="non-healer (QC drop)"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor=POS,
               markeredgecolor="black", markersize=16,
               label="healer (QC drop)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=NEG,
               markeredgecolor="black", markersize=16, alpha=0.5,
               label="SAE-excluded"),
    ]
    ax.legend(handles=legend_h, loc="lower right", fontsize=15, ncol=1,
              framealpha=0.95)

    # ---- (b) samples per visit ----
    ax = fig.add_subplot(gs[1, 0])
    visits = sorted({int(x) for x in v_all})
    by_v0 = [int(((v_all == vi) & (y_all == 0)).sum()) for vi in visits]
    by_v1 = [int(((v_all == vi) & (y_all == 1)).sum()) for vi in visits]
    by_v0_a = [int(((v_all == vi) & (y_all == 0) & analytical).sum())
               for vi in visits]
    by_v1_a = [int(((v_all == vi) & (y_all == 1) & analytical).sum())
               for vi in visits]
    xs = np.arange(len(visits)); w = 0.36
    ax.bar(xs - w / 2, by_v0, w, color=NEG, alpha=0.35, edgecolor="black",
           linewidth=0.6, label="non-healer (enrolled)")
    ax.bar(xs + w / 2, by_v1, w, color=POS, alpha=0.35, edgecolor="black",
           linewidth=0.6, label="healer (enrolled)")
    ax.bar(xs - w / 2, by_v0_a, w, color=NEG, alpha=0.95, edgecolor="black",
           linewidth=0.6, label="non-healer (analyzed)")
    ax.bar(xs + w / 2, by_v1_a, w, color=POS, alpha=0.95, edgecolor="black",
           linewidth=0.6, label="healer (analyzed)")
    for i in range(len(visits)):
        ax.text(i - w / 2, by_v0[i] + 0.2, f"{by_v0_a[i]}/{by_v0[i]}",
                ha="center", fontsize=15)
        ax.text(i + w / 2, by_v1[i] + 0.2, f"{by_v1_a[i]}/{by_v1[i]}",
                ha="center", fontsize=15)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"v{vi}" for vi in visits], fontsize=17)
    ax.tick_params(axis="y", labelsize=17)
    ax.set_ylabel("Number of samples", fontsize=18)
    ax.set_title("(b) Samples per visit (enrolled / analyzed)",
                 fontsize=20, loc="left")
    ax.set_ylim(0, max(max(by_v0), max(by_v1)) * 1.35)
    ax.legend(fontsize=15, loc="upper right", ncol=2, columnspacing=0.9,
              handlelength=1.4, framealpha=0.95)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    # ---- (c) samples per week bin ----
    ax = fig.add_subplot(gs[1, 1])
    bins = np.arange(0, 36, 4)
    ax.hist([w_all[y_all == 0], w_all[y_all == 1]], bins=bins,
            color=[NEG, POS], edgecolor="white", linewidth=0.5,
            label=["non-healer", "healer"], stacked=True)
    ax.set_xticks(np.arange(0, 36, 4))
    ax.tick_params(axis="both", labelsize=17)
    ax.set_xlabel("Week of sample collection", fontsize=18)
    ax.set_ylabel("Number of samples", fontsize=18)
    ax.set_title("(c) Samples per week bin (stacked, all enrolled)",
                 fontsize=20, loc="left")
    ax.legend(fontsize=16)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    out1 = FIG / "fig1_cohort_overview.pdf"
    fig.savefig(out1, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out1}")

    # ---- baseline-coverage table (analytical cohort only) ----
    rows = []
    for p in patients:
        if p in SAE_EXCLUDED_PATIENTS:
            continue
        mask = g_all == p
        rows.append({
            "patient_id": p, "class": pat_class[p],
            "n_samples_total": int(mask.sum()),
            "n_samples_kept": int((mask & analytical).sum()),
            "visits_all": ";".join(str(int(v)) for v in
                                   v_all[mask][np.argsort(w_all[mask])]),
            "weeks_all": ";".join(str(int(w)) for w in sorted(w_all[mask])),
            "has_baseline": bool(((v_all == 1) & mask).any()),
            "baseline_kept": bool(((v_all == 1) & mask & analytical).any()),
        })
    pd.DataFrame(rows).to_csv(OUT / "baseline_coverage.csv", index=False)
    print(f"Saved {OUT/'baseline_coverage.csv'}")


if __name__ == "__main__":
    main()
