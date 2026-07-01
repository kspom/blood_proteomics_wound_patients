"""Figure S8 and Table S14 -- PHS axis decomposition and per-protein
contribution.

For each of the 4 comparison panels we split the Protein Healing Score
into two sub-sums:

  PHS  =  pos_subsum  +  neg_subsum  +  intercept

where pos_subsum collects proteins with positive LDA coefficients and
neg_subsum collects proteins with negative LDA coefficients. We plot
(pos_subsum, |neg_subsum|) per sample, colored by class, with the
classification boundary drawn (PHS = 0).

For each protein we also compute its per-protein contribution
(axis coefficient * mean intensity) and express it as a percentage of
its axis total.

Outputs:
  figures/figS8_phs_decomposition.pdf
  output/tableS14_phs_contribution.csv
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from lib.data_utils import load_longitudinal
from lib.panels import PANELS, POS, NEG

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)
OUT = HERE / "output"
OUT.mkdir(parents=True, exist_ok=True)


def scale_factor(coefs):
    a = [abs(c) for c in coefs if c != 0]
    if not a:
        return 1.0
    return 10.0 ** (-math.floor(math.log10(min(a))))


def axis_display(X_sub, coefs, proteins):
    """Display values, axis label, and per-protein axis coefficients."""
    n = len(proteins)
    if n == 0:
        return None, "(none)", []
    abs_coefs = np.abs(coefs)
    if n == 1:
        displayed = X_sub[:, 0]
        label = proteins[0]
        per_protein = [(proteins[0], 1.0)]
        return displayed, label, per_protein
    k = scale_factor(coefs)
    raw = X_sub @ abs_coefs
    displayed = raw * k
    terms = [f"{c * k:.2g}*{p}" for p, c in zip(proteins, abs_coefs)]
    per_protein = [(p, float(c * k)) for p, c in zip(proteins, abs_coefs)]
    return displayed, " + ".join(terms), per_protein


def plot_panel(ax, x_disp, y_disp, x_label, y_label, intercept, y, sx, sy,
               title):
    ax.scatter(x_disp[y == 0], y_disp[y == 0], c=NEG, s=160, alpha=0.85,
               edgecolors="black", linewidths=0.8)
    ax.scatter(x_disp[y == 1], y_disp[y == 1], c=POS, s=160, alpha=0.85,
               edgecolors="black", linewidths=0.8)
    if x_disp.size > 0 and x_disp.max() > 0:
        xs = np.array([x_disp.min() * 0.95, x_disp.max() * 1.05])
        ys = (sy / sx) * xs + sy * intercept
        ax.plot(xs, ys, "--", color="#333", linewidth=2.8)
    ax.set_xlabel(x_label, fontsize=30, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=30, fontweight="bold")
    ax.set_title(title, fontsize=38, loc="left", fontweight="bold")
    ax.tick_params(axis="both", labelsize=24)
    ax.grid(True, linestyle="--", alpha=0.3)


def main():
    m, y, g, vis, wk, names, info = load_longitudinal(
        apply_per_protein_qc=False, apply_per_sample_qc=True)
    name_to_idx = {n: i for i, n in enumerate(names)}

    # ---- contribution table ----
    table_rows = []
    panel_decomp = {}
    for pname, proteins in PANELS.items():
        idxs = [name_to_idx[p] for p in proteins]
        X = m[idxs].T
        clf = LinearDiscriminantAnalysis().fit(X, y)
        coefs = clf.coef_.ravel()
        intercept = float(clf.intercept_.ravel()[0])

        pos_mask = coefs > 0
        neg_mask = coefs < 0
        pos_proteins = [p for p, k in zip(proteins, pos_mask) if k]
        neg_proteins = [p for p, k in zip(proteins, neg_mask) if k]

        x_disp, x_label, pos_per_prot = axis_display(
            X[:, pos_mask], coefs[pos_mask], pos_proteins)
        y_disp, y_label, neg_per_prot = axis_display(
            X[:, neg_mask], coefs[neg_mask], neg_proteins)
        if x_disp is None:
            x_disp = np.zeros(X.shape[0])
        if y_disp is None:
            y_disp = np.zeros(X.shape[0])
        sx = 1.0 if not pos_per_prot else (
            scale_factor(coefs[pos_mask]) if len(pos_per_prot) > 1
            else 1.0 / abs(coefs[pos_mask][0]))
        sy = 1.0 if not neg_per_prot else (
            scale_factor(coefs[neg_mask]) if len(neg_per_prot) > 1
            else 1.0 / abs(coefs[neg_mask][0]))
        panel_decomp[pname] = {
            "x_disp": x_disp, "y_disp": y_disp,
            "x_label": x_label, "y_label": y_label,
            "intercept": intercept, "sx": sx, "sy": sy,
            "proteins": proteins, "n": len(proteins),
        }

        for axis_name, per_prot in [("positive (x-axis)", pos_per_prot),
                                     ("negative (y-axis)", neg_per_prot)]:
            if not per_prot:
                continue
            products = []
            for prot, axis_coef in per_prot:
                mean_val = float(m[name_to_idx[prot]].mean())
                product = axis_coef * mean_val
                products.append((prot, axis_coef, mean_val, product))
            total = sum(p for _, _, _, p in products)
            for prot, axis_coef, mean_val, product in products:
                pct = (product / total * 100.0) if total > 0 else 0.0
                table_rows.append({
                    "Panel": pname,
                    "Axis": axis_name,
                    "Protein": prot,
                    "Axis coefficient": (f"{axis_coef:.2g}"
                                         if len(per_prot) > 1
                                         else "1 (raw intensity)"),
                    "Mean intensity (across 68 samples)":
                        round(mean_val, 0),
                    "Product (coef * mean)": round(product, 0),
                    "Contribution (% of axis total)":
                        f"{pct:.1f}%",
                })

    df = pd.DataFrame(table_rows)
    out_csv = OUT / "tableS14_phs_contribution.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")
    print(df.to_string(index=False))
    print()

    # ---- figure ----
    fig, axes = plt.subplots(2, 2, figsize=(30, 22))
    for ax, (pname, d) in zip(axes.flat, panel_decomp.items()):
        plot_panel(ax, d["x_disp"], d["y_disp"],
                   d["x_label"], d["y_label"], d["intercept"], y,
                   d["sx"], d["sy"],
                   f"{pname}  (size {d['n']})")

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=NEG,
               markeredgecolor="black", markersize=22, label="non-healer"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=POS,
               markeredgecolor="black", markersize=22, label="healer"),
    ]
    fig.legend(handles=legend_handles, loc="center right",
               bbox_to_anchor=(0.995, 0.5),
               fontsize=32, framealpha=0.95)
    fig.subplots_adjust(wspace=0.30, hspace=0.30, right=0.87)
    out_pdf = FIG / "figS8_phs_decomposition.pdf"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
