"""Table 1, Tables S1-S2, Figure S3 -- differential expression analysis.

Two complementary tests:

  (1) Longitudinal (all 71 QC-passed samples).
        - Mann-Whitney U on per-patient mean intensities (one observation
          per patient, exact under the null, immune to within-patient
          repetition).
        - Mixed-effects model log2(intensity+1) ~ class + (1 | patient)
          for proteins where the random-intercept variance does not
          collapse to the boundary.
        - Both q-values (Benjamini-Hochberg over the 449-protein pool)
          are written to Table S1.

  (2) Visit-1 only (13 baseline samples).
        - Welch t-test on log2(intensity+1).
        - Mann-Whitney and a 10 000-iteration label-permutation test on
          log2 fold-change as robustness checks.
        - Written to Table S2.

Table 1 = the union of proteins reaching FDR <= 0.05 in either analysis.
Figure S3 = volcano plots (longitudinal MWU and baseline Welch).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf

from lib.data_utils import load_longitudinal
from lib.panels import POS, NEG

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)
OUT = HERE / "output"
OUT.mkdir(parents=True, exist_ok=True)


def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    valid = ~np.isnan(p)
    q = np.full_like(p, np.nan)
    if valid.sum() == 0:
        return q
    _, q_v, _, _ = multipletests(p[valid], method="fdr_bh")
    q[valid] = q_v
    return q


def log2fc_healer_vs_nonhealer(intensity_row, labels):
    """log2((median[healer] + 1) / (median[non-healer] + 1)) on raw intensity."""
    a = intensity_row[labels == 1]
    b = intensity_row[labels == 0]
    return float(np.log2((np.median(a) + 1.0) / (np.median(b) + 1.0)))


def main():
    m, y, g, vis, wk, names, info = load_longitudinal(
        apply_per_protein_qc=True, apply_per_sample_qc=True)
    n_prot = m.shape[0]

    # ---- Longitudinal MWU on per-patient means ----
    pats = sorted({int(p) for p in g})
    pat_means = np.zeros((n_prot, len(pats)))
    pat_class = np.zeros(len(pats), dtype=int)
    for j, p in enumerate(pats):
        mask = g == p
        pat_means[:, j] = m[:, mask].mean(axis=1)
        pat_class[j] = int(y[mask][0])

    mw_p_pm = np.full(n_prot, np.nan)
    for i in range(n_prot):
        a = pat_means[i, pat_class == 1]; b = pat_means[i, pat_class == 0]
        if len(a) > 0 and len(b) > 0:
            try:
                mw_p_pm[i] = mannwhitneyu(a, b, alternative="two-sided").pvalue
            except ValueError:
                pass
    mw_q_pm = bh_fdr(mw_p_pm)

    # naive sample-level MWU (reported only as reference; FDR inflated)
    mw_p_naive = np.full(n_prot, np.nan)
    for i in range(n_prot):
        a = m[i, y == 1]; b = m[i, y == 0]
        try:
            mw_p_naive[i] = mannwhitneyu(a, b, alternative="two-sided").pvalue
        except ValueError:
            pass
    mw_q_naive = bh_fdr(mw_p_naive)

    # mixed-effects model, log2(x+1) ~ class + (1 | patient)
    mixed_p = np.full(n_prot, np.nan)
    mixed_status = np.full(n_prot, "boundary", dtype=object)
    df_long_base = pd.DataFrame({"class": y.astype(int),
                                 "patient": g.astype(int)})
    for i in range(n_prot):
        df_long_base["v"] = np.log2(m[i] + 1.0)
        try:
            md = smf.mixedlm("v ~ Q('class')", df_long_base,
                             groups=df_long_base["patient"]).fit(
                disp=False, method="lbfgs")
            if md.cov_re.values[0, 0] > 1e-8:
                mixed_p[i] = md.pvalues.get("Q('class')", np.nan)
                mixed_status[i] = "ok"
        except Exception:
            mixed_status[i] = "error"
    mixed_q = bh_fdr(mixed_p)

    log2fc_long = np.array([log2fc_healer_vs_nonhealer(m[i], y)
                            for i in range(n_prot)])

    table_s1 = pd.DataFrame({
        "protein": names,
        "log2FC_healer_vs_nonhealer": log2fc_long,
        "mw_p_naive": mw_p_naive, "mw_q_naive": mw_q_naive,
        "mw_p_patient_mean": mw_p_pm, "mw_q_patient_mean": mw_q_pm,
        "mixedlm_p": mixed_p, "mixedlm_q": mixed_q,
        "mixedlm_status": mixed_status,
    }).sort_values("mw_q_patient_mean").reset_index(drop=True)
    table_s1.to_csv(OUT / "tableS1_de_longitudinal_all.csv", index=False)
    print(f"Saved {OUT/'tableS1_de_longitudinal_all.csv'}")

    # ---- Visit-1 only Welch + MWU + permutation ----
    base = vis == 1
    mb = m[:, base]; yb = y[base]
    log2fc_base = np.array([log2fc_healer_vs_nonhealer(mb[i], yb)
                            for i in range(n_prot)])
    welch_p = np.full(n_prot, np.nan); mw_p_b = np.full(n_prot, np.nan)
    for i in range(n_prot):
        a = np.log2(mb[i, yb == 1] + 1.0); b = np.log2(mb[i, yb == 0] + 1.0)
        if len(a) >= 2 and len(b) >= 2:
            welch_p[i] = ttest_ind(a, b, equal_var=False).pvalue
            try:
                mw_p_b[i] = mannwhitneyu(a, b, alternative="two-sided").pvalue
            except ValueError:
                pass
    welch_q = bh_fdr(welch_p); mw_q_b = bh_fdr(mw_p_b)

    # 10 000-iter label-permutation test on |log2FC| (Methods).
    # Vectorised across proteins for speed; expect ~30 s on this cohort.
    rng = np.random.default_rng(0)
    n_perm = 10000
    obs_lfc = np.abs(log2fc_base)
    perm_p = np.zeros(n_prot)
    log_intens = np.log2(mb + 1.0)
    for k in range(n_perm):
        yp = rng.permutation(yb)
        a = mb[:, yp == 1]; b = mb[:, yp == 0]
        med_a = np.median(a, axis=1); med_b = np.median(b, axis=1)
        lfc_p = np.abs(np.log2((med_a + 1.0) / (med_b + 1.0)))
        perm_p += (lfc_p >= obs_lfc)
    perm_p = (perm_p + 1.0) / (n_perm + 1.0)
    perm_q = bh_fdr(perm_p)

    table_s2 = pd.DataFrame({
        "protein": names,
        "log2FC_healer_vs_nonhealer": log2fc_base,
        "welch_p": welch_p, "welch_q": welch_q,
        "mw_p": mw_p_b, "mw_q": mw_q_b,
        "perm_p": perm_p, "perm_q": perm_q,
    }).sort_values("welch_q").reset_index(drop=True)
    table_s2.to_csv(OUT / "tableS2_de_baseline_all.csv", index=False)
    print(f"Saved {OUT/'tableS2_de_baseline_all.csv'}")

    # ---- Table 1: union of FDR <= 0.05 ----
    sig_long = set(table_s1[table_s1["mw_q_patient_mean"] <= 0.05]["protein"])
    sig_base = set(table_s2[table_s2["welch_q"] <= 0.05]["protein"])
    sig_union = sorted(sig_long | sig_base)
    rows = []
    for p in sig_union:
        r1 = table_s1[table_s1["protein"] == p].iloc[0]
        r2 = table_s2[table_s2["protein"] == p].iloc[0]
        rows.append({
            "protein": p,
            "log2 fold-change, longitudinal":
                f"{float(r1['log2FC_healer_vs_nonhealer']):+.2f}",
            "log2 fold-change, visit-1 only":
                f"{float(r2['log2FC_healer_vs_nonhealer']):+.2f}",
            "q-value, longitudinal": f"{float(r1['mw_q_patient_mean']):.3g}",
            "q-value, visit-1 only": f"{float(r2['welch_q']):.3g}",
        })
    rows.sort(key=lambda r: float(r["q-value, longitudinal"]))
    table1 = pd.DataFrame(rows)
    table1.to_csv(OUT / "table1_differentially_expressed_proteins.csv",
                  index=False)
    print(f"Saved {OUT/'table1_differentially_expressed_proteins.csv'}")
    print()
    print(table1.to_string(index=False))

    # ---- Figure S3: volcano plots ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    qq = table_s1["mw_q_patient_mean"].to_numpy()
    lfc = table_s1["log2FC_healer_vs_nonhealer"].to_numpy()
    sig = qq <= 0.05
    ax.scatter(lfc[~sig], -np.log10(qq[~sig]), c="#bbb", s=20, alpha=0.7,
               edgecolor="none")
    ax.scatter(lfc[sig], -np.log10(qq[sig]), c="#b2182b", s=40, alpha=0.95,
               edgecolor="black", linewidth=0.4)
    for i in np.where(sig)[0]:
        ax.annotate(table_s1["protein"].iloc[i], (lfc[i], -np.log10(qq[i])),
                    fontsize=9, xytext=(4, 0), textcoords="offset points")
    ax.axhline(-np.log10(0.05), color="#555", linestyle=":", linewidth=0.8)
    ax.axvline(0, color="#aaa", linewidth=0.5)
    ax.set_xlabel("log2 fold-change (healer - non-healer)", fontsize=11)
    ax.set_ylabel("-log10(q)", fontsize=11)
    ax.set_title("(a) Longitudinal MWU on per-patient means", fontsize=12,
                 loc="left")
    ax.grid(True, linestyle="--", alpha=0.3)

    ax = axes[1]
    qq = table_s2["welch_q"].to_numpy()
    lfc = table_s2["log2FC_healer_vs_nonhealer"].to_numpy()
    sig = qq <= 0.05
    ax.scatter(lfc[~sig], -np.log10(qq[~sig]), c="#bbb", s=20, alpha=0.7,
               edgecolor="none")
    ax.scatter(lfc[sig], -np.log10(qq[sig]), c="#b2182b", s=40, alpha=0.95,
               edgecolor="black", linewidth=0.4)
    for i in np.where(sig)[0]:
        ax.annotate(table_s2["protein"].iloc[i], (lfc[i], -np.log10(qq[i])),
                    fontsize=9, xytext=(4, 0), textcoords="offset points")
    ax.axhline(-np.log10(0.05), color="#555", linestyle=":", linewidth=0.8)
    ax.axvline(0, color="#aaa", linewidth=0.5)
    ax.set_xlabel("log2 fold-change (healer - non-healer)", fontsize=11)
    ax.set_ylabel("-log10(q)", fontsize=11)
    ax.set_title("(b) Visit-1 only Welch t-test", fontsize=12, loc="left")
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    out = FIG / "figS3_volcano_plots.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
