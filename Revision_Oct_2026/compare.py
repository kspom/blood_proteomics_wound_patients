"""Compare the tables of the previous revision with a re-run of its code.

Reads the tables of the previous revision (previous_revision/) and the outputs
of the code of the previous revision (GitHub) re-run in a new Python environment
(rerun/). Only tables of the previous revision that the code produces are compared.
Writes comparison tables to comparison/. Run from this folder:
    python compare.py
"""

from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
PREV = HERE / "previous_revision"
NEW = HERE / "rerun"
OUT = HERE / "comparison"
OUT.mkdir(exist_ok=True)


def sub(name):
    return pd.read_csv(PREV / name)


def new(name):
    return pd.read_csv(NEW / name)


def same_content(a, b):
    """True if the two tables hold the same rows, in any order."""
    if a.shape != b.shape or list(a.columns) != list(b.columns):
        return False
    key = list(a.columns)
    return (a.sort_values(key).reset_index(drop=True)
            .equals(b.sort_values(key).reset_index(drop=True)))


# 1. table-level comparison: every table of the previous revision that the code produces
rows = []
for p in sorted(PREV.glob("*.csv")):
    q = NEW / p.name
    if not q.exists():
        continue
    if p.read_bytes() == q.read_bytes():
        status = "identical"
    elif same_content(pd.read_csv(p), pd.read_csv(q)):
        status = "same content, different row order"
    else:
        status = "different"
    rows.append({"table": p.name, "status": status})
pd.DataFrame(rows).to_csv(OUT / "01_table_level.csv", index=False)

# 2. top-10 LOPPO panels side by side (Table 2)
a = sub("table2_top10_panels_LOPPO.csv")
b = new("table2_top10_panels_LOPPO.csv")
top = pd.DataFrame({
    "rank": range(1, 11),
    "previous_panel": a["proteins"].head(10).values,
    "previous_cv_error": a["cv_error"].head(10).values,
    "rerun_panel": b["proteins"].head(10).values,
    "rerun_cv_error": b["cv_error"].head(10).values,
})
top["same_panel"] = top["previous_panel"] == top["rerun_panel"]
top.to_csv(OUT / "02_top10_panels_LOPPO.csv", index=False)

# 3. zero-error panels under both search schemes (Tables S5, S6)
rows = []
for scheme, f in [("all-68-LOPPO", "tableS5_unique_sets_LOPPO.csv"),
                  ("all-68-LOOCV", "tableS6_unique_sets_LOOCV.csv")]:
    a, b = sub(f), new(f)
    za = set(a.loc[a["cv_error"] == 0, "proteins"])
    zb = set(b.loc[b["cv_error"] == 0, "proteins"])
    for p in sorted(za | zb):
        rows.append({"scheme": scheme, "panel": p,
                     "zero_error_previous": p in za, "zero_error_rerun": p in zb})
pd.DataFrame(rows).to_csv(OUT / "03_zero_error_panels.csv", index=False)

# 4. LDA coefficients of the four named panels (Table S7)
a = sub("tableS7_panel_coefficients_long.csv")
b = new("tableS7_panel_coefficients_long.csv")
coef = a.merge(b, on=["panel", "protein"], suffixes=("_previous", "_rerun"))
coef["ratio_rerun_to_previous"] = (coef["coefficient_rerun"]
                                    / coef["coefficient_previous"]).round(4)
coef["same_sign"] = (coef["coefficient_previous"] * coef["coefficient_rerun"]) > 0
coef.to_csv(OUT / "04_panel_coefficients.csv", index=False)

# 5. Table 4 (panel performance under seven CV schemes)
a = sub("table4_panel_cv_evaluation.csv").set_index("panel")
b = new("table4_panel_cv_evaluation.csv").set_index("panel")
t4 = [{"panel": panel, "scheme": scheme,
       "previous": a.at[panel, scheme], "rerun": b.at[panel, scheme],
       "same": a.at[panel, scheme] == b.at[panel, scheme]}
      for panel in a.index for scheme in a.columns]
pd.DataFrame(t4).to_csv(OUT / "05_table4_cv_schemes.csv", index=False)

# 6. summary counts, all derived from tables of the previous revision
def n_seeds(df):
    return int(df["seed_rank"].max())


def n_sets(df):
    return len(df), int((df["cv_error"] == 0).sum())


s4a, s4b = sub("tableS4_all_branches_summary.csv"), new("tableS4_all_branches_summary.csv")
s5a, s5b = n_sets(sub("tableS5_unique_sets_LOPPO.csv")), n_sets(new("tableS5_unique_sets_LOPPO.csv"))
s6a, s6b = n_sets(sub("tableS6_unique_sets_LOOCV.csv")), n_sets(new("tableS6_unique_sets_LOOCV.csv"))
summary = pd.DataFrame([
    ("seed pairs (train error <= 5), Table S4", n_seeds(s4a), n_seeds(s4b), ""),
    ("unique panels, LOPPO search, Table S5", s5a[0], s5b[0], ""),
    ("zero-error panels, LOPPO search, Table S5", s5a[1], s5b[1], ""),
    ("unique panels, LOOCV search, Table S6", s6a[0], s6b[0], ""),
    ("zero-error panels, LOOCV search, Table S6", s6a[1], s6b[1], ""),
    ("identical panels in top-10 (same rank), Table 2", "", "",
     f"{int(top['same_panel'].sum())} of 10"),
    ("coefficient ratio rerun / previous, Table S7", "", "",
     f"{coef['ratio_rerun_to_previous'].min()}-{coef['ratio_rerun_to_previous'].max()}"),
], columns=["quantity", "previous", "rerun", "difference_between_runs"])
summary.to_csv(OUT / "00_summary.csv", index=False)
print(summary.to_string(index=False))
