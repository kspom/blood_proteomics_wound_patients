# Reproduction code for *"A Three-to-Five-Protein Healing Score Stratifies Diabetic Foot Ulcer Outcomes from Blood Proteomics"*

This folder contains the Python scripts that reproduce every figure and
table of the paper, in the order they appear. All scripts read from
`dataset/` and write to `figures/` (PDF) or `output/` (CSV).

## Folder layout

```
paper_code/
├── README.md
├── run_all.sh
├── dataset/                        # input data
│   ├── Data.csv                    # 1326 x 77 protein intensity matrix
│   ├── ClassLabel.csv              # 77 rows: class | patient_id | visit | week | included
│   ├── ProteinNames.txt            # 1326 protein names, one per line
│   └── filtered_proteins.csv       # 449 protein names kept after per-protein QC
├── lib/                            # shared modules
│   ├── data_utils.py               # central data loader; applies QC + cohort mask
│   ├── panels.py                   # four comparison panels and class colours
│   ├── classifier_search.py        # LDA + LOPPO/LOOCV machinery
│   ├── branching.py                # greedy forward branching search
│   └── search_log.py               # search-log dataclass
└── 01_*.py … 27_*.py               # numbered scripts in paper order
```

The `figures/` and `output/` directories are created automatically when
scripts are run.

## How to run

Each script is self-contained and can be run from this folder:

```bash
cd paper_code
python 01_figure1_cohort_overview.py
python 02_figureS1_per_sample_zero_fraction.py
...
python 26_lopo_metrics.py
```

Or run the whole pipeline at once:

```bash
bash run_all.sh
```

Prerequisites: Python 3.9+ with `numpy`, `pandas`, `scipy`,
`scikit-learn`, `statsmodels`, `matplotlib`.

## Cohort

Of the 18 patients originally enrolled, two non-healer patients (P13 and
P14) were excluded post hoc due to wound-site serious adverse events
unrelated to the natural-history outcome being modelled. The analytical
cohort therefore comprises **16 patients (7 healers + 9 non-healers)**
with 73 mass-spectrometry samples, of which **68 passed quality
control** (42 from non-healers, 26 from healers; 12 baseline samples).
The exclusion is implemented via the `included` column in
`dataset/ClassLabel.csv` and applied automatically by the data loader.

## Running order and dependencies

Scripts depend on each other via the CSV files they write to `output/`.
Running them in numeric order satisfies all dependencies. Key edges:

- **04** must run before **17** (uses Tables S1, S2)
- **09** must run before **17, 18, 19, 20, 22, 23** (uses Tables S5, top-10, panel coefficients)
- **14** must run before **21** (re-uses CV functions)
- **25** must run before **26** (LOPO metrics consume LOPO search output)
- **12b** must run before **12** (Figure S6 reads addition histories)

**Slow scripts** (expect 5-15 min each):
- `09_table2_panel_search_loppo.py` (greedy branching from 45 seed pairs)
- `10_tableS6_panel_search_loocv.py` (same with LOOCV)
- `12b_top5_branch_histories.py` (re-runs LOPPO with full step-by-step logging)
- `25_lopo_sensitivity.py` (re-runs the full pipeline 16 times for the leave-one-patient-out sensitivity analysis; ~30-60 min)

All other scripts run in under a minute.

## What each script produces

### Shared modules (`lib/`)

| File | Purpose |
|---|---|
| `lib/data_utils.py` | Central data loader. Reads `dataset/Data.csv`, `ClassLabel.csv`, `ProteinNames.txt`, `filtered_proteins.csv`. Applies the SAE-exclusion mask (`included` column), the per-protein zero-fraction filter (<=25% zero samples) and the per-sample zero-fraction band ([0.10, 0.80]). Returns the QC-passed 449 x 68 matrix plus per-sample metadata. |
| `lib/panels.py` | Defines the four comparison panels (Complement-5, Metabolic-4, Core-3, Clinical-3) and the red/blue class colours. |
| `lib/classifier_search.py` | LDA training- and CV-error helpers. Implements patient-pair LOPPO (one healer + one non-healer held out per fold) and sample-level LOOCV. Provides `scan_all_pairs()` and `find_best_addition()`. |
| `lib/branching.py` | Greedy forward-branching search built on `classifier_search.py`. Takes seed pairs, returns the multi-branch family of panels reaching zero-error, no-improvement, or max-iterations. |
| `lib/search_log.py` | Dataclass for the audit log of every protein set evaluated during the search. |

### Pipeline scripts (in paper order)

| # | Script | Produces | Paper element |
|---|---|---|---|
| 01 | `01_figure1_cohort_overview.py` | `figures/fig1_cohort_overview.pdf`, `output/baseline_coverage.csv` | **Figure 1** (cohort design, samples per visit/week) |
| 02 | `02_figureS1_per_sample_zero_fraction.py` | `figS1_per_sample_zero_fraction.pdf` | **Figure S1** (per-sample zero-fraction vs week, QC band) |
| 03 | `03_figureS2_qc_distributions.py` | `figS2_qc_distributions.pdf` | **Figure S2** (per-protein and per-sample zero-fraction distributions) |
| 04 | `04_table1_differential_expression.py` | `table1_differentially_expressed_proteins.csv`, `tableS1_de_longitudinal_all.csv`, `tableS2_de_baseline_all.csv`, `figS3_volcano_plots.pdf` | **Table 1** (DE proteins), **Tables S1, S2**, **Figure S3** |
| 05 | `05_figure2_panel_protein_trajectories.py` | `fig4_panel_protein_trajectories.pdf` | **Figure 4** (per-patient time series of 6 panel-contributing proteins) |
| 06 | `06_figure3_seed_pairs.py` | `fig2_seed_pairs.pdf` | **Figure 2** (representative seed pairs) |
| 07 | `07_figureS4_seed_pair_error_histogram.py` | `figS4_seed_pair_error_histogram.pdf` | **Figure S4** (LDA training-error histogram across pairs; train_error <= 5 cutoff = 45 seed pairs) |
| 08 | `08_figureS5_top20_seed_pair_scatter.py` | `figS5_top20_seed_pairs.pdf` | **Figure S5** (top-20 seed pairs) |
| 09 | `09_table2_panel_search_loppo.py` | `loppo_pair_scan.csv`, `tableS4_all_branches_summary.csv`, `tableS5_unique_sets_LOPPO.csv`, `table2_top10_panels_LOPPO.csv` | **Table 2** (top-10 LOPPO panels), **Tables S4, S5**. **Slow.** |
| 10 | `10_tableS6_panel_search_loocv.py` | `loocv_pair_scan.csv`, `tableS6_unique_sets_LOOCV.csv` | **Table S6** (unique panels under all-68-LOOCV). **Slow.** |
| 11 | `11_figure4_pairwise1_boxplots.py` | `fig3_pairwise1_boxplots.pdf` | **Figure 3** (per-protein and combined-PHS boxplots for Complement-5) |
| 12b | `12b_top5_branch_histories.py` | `top5_branch_histories.csv` | step-by-step addition history for top-5 LOPPO panels (input to script 12) |
| 12 | `12_figureS6_performance_trajectory.py` | `figS6_performance_trajectory.pdf` | **Figure S6** (training and CV error trajectory during branching, top-5 size-5 panels) |
| 13 | `13_table3_panel_roster.py` | `table3_panel_roster.csv` | **Table 3** (the four comparison panels with selection criteria) |
| 14 | `14_table4_panel_cv_evaluation.py` | `table4_panel_cv_evaluation.csv` | **Table 4** (each panel's misclassification rate under seven CV schemes) |
| 15 | `15_figure5_phs_trajectories.py` | `fig5_phs_trajectories.pdf` | **Figure 5** (per-patient PHS trajectories for the four panels) |
| 16 | `16_figureS7_per_patient_range.py` | `figS7_per_patient_range.pdf` | **Figure S7** (per-patient PHS range under baseline-fit) |
| 17 | `17_table5_recurrent_proteins.py` | `table5_recurrent_proteins.csv` | **Table 5** (recurrent panel-contributing proteins) |
| 18 | `18_tableS8_S9_protein_frequency.py` | `tableS8_protein_freq_seed_pairs.csv`, `tableS9_protein_freq_top10_panels.csv` | **Tables S8, S9** (protein occurrence frequencies) |
| 19 | `19_tableS10_axis_partial_core3.py` | `tableS10_axis_only_one_core3_panels.csv` | **Table S10** (panels containing exactly one of CFB / IGHV3-64 / PI16) |
| 20 | `20_tableS11_axis_zero_core3.py` | `tableS11_axis_zero_core3_panels.csv` | **Table S11** (panels containing none of the Core-3 proteins) |
| 21 | `21_tableS12_S13_clinical3_summary.py` | `tableS12_clinical3_coefficients.csv`, `tableS13_clinical3_cv_errors.csv` | **Tables S12, S13** (Clinical-3 coefficients and CV errors) |
| 22 | `22_tableS7_panel_coefficients.py` | `tableS7_panel_coefficients_long.csv` | **Table S7** (LDA coefficients for all four panels) |
| 23 | `23_figureS8_tableS14_phs_decomposition.py` | `figS8_phs_decomposition.pdf`, `tableS14_phs_contribution.csv` | **Figure S8, Table S14** (per-panel decomposition of PHS into positive/negative sub-sums) |
| 24 | `24_tableS15_panel_protein_univariate.py` | `tableS15_panel_protein_univariate_classification.csv` | **Table S15** (univariate vs multivariate role of each panel protein) |
| 25 | `25_lopo_sensitivity.py` | `lopo_all_pair_scans.csv`, `lopo_all_unique_sets.csv`, `lopo_top5_per_run.csv` | leave-one-patient-out (LOPO) re-search; produces inputs for Tables S16-S19. **Slow.** |
| 26 | `26_lopo_metrics.py` | `tableS16_lopo_protein_frequency.csv`, `tableS17_lopo_robust_seed_pairs.csv`, `tableS18_lopo_protein_hub_centrality.csv`, `tableS19_lopo_frequent_subsets.csv` | **Tables S16-S19** (LOPO-sensitivity metrics: per-protein frequency, robust seed pairs, hub centrality, frequent subsets) |

## Notes

- **Cohort numbers.** With the default QC pipeline (SAE-exclusion + per-protein + per-sample filters), the loader returns a 449 x 68 matrix (449 proteins after per-protein zero-fraction filter; 68 of 73 samples after per-sample QC), with class 0 / class 1 = 42 / 26 samples and 16 patients (9 non-healers + 7 healers). 12 baseline samples (6 + 6).

- **Reproducibility.** All randomised steps fix a NumPy generator seed (`rng = np.random.default_rng(0)`). Re-running on the same data reproduces the same numbers.

- **`apply_per_protein_qc` flag.** Differential expression and panel search use the 449-protein filtered pool (`apply_per_protein_qc=True`). Score computations on the four named panels (Figures 3, 4, 5, S7, Tables 4, S7, S12, S13) call `load_longitudinal(apply_per_protein_qc=False, apply_per_sample_qc=True)` since they only need specific known proteins.

- **Working directory.** Run scripts from `paper_code/`. They use `from lib.data_utils import ...` and `Path(__file__).resolve().parent` for outputs.

- **No external dataset paths.** This pipeline reads only `dataset/`. No code reads from any other location in the repository.
