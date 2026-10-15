#!/usr/bin/env bash
# Run every script in an order that satisfies all dependencies (see README.md).
# Stops on the first failure.
# Slow steps: 09, 10, 12b (5-15 min each), 25 (30-60 min),
# 29 (~1.7 h on 22 cores) and 31 (~5 min on 16 cores).

set -euo pipefail
cd "$(dirname "$0")"

scripts=(
  01_figure1_cohort_overview.py
  02_figureS1_per_sample_zero_fraction.py
  03_figureS2_qc_distributions.py
  04_table1_differential_expression.py
  05_figure4_panel_protein_trajectories.py
  06_figure2_seed_pairs.py
  07_figureS4_seed_pair_error_histogram.py
  08_figureS5_top20_seed_pair_scatter.py
  09_table2_panel_search_loppo.py
  10_tableS6_panel_search_loocv.py
  11_figure3_pairwise1_boxplots.py
  12b_top5_branch_histories.py
  12_figureS6_performance_trajectory.py
  13_table3_panel_roster.py
  14_table4_panel_cv_evaluation.py
  15_figure5_phs_trajectories.py
  16_figureS7_per_patient_range.py
  17_table5_recurrent_proteins.py
  18_tableS8_S9_protein_frequency.py
  19_tableS10_axis_partial_core3.py
  20_tableS11_axis_zero_core3.py
  21_tableS12_S13_clinical3_summary.py
  22_tableS7_panel_coefficients.py
  23_figureS8_tableS14_phs_decomposition.py
  24_tableS15_panel_protein_univariate.py
  25_lopo_sensitivity.py
  26_lopo_metrics.py
  27_tableS21_shrinkage_lda.py
  28_tableS22_nested_lopo.py
  29_tableS23_permutation_test.py
  30_figureS9_permutation_null.py
  31_nested_lopo_svm.py
  32_tableS24_benchmark_methods.py
)

for s in "${scripts[@]}"; do
  echo "=== running $s ==="
  python "$s"
done
echo "All scripts finished. Outputs in figures/ and output/."
