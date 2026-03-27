# Punjab Paper Results Package

## Recommended reporting statement

The Punjab real-data inversion is reportable at this stage as a Phase 1 baseline plus external-prior ablations. The expanded grouped-support baseline is the reference result. W3RA and GRACE were integrated successfully, but the formulations tested so far did not materially improve the main deformation-fit metrics.

## Main result to report

- Phase 1 baseline `val_forward = 1.4286398`
- Phase 1 baseline normalized forward RMSE mean `= 0.9919191`

## What to show in the paper

- Summary table: `/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_paper_results_summary.csv`
- Comparison figure: `/home/ubuntu/work/punjab/outputs/punjab_prior/figures/punjab_paper_results_comparison.png`
- Baseline residual gallery: `/home/ubuntu/work/punjab/outputs/punjab_prior/figures/punjab_phase1_pilot_residuals_grouped_support_expanded.png`
- GRACE regional series: `/home/ubuntu/work/punjab/outputs/punjab_prior/figures/punjab_phase2_grace_region_timeseries.png`
- W3RA regional series: `/home/ubuntu/work/punjab/outputs/punjab_prior/figures/punjab_phase2_w3ra_basin_timeseries.png`

## Interpretation

- The expanded grouped-support baseline is the best Punjab result so far.
- W3RA and GRACE hooks are valid ablations but did not materially change the main inversion metrics.
- The paper should present these as useful negative or near-null external-prior results, not as improvements.