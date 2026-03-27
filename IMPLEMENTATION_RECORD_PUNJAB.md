# Punjab Phase Implementation Record

## Purpose
This file records the current real-data Punjab inversion phase.

## Primary notebook
- [punjab_prior_constrained_inversion.ipynb](/home/ubuntu/work/punjab/punjab_prior_constrained_inversion.ipynb)

## Phase definition
- Phase 1: generic-prior Punjab inversion baseline
- Phase 2: external-prior integration on top of the frozen Phase 1 baseline
- Active priors:
  - forward consistency
  - spatial regularization
  - temporal regularization
- Active external-prior experiments:
  - `W3RA`
  - `GRACE`
- Deferred to later integration:
  - `GPS`

## Reusable implementation base
- [punjab_inversion/physics.py](/home/ubuntu/work/punjab/punjab_inversion/physics.py)
- [punjab_inversion/metrics.py](/home/ubuntu/work/punjab/punjab_inversion/metrics.py)
- [punjab_inversion/models.py](/home/ubuntu/work/punjab/punjab_inversion/models.py)

## Current status
- The Punjab notebook exists and compiles.
- It is wired to the real deformation stack in metadata-first mode.
- The Phase 1 generic-prior baseline is now frozen around the expanded grouped-support pilot.
- The clipped W3RA reference is aligned to Punjab acquisition dates and has already been tested as a weak and stronger external prior.
- The local JPL GRACE mascon file has been checked, aligned to Punjab dates, and tested as a first weak regional prior.
- The stronger GRACE prior hook has now also been tested and stays stable, but remains a tie on the main deformation-fit metrics.
- A reformulated GRACE prior using uncertainty-weighted regional anomalies has also been tested and remains effectively a tie on the main deformation-fit metrics.
- An interactive baseline inversion viewer is now available for the six expanded grouped-support baseline tiles across the full Punjab time span.

## Confirmed data sources
- Punjab deformation root:
  - `/mnt/data/aoi_punjab`
- Clipped W3RA reference:
  - [outputs/W3RA_punjab_preclipped.nc](/home/ubuntu/work/punjab/outputs/W3RA_punjab_preclipped.nc)

## Confirmed metadata
- Punjab deformation stack:
  - shape `(225, 1130, 1850)`
- Punjab acquisition dates:
  - `2016-01-06` to `2024-01-24`
- Clipped W3RA reference:
  - span `2003-01-01` to `2024-12-01`
  - `S0`/`Sg` shape `(264, 41, 41)`

## Current model choice
- Default model for Phase 1:
  - plain dual-decoder frequency-separated model
- Noise conditioning is disabled for now because we do not yet have a real Punjab noise proxy defined.

## Current loss for Phase 1
\[
L_{phase1} =
\lambda_f L_{fwd}
+ \lambda_{s0} L_{spatial}^{S_0}
+ \lambda_{sg} L_{spatial}^{S_g}
+ \lambda_{t0} L_{temporal}^{S_0}
+ \lambda_{tg} L_{temporal}^{S_g}
\]

## Phase 1 Baseline Ladder
The current Punjab Phase 1 baselines all use the same two-layer inverse map:
\[
(\hat S_0,\hat S_g)=f_\theta(d)
\]
with deformation reconstructed as:
\[
\hat d = G_{\mathrm{load}}(\hat S_0) + G_{\mathrm{poro}}(\hat S_g).
\]

The forward term is evaluated in normalized deformation space:
\[
L_{fwd}=\mathrm{MSE}(\hat d^*, d^*),
\]
and the common generic-prior objective is:
\[
L_{phase1}
=
\lambda_f L_{fwd}
+ \lambda_{s0}L^{S_0}_{spatial}
+ \lambda_{sg}L^{S_g}_{spatial}
+ \lambda_{t0}L^{S_0}_{temporal}
+ \lambda_{tg}L^{S_g}_{temporal}.
\]

The spatial support mask shared by these runs is:
\[
M(i,j)=\mathbf{1}[\mathrm{coh}(i,j)\ge 0.20]\cdot \mathbf{1}[f_{\mathrm{valid}}(i,j)\ge 0.05].
\]

### Baseline 1: Expanded grouped-support pilot
This is the current Phase 1 reference baseline.

Sampler:
\[
\mathcal{I}_{\mathrm{expanded}}
=
\{(k,t):k\in\mathcal{K}_{6},\ t\in\mathcal{T}^{train}_{20}\},
\qquad
\mathcal{I}_{\mathrm{expanded,val}}
=
\{(k,t):k\in\mathcal{K}_{6},\ t\in\mathcal{T}^{val}_{10}\}.
\]

Meaning:
- `6` train tiles and `6` validation tiles
- `20` consecutive train windows per tile
- `10` consecutive validation windows per tile

Result:
- `val_forward = 1.4286`
- normalized forward RMSE mean `0.9919`

Reference outputs:
- [punjab_phase1_pilot_summary_grouped_support_expanded.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_pilot_summary_grouped_support_expanded.json)
- [punjab_phase1_pilot_diagnostics_grouped_support_expanded.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_pilot_diagnostics_grouped_support_expanded.json)

### Baseline 2: Fuller grouped run
This kept the same model, loss, and support mask, but scaled the grouped sampler.

Sampler:
\[
\mathcal{I}_{\mathrm{fuller}}
=
\{(k,t):k\in\mathcal{K}_{8},\ t\in\mathcal{T}^{train}_{24}\},
\qquad
\mathcal{I}_{\mathrm{fuller,val}}
=
\{(k,t):k\in\mathcal{K}_{8},\ t\in\mathcal{T}^{val}_{12}\}.
\]

Meaning:
- `8` train tiles and `8` validation tiles
- `24` consecutive train windows per tile
- `12` consecutive validation windows per tile
- `6` training epochs

Result:
- `val_forward = 2.4986`
- normalized forward RMSE mean `1.3493`

Interpretation:
- scaling grouped support breadth and duration alone degraded the run
- the expanded grouped-support pilot stayed better

Reference outputs:
- [punjab_phase1_fuller_summary.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_fuller_summary.json)
- [punjab_phase1_fuller_diagnostics.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_fuller_diagnostics.json)

### Baseline 3: Fuller warm-start fine-tune
This tested whether the fuller degradation was only an initialization problem.

Initialization:
\[
\theta_0 \leftarrow \theta_{\mathrm{expanded}}
\]
followed by:
\[
\theta_{\mathrm{finetune}}
=
\mathrm{Train}(\mathcal{I}_{\mathrm{fuller}}\mid\theta_0,\ \eta=5\times10^{-5}).
\]

Result:
- `val_forward = 2.4986`
- normalized forward RMSE mean `1.3493`

Interpretation:
- effectively identical to the scratch fuller run
- warm-starting did not recover the lost performance

Reference outputs:
- [punjab_phase1_fuller_finetune_summary.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_fuller_finetune_summary.json)
- [punjab_phase1_fuller_finetune_diagnostics.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_fuller_finetune_diagnostics.json)

### Baseline conclusion
Current ranking:
\[
\text{expanded grouped-support} \;>\; \text{fuller grouped} \approx \text{fuller warm-start}.
\]

This means the best current Punjab Phase 1 baseline is still the expanded grouped-support pilot. The next change should target loss design or reliability handling rather than simply scaling grouped sample breadth.

## Current plan link
- [STEPWISE_PLAN.md](/home/ubuntu/work/punjab/STEPWISE_PLAN.md)

## Next actions
1. Keep the expanded grouped-support sampler as the frozen Phase 1 baseline.
2. Keep the completed `W3RA` and weak `GRACE` runs as the first Phase 2 proof-of-integration results.
3. Decide whether the next external-prior step should strengthen or reformulate `GRACE`, or pivot to `GPS` preparation.
4. Keep `GPS` as the remaining deferred external prior.

## 2026-03-14 Data Wiring And Dataset Scaffold Update
- Change summary:
  - Added real Punjab data wiring in metadata-first mode to the new notebook.
  - Added coverage diagnostics based on deformation sampling, coherence, velocity, and a support mask.
  - Added chronological train/validation/test split generation for the Punjab acquisition timeline.
  - Added the first patch-based dataset scaffold for 64x64 tiles over 12-step temporal windows.
  - Exported phase-1 coverage and split artifacts to `outputs/punjab_prior/`.
- Files/cells touched:
  - `punjab_prior_constrained_inversion.ipynb` Cells 5, 7, and 9
  - `IMPLEMENTATION_RECORD_PUNJAB.md`
  - `STEPWISE_PLAN.md`
- Verification result:
  - Notebook cells executed successfully in `swin_env`.
  - Confirmed metadata:
    - deformation stack `(225, 1130, 1850)`
    - acquisition dates `2016-01-06` to `2024-01-24`
  - Coverage diagnostics:
    - sampled deformation valid fraction `0.1159`
    - support mask fraction `0.0543`
  - Chronological split summary:
    - train windows `149`
    - val windows `32`
    - test windows `33`
  - Tile inventory:
    - usable 64x64 tiles at current settings: `8`
  - Dataset scaffold:
    - sample input shape `(1, 12, 64, 64)`
    - sample observed deformation shape `(1, 64, 64)`
- Output paths:
  - [punjab_phase1_coverage_summary.csv](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_coverage_summary.csv)
  - [punjab_phase1_coverage_summary.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_coverage_summary.json)
  - [punjab_phase1_coverage_diagnostics.png](/home/ubuntu/work/punjab/outputs/punjab_prior/figures/punjab_phase1_coverage_diagnostics.png)
  - [punjab_phase1_split_summary.csv](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_split_summary.csv)
  - [punjab_phase1_tile_inventory.csv](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_tile_inventory.csv)
- Next action:
  - Add normalization and batching for the tile dataset.
  - Build the first end-to-end phase-1 training scaffold with forward + spatial + temporal losses.

## 2026-03-14 Normalization And Smoke-Training Update
- Change summary:
  - Added train-only input normalization for the patch-based Punjab dataset.
  - Added sequential `DataLoader` construction that preserves temporal order for temporal regularization.
  - Added the first phase-1 epoch runner using:
    - forward consistency,
    - spatial regularization,
    - temporal regularization.
  - Added a small end-to-end smoke-training routine and executed it successfully.
  - Added artifact paths to the notebook evaluation template.
- Files/cells touched:
  - `punjab_prior_constrained_inversion.ipynb` Cells 11, 17, and 19
  - `IMPLEMENTATION_RECORD_PUNJAB.md`
- Verification result:
  - Notebook compile check passed.
  - Smoke training executed in `swin_env` on the real Punjab tile dataset.
  - Normalization summary:
    - `x_mean = 1.2609`
    - `x_std = 13.1266`
    - train/val/test samples: `1192 / 256 / 264`
  - Smoke-run history:
    - epoch 1:
      - train forward `25.4667`
      - val forward `495.4593`
    - epoch 2:
      - train forward `25.4667`
      - val forward `495.4592`
  - This confirms the full phase-1 path runs end-to-end on real Punjab data, but also indicates the current forward-loss scale is likely too large or poorly conditioned for a full run without another calibration pass.
- Output paths:
  - [punjab_phase1_normalization.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_normalization.json)
  - [punjab_phase1_smoke_history.csv](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_smoke_history.csv)
  - [punjab_phase1_smoke_summary.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_smoke_summary.json)
- Next action:
  - Inspect and likely rescale the forward-loss term before starting a longer Punjab run.
  - Add first predicted-field and residual visualization cells for the smoke run.

## 2026-03-14 Forward-Normalization And Residual-Diagnostics Update
- Change summary:
  - Added train-based deformation normalization for the forward loss term.
  - Re-ran the Punjab smoke training with normalized forward loss.
  - Added first smoke residual diagnostics and exported a residual figure plus summary JSON.
  - Updated the notebook artifact manifest to include the smoke diagnostics.
- Files/cells touched:
  - `punjab_prior_constrained_inversion.ipynb` Cells 3, 11, 17, 19, and 21
  - `IMPLEMENTATION_RECORD_PUNJAB.md`
  - `STEPWISE_PLAN.md`
- Verification result:
  - Forward-loss scaling improved substantially after normalization.
  - Updated smoke-run summary:
    - epoch 2 train forward `0.1339`
    - epoch 2 val forward `2.6058`
  - Diagnostic summary:
    - raw forward RMSE `22.2053`
    - normalized forward RMSE `1.6103`
  - This makes the phase-1 training objective much better conditioned for a longer run.
- Output paths:
  - [punjab_phase1_smoke_history.csv](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_smoke_history.csv)
  - [punjab_phase1_smoke_summary.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_smoke_summary.json)
  - [punjab_phase1_smoke_diagnostics.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_smoke_diagnostics.json)
  - [punjab_phase1_smoke_residuals.png](/home/ubuntu/work/punjab/outputs/punjab_prior/figures/punjab_phase1_smoke_residuals.png)
- Next action:
  - Treat forward-normalized phase-1 loss as the new baseline.
  - Move next to a longer Punjab phase-1 run with checkpointing and broader residual inspection.

## 2026-03-14 Checkpointed Pilot-Run Update
- Change summary:
  - Extended the Punjab phase-1 notebook from smoke-only training to a checkpointed pilot run.
  - Added a bounded longer training routine with:
    - `4` epochs,
    - `16` train batches per epoch,
    - `8` validation batches per epoch.
  - Added best-checkpoint saving based on validation forward loss.
  - Added broader pilot diagnostics over multiple validation batches and exported a pilot residual figure.
  - Updated the notebook artifact manifest to include pilot outputs.
- Files/cells touched:
  - `punjab_prior_constrained_inversion.ipynb` Cells 17, 19, and 21
  - `IMPLEMENTATION_RECORD_PUNJAB.md`
  - `STEPWISE_PLAN.md`
- Verification result:
  - Pilot run executed successfully in `swin_env`.
  - Best pilot validation forward loss:
    - `2.4672086010687053`
  - Final pilot summary:
    - epoch `4`
    - train total `0.1069803`
    - val total `2.4672402`
    - train forward `0.1069715`
    - val forward `2.4672086`
  - Pilot diagnostics across `4` validation batches:
    - raw forward RMSE mean `22.0526`
    - normalized forward RMSE mean `1.5993`
  - A best checkpoint was written successfully.
- Output paths:
  - [punjab_phase1_pilot_history.csv](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_pilot_history.csv)
  - [punjab_phase1_pilot_summary.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_pilot_summary.json)
  - [punjab_phase1_pilot_best.pt](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_pilot_best.pt)
  - [punjab_phase1_pilot_diagnostics.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_pilot_diagnostics.json)
  - [punjab_phase1_pilot_residuals.png](/home/ubuntu/work/punjab/outputs/punjab_prior/figures/punjab_phase1_pilot_residuals.png)
- Next action:
  - Treat the checkpointed pilot as the first real Punjab baseline.
  - Decide whether to scale the batch budget upward or first revise the spatial support/tile strategy.

## 2026-03-14 Support-Strategy Revision Update
- Change summary:
  - Added a support-mask strategy sweep to the Punjab notebook.
  - Replaced the overly restrictive velocity-heavy support rule with a revised phase-1 selection based on:
    - temporal support from the deformation stack, and
    - a relaxed coherence threshold.
  - Updated tile selection and normalization using the revised support strategy.
- Selected phase-1 support strategy:
  - `mask_strategy = coherence_and_temporal_support`
  - `coherence_threshold = 0.20`
  - `time_valid_fraction_threshold = 0.05`
  - `tile_size = 64`
  - `tile_min_valid_fraction = 0.20`
- Verification result:
  - Support mask fraction increased from `0.0543` to `0.0933`.
  - Usable `64x64` tiles increased from `8` to `65`.
  - Updated dataset sizes:
    - train `9685`
    - val `2080`
    - test `2145`
  - Updated normalization:
    - `x_mean = 5.2309`
    - `x_std = 14.2448`
    - `obs_mean = 5.5019`
    - `obs_std = 14.8767`
- Files/cells touched:
  - `punjab_prior_constrained_inversion.ipynb` Cells 3, 7, 9, and 11
  - `IMPLEMENTATION_RECORD_PUNJAB.md`
  - `STEPWISE_PLAN.md`
- Output paths:
  - [punjab_phase1_coverage_summary.csv](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_coverage_summary.csv)
  - [punjab_phase1_support_strategy_sweep.csv](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_support_strategy_sweep.csv)
  - [punjab_phase1_tile_inventory.csv](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_tile_inventory.csv)
  - [punjab_phase1_normalization.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_normalization.json)
- Next action:
  - Rerun the pilot training with the revised support-mask strategy before scaling to a fuller Punjab run.

## 2026-03-14 Revised-Support Pilot Comparison Update
- Change summary:
  - Preserved the earlier sparse-support pilot outputs.
  - Reran the pilot training with the revised support-mask strategy.
  - Preserved the revised-support pilot outputs separately.
  - Generated a direct support-strategy comparison table and JSON report.
- Verification result:
  - Revised-support pilot ran successfully with the expanded support mask.
  - Comparison against the earlier sparse-support pilot shows:
    - usable tiles increased from `8` to `65`
    - but the bounded pilot metrics became worse under the current sampling scheme
  - Pilot comparison:
    - sparse-support `val_forward = 2.4672`
    - revised-support `val_forward = 3.6048`
    - sparse-support normalized forward RMSE mean `1.5993`
    - revised-support normalized forward RMSE mean `1.6829`
- Interpretation:
  - The revised support mask is still the better spatial-coverage choice.
  - However, with the current bounded-batch pilot, the expanded tile pool appears to dilute temporal continuity.
  - This is consistent with the observed temporal terms staying near zero in the revised-support pilot.
  - So the next issue is no longer just spatial support. It is the batching/sampling strategy for temporal regularization.
- Output paths:
  - [punjab_phase1_pilot_history_sparse_support.csv](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_pilot_history_sparse_support.csv)
  - [punjab_phase1_pilot_summary_sparse_support.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_pilot_summary_sparse_support.json)
  - [punjab_phase1_pilot_diagnostics_sparse_support.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_pilot_diagnostics_sparse_support.json)
  - [punjab_phase1_pilot_history_revised_support.csv](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_pilot_history_revised_support.csv)
  - [punjab_phase1_pilot_summary_revised_support.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_pilot_summary_revised_support.json)
  - [punjab_phase1_pilot_diagnostics_revised_support.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_pilot_diagnostics_revised_support.json)
  - [punjab_phase1_pilot_support_comparison.csv](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_pilot_support_comparison.csv)
  - [punjab_phase1_pilot_support_comparison.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_pilot_support_comparison.json)
- Next action:
  - Revise the pilot batching so the model actually experiences temporal progression for the same tiles, then rerun the revised-support pilot.

## 2026-03-14 Grouped-Support Temporal-Batching Update
- Change summary:
  - Reworked the bounded Phase 1 pilot so samples are grouped by tile instead of diffusing across the full revised-support tile pool.
  - Added a grouped-support pilot schedule that preserves consecutive temporal windows for the same tile within each bounded run.
  - Refreshed the canonical pilot outputs to point to the new grouped-support run while preserving the older revised-support outputs separately.
  - Exported a direct grouped-versus-revised comparison report.
- Selected grouped pilot strategy:
  - `strategy = tile_grouped_temporal_progression`
  - train tiles per bounded pilot: `4`
  - validation tiles per bounded pilot: `4`
  - train windows per tile: `16`
  - validation windows per tile: `8`
- Files/cells touched:
  - `punjab_prior_constrained_inversion.ipynb` Cells 3, 11, 17, 19, and 21
  - `IMPLEMENTATION_RECORD_PUNJAB.md`
  - `STEPWISE_PLAN.md`
- Verification result:
  - Notebook compile check passed.
  - Full notebook rerun completed successfully in `swin_env`.
  - Grouped-support pilot summary:
    - epoch `4`
    - train forward `0.0953`
    - val forward `2.1528`
    - train temporal `S0 = 0.00119`
    - train temporal `Sg = 0.00095`
    - val temporal `S0 = 0.00022`
    - val temporal `Sg = 0.00018`
  - Grouped-support diagnostics across `4` validation batches:
    - raw forward RMSE mean `18.7686`
    - normalized forward RMSE mean `1.2616`
  - Direct comparison against the earlier revised-support bounded pilot:
    - revised-support `val_forward = 3.6048`
    - grouped-support `val_forward = 2.1528`
    - revised-support normalized forward RMSE mean `1.6829`
    - grouped-support normalized forward RMSE mean `1.2616`
- Interpretation:
  - The revised support mask was not the main issue.
  - The main bottleneck was the bounded pilot sampler, which had been starving the temporal prior of consecutive windows from the same tile.
  - Grouped tile-wise temporal batching materially improved the revised-support pilot and activated the temporal regularization terms.
  - This is now the best Punjab Phase 1 pilot configuration so far.
- Output paths:
  - [punjab_phase1_pilot_history_grouped_support.csv](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_pilot_history_grouped_support.csv)
  - [punjab_phase1_pilot_summary_grouped_support.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_pilot_summary_grouped_support.json)
  - [punjab_phase1_pilot_best_grouped_support.pt](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_pilot_best_grouped_support.pt)
  - [punjab_phase1_pilot_diagnostics_grouped_support.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_pilot_diagnostics_grouped_support.json)
  - [punjab_phase1_pilot_residuals_grouped_support.png](/home/ubuntu/work/punjab/outputs/punjab_prior/figures/punjab_phase1_pilot_residuals_grouped_support.png)
  - [punjab_phase1_pilot_grouped_comparison.csv](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_pilot_grouped_comparison.csv)
  - [punjab_phase1_pilot_grouped_comparison.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_pilot_grouped_comparison.json)
- Next action:
  - Inspect grouped-support predicted fields across more validation examples, then test a slightly larger bounded grouped pilot before scaling to a fuller run.

## 2026-03-14 Expanded Grouped-Pilot Validation Update
- Change summary:
  - Added broader grouped-support diagnostics so pilot validation now saves multiple examples rather than only the first batch example.
  - Added a slightly larger grouped-support pilot using more tiles and more consecutive time windows per tile.
  - Compared three bounded Phase 1 pilot variants directly:
    - revised-support bounded sampler,
    - grouped-support temporal sampler,
    - expanded grouped-support temporal sampler.
  - Promoted the expanded grouped-support pilot to the canonical Phase 1 pilot outputs because it is the current best bounded run.
- Selected expanded grouped pilot strategy:
  - `strategy = tile_grouped_temporal_progression_expanded`
  - train tiles per bounded pilot: `6`
  - validation tiles per bounded pilot: `6`
  - train windows per tile: `20`
  - validation windows per tile: `10`
- Files/cells touched:
  - `punjab_prior_constrained_inversion.ipynb` Cells 3, 11, 17, 19, and 21
  - `IMPLEMENTATION_RECORD_PUNJAB.md`
  - `STEPWISE_PLAN.md`
- Verification result:
  - Full notebook rerun completed successfully in `swin_env`.
  - Expanded grouped-support pilot summary:
    - epoch `4`
    - train forward `0.1700`
    - val forward `1.4286`
    - train temporal `S0 = 0.00106`
    - train temporal `Sg = 0.00102`
    - val temporal `S0 = 0.00030`
    - val temporal `Sg = 0.00032`
  - Expanded grouped-support diagnostics across `6` validation batches:
    - raw forward RMSE mean `14.7565`
    - normalized forward RMSE mean `0.9919`
  - Direct bounded-pilot comparison:
    - revised-support bounded `val_forward = 3.6048`
    - grouped-support temporal `val_forward = 2.1528`
    - expanded grouped-support `val_forward = 1.4286`
  - The expanded grouped-support run is now the best Punjab Phase 1 pilot so far.
- Interpretation:
  - The grouped tile-wise temporal sampler continues to be the decisive improvement.
  - Increasing the grouped pilot budget helps rather than hurts, which suggests the gain is not just a small-sample artifact.
  - The expanded grouped-support sampler is now the correct starting point for the first fuller Phase 1 Punjab run.
- Output paths:
  - [punjab_phase1_pilot_history_grouped_support_expanded.csv](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_pilot_history_grouped_support_expanded.csv)
  - [punjab_phase1_pilot_summary_grouped_support_expanded.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_pilot_summary_grouped_support_expanded.json)
  - [punjab_phase1_pilot_best_grouped_support_expanded.pt](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_pilot_best_grouped_support_expanded.pt)
  - [punjab_phase1_pilot_diagnostics_grouped_support_expanded.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_pilot_diagnostics_grouped_support_expanded.json)
  - [punjab_phase1_pilot_residuals_grouped_support_expanded.png](/home/ubuntu/work/punjab/outputs/punjab_prior/figures/punjab_phase1_pilot_residuals_grouped_support_expanded.png)
  - [punjab_phase1_pilot_grouped_comparison.csv](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_pilot_grouped_comparison.csv)
  - [punjab_phase1_pilot_grouped_comparison.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_pilot_grouped_comparison.json)
  - canonical pilot outputs now point to the expanded grouped-support run:
    - [punjab_phase1_pilot_history.csv](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_pilot_history.csv)
    - [punjab_phase1_pilot_summary.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_pilot_summary.json)
    - [punjab_phase1_pilot_best.pt](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_pilot_best.pt)
    - [punjab_phase1_pilot_diagnostics.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_pilot_diagnostics.json)
    - [punjab_phase1_pilot_residuals.png](/home/ubuntu/work/punjab/outputs/punjab_prior/figures/punjab_phase1_pilot_residuals.png)
- Next action:
  - Use the expanded grouped-support sampler as the basis for the first fuller Phase 1 Punjab run.

## 2026-03-14 Fuller Phase-1 Run Update
- Change summary:
  - Added a first fuller Phase 1 training configuration on top of the grouped-support strategy.
  - Kept the expanded grouped-support pilot as the bounded baseline and ran a larger grouped schedule separately.
  - Added fuller-run diagnostics and residual galleries.
  - Compared the fuller run directly against all bounded grouped-support variants.
- Selected fuller configuration:
  - `strategy = phase1_fuller_grouped_run`
  - train tiles: `8`
  - validation tiles: `8`
  - train windows per tile: `24`
  - validation windows per tile: `12`
  - epochs: `6`
- Files/cells touched:
  - `punjab_prior_constrained_inversion.ipynb` Cells 3, 11, 17, 19, and 21
  - `IMPLEMENTATION_RECORD_PUNJAB.md`
  - `STEPWISE_PLAN.md`
- Verification result:
  - Full notebook rerun completed successfully in `swin_env`.
  - Fuller Phase 1 summary:
    - epoch `6`
    - train forward `0.2342`
    - val forward `2.4986`
    - train temporal `S0 = 0.00054`
    - train temporal `Sg = 0.00058`
    - val temporal `S0 = 0.00021`
    - val temporal `Sg = 0.00022`
  - Fuller Phase 1 diagnostics across `6` validation batches:
    - raw forward RMSE mean `20.0724`
    - normalized forward RMSE mean `1.3493`
  - Direct comparison:
    - revised-support bounded `val_forward = 3.6048`
    - grouped-support temporal `val_forward = 2.1528`
    - expanded grouped-support `val_forward = 1.4286`
    - fuller Phase 1 `val_forward = 2.4986`
- Interpretation:
  - The fuller grouped run did not improve over the expanded grouped-support pilot.
  - This means simply increasing grouped support breadth and time span is not enough by itself.
  - The expanded grouped-support pilot remains the best Phase 1 baseline so far.
  - The next scaling step should change optimization or schedule, not only grouped support size.
- Output paths:
  - [punjab_phase1_fuller_history.csv](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_fuller_history.csv)
  - [punjab_phase1_fuller_summary.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_fuller_summary.json)
  - [punjab_phase1_fuller_best.pt](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_fuller_best.pt)
  - [punjab_phase1_fuller_diagnostics.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_fuller_diagnostics.json)
  - [punjab_phase1_fuller_residuals.png](/home/ubuntu/work/punjab/outputs/punjab_prior/figures/punjab_phase1_fuller_residuals.png)
  - [punjab_phase1_pilot_grouped_comparison.csv](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_pilot_grouped_comparison.csv)
  - [punjab_phase1_pilot_grouped_comparison.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_pilot_grouped_comparison.json)
- Next action:
  - Keep the expanded grouped-support pilot as the current baseline and diagnose why the fuller grouped run degraded before scaling further.

## 2026-03-14 Fuller Warm-Start Fine-Tuning Update
- Change summary:
  - Added a warm-start fine-tuning variant for the fuller Phase 1 run.
  - Initialized the fuller grouped run from the best expanded grouped-support checkpoint.
  - Reduced learning rate during the warm-started fuller run to test whether the scratch fuller degradation was mainly an optimization issue.
  - Exported separate history, summary, checkpoint, diagnostics, and residual galleries for the fine-tuned fuller run.
- Selected fine-tune configuration:
  - `strategy = phase1_fuller_warmstart_finetune`
  - initialization: best `grouped_support_expanded` checkpoint
  - learning rate: `5e-5`
  - train tiles: `8`
  - validation tiles: `8`
  - train windows per tile: `24`
  - validation windows per tile: `12`
  - epochs: `6`
- Files/cells touched:
  - `punjab_prior_constrained_inversion.ipynb` Cells 3, 11, 17, 19, and 21
  - `IMPLEMENTATION_RECORD_PUNJAB.md`
  - `STEPWISE_PLAN.md`
- Verification result:
  - Partial notebook rerun completed successfully in `swin_env`.
  - Warm-start fuller fine-tune summary:
    - epoch `6`
    - train forward `0.2342`
    - val forward `2.4986`
    - train temporal `S0 = 0.00068`
    - train temporal `Sg = 0.00069`
    - val temporal `S0 = 0.00027`
    - val temporal `Sg = 0.00029`
  - Warm-start diagnostics across `6` validation batches:
    - raw forward RMSE mean `20.0724`
    - normalized forward RMSE mean `1.3493`
  - Direct comparison:
    - expanded grouped-support `val_forward = 1.4286`
    - scratch fuller `val_forward = 2.4986`
    - warm-start fuller fine-tune `val_forward = 2.4986`
- Interpretation:
  - Warm-starting from the best expanded grouped checkpoint did not improve the fuller grouped run.
  - This suggests the degradation is not just due to poor random initialization.
  - The current bottleneck is therefore deeper than simple optimizer start-up; the next change should alter the loss balance, curriculum, or input representation rather than only warm-starting.
  - The expanded grouped-support pilot remains the best Phase 1 baseline.
- Output paths:
  - [punjab_phase1_fuller_finetune_history.csv](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_fuller_finetune_history.csv)
  - [punjab_phase1_fuller_finetune_summary.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_fuller_finetune_summary.json)
  - [punjab_phase1_fuller_finetune_best.pt](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_fuller_finetune_best.pt)
  - [punjab_phase1_fuller_finetune_diagnostics.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_fuller_finetune_diagnostics.json)
  - [punjab_phase1_fuller_finetune_residuals.png](/home/ubuntu/work/punjab/outputs/punjab_prior/figures/punjab_phase1_fuller_finetune_residuals.png)
  - [punjab_phase1_pilot_grouped_comparison.csv](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_pilot_grouped_comparison.csv)
  - [punjab_phase1_pilot_grouped_comparison.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_pilot_grouped_comparison.json)
- Next action:
  - Keep the expanded grouped-support pilot as the current baseline and choose a more substantive training change before attempting another larger Phase 1 run.

## 2026-03-14 Reliability-Weighted Expanded Baseline Update
- Change summary:
  - Added a reliability-weighted forward-loss branch on top of the current best expanded grouped-support baseline.
  - Kept the sampler fixed and changed only the forward term so that high-coherence, high-temporal-support pixels contribute more strongly.
  - Added a notebook section documenting the weighted formulation and its outputs.
- Weighted formulation:
  - Reliability proxy:
    \[
    \tilde c(i,j)=\mathrm{clip}\left(\frac{\mathrm{coh}(i,j)-0.20}{1-0.20},0,1\right),
    \qquad
    \tilde f(i,j)=\mathrm{clip}\left(\frac{f_{\mathrm{valid}}(i,j)-0.05}{1-0.05},0,1\right)
    \]
    \[
    w(i,j)=M(i,j)\left[0.25 + 0.75\left(\tilde c(i,j)\tilde f(i,j)\right)^{1/2}\right]
    \]
  - Weighted forward loss:
    \[
    L^{(w)}_{fwd}
    =
    \frac{\sum_{i,j} w(i,j)\left(\hat d^*(i,j)-d^*(i,j)\right)^2}{\sum_{i,j} w(i,j)}
    \]
  - Total objective:
    \[
    L^{(w)}_{phase1}
    =
    \lambda_f L^{(w)}_{fwd}
    + \lambda_{s0}L^{S_0}_{spatial}
    + \lambda_{sg}L^{S_g}_{spatial}
    + \lambda_{t0}L^{S_0}_{temporal}
    + \lambda_{tg}L^{S_g}_{temporal}
    \]
- Files/cells touched:
  - `punjab_prior_constrained_inversion.ipynb` new end-section:
    - `## Reliability-Weighted Expanded Baseline`
    - `## Reliability-Weighted Results Summary`
  - `IMPLEMENTATION_RECORD_PUNJAB.md`
  - `STEPWISE_PLAN.md`
- Verification result:
  - The initial weighted branch exposed a real implementation issue: invalid coherence pixels propagated `NaN` values into the reliability map.
  - That bug was fixed by zero-filling non-finite reliability weights before training.
  - After the fix, the weighted expanded run still collapsed numerically:
    - training totals became `NaN`
    - validation totals became `NaN`
    - diagnostics also became `NaN`
  - Reliability statistics themselves were reasonable:
    - support fraction `0.0933`
    - weight min `0.2505`
    - weight mean `0.5532`
    - weight max `0.9901`
- Interpretation:
  - The experiment is still useful because it cleanly shows that simple reliability weighting is too aggressive for the current Punjab Phase 1 optimization setup.
  - The expanded grouped-support baseline remains the best real-data Phase 1 baseline.
  - The next change should be milder than direct weighted forward replacement, for example:
    - loss rebalance,
    - schedule change,
    - or a gentler residual-style reliability correction.
- Output paths:
  - [punjab_phase1_pilot_history_grouped_support_expanded_reliability.csv](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_pilot_history_grouped_support_expanded_reliability.csv)
  - [punjab_phase1_pilot_summary_grouped_support_expanded_reliability.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_pilot_summary_grouped_support_expanded_reliability.json)
  - [punjab_phase1_pilot_diagnostics_grouped_support_expanded_reliability.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_pilot_diagnostics_grouped_support_expanded_reliability.json)
  - [punjab_phase1_pilot_residuals_grouped_support_expanded_reliability.png](/home/ubuntu/work/punjab/outputs/punjab_prior/figures/punjab_phase1_pilot_residuals_grouped_support_expanded_reliability.png)
  - [punjab_phase1_reliability_weight_comparison.csv](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_reliability_weight_comparison.csv)
  - [punjab_phase1_reliability_weight_comparison.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_reliability_weight_comparison.json)
  - [punjab_phase1_reliability_weight_stats.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_reliability_weight_stats.json)
- Next action:
  - Keep the expanded grouped-support pilot as the Phase 1 baseline and choose a milder targeted change than direct reliability-weighted forward replacement.

## 2026-03-14 Rebalanced Expanded Baseline Update
- Change summary:
  - Added a milder Phase 1 refinement on top of the expanded grouped-support baseline.
  - Kept the same grouped sampler and same data support.
  - Warm-started from the best expanded grouped-support checkpoint.
  - Lowered the learning rate and increased the generic-prior weights so the spatial and temporal terms contribute meaningfully without replacing the forward term.
- Rebalanced formulation:
  - Initialization and optimizer:
    \[
    \theta_0 \leftarrow \theta_{\mathrm{expanded}},
    \qquad
    \eta = 5 \times 10^{-5}
    \]
  - Loss:
    \[
    L^{(reb)}_{phase1}
    =
    \lambda_f L_{fwd}
    + \lambda_{s0}L^{S_0}_{spatial}
    + \lambda_{sg}L^{S_g}_{spatial}
    + \lambda_{t0}L^{S_0}_{temporal}
    + \lambda_{tg}L^{S_g}_{temporal}
    \]
  - Rebalanced weights:
    - `forward = 1.0`
    - `spatial_s0 = 5.0`
    - `spatial_sg = 5.0`
    - `temporal_s0 = 1.0`
    - `temporal_sg = 1.0`
- Files/cells touched:
  - `punjab_prior_constrained_inversion.ipynb` new end-section:
    - `## Rebalanced Expanded Baseline`
    - `## Rebalanced Results Summary`
  - `IMPLEMENTATION_RECORD_PUNJAB.md`
  - `STEPWISE_PLAN.md`
- Verification result:
  - The rebalanced branch ran successfully in `swin_env`.
  - It remained numerically stable, unlike the direct reliability-weighted branch.
  - Final rebalanced summary:
    - `val_forward = 1.4286399`
    - normalized forward RMSE mean `0.9919191`
  - Direct comparison against the expanded grouped-support baseline:
    - baseline `val_forward = 1.4286398`
    - rebalanced `val_forward = 1.4286399`
    - baseline normalized forward RMSE mean `0.9919191`
    - rebalanced normalized forward RMSE mean `0.9919191`
  - Regularization terms decreased modestly under the rebalanced branch:
    - baseline val spatial `S0/Sg = 0.01053 / 0.01019`
    - rebalanced val spatial `S0/Sg = 0.00882 / 0.00882`
    - baseline val temporal `S0/Sg = 0.000304 / 0.000317`
    - rebalanced val temporal `S0/Sg = 0.000282 / 0.000296`
- Interpretation:
  - The rebalanced branch is stable and slightly smoother, but it does not improve the main deformation-fit metrics.
  - The expanded grouped-support pilot remains the best Punjab Phase 1 baseline on the primary forward criteria.
  - This is still a useful result because it shows the current baseline is already near a local optimum for the present Phase 1 setup.
- Output paths:
  - [punjab_phase1_pilot_history_grouped_support_expanded_rebalanced.csv](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_pilot_history_grouped_support_expanded_rebalanced.csv)
  - [punjab_phase1_pilot_summary_grouped_support_expanded_rebalanced.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_pilot_summary_grouped_support_expanded_rebalanced.json)
  - [punjab_phase1_pilot_best_grouped_support_expanded_rebalanced.pt](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_pilot_best_grouped_support_expanded_rebalanced.pt)
  - [punjab_phase1_pilot_diagnostics_grouped_support_expanded_rebalanced.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_pilot_diagnostics_grouped_support_expanded_rebalanced.json)
  - [punjab_phase1_pilot_residuals_grouped_support_expanded_rebalanced.png](/home/ubuntu/work/punjab/outputs/punjab_prior/figures/punjab_phase1_pilot_residuals_grouped_support_expanded_rebalanced.png)
  - [punjab_phase1_rebalanced_comparison.csv](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_rebalanced_comparison.csv)
  - [punjab_phase1_rebalanced_comparison.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_rebalanced_comparison.json)
- Next action:
  - Keep the expanded grouped-support pilot as the main Phase 1 baseline.
  - If we change Phase 1 again before external priors, the next change should be a schedule or representation change rather than another mild loss rebalance.

## 2026-03-14 W3RA Prior Preparation Update
- Change summary:
  - Started the external-prior preparation phase after freezing the generic-prior Punjab baseline.
  - Added reusable W3RA alignment helpers to the shared package.
  - Added a new notebook section that aligns the clipped Punjab W3RA fields to Punjab acquisition dates, computes overlap-period anomalies, and exports Phase 2-ready artifacts.
  - Kept W3RA inactive in the Punjab loss for now; this step is preparation only.
- External-prior preparation formulas:
  - Month alignment:
    \[
    t_{\mathrm{month}} = \mathrm{month\_start}(t_{\mathrm{Punjab}})
    \]
    \[
    (S_0^{W3RA}, S_g^{W3RA})(t_{\mathrm{Punjab}})
    =
    (S_0, S_g)_{W3RA}(t_{\mathrm{month}})
    \]
  - Overlap-period anomalies:
    \[
    S_k^{\prime}(t)=S_k(t)-\overline{S_k}^{\,\mathrm{overlap}},
    \qquad k\in\{0,g\}
    \]
- Files/cells touched:
  - [punjab_inversion/priors.py](/home/ubuntu/work/punjab/punjab_inversion/priors.py)
  - [punjab_inversion/__init__.py](/home/ubuntu/work/punjab/punjab_inversion/__init__.py)
  - `punjab_prior_constrained_inversion.ipynb` new end-section:
    - `## W3RA Prior Preparation`
  - `IMPLEMENTATION_RECORD_PUNJAB.md`
  - `STEPWISE_PLAN.md`
- Verification result:
  - W3RA prep ran successfully in `swin_env`.
  - Punjab overlap:
    - `n_times = 225`
    - `date_start = 2016-01-06`
    - `date_end = 2024-01-24`
    - `source_month_start = 2016-01-01`
    - `source_month_end = 2024-01-01`
  - Basin-mean aligned statistics:
    - `S0_mean = 0.03919`
    - `S0_std = 0.01932`
    - `S0_anom_std = 0.01669`
    - `Sg_mean = 0.01212`
    - `Sg_std = 0.02735`
    - `Sg_anom_std = 0.02101`
- Interpretation:
  - The first external prior source is now aligned and exportable without changing the active Punjab inversion.
  - This gives us a clean Phase 2 starting point: the next step is to add a weak `W3RA` prior hook on top of the frozen Phase 1 baseline rather than continuing generic-prior tuning.
- Output paths:
  - [punjab_phase2_w3ra_aligned.nc](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase2_w3ra_aligned.nc)
  - [punjab_phase2_w3ra_aligned_anomalies.nc](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase2_w3ra_aligned_anomalies.nc)
  - [punjab_phase2_w3ra_alignment_summary.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase2_w3ra_alignment_summary.json)
  - [punjab_phase2_w3ra_alignment_table.csv](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase2_w3ra_alignment_table.csv)
  - [punjab_phase2_w3ra_basin_timeseries.csv](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase2_w3ra_basin_timeseries.csv)
  - [punjab_phase2_w3ra_basin_timeseries.png](/home/ubuntu/work/punjab/outputs/punjab_prior/figures/punjab_phase2_w3ra_basin_timeseries.png)
- Next action:
  - Add the weak `W3RA` prior hook to the Punjab loss while keeping the expanded grouped-support baseline frozen as the reference run.

## 2026-03-14 Weak W3RA Prior Pilot Update
- Change summary:
  - Activated the first external prior on top of the frozen expanded grouped-support baseline.
  - Added a weak W3RA anomaly prior for `S0` and `Sg` while keeping:
    - the sampler,
    - the support mask,
    - the forward model,
    - and the warm-start baseline checkpoint
    fixed.
- Weak W3RA prior formulation:
  - Total loss:
    \[
    L_{phase2,W3RA} = L_{phase1} + \lambda_w L_{W3RA}
    \]
  - Hydrology prior term:
    \[
    L_{W3RA} = \mathrm{MSE}(\hat S, S^{W3RA}_{anom})
    \]
  - Settings:
    - `\lambda_w = 0.05`
    - warm start from the best expanded grouped-support checkpoint
    - learning rate `5e-5`
    - W3RA targets interpolated tile-by-tile from aligned anomaly fields
- Files/cells touched:
  - `punjab_prior_constrained_inversion.ipynb` new end-section:
    - `## Weak W3RA Prior Pilot`
    - `## Weak W3RA Prior Results Summary`
  - `IMPLEMENTATION_RECORD_PUNJAB.md`
  - `STEPWISE_PLAN.md`
- Verification result:
  - The weak W3RA prior pilot ran successfully in `swin_env`.
  - Final summary:
    - `val_forward = 1.4286398`
    - normalized forward RMSE mean `0.9919191`
  - Direct comparison against the frozen Phase 1 baseline:
    - baseline `val_forward = 1.4286398`
    - weak W3RA `val_forward = 1.4286398`
    - baseline normalized forward RMSE mean `0.9919191`
    - weak W3RA normalized forward RMSE mean `0.9919191`
  - The hydrology term decreased steadily:
    - train hydrology `0.0001180 -> 0.0000541`
    - val hydrology `0.0006363 -> 0.0004132`
- Interpretation:
  - The first external prior hook is stable.
  - At weak strength, W3RA does not materially change the primary deformation-fit metrics.
  - This is still a good result because it establishes a working Phase 2 external-prior path without destabilizing the inversion.
  - The next question is no longer “can W3RA be added?” but “what prior strength or formulation changes the solution meaningfully without hurting the fit?”
- Output paths:
  - [punjab_phase2_pilot_history_grouped_support_expanded_w3ra_weak.csv](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase2_pilot_history_grouped_support_expanded_w3ra_weak.csv)
  - [punjab_phase2_pilot_summary_grouped_support_expanded_w3ra_weak.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase2_pilot_summary_grouped_support_expanded_w3ra_weak.json)
  - [punjab_phase2_pilot_best_grouped_support_expanded_w3ra_weak.pt](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase2_pilot_best_grouped_support_expanded_w3ra_weak.pt)
  - [punjab_phase2_pilot_diagnostics_grouped_support_expanded_w3ra_weak.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase2_pilot_diagnostics_grouped_support_expanded_w3ra_weak.json)
  - [punjab_phase2_pilot_residuals_grouped_support_expanded_w3ra_weak.png](/home/ubuntu/work/punjab/outputs/punjab_prior/figures/punjab_phase2_pilot_residuals_grouped_support_expanded_w3ra_weak.png)
  - [punjab_phase2_w3ra_comparison.csv](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase2_w3ra_comparison.csv)
  - [punjab_phase2_w3ra_comparison.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase2_w3ra_comparison.json)
- Next action:
  - Decide whether to strengthen or reformulate the W3RA prior before moving on to `GRACE`.

## 2026-03-14 Stronger W3RA Prior Pilot Update
- Change summary:
  - Reused the same Phase 2 W3RA anomaly-prior branch and increased only the hydrology weight.
  - Kept:
    - the frozen expanded grouped-support baseline,
    - the warm-start checkpoint,
    - the sampler,
    - the support mask,
    - and the anomaly-space W3RA targets
    fixed.
- Stronger W3RA prior formulation:
  - Total loss:
    \[
    L_{phase2,W3RA} = L_{phase1} + \lambda_w L_{W3RA}
    \]
  - Same anomaly-space hydrology term:
    \[
    L_{W3RA} = \mathrm{MSE}(\hat S, S^{W3RA}_{anom})
    \]
  - Updated setting:
    - `\lambda_w = 0.20`
- Files/cells touched:
  - `punjab_prior_constrained_inversion.ipynb` new end-section:
    - `## Stronger W3RA Prior Results Summary`
  - `IMPLEMENTATION_RECORD_PUNJAB.md`
  - `STEPWISE_PLAN.md`
- Verification result:
  - The stronger W3RA prior pilot also ran successfully in `swin_env`.
  - Final summary:
    - `val_forward = 1.4286398`
    - normalized forward RMSE mean `0.9919191`
  - Direct comparison against the frozen Phase 1 baseline:
    - baseline `val_forward = 1.4286398`
    - stronger W3RA `val_forward = 1.4286398`
    - baseline normalized forward RMSE mean `0.9919191`
    - stronger W3RA normalized forward RMSE mean `0.9919191`
  - Hydrology term decreased slightly further than in the weak W3RA run:
    - stronger val hydrology `0.0003928`
    - weak val hydrology `0.0004132`
- Interpretation:
  - W3RA prior integration remains stable when the weight is increased from `0.05` to `0.20`.
  - The stronger W3RA branch still does not materially change the main deformation-fit metrics.
  - This suggests the next meaningful W3RA experiment should be a reformulation rather than another simple weight increase.
- Output paths:
  - [punjab_phase2_pilot_history_grouped_support_expanded_w3ra_stronger.csv](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase2_pilot_history_grouped_support_expanded_w3ra_stronger.csv)
  - [punjab_phase2_pilot_summary_grouped_support_expanded_w3ra_stronger.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase2_pilot_summary_grouped_support_expanded_w3ra_stronger.json)
  - [punjab_phase2_pilot_best_grouped_support_expanded_w3ra_stronger.pt](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase2_pilot_best_grouped_support_expanded_w3ra_stronger.pt)
  - [punjab_phase2_pilot_diagnostics_grouped_support_expanded_w3ra_stronger.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase2_pilot_diagnostics_grouped_support_expanded_w3ra_stronger.json)
  - [punjab_phase2_pilot_residuals_grouped_support_expanded_w3ra_stronger.png](/home/ubuntu/work/punjab/outputs/punjab_prior/figures/punjab_phase2_pilot_residuals_grouped_support_expanded_w3ra_stronger.png)
- Next action:
  - Choose whether to reformulate the W3RA prior or move on to GRACE preparation.

## 2026-03-14 GRACE Preparation Update
- Change summary:
  - Added reusable GRACE discovery and alignment helpers to the shared prior module.
  - Added a new GRACE preparation section to the Punjab notebook.
  - Ran the GRACE preparation block to search for local GRACE/mascon inputs before attempting any prior integration.
- GRACE preparation target formulation:
  - Planned GRACE role:
    \[
    L_{GRACE} = \mathrm{MSE}\left(A(\hat S_{tot}), TWS_{GRACE}\right),
    \qquad
    \hat S_{tot} = \hat S_0 + \hat S_g
    \]
  - Current step:
    - discovery and alignment hook only
    - no GRACE term added to the Punjab loss yet
- Files/cells touched:
  - [punjab_inversion/priors.py](/home/ubuntu/work/punjab/punjab_inversion/priors.py)
  - [punjab_inversion/__init__.py](/home/ubuntu/work/punjab/punjab_inversion/__init__.py)
  - `punjab_prior_constrained_inversion.ipynb` new end-section:
    - `## GRACE Prior Preparation`
    - `## GRACE Preparation Results Summary`
  - `IMPLEMENTATION_RECORD_PUNJAB.md`
  - `STEPWISE_PLAN.md`
- Verification result:
  - GRACE preparation ran successfully in `swin_env`.
  - Discovery record:
    - search roots:
      - `/home/ubuntu/work/punjab`
      - `/mnt/data`
    - candidate count: `0`
  - No local GRACE/mascon dataset was found.
- Interpretation:
  - The GRACE alignment hook is now implemented and ready.
  - GRACE cannot yet be activated as a Punjab prior because there is no local mascon dataset to align.
  - This is a clean and useful stopping point for GRACE preparation: the next step depends on adding a real GRACE/mascon file to the workspace.
- Output paths:
  - [punjab_phase2_grace_discovery.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase2_grace_discovery.json)
- Next action:
  - Add a local GRACE/mascon dataset, then rerun the notebook’s GRACE preparation block to produce aligned GRACE artifacts before wiring a GRACE prior term.

## 2026-03-14 GRACE File Check And First Plot Update
- Change summary:
  - Located the local JPL RL06.3 mascon file under `/mnt/data/punjab_grace_mascon_l3/`.
  - Added a first GRACE check section to the Punjab notebook that:
    - opens the mascon file,
    - subsets a Punjab-region bounding box,
    - plots overlap-period basin-mean `lwe_thickness` and `uncertainty`,
    - and exports a first summary and figure.
- Files/cells touched:
  - `punjab_prior_constrained_inversion.ipynb` new end-section:
    - `## GRACE File Check And First Plot`
  - `IMPLEMENTATION_RECORD_PUNJAB.md`
- Verification result:
  - The GRACE file opened successfully in `swin_env`.
  - Punjab-region overlap summary:
    - file: `GRCTellus.JPL.200204_202512.GLO.RL06.3M.MSCNv04CRI.nc`
    - overlap times: `80`
    - overlap dates: `2016-01-16` to `2024-01-16`
    - regional lat range: `29.25` to `32.75`
    - regional lon range: `73.25` to `76.75`
    - overlap mean `lwe_thickness = -39.8209`
    - overlap std `lwe_thickness = 21.9940`
    - overlap mean `uncertainty = 2.4031`
- Interpretation:
  - The GRACE mascon file is real, readable, and spatially overlaps Punjab.
  - We now have the first GRACE plot and summary needed before alignment and prior wiring.
- Output paths:
  - [punjab_phase2_grace_first_check_summary.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase2_grace_first_check_summary.json)
  - [punjab_phase2_grace_first_check_timeseries.csv](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase2_grace_first_check_timeseries.csv)
  - [punjab_phase2_grace_first_check.png](/home/ubuntu/work/punjab/outputs/punjab_prior/figures/punjab_phase2_grace_first_check.png)
- Next action:
  - Align the GRACE mascon product to Punjab dates and build the first GRACE prior hook.

## 2026-03-14 GRACE Alignment And Weak Prior Pilot Update
- Change summary:
  - Extended the GRACE alignment helper to handle:
    - duplicate source timestamps within the same month,
    - missing monthly bins inside the Punjab overlap period,
    - and expansion back from monthly GRACE support to the native Punjab acquisition dates.
  - Added and ran the new notebook sections:
    - `## GRACE Alignment And Regional Series`
    - `## Weak GRACE Prior Pilot`
  - Used the frozen expanded grouped-support Phase 1 checkpoint as the initialization for the first GRACE prior run.
- GRACE alignment and prior formulation:
  - Regional GRACE alignment:
    \[
    g_{\mathrm{GRACE}}(t)=\mathrm{mean}_{\Omega_{\mathrm{Punjab}}}\big(TWS'_{\mathrm{GRACE}}(t)\big)
    \]
    where the GRACE field is first collapsed to unique monthly bins, then linearly interpolated across internal monthly gaps, and finally mapped back to the Punjab acquisition dates.
  - Weak GRACE prior pilot:
    \[
    L_{phase2,GRACE}=L_{phase1}+\lambda_g L_{\mathrm{GRACE}}
    \]
    with:
    \[
    L_{\mathrm{GRACE}}
    =
    \mathrm{MSE}\!\left(
    z\!\left(\overline{\hat S_0+\hat S_g}\right),
    z\!\left(g_{\mathrm{GRACE}}\right)
    \right)
    \]
    where \(z(\cdot)\) denotes batch standardization and \(\overline{\hat S_0+\hat S_g}\) is the tile-mean predicted total storage for each sample in the batch.
  - Settings:
    - `\lambda_g = 0.05`
    - learning rate `5e-5`
    - `4` epochs
    - warm start from `punjab_phase1_pilot_best_grouped_support_expanded.pt`
- Files/cells touched:
  - [punjab_inversion/priors.py](/home/ubuntu/work/punjab/punjab_inversion/priors.py)
  - [punjab_inversion/__init__.py](/home/ubuntu/work/punjab/punjab_inversion/__init__.py)
  - `punjab_prior_constrained_inversion.ipynb` new end-sections:
    - `## GRACE Alignment And Regional Series`
    - `## Weak GRACE Prior Pilot`
  - `IMPLEMENTATION_RECORD_PUNJAB.md`
  - `STEPWISE_PLAN.md`
- Verification result:
  - The aligned GRACE series now spans the full Punjab acquisition period:
    - `n_times = 225`
    - aligned dates `2016-01-06` to `2024-01-24`
    - source months `2016-01-01` to `2024-01-01`
    - aligned anomaly std `8.9884`
  - The weak GRACE pilot ran successfully in `swin_env`.
  - Final GRACE-prior summary:
    - `val_forward = 1.4286390`
    - normalized forward RMSE mean `0.9919191`
    - `val_grace = 1.9371`
  - Direct comparison against the frozen Phase 1 baseline:
    - baseline `val_forward = 1.4286398`
    - weak GRACE `val_forward = 1.4286390`
    - baseline normalized forward RMSE mean `0.9919191`
    - weak GRACE normalized forward RMSE mean `0.9919191`
- Interpretation:
  - GRACE prior integration now works end-to-end on the real Punjab notebook.
  - The first weak GRACE prior is stable.
  - At this strength and formulation, it behaves as an effective tie with the frozen expanded grouped-support baseline on the main deformation-fit metrics.
  - This mirrors the earlier W3RA result: external-prior plumbing works, but the first weak formulation does not yet materially move the main solution.
- Output paths:
  - [punjab_phase2_grace_aligned.nc](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase2_grace_aligned.nc)
  - [punjab_phase2_grace_aligned_anomalies.nc](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase2_grace_aligned_anomalies.nc)
  - [punjab_phase2_grace_alignment_summary.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase2_grace_alignment_summary.json)
  - [punjab_phase2_grace_alignment_table.csv](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase2_grace_alignment_table.csv)
  - [punjab_phase2_grace_region_timeseries.csv](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase2_grace_region_timeseries.csv)
  - [punjab_phase2_grace_region_timeseries.png](/home/ubuntu/work/punjab/outputs/punjab_prior/figures/punjab_phase2_grace_region_timeseries.png)
  - [punjab_phase2_pilot_history_grouped_support_expanded_grace_weak.csv](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase2_pilot_history_grouped_support_expanded_grace_weak.csv)
  - [punjab_phase2_pilot_summary_grouped_support_expanded_grace_weak.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase2_pilot_summary_grouped_support_expanded_grace_weak.json)
  - [punjab_phase2_pilot_best_grouped_support_expanded_grace_weak.pt](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase2_pilot_best_grouped_support_expanded_grace_weak.pt)
  - [punjab_phase2_pilot_diagnostics_grouped_support_expanded_grace_weak.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase2_pilot_diagnostics_grouped_support_expanded_grace_weak.json)
  - [punjab_phase2_pilot_residuals_grouped_support_expanded_grace_weak.png](/home/ubuntu/work/punjab/outputs/punjab_prior/figures/punjab_phase2_pilot_residuals_grouped_support_expanded_grace_weak.png)
  - [punjab_phase2_grace_comparison.csv](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase2_grace_comparison.csv)
  - [punjab_phase2_grace_comparison.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase2_grace_comparison.json)
- Next action:
  - Decide whether to strengthen or reformulate the GRACE prior, or pivot to `GPS` preparation as the next external-prior branch.

## 2026-03-14 Stronger GRACE Prior Update
- Change summary:
  - Kept the same GRACE regional-anomaly formulation and the same frozen expanded grouped-support baseline.
  - Increased the GRACE prior weight from `\lambda_g = 0.05` to `\lambda_g = 0.20`.
  - Warm-started again from `punjab_phase1_pilot_best_grouped_support_expanded.pt`.
- Stronger GRACE formulation:
  \[
  L_{phase2,GRACE}=L_{phase1}+\lambda_g L_{\mathrm{GRACE}}
  \]
  with the same regional batch-standardized term:
  \[
  L_{\mathrm{GRACE}}
  =
  \mathrm{MSE}\!\left(
  z\!\left(\overline{\hat S_0+\hat S_g}\right),
  z\!\left(g_{\mathrm{GRACE}}\right)
  \right).
  \]
- Verification result:
  - The stronger GRACE pilot ran successfully in `swin_env`.
  - Final stronger-GRACE summary:
    - `val_forward = 1.4286390`
    - normalized forward RMSE mean `0.9919191`
    - `val_grace = 1.8649`
  - Direct comparison against the frozen Phase 1 baseline:
    - baseline `val_forward = 1.4286398`
    - stronger GRACE `val_forward = 1.4286390`
    - baseline normalized forward RMSE mean `0.9919191`
    - stronger GRACE normalized forward RMSE mean `0.9919191`
- Interpretation:
  - The stronger GRACE branch stays stable.
  - It lowers the GRACE loss slightly, but still does not materially change the main deformation-fit metrics.
  - This now matches the same pattern already seen for W3RA: simple weight increases are not enough by themselves.
- Output paths:
  - [punjab_phase2_pilot_history_grouped_support_expanded_grace_stronger.csv](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase2_pilot_history_grouped_support_expanded_grace_stronger.csv)
  - [punjab_phase2_pilot_summary_grouped_support_expanded_grace_stronger.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase2_pilot_summary_grouped_support_expanded_grace_stronger.json)
  - [punjab_phase2_pilot_best_grouped_support_expanded_grace_stronger.pt](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase2_pilot_best_grouped_support_expanded_grace_stronger.pt)
  - [punjab_phase2_pilot_diagnostics_grouped_support_expanded_grace_stronger.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase2_pilot_diagnostics_grouped_support_expanded_grace_stronger.json)
  - [punjab_phase2_pilot_residuals_grouped_support_expanded_grace_stronger.png](/home/ubuntu/work/punjab/outputs/punjab_prior/figures/punjab_phase2_pilot_residuals_grouped_support_expanded_grace_stronger.png)
  - [punjab_phase2_grace_stronger_comparison.csv](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase2_grace_stronger_comparison.csv)
  - [punjab_phase2_grace_stronger_comparison.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase2_grace_stronger_comparison.json)
- Next action:
  - Stop simple GRACE weight tuning and either reformulate the GRACE term or pivot to `GPS` preparation.

## 2026-03-14 Reformulated GRACE Prior Update
- Change summary:
  - Replaced the earlier batch-standardized GRACE level-matching term with an uncertainty-weighted regional anomaly term.
  - Kept the same frozen expanded grouped-support baseline and warm-start checkpoint.
  - Used aligned GRACE basin anomalies together with aligned GRACE regional uncertainty.
- Reformulated GRACE prior:
  \[
  L_{phase2,GRACE}=L_{phase1}+\lambda_g L_{\mathrm{GRACE,uw}}
  \]
  with:
  \[
  L_{\mathrm{GRACE,uw}}
  =
  \mathrm{mean}\!\left[
  \left(
  \frac{
  (\overline{\hat S_0+\hat S_g}-\mathrm{mean}(\overline{\hat S_0+\hat S_g}))-g_{\mathrm{GRACE}}
  }{\sigma_{\mathrm{GRACE}}}
  \right)^2
  \right]
  \]
  where \(g_{\mathrm{GRACE}}\) is the aligned Punjab-region GRACE anomaly series and \(\sigma_{\mathrm{GRACE}}\) is the aligned regional GRACE uncertainty with a small floor.
- Settings:
  - `\lambda_g = 0.05`
  - learning rate `5e-5`
  - `4` epochs
  - warm start from `punjab_phase1_pilot_best_grouped_support_expanded.pt`
- Verification result:
  - The reformulated GRACE pilot ran successfully in `swin_env`.
  - Final summary:
    - `val_forward = 1.4286387`
    - normalized forward RMSE mean `0.991918`
    - `val_grace = 21.3698`
  - Direct comparison against the frozen Phase 1 baseline:
    - baseline `val_forward = 1.4286398`
    - reformulated GRACE `val_forward = 1.4286387`
    - baseline normalized forward RMSE mean `0.9919191`
    - reformulated GRACE normalized forward RMSE mean `0.991918`
- Interpretation:
  - The reformulated GRACE prior is stable.
  - It produces only a tiny numerical nudge in the main validation metrics, not a practically meaningful improvement.
  - The large residual GRACE term indicates the current regional anomaly formulation is still too weakly informative relative to the deformation objective.
- Output paths:
  - [punjab_phase2_pilot_history_grouped_support_expanded_grace_uncertainty_anom.csv](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase2_pilot_history_grouped_support_expanded_grace_uncertainty_anom.csv)
  - [punjab_phase2_pilot_summary_grouped_support_expanded_grace_uncertainty_anom.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase2_pilot_summary_grouped_support_expanded_grace_uncertainty_anom.json)
  - [punjab_phase2_pilot_best_grouped_support_expanded_grace_uncertainty_anom.pt](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase2_pilot_best_grouped_support_expanded_grace_uncertainty_anom.pt)
  - [punjab_phase2_pilot_diagnostics_grouped_support_expanded_grace_uncertainty_anom.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase2_pilot_diagnostics_grouped_support_expanded_grace_uncertainty_anom.json)
  - [punjab_phase2_pilot_residuals_grouped_support_expanded_grace_uncertainty_anom.png](/home/ubuntu/work/punjab/outputs/punjab_prior/figures/punjab_phase2_pilot_residuals_grouped_support_expanded_grace_uncertainty_anom.png)
  - [punjab_phase2_grace_reformulated_comparison.csv](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase2_grace_reformulated_comparison.csv)
  - [punjab_phase2_grace_reformulated_comparison.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase2_grace_reformulated_comparison.json)
- Next action:
  - Stop tuning the current regional GRACE family and decide between:
    - a more structural GRACE formulation, or
    - pivoting to `GPS` preparation.

## 2026-03-15 Interactive Baseline Inversion Viewer Update
- Change summary:
  - Added a compact baseline prediction archive builder for the frozen expanded grouped-support Punjab model.
  - Generated a baseline prediction archive over the six reported baseline tiles across the full Punjab time span.
  - Added a notebook launch section for an interactive map that shows pixelwise predicted `S0` and `Sg` time series on click.
- Archive formulation:
  - Model:
    \[
    (\hat S_0,\hat S_g)=f_\theta(d)
    \]
  - Archive scope:
    - six expanded grouped-support baseline tiles
    - all valid Punjab end dates
    - stored as pixelwise predicted `S0` and `Sg` time series
- Files/cells touched:
  - [punjab_inversion/punjab_prediction_viewer.py](/home/ubuntu/work/punjab/punjab_inversion/punjab_prediction_viewer.py)
  - [punjab_inversion/__init__.py](/home/ubuntu/work/punjab/punjab_inversion/__init__.py)
  - `punjab_prior_constrained_inversion.ipynb` new end-sections:
    - `## Interactive Baseline Inversion Map`
    - viewer launch code cell
- Verification result:
  - Built archive:
    - `n_times = 214`
    - `n_tiles = 6`
    - `n_supported_pixels = 6768`
    - date span `2016-10-20` to `2024-01-24`
  - Verified archive loading:
    - map reconstruction works
    - pixel-series extraction works for supported pixels
- Output paths:
  - [punjab_phase1_baseline_prediction_archive_expanded_tiles.h5](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_baseline_prediction_archive_expanded_tiles.h5)
  - [punjab_phase1_baseline_prediction_archive_expanded_tiles_summary.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_baseline_prediction_archive_expanded_tiles_summary.json)
  - [punjab_phase1_baseline_prediction_archive_expanded_tiles.png](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_baseline_prediction_archive_expanded_tiles.png)
- How to use:
  - Open [punjab_prior_constrained_inversion.ipynb](/home/ubuntu/work/punjab/punjab_prior_constrained_inversion.ipynb)
  - Go to the last section:
    - `## Interactive Baseline Inversion Map`
  - If click events are not active, run `%matplotlib widget` in a notebook cell first.
  - Run the viewer cell and click a supported pixel to plot `S0` and `Sg` through time.

## 2026-03-15 Baseline Export Files And Plotting Update
- Change summary:
  - Rebuilt the reduced baseline prediction archive with clearer metadata and timestamps.
  - Added explicit `HDF5` keys for:
    - `S0_pred`
    - `Sg_pred`
    - `time_iso`
    - `time_unix_ns`
    - `lat`
    - `lon`
  - Exported the frozen Phase 1 baseline to two `NetCDF` files:
    - one for `S0`
    - one for `Sg`
  - Added notebook cells that read the saved `nc`/`h5` files directly and produce:
    - baseline map panels
    - representative pixel time series
- Files/cells touched:
  - [punjab_inversion/punjab_prediction_viewer.py](/home/ubuntu/work/punjab/punjab_inversion/punjab_prediction_viewer.py)
  - [punjab_inversion/__init__.py](/home/ubuntu/work/punjab/punjab_inversion/__init__.py)
  - `punjab_prior_constrained_inversion.ipynb` new end-section:
    - `## Baseline Export Plots`
- Verification result:
  - `HDF5` archive keys now include clear aliases and timestamps.
  - `NetCDF` exports now include:
    - variables `S0_pred` / `Sg_pred`
    - coordinates `time`, `y`, `x`, `lat`, `lon`
    - time span `2016-10-20` to `2024-01-24`
  - Generated plotting outputs from the saved files successfully.
- Output paths:
  - [punjab_phase1_baseline_prediction_archive_expanded_tiles.h5](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_baseline_prediction_archive_expanded_tiles.h5)
  - [punjab_phase1_baseline_prediction_archive_expanded_tiles_summary.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_baseline_prediction_archive_expanded_tiles_summary.json)
  - [punjab_phase1_baseline_prediction_expanded_tiles_S0.nc](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_baseline_prediction_expanded_tiles_S0.nc)
  - [punjab_phase1_baseline_prediction_expanded_tiles_Sg.nc](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_baseline_prediction_expanded_tiles_Sg.nc)
  - [punjab_phase1_baseline_export_maps.png](/home/ubuntu/work/punjab/outputs/punjab_prior/figures/punjab_phase1_baseline_export_maps.png)
  - [punjab_phase1_baseline_export_timeseries.png](/home/ubuntu/work/punjab/outputs/punjab_prior/figures/punjab_phase1_baseline_export_timeseries.png)
  - [punjab_phase1_baseline_export_sample_pixels.csv](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_baseline_export_sample_pixels.csv)

## 2026-03-15 Paper Figure Notebook For Punjab Results
- Change summary:
  - Added a dedicated notebook to regenerate the Punjab paper figures directly from saved data and saved model outputs.
  - Added a small plotting helper module so the same figure code can be reused outside the notebook.
  - Updated the figure set so Figure 5 explicitly includes the InSAR velocity map alongside coherence, temporal support, and the final support mask.
- Files/cells touched:
  - [punjab_paper_figures.ipynb](/home/ubuntu/work/punjab/punjab_paper_figures.ipynb)
  - [punjab_inversion/paper_figures.py](/home/ubuntu/work/punjab/punjab_inversion/paper_figures.py)
  - [punjab_inversion/__init__.py](/home/ubuntu/work/punjab/punjab_inversion/__init__.py)
- Figure scope in the new notebook:
  - Figure 5:
    - Punjab data support and masking
    - includes InSAR velocity
  - Figure 6:
    - frozen expanded grouped-support baseline residual gallery
  - Figure 7:
    - Punjab baseline versus W3RA and GRACE prior ablations
  - Figure 8:
    - baseline export panel with `S0`/`Sg` mean maps, last-date maps, and representative pixel time series
- Verification result:
  - The notebook paths and helper functions were validated against:
    - raw Punjab HDF5 files
    - saved baseline archive
    - saved NetCDF exports
    - saved paper-results summary
  - Paper-facing outputs were generated successfully.
- Output paths:
  - [punjab_paper_figure5_support_mask_velocity.png](/home/ubuntu/work/punjab/outputs/punjab_prior/figures/punjab_paper_figure5_support_mask_velocity.png)
  - [punjab_paper_figure6_baseline_residual_gallery.png](/home/ubuntu/work/punjab/outputs/punjab_prior/figures/punjab_paper_figure6_baseline_residual_gallery.png)
  - [punjab_paper_figure7_prior_ablation_comparison.png](/home/ubuntu/work/punjab/outputs/punjab_prior/figures/punjab_paper_figure7_prior_ablation_comparison.png)
  - [punjab_paper_figure8_baseline_export_panel.png](/home/ubuntu/work/punjab/outputs/punjab_prior/figures/punjab_paper_figure8_baseline_export_panel.png)

## 2026-03-15 Punjab Paper Figure Notebook Revision
- Change summary:
  - Reworked the paper-figure notebook so the figure-generation functions now live inside the notebook itself rather than only in an imported helper module.
  - Added editable top-level unit-label constants for:
    - velocity
    - deformation
    - normalized deformation
    - storage
  - Regenerated Figure 6 inside the notebook from saved predictions plus the two-layer forward operator instead of just copying an existing PNG.
  - Reframed Figure 7 as a null-result prior-ablation figure using differences relative to the frozen baseline.
  - Improved Figure 8 by cropping to the active supported domain and labeling the storage maps and time series explicitly as relative model-unit outputs.
- Files/cells touched:
  - [punjab_paper_figures.ipynb](/home/ubuntu/work/punjab/punjab_paper_figures.ipynb)
- Verification result:
  - The updated notebook code was executed directly in Python to regenerate all four Punjab paper figures.
  - Inline display calls were added so the figures appear directly inside the notebook when run interactively.
  - Jupyter kernel execution could not be run inside the sandbox because local kernel ports are blocked, but the notebook code itself was run successfully outside the Jupyter execution path.
- Output paths refreshed:
  - [punjab_paper_figure5_support_mask_velocity.png](/home/ubuntu/work/punjab/outputs/punjab_prior/figures/punjab_paper_figure5_support_mask_velocity.png)
  - [punjab_paper_figure6_baseline_residual_gallery.png](/home/ubuntu/work/punjab/outputs/punjab_prior/figures/punjab_paper_figure6_baseline_residual_gallery.png)
  - [punjab_paper_figure7_prior_ablation_comparison.png](/home/ubuntu/work/punjab/outputs/punjab_prior/figures/punjab_paper_figure7_prior_ablation_comparison.png)
  - [punjab_paper_figure8_baseline_export_panel.png](/home/ubuntu/work/punjab/outputs/punjab_prior/figures/punjab_paper_figure8_baseline_export_panel.png)

## 2026-03-24 Punjab Source Comparison Panels
- Change summary:
  - Added a dedicated notebook to generate cleaner source-comparison figures for Punjab with explicit unit labels and a more readable InSAR velocity stretch.
  - Added a reusable helper module for direct HDF5/NetCDF-HDF5 loading, robust symmetric color limits, latest-slice extraction, and comparison panel generation.
  - Generated a 6-panel comparison map figure covering:
    - InSAR velocity
    - GRACE TWS anomaly
    - W3RA `S0` anomaly
    - W3RA `Sg` anomaly
    - inversion `S0`
    - inversion `Sg`
  - Generated a companion time-series figure covering:
    - GRACE basin-mean TWS anomaly
    - W3RA basin-mean `S0` and `Sg` anomalies
    - support-area mean inversion `S0` and `Sg`
- Files/cells touched:
  - [punjab_source_comparison_panels.ipynb](/home/ubuntu/work/punjab/punjab_source_comparison_panels.ipynb)
  - [punjab_inversion/comparison_figures.py](/home/ubuntu/work/punjab/punjab_inversion/comparison_figures.py)
  - [punjab_inversion/__init__.py](/home/ubuntu/work/punjab/punjab_inversion/__init__.py)
  - [tools/create_punjab_source_comparison_notebook.py](/home/ubuntu/work/punjab/tools/create_punjab_source_comparison_notebook.py)
- Verification result:
  - The comparison figures were generated successfully outside Jupyter using the same helper functions used by the notebook.
  - The InSAR velocity stretch was set with a robust symmetric limit based on the 95th percentile of the absolute finite values.
  - Latest aligned dates used by the map panel are all `2024-01-24` for GRACE, W3RA, and inversion outputs.
- Output paths:
  - [punjab_source_comparison_maps.png](/home/ubuntu/work/punjab/outputs/punjab_prior/figures/punjab_source_comparison_maps.png)
  - [punjab_source_comparison_timeseries.png](/home/ubuntu/work/punjab/outputs/punjab_prior/figures/punjab_source_comparison_timeseries.png)
  - [punjab_source_comparison_maps_summary.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_source_comparison_maps_summary.json)
  - [punjab_source_comparison_timeseries_summary.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_source_comparison_timeseries_summary.json)
  - [Fig10_punjab_source_comparison_maps.png](/home/ubuntu/work/punjab/paper_figures/Fig10_punjab_source_comparison_maps.png)
  - [Fig11_punjab_source_comparison_timeseries.png](/home/ubuntu/work/punjab/paper_figures/Fig11_punjab_source_comparison_timeseries.png)

## 2026-03-24 Re-export Baseline Inversion With All Tile Values
- Change summary:
  - Verified that the previous `S0`/`Sg` NetCDF exports were sparse because the saved products themselves were clipped to the support mask rather than only being masked at plot time.
  - Extended the baseline archive/export path so the selected expanded grouped-support tiles can be re-exported while keeping all tile values instead of only the support-mask subset.
  - Preserved the original coherence/temporal support footprint separately inside the archive as `source_support_mask`.
  - Regenerated the Punjab source-comparison notebook outputs against the new all-values exports.
- Files/cells touched:
  - [punjab_inversion/punjab_prediction_viewer.py](/home/ubuntu/work/punjab/punjab_inversion/punjab_prediction_viewer.py)
  - [tools/reexport_expanded_tiles_all_values.py](/home/ubuntu/work/punjab/tools/reexport_expanded_tiles_all_values.py)
  - [punjab_source_comparison_panels.ipynb](/home/ubuntu/work/punjab/punjab_source_comparison_panels.ipynb)
- Verification result:
  - Old export finite footprint fraction: `0.0032375`
  - New all-values export finite footprint fraction: `0.0117560`
  - The re-export therefore preserves about `3.63x` more inversion values on the same six reported baseline tiles.
  - The regenerated comparison maps and panel figures now read from the all-values NetCDF exports instead of the older support-clipped exports.
- Output paths:
  - [punjab_phase1_baseline_prediction_archive_expanded_tiles_all_values.h5](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_baseline_prediction_archive_expanded_tiles_all_values.h5)
  - [punjab_phase1_baseline_prediction_archive_expanded_tiles_all_values_summary.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_baseline_prediction_archive_expanded_tiles_all_values_summary.json)
  - [punjab_phase1_baseline_prediction_expanded_tiles_all_values_S0.nc](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_baseline_prediction_expanded_tiles_all_values_S0.nc)
  - [punjab_phase1_baseline_prediction_expanded_tiles_all_values_Sg.nc](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_baseline_prediction_expanded_tiles_all_values_Sg.nc)
  - [punjab_source_comparison_maps.png](/home/ubuntu/work/punjab/outputs/punjab_prior/figures/punjab_source_comparison_maps.png)
  - [punjab_source_comparison_timeseries.png](/home/ubuntu/work/punjab/outputs/punjab_prior/figures/punjab_source_comparison_timeseries.png)
  - [velocity.png](/home/ubuntu/work/punjab/outputs/punjab_prior/figures/punjab_source_comparison_panels/velocity.png)
  - [grace_tws.png](/home/ubuntu/work/punjab/outputs/punjab_prior/figures/punjab_source_comparison_panels/grace_tws.png)
  - [w3ra_s0.png](/home/ubuntu/work/punjab/outputs/punjab_prior/figures/punjab_source_comparison_panels/w3ra_s0.png)
  - [w3ra_sg.png](/home/ubuntu/work/punjab/outputs/punjab_prior/figures/punjab_source_comparison_panels/w3ra_sg.png)
  - [inversion_s0.png](/home/ubuntu/work/punjab/outputs/punjab_prior/figures/punjab_source_comparison_panels/inversion_s0.png)
  - [inversion_sg.png](/home/ubuntu/work/punjab/outputs/punjab_prior/figures/punjab_source_comparison_panels/inversion_sg.png)

## 2026-03-24 Re-export Baseline Inversion Over All Valid Tiles
- Change summary:
  - Extended the all-values export from the six reported baseline tiles to the full set of `65` valid tiles under the current Punjab support/tile configuration.
  - Updated the source-comparison notebook so the inversion panels now read from the all-tile NetCDFs rather than the smaller six-tile export.
  - Regenerated the comparison composite, the companion time-series figure, and the per-panel PNG outputs for manuscript use.
- Verification result:
  - Six-tile all-values finite footprint fraction: `0.0117560`
  - All-tile all-values finite footprint fraction: `0.1273571`
  - The full-tile re-export therefore preserves about `10.83x` more inversion values than the earlier six-tile all-values export.
- Files/cells touched:
  - [tools/reexport_all_tiles_all_values.py](/home/ubuntu/work/punjab/tools/reexport_all_tiles_all_values.py)
  - [punjab_source_comparison_panels.ipynb](/home/ubuntu/work/punjab/punjab_source_comparison_panels.ipynb)
  - [tools/create_punjab_source_comparison_notebook.py](/home/ubuntu/work/punjab/tools/create_punjab_source_comparison_notebook.py)
- Output paths:
  - [punjab_phase1_baseline_prediction_archive_all_tiles_all_values.h5](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_baseline_prediction_archive_all_tiles_all_values.h5)
  - [punjab_phase1_baseline_prediction_archive_all_tiles_all_values_summary.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_baseline_prediction_archive_all_tiles_all_values_summary.json)
  - [punjab_phase1_baseline_prediction_all_tiles_all_values_S0.nc](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_baseline_prediction_all_tiles_all_values_S0.nc)
  - [punjab_phase1_baseline_prediction_all_tiles_all_values_Sg.nc](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_baseline_prediction_all_tiles_all_values_Sg.nc)
  - [punjab_source_comparison_maps.png](/home/ubuntu/work/punjab/outputs/punjab_prior/figures/punjab_source_comparison_maps.png)
  - [punjab_source_comparison_timeseries.png](/home/ubuntu/work/punjab/outputs/punjab_prior/figures/punjab_source_comparison_timeseries.png)
  - [Fig10_punjab_source_comparison_maps.png](/home/ubuntu/work/punjab/paper_figures/Fig10_punjab_source_comparison_maps.png)
  - [Fig11_punjab_source_comparison_timeseries.png](/home/ubuntu/work/punjab/paper_figures/Fig11_punjab_source_comparison_timeseries.png)

## 2026-03-24 Full-Scene No-Mask Latest-Date Map Export
- Change summary:
  - Added a dedicated latest-date export that runs the baseline model over the full tiling of the Punjab scene, including edge tiles, without applying the coherence-derived support mask to the inputs.
  - Kept the broader all-tile time-stack export for time-series comparison, but switched the source-comparison map notebook to use the new full-scene latest-date export for the inversion map panels.
  - This gives a full-domain visualization for `S0` and `Sg` map panels while preserving the longer time-range comparison series in the lower figure.
- Verification result:
  - The latest-date full-scene no-mask export has finite coverage fraction `1.0` in the exported `S0` cube.
  - The refreshed comparison map reports inversion footprint rows `[0, 1130]` and columns `[0, 1850]`, so the inversion panels now span the full map domain.
- Files/cells touched:
  - [tools/reexport_full_scene_no_mask_latest_only.py](/home/ubuntu/work/punjab/tools/reexport_full_scene_no_mask_latest_only.py)
  - [punjab_inversion/punjab_prediction_viewer.py](/home/ubuntu/work/punjab/punjab_inversion/punjab_prediction_viewer.py)
  - [tools/create_punjab_source_comparison_notebook.py](/home/ubuntu/work/punjab/tools/create_punjab_source_comparison_notebook.py)
  - [punjab_source_comparison_panels.ipynb](/home/ubuntu/work/punjab/punjab_source_comparison_panels.ipynb)
- Output paths:
  - [punjab_phase1_baseline_prediction_full_scene_no_mask_latest_only.h5](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_baseline_prediction_full_scene_no_mask_latest_only.h5)
  - [punjab_phase1_baseline_prediction_full_scene_no_mask_latest_only_S0.nc](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_baseline_prediction_full_scene_no_mask_latest_only_S0.nc)
  - [punjab_phase1_baseline_prediction_full_scene_no_mask_latest_only_Sg.nc](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_baseline_prediction_full_scene_no_mask_latest_only_Sg.nc)
  - [punjab_source_comparison_maps.png](/home/ubuntu/work/punjab/outputs/punjab_prior/figures/punjab_source_comparison_maps.png)
  - [inversion_s0.png](/home/ubuntu/work/punjab/outputs/punjab_prior/figures/punjab_source_comparison_panels/inversion_s0.png)
  - [inversion_sg.png](/home/ubuntu/work/punjab/outputs/punjab_prior/figures/punjab_source_comparison_panels/inversion_sg.png)

## 2026-03-25 Seasonal Maps And Two Pixel Time-Series Block
- Change summary:
  - Added a new visualization block for the source-comparison notebook that shows `S0` and `Sg` over three seasonal dates together with two pixelwise time series.
  - The seasonal map panels use a full-scene no-mask export at three dates: `2023-03-30`, `2023-07-16`, and `2023-11-13`.
  - The two time series use a compact no-mask archive built only for the tiles containing the chosen pixels:
    - the highest-coherence pixel in the static coherence raster
    - the nearest grid point to `N31.229126°, E76.564970°`
- Verification result:
  - Seasonal `S0` and `Sg` exports each have shape `(3, 1130, 1850)`.
  - The two-pixel time-series export includes both requested labels:
    - `high_coherence`
    - `river_reference`
  - Pixel coordinates used are:
    - high coherence: `lat=31.378611`, `lon=76.035069`
    - river reference: `lat=31.229167`, `lon=76.564931`
- Files/cells touched:
  - [tools/reexport_full_scene_no_mask_seasonal_dates.py](/home/ubuntu/work/punjab/tools/reexport_full_scene_no_mask_seasonal_dates.py)
  - [tools/export_two_pixel_timeseries_no_mask.py](/home/ubuntu/work/punjab/tools/export_two_pixel_timeseries_no_mask.py)
  - [tools/create_punjab_source_comparison_notebook.py](/home/ubuntu/work/punjab/tools/create_punjab_source_comparison_notebook.py)
  - [punjab_source_comparison_panels.ipynb](/home/ubuntu/work/punjab/punjab_source_comparison_panels.ipynb)
- Output paths:
  - [punjab_phase1_baseline_prediction_full_scene_no_mask_seasonal_dates_S0.nc](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_baseline_prediction_full_scene_no_mask_seasonal_dates_S0.nc)
  - [punjab_phase1_baseline_prediction_full_scene_no_mask_seasonal_dates_Sg.nc](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_baseline_prediction_full_scene_no_mask_seasonal_dates_Sg.nc)
  - [punjab_phase1_two_pixel_timeseries_no_mask.csv](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_two_pixel_timeseries_no_mask.csv)
  - [punjab_phase1_two_pixel_timeseries_no_mask_summary.json](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_two_pixel_timeseries_no_mask_summary.json)

## 2026-03-25 Interactive Velocity Map For Clicked Inversion Time Series
- Change summary:
  - Added a new interactive section to the Punjab source-comparison notebook with the InSAR velocity map as the clickable base layer.
  - On click, the notebook now updates three linked quantities at the selected location:
    - predicted `S0`
    - predicted `Sg`
    - locally forward-modeled deformation from the saved inversion outputs
  - The same panel also shows the observed deformation time series at the clicked pixel for context.
  - The notebook generator metadata was updated to target the `swin_env` kernel because the interactive deformation reconstruction uses `torch`.
- Implementation notes:
  - The new cell prefers the full-scene multi-date archive `punjab_phase1_baseline_prediction_full_scene_no_mask_all_values.h5` when available.
  - If that archive is not present yet, it falls back to [punjab_phase1_baseline_prediction_archive_all_tiles_all_values.h5](/home/ubuntu/work/punjab/outputs/punjab_prior/punjab_phase1_baseline_prediction_archive_all_tiles_all_values.h5).
  - In fallback mode, clicks outside the stored multi-date footprint snap to the nearest archived pixel and the status text reports that explicitly.
  - The deformation series is computed by applying the existing two-layer forward model over a local window centered on the clicked pixel.
- Verification result:
  - The interactive cell was regenerated into [punjab_source_comparison_panels.ipynb](/home/ubuntu/work/punjab/punjab_source_comparison_panels.ipynb).
  - The local forward reconstruction path was sanity-checked against the fallback archive and returns non-trivial modeled deformation values.
- Files/cells touched:
  - [tools/create_punjab_source_comparison_notebook.py](/home/ubuntu/work/punjab/tools/create_punjab_source_comparison_notebook.py)
  - [punjab_source_comparison_panels.ipynb](/home/ubuntu/work/punjab/punjab_source_comparison_panels.ipynb)
