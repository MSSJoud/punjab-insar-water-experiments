# Punjab Prior-Constrained Inversion Stepwise Plan

## Scope
- Active notebook: `punjab_prior_constrained_inversion.ipynb`
- Shared code package: `punjab_inversion/`
- Phase 1 goal: build and freeze the first Punjab inversion using only generic priors
- Phase 2 goal: add external priors in a controlled sequence on top of the frozen Phase 1 baseline
- External prior branches now touched:
  - `W3RA`
  - `GRACE`
- Remaining external prior branch:
  - `GPS`

## Current strategy
Use a two-signal inversion with the clean-anchor synthetic architecture:
- model: dual-decoder frequency-separated SWIN-style network
- physics: elastic + poroelastic forward operator
- loss:
  - forward consistency
  - spatial regularization
  - temporal regularization

Formally:

\[
L_{phase1} =
\lambda_f L_{fwd}
+ \lambda_{s0} L_{spatial}^{S_0}
+ \lambda_{sg} L_{spatial}^{S_g}
+ \lambda_{t0} L_{temporal}^{S_0}
+ \lambda_{tg} L_{temporal}^{S_g}
\]

## Phase 1 execution blocks

### 1. Punjab data wiring
1. Load the Punjab deformation stack from `/mnt/data/aoi_punjab/`.
2. Read:
   - `disp_all_ll.h5`
   - `coh_ll.h5`
   - `vel_ll.h5`
   - `aquisition_dates_ll.h5`
3. Confirm:
   - shapes
   - date span
   - grid extents
   - valid-data coverage
4. Keep the first loader metadata-first to avoid pulling the full cube into memory unnecessarily.

### 2. Grid and split definition
1. Define the canonical Punjab grid from the HDF5 stack.
2. Build the time axis from acquisition dates.
3. Choose chronological train/validation/test splits.
4. Decide whether the inversion target is:
   - monthly windows resampled from acquisition dates, or
   - native acquisition-time windows.

### 3. Phase-1 prior setup
1. Implement `L_fwd` from the two-layer forward operator.
2. Implement spatial regularization on `S0` and `Sg`.
3. Implement temporal smoothness between consecutive predictions.
4. Keep all external-prior weights at zero in this phase.

### 4. Punjab dataloader construction
1. Build windowed tensors from deformation time series.
2. Add masks/coherence handling if needed.
3. Decide whether to:
   - use only deformation as input, or
   - include a simple noise proxy channel later.
4. Start with the plain dual-decoder model unless a real noise proxy is defined.

### 5. First inversion run
1. Build FFT kernels for the Punjab grid.
2. Run the phase-1 prior-constrained loss.
3. Save:
   - checkpoints
   - configuration
   - predicted fields
   - forward-reconstructed deformation
   - residual maps/time series

### 6. Evaluation
1. Summarize deformation reconstruction quality.
2. Inspect spatial smoothness and temporal continuity of predicted fields.
3. Export:
   - maps
   - time series
   - residual figures
   - JSON/CSV summaries

## Exit criteria for Phase 1
Phase 1 is considered complete when:
1. The Punjab data loader is stable and reproducible.
2. The first generic-prior inversion runs end-to-end.
3. Output artifacts are exported to `outputs/punjab_prior/`.
4. We have a clean baseline before adding GRACE/W3RA/GPS.

## Phase 2 external priors
After Phase 1 is frozen, add external priors in this order:
1. `W3RA` as weak hydrologic structure prior
2. `GRACE` as coarse TWS prior
3. `GPS` as deformation-space constraint or validation

Current Phase 2 status:
1. `W3RA` weak hook: complete and stable tie
2. `W3RA` stronger hook: complete and stable tie
3. `GRACE` file check: complete
4. `GRACE` alignment to Punjab dates: complete
5. `GRACE` weak regional prior hook: complete and stable tie
6. `GRACE` stronger regional prior hook: complete and stable tie
7. `GRACE` reformulated uncertainty-weighted regional anomaly hook: complete and near-tie
8. `GPS`: not started

## Immediate next tasks
1. Freeze the expanded grouped-support pilot as the Phase 1 Punjab baseline.
2. Treat the fuller run, fuller warm-start, and direct reliability-weighted branch as negative follow-up results.
3. Treat the rebalanced warm-start branch as a stable tie, not a new best baseline.
4. Keep the new W3RA-preparation artifacts as the first Phase 2-ready external-prior package.
5. Treat the weak `W3RA` prior hook as Phase 2 proof of integration:
   - stable
   - no meaningful change yet in the main forward-fit metrics
6. Treat the stronger `W3RA` prior hook as the same conclusion at higher weight:
   - stable
   - still no meaningful change in the main forward-fit metrics
7. Treat the GRACE alignment package as complete:
   - the local JPL mascon file is readable
   - Punjab-region GRACE has been aligned to Punjab dates
   - aligned anomaly series and figures exist
8. Treat the weak `GRACE` prior hook as the second external-prior proof of integration:
   - stable
   - no meaningful change yet in the main forward-fit metrics
9. Treat the stronger `GRACE` prior hook as the same conclusion at higher weight:
   - stable
   - still no meaningful change in the main forward-fit metrics
10. Treat the reformulated GRACE anomaly hook as a slightly more informative but still near-tie result:
   - stable
   - not yet a practically meaningful improvement
11. Next technical choice:
   - design a more structural GRACE formulation, or
   - pivot to `GPS` preparation
12. Visualization support now exists for paper exploration:
   - compact baseline prediction archive over the six reported baseline tiles
   - interactive notebook map with pixelwise `S0`/`Sg` time series on click
13. Standard analysis exports now also exist:
   - `HDF5` archive with clear keys and timestamps
   - separate `NetCDF` files for `S0` and `Sg`
   - notebook cells that plot maps and representative pixel time series directly from the saved files
14. A dedicated Punjab paper-figure notebook now exists:
   - [punjab_paper_figures.ipynb](/home/ubuntu/work/punjab/punjab_paper_figures.ipynb)
   - regenerates Figure 5 through Figure 8 from saved files
   - Figure 5 explicitly includes the InSAR velocity layer
15. A dedicated Punjab source-comparison notebook now exists:
   - [punjab_source_comparison_panels.ipynb](/home/ubuntu/work/punjab/punjab_source_comparison_panels.ipynb)
   - generates a cleaner InSAR velocity panel with editable color stretch
   - generates comparison maps for GRACE, W3RA `S0`/`Sg`, and inversion `S0`/`Sg`
   - generates companion basin-mean/support-mean time-series comparisons
   - copies the resulting figures into `paper_figures/` for manuscript use
16. A denser re-export of the baseline inversion now exists:
   - keeps all values on the six expanded grouped-support baseline tiles
   - preserves the original source support separately for provenance
   - provides updated `HDF5` and `NetCDF` outputs for `S0` and `Sg`
   - feeds the refreshed source-comparison notebook and paper figures
17. A full valid-tile re-export now exists for figure work:
   - covers all `65` valid tiles instead of the six pilot tiles
   - produces denser `S0`/`Sg` inversion maps for Punjab paper figures
   - now underlies the refreshed source-comparison notebook outputs
18. A full-scene no-mask latest-date export now exists for map panels:
   - covers the full tiling of the Punjab scene including edge tiles
   - does not apply the coherence-derived support mask during inference export
   - now drives the source-comparison inversion map panels while time-series still use the broader multi-date all-tile export
19. The source-comparison notebook now also includes a seasonal and pointwise inspection block:
   - three seasonal full-scene `S0`/`Sg` maps
   - one time series from the highest-coherence area
   - one time series from the nearest grid point to the requested river-reference coordinate
20. The source-comparison notebook now also includes an interactive velocity-map inspection cell:
   - click on the InSAR velocity map
   - update `S0` and `Sg` time series at the selected location
   - compute a local forward-modeled deformation series from the inversion outputs
   - compare that modeled deformation with the observed deformation at the same pixel
   - automatically fall back to the all-tile multi-date archive when the full-scene multi-date archive is not present yet
