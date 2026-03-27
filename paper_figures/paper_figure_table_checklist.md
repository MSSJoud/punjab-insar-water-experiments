# Paper Figures And Tables Checklist

This folder collects the current paper-facing figure files and table source files.

## Main Figures

### Figure 1. Synthetic Conditioning Problem
- Section: Synthetic Experimental Design
- Subsection: Conditioning of the synthetic forward model
- File: `Fig01_synthetic_conditioning_problem.png`
- Purpose: show the elastic-versus-poroelastic magnitude mismatch that made the original synthetic setup poorly conditioned.
- Main message: balancing the forward contributions was necessary before a meaningful two-signal inversion could be tested.
- Draft caption: Conditioning diagnostic for the original synthetic forward model. Panel A shows the elastic, poroelastic, and total deformation RMS values per synthetic month on a logarithmic scale so that both component magnitudes remain visible. Panel B summarizes the median RMS contrast, showing that the poroelastic contribution exceeds the elastic contribution by several orders of magnitude. This strong scale imbalance motivated the balanced synthetic benchmark used in the subsequent experiments.

### Figure 2. Compact 3D Swin Dual-Decoder Architecture
- Section: Method
- Subsection: Network architecture
- File: manual figure, not copied here yet
- Purpose: show the frequency split, dual patch embedding, shared encoder/bottleneck, and dual decoders for `S0` and `Sg`.
- Main message: the final model is a compact 3D Swin U-Net with a shared encoder and separate output branches.
- Note: the architecture diagram was fixed manually by the user and should be inserted here once its final file is saved.
- Draft caption: Overview of the compact 3D Swin Transformer U-Net used in this study. The model decomposes the deformation input into low-frequency and residual branches, embeds each branch using spatiotemporal patches, processes the tokens with a shared encoder and bottleneck, and then splits into two decoders that predict `S0` and `Sg`. The architecture preserves a shared latent representation while allowing component-specific decoding.

### Figure 3. Synthetic Architecture Progression
- Section: Synthetic Experiments
- Subsection: Clean-case model selection
- File: `Fig03_synthetic_architecture_progression.png`
- Purpose: compare the clean synthetic baselines and show why the dual-decoder frequency-separated model was chosen.
- Main message: decoder specialization and frequency-aware inputs improved recoverability in the balanced synthetic benchmark.
- Draft caption: Clean synthetic benchmark comparison for the main architecture variants. The frequency-separated dual-decoder model produced the strongest overall recovery in the balanced two-signal benchmark, supporting its selection as the clean synthetic reference architecture.

### Figure 4. Synthetic Best Robustness Branch
- Section: Synthetic Experiments
- Subsection: Noise robustness
- File: `Fig04_synthetic_best_robustness_branch.png`
- Purpose: compare the best noise-aware branch against the synthetic baselines.
- Main message: hybrid conditioning with noisy-stage `Sg` emphasis produced the strongest single-run noisy synthetic result.
- Draft caption: Comparison of the best noise-aware synthetic branch against the main baselines. Hybrid direct conditioning with noisy-stage `Sg` emphasis yielded the strongest single-run improvement under noisy synthetic conditions, particularly for groundwater recovery at moderate noise.

### Figure 5. Synthetic Multi-Seed Validation
- Section: Synthetic Results
- Subsection: Stability under noise
- File: `Fig05_synthetic_multiseed_validation.png`
- Purpose: summarize mean and spread across seeds for the best synthetic robustness branch.
- Main message: noisy synthetic gains were real in the best run but did not remain robust across seeds.
- Draft caption: Multi-seed validation of the final synthetic robustness branch. Although the best run showed meaningful gains under noise, the spread across seeds remained substantial, indicating that noisy recovery is still unstable and the inverse problem remains weakly conditioned in the noisy regime.

### Figure 6A. Punjab Study Region
- Section: Punjab Real-Data Inversion Setup
- Subsection: Study area and analysis domain
- File: `Fig06a_punjab_study_region.png`
- Purpose: show the Punjab test bed within the broader regional crop and document the exact analysis box used for the InSAR-driven inversion.
- Main message: the real-data experiments focus on a sub-area test bed in Punjab, and the satellite-backed analysis extent is tightly defined and reproducible.
- Draft caption: Study region used as the Punjab real-data test bed. Panel A shows a regional locator map with country borders and the Punjab test bed highlighted in red. Panel B shows a Sentinel-2 true-color image for the same test bed, with the exact InSAR and inversion analysis box overlaid. The analysis extent spans \([75.5972, 76.8819]^\circ\)E and \([31.1222, 31.7500]^\circ\)N and serves as the common spatial reference for the real-data inversion figures.

### Figure 6. Punjab Data Support And Masking
- Section: Punjab Real-Data Inversion Setup
- Subsection: Data support and spatial masking
- File: `Fig06_punjab_support_and_masking.png`
- Purpose: show the InSAR velocity field, coherence/support diagnostics, and the spatial support used by the inversion.
- Main message: the inversion operates only on a limited, coherence-supported subset of the Punjab domain.
- Draft caption: Punjab InSAR data-quality and support diagnostics. Panel A shows the InSAR linear-rate field with a robust percentile-based stretch for readability. Panel B shows the static coherence raster using an actual-value display range clipped to the 5th to 99th percentiles, which makes the low-to-moderate coherence structure visible without altering the underlying values. Panel C shows the temporal observation fraction, and Panel D shows the binary support mask used during the baseline training stage. The mask itself still uses the fixed thresholds described in the text; only the coherence display range was tightened for visualization.

### Figure 7. Punjab Baseline Residual Gallery
- Section: Punjab Baseline Results
- Subsection: Local inversion behavior
- File: `Fig07_punjab_baseline_residual_gallery.png`
- Purpose: show observed deformation, residual, predicted `S0`, and predicted `Sg` for selected validation examples.
- Main message: the baseline produces structured latent fields and a deformation-consistent real-data inversion, while residual structure remains.
- Draft caption: Selected validation examples from the frozen Punjab baseline inversion. For each example, the observed deformation, residual field, and predicted latent components `S0` and `Sg` are shown. The baseline produces structured latent fields and a deformation-consistent inversion, although residual structure remains in the real-data fit.

### Figure 8. Punjab Prior-Ablation Null Result
- Section: External-Prior Ablations
- Subsection: W3RA and GRACE tests
- File: `Fig08_punjab_prior_ablation_null_result.png`
- Purpose: compare the baseline against W3RA and GRACE ablations.
- Main message: the tested external-prior formulations were stable but remained effectively tied with the baseline on the main validation metrics.
- Draft caption: Comparison of the Punjab internal-prior baseline against the tested W3RA and GRACE ablations. The external-prior variants were technically stable but remained effectively tied with the baseline on the main validation metrics, indicating that the tested formulations did not materially improve the real-data inversion.

### Figure 9. Punjab Seasonal Baseline Maps And Reference Time Series
- Section: Punjab Baseline Results
- Subsection: Baseline inversion outputs
- File: `Fig09_punjab_baseline_export_panel.png`
- Purpose: show the seasonal evolution of the baseline inversion over three dates together with two representative pointwise storage time series.
- Main message: the Punjab baseline produces spatially coherent seasonal variations in both `S0` and `Sg`, and the temporal behavior can be inspected at a high-coherence location and a river-adjacent location.
- Draft caption: Seasonal visualization of the Punjab baseline inversion. The first row shows predicted `S_0` and the second row shows predicted `S_g` for three seasonal dates (`2023-03-30`, `2023-07-16`, `2023-11-13`) from the full-scene no-mask export, with shared row-wise colorbars in meters of equivalent water height. Markers indicate the two reference pixels used for the lower panels: a high-coherence location and a river-adjacent location nearest to `N31.229126°, E76.564970°` with non-trivial temporal variability. The lower panels show the corresponding `S_0` and `S_g` time series, illustrating how the baseline latent-storage fields evolve seasonally at representative locations.

## Main Tables

### Table 1. Synthetic Experiment Ladder Summary
- Section: Synthetic Experiments
- Subsection: Experiment ladder
- File: `Tab01_synthetic_experiment_ladder_source.csv`
- Purpose: summarize the main synthetic model progression.
- Suggested columns: experiment, main change, representative clean metric, representative noisy metric, conclusion.
- Main message: the synthetic phase was a structured model-selection process rather than a single benchmark run.
- Draft caption: Summary of the main synthetic experiment ladder. Each row reports a representative model variant, the primary architectural or training change introduced at that stage, and the corresponding clean and noisy benchmark behavior. The table highlights the progression from baseline inversion to the final selected synthetic model family.

### Table 2. Synthetic Multi-Seed Summary
- Section: Synthetic Results
- Subsection: Stability under noise
- File: `Tab02_synthetic_multiseed_source.csv`
- Purpose: report mean and spread across seeds for the final synthetic robustness branch.
- Suggested columns: noise level, layer, mean `R^2`, mean correlation, mean slope, standard deviation.
- Main message: noisy synthetic recovery remains fragile across seeds.
- Draft caption: Multi-seed summary for the final synthetic robustness branch. Mean and spread across seeds are reported by noise level and latent component. The table shows that, despite the best-run gains, noisy recovery remains sensitive to initialization and optimization.

### Table 3. Punjab Baseline And Prior-Ablation Summary
- Section: Punjab Baseline Results / External-Prior Ablations
- Subsection: Baseline and ablation comparison
- File: `Tab03_punjab_baseline_prior_ablation_source.csv`
- Purpose: report the Punjab baseline and the tested W3RA/GRACE ablations.
- Suggested columns: run label, validation forward loss, normalized forward RMSE, extra-prior term, interpretation.
- Main message: the internal-prior baseline is the best current Punjab result, while the tested external priors remained near-null.
- Draft caption: Summary of the Punjab baseline inversion and the tested W3RA and GRACE prior ablations. The internal-prior baseline is the best current real-data result, while the external-prior variants remained stable but did not materially improve the principal validation metrics.

## Optional / Supplementary Table

### Table S1. Punjab Sample Pixels Used In Export Panel
- Section: Supplement
- Subsection: Visualization details
- File: `TabS01_punjab_sample_pixels_source.csv`
- Purpose: document the representative pixels used in the Punjab baseline export panel.
- Main message: the reported time series are reproducible and tied to fixed sampled locations.
- Draft caption: Coordinates and identifiers for the representative pixels used in the Punjab baseline export panel. This table documents the locations corresponding to the reported example time series.

## Suggested Section Mapping

1. Method
   - Figure 2
2. Synthetic Experimental Design
   - Figure 1
3. Synthetic Experiments
   - Figure 3
   - Figure 4
   - Table 1
4. Synthetic Results
   - Figure 5
   - Table 2
5. Punjab Real-Data Inversion Setup
   - Figure 6A
   - Figure 6
6. Punjab Baseline Results
   - Figure 7
   - Figure 9
7. External-Prior Ablations
   - Figure 8
   - Table 3
