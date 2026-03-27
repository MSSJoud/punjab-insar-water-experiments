# Paper Writeup Ready

## Readiness Statement

Yes, the project is ready for paper writing with the following scope.

- The synthetic part is ready as a completed model-development and validation section.
- The Punjab part is ready as a real-data baseline inversion section plus prior-ablation results.
- The paper should not claim that the final prior-constrained real-data solution is fully solved.
- The paper can claim that:
  - the synthetic study identified a viable inversion architecture and clarified the conditioning problem,
  - the Punjab pipeline runs end-to-end on real data,
  - the best current Punjab result is the internal-prior baseline,
  - the tested `W3RA` and `GRACE` prior formulations were technically valid but did not materially improve the baseline.

## Research Question

The paper addresses the following central question:

\[
\text{Can surface deformation be inverted to recover separable hydrologic storage signals in a physics-aware learning framework?}
\]

In the synthetic setting, the problem is posed as recovery of two latent components from deformation:

\[
(\hat S_0,\hat S_g) = f_\theta(d),
\]

with forward consistency enforced through

\[
\hat d = G_{\mathrm{load}}(\hat S_0) + G_{\mathrm{poro}}(\hat S_g).
\]

The synthetic experiments answer whether this inversion is learnable under controlled conditions, which architecture is best, and how robustness degrades under noise.

The Punjab section then asks a more practical question:

\[
\text{Can the same inversion framework be carried to real deformation data with internal regularization and external-prior ablations?}
\]

## Objective Of The Synthetic Phase

The objective of the synthetic phase was not to produce the final applied result, but to create a controlled environment in which the inverse problem could be understood before moving to Punjab real data.

More specifically, the synthetic simulations were designed to answer four questions:

1. Is the two-signal inversion learnable at all when the true latent fields are known?
2. Is the inverse problem fundamentally failing because of non-identifiability, or because of poor conditioning and model choice?
3. Which model architecture is the strongest candidate for recovering both \(S_0\) and \(S_g\)?
4. How much of the synthetic success survives once realistic noise is introduced?

The reason this stage was necessary is that real Punjab data do not provide direct ground truth for the hidden storage components. Without a synthetic benchmark, it would be impossible to tell whether a failure on real data came from:

- a fundamentally ill-conditioned inverse problem,
- a poor forward-model scaling,
- an unsuitable network architecture,
- or insufficient robustness to noise.

Accordingly, the synthetic phase served as a model-selection and identifiability phase. It allowed us to test the inversion under known truth, diagnose why some formulations failed, and select the best inversion strategy before carrying the method to real data.

In short, the synthetic phase had the following objective:

\[
\text{use controlled simulations to establish identifiability, select the best inversion model, and define the limits of unconstrained recovery before real-data deployment.}
\]

## What We Did To Answer The Question

## 1. Synthetic Forward-Model Design

We first built a controlled synthetic benchmark with known latent storage fields and known deformation physics. The two-component forward model is:

\[
d = G_{\mathrm{load}}(S_0) + G_{\mathrm{poro}}(S_g) + \varepsilon.
\]

The first important finding was that the original synthetic setup was poorly conditioned because the elastic and poroelastic responses differed strongly in magnitude. To make the inversion meaningful, we introduced a balanced synthetic formulation:

\[
d_{\mathrm{bal}} = \alpha\,G_{\mathrm{load}}(S_0) + \beta\,G_{\mathrm{poro}}(S_g) + \varepsilon,
\]

with balancing chosen so that the two forward contributions were comparable in scale.

This section answers:

- why balancing was necessary,
- what the synthetic inverse problem actually is,
- and why the synthetic benchmark is informative rather than arbitrary.

## 2. Synthetic Inversion Experiment Ladder

We then carried out a structured architecture and training progression rather than a single model test.

The base inverse map is:

\[
(\hat S_0,\hat S_g) = f_\theta(d),
\]

or, in conditioned variants,

\[
(\hat S_0,\hat S_g) = f_\theta(d,\nu),
\]

where \(\nu\) is the known synthetic noise-level condition.

The main synthetic progression was:

1. baseline unconstrained inversion,
2. balanced forward-model benchmark,
3. frequency-separated architectures,
4. dual-decoder architectures,
5. direct noise-conditioned variants,
6. hybrid conditioned refinement,
7. noisy-stage \(S_g\)-emphasis refinement,
8. multi-seed validation.

The synthetic loss family was

\[
L =
w_{0}L^{S_0}_{\mathrm{state}} +
w_{g}L^{S_g}_{\mathrm{state}} +
\lambda_f L_{\mathrm{fwd}} +
\lambda_{c0}L^{S_0}_{\mathrm{corr}} +
\lambda_{cg}L^{S_g}_{\mathrm{corr}} +
\lambda_{a0}L^{S_0}_{\mathrm{amp}} +
\lambda_{ag}L^{S_g}_{\mathrm{amp}}.
\]

This section answers:

- which model changes mattered,
- which robustness ideas failed,
- and which branch became the best synthetic reference.

## 3. Synthetic Findings

The synthetic conclusions are:

- balancing the forward problem was necessary,
- the clean two-signal inversion is learnable,
- the best clean model is the dual-decoder frequency-separated architecture,
- the best robustness branch is the hybrid direct-conditioned model with noisy-stage \(S_g\) emphasis,
- the noisy stages remain unstable across seeds.

Thus the synthetic stage should be framed as:

\[
\text{proof of concept + model selection},
\]

not as a fully solved robust inversion.

## What The Synthetic Phase Achieved

The synthetic phase achieved three important things for the paper.

First, it showed that the inversion is genuinely feasible in a controlled setting. Once the elastic and poroelastic contributions were balanced, both latent signals could be recovered in the clean two-component benchmark. This means the method is not fundamentally broken.

Second, it identified the best model family for the inversion task. The dual-decoder frequency-separated architecture became the clean-case reference model, and the hybrid conditioned branch with noisy-stage \(S_g\) emphasis became the strongest robustness-oriented synthetic branch.

Third, it defined the real limitation of the unconstrained approach. The multi-seed validation showed that performance degrades substantially and becomes unstable under noise, especially for the noisy synthetic cases. This gave a clear and defensible reason not to overclaim robustness when moving to Punjab.

Therefore, the synthetic phase did not merely generate preliminary examples. It established:

- that the inverse problem is learnable in principle,
- that balancing the physics is necessary,
- that architecture choice matters strongly,
- which model should be carried into the real-data stage,
- and what limitation must be acknowledged in the paper.

The most concise summary is:

\[
\text{the synthetic phase established feasibility, selected the inversion architecture, and quantified the robustness limits of unconstrained recovery.}
\]

## 4. Punjab Real-Data Phase 1 Baseline

The Punjab real-data phase was designed first as an internal-prior baseline, without requiring external datasets to carry the inversion.

The Phase 1 loss is:

\[
L_{\mathrm{phase1}} =
\lambda_f L_{\mathrm{fwd}} +
\lambda_{s0}L^{S_0}_{\mathrm{spatial}} +
\lambda_{sg}L^{S_g}_{\mathrm{spatial}} +
\lambda_{t0}L^{S_0}_{\mathrm{temporal}} +
\lambda_{tg}L^{S_g}_{\mathrm{temporal}}.
\]

This includes:

- forward consistency,
- spatial regularization,
- temporal regularization.

The support mask is defined from coherence and temporal support:

\[
M(i,j) =
\mathbf{1}\!\left[\mathrm{coh}(i,j) \ge 0.20\right]
\cdot
\mathbf{1}\!\left[f_{\mathrm{valid}}(i,j) \ge 0.05\right].
\]

We then tested multiple Punjab training/sampling variants and found that the best real-data baseline is the expanded grouped-support pilot.

This section answers:

- how the Punjab inversion was made operational,
- how the support mask and grouped temporal batching were chosen,
- and what the best real-data baseline is.

## 5. Punjab External-Prior Ablations

After freezing the internal-prior baseline, we tested external-prior ablations.

### W3RA

The W3RA-augmented form was:

\[
L_{\mathrm{phase2,W3RA}} = L_{\mathrm{phase1}} + \lambda_w L_{\mathrm{W3RA}},
\]

with weak and stronger anomaly-based W3RA penalties.

### GRACE

The GRACE-augmented forms included:

\[
L_{\mathrm{phase2,GRACE}} = L_{\mathrm{phase1}} + \lambda_g L_{\mathrm{GRACE}},
\]

and a reformulated uncertainty-weighted anomaly form.

These experiments showed that both prior families were implementable and stable, but neither materially improved the main Punjab fit metrics under the formulations tested.

This section answers:

- whether external priors can be integrated,
- whether they changed the current best Punjab result,
- and what their present role is in the paper.

## What To Show In The Paper

## Main Tables

### Table 1. Synthetic Experiment Ladder Summary

Purpose:

- summarize the synthetic progression from baseline to candidate model.

Suggested columns:

- experiment,
- main change,
- clean-case metric,
- noisy-case metric,
- conclusion.

### Table 2. Synthetic Multi-Seed Validation

Purpose:

- show that the best synthetic robustness branch is still seed-sensitive under noise.

Suggested columns:

- noise level,
- layer,
- mean \(R^2\),
- mean correlation,
- mean slope,
- standard deviations across seeds.

Recommended source:

- `outputs/synthetic_two_layer_balanced_conditioned_dualdec_hybrid_sg_multiseed_summary.csv`

### Table 3. Punjab Baseline And Prior-Ablation Summary

Purpose:

- summarize the real-data baseline and the external-prior near-null results.

Suggested columns:

- run label,
- validation forward loss,
- normalized forward RMSE,
- extra-prior term,
- interpretation.

Recommended source:

- `outputs/punjab_prior/punjab_paper_results_summary.csv`

## Main Figures

### Figure 1. Synthetic Conditioning Problem

Show:

- elastic versus poroelastic magnitude mismatch.

Purpose:

- motivate the balanced benchmark.

Recommended file:

- `outputs/figures/synthetic_elastic_vs_poro_magnitude.png`

### Figure 2. Synthetic Architecture Progression

Show:

- the clean-model comparison leading to the dual-decoder frequency-separated anchor.

Purpose:

- justify the selected clean synthetic architecture.

Recommended file:

- `outputs/figures/synthetic_two_layer_balanced_dualdec_frequency_vs_baselines.png`

### Figure 3. Synthetic Robustness Branch Comparison

Show:

- the best robustness branch against the main synthetic baselines.

Purpose:

- justify the selected robustness branch.

Recommended file:

- `outputs/figures/synthetic_two_layer_balanced_conditioned_dualdec_hybrid_sg_vs_baselines.png`

### Figure 4. Synthetic Multi-Seed Validation

Show:

- multi-seed mean \(\pm\) spread for the best robustness branch.

Purpose:

- support the claim that the synthetic method is promising but still noise-fragile.

Recommended file:

- `outputs/figures/synthetic_two_layer_balanced_conditioned_dualdec_hybrid_sg_multiseed.png`

### Figure 5. Punjab Data Support And Masking

Show:

- InSAR velocity,
- coherence,
- temporal support fraction,
- final support mask.

Purpose:

- explain the spatial domain actually used by the inversion.

Recommended file:

- `outputs/punjab_prior/figures/punjab_paper_figure5_support_mask_velocity.png`

### Figure 6. Punjab Baseline Residual Gallery

Show:

- observed deformation,
- residual,
- predicted \(S_0\),
- predicted \(S_g\),

for selected validation examples from the frozen expanded grouped-support baseline.

Purpose:

- illustrate what the real-data baseline is actually doing.

Recommended file:

- `outputs/punjab_prior/figures/punjab_paper_figure6_baseline_residual_gallery.png`

### Figure 7. Punjab Prior-Ablation Null Result

Show:

- changes in validation forward loss and normalized forward RMSE relative to the frozen baseline.

Purpose:

- show clearly that the tested W3RA and GRACE formulations remained effectively tied with the baseline.

Recommended file:

- `outputs/punjab_prior/figures/punjab_paper_figure7_prior_ablation_comparison.png`

### Figure 8. Punjab Baseline Export Panel

Show:

- mean predicted \(S_0\),
- mean predicted \(S_g\),
- last-date predicted \(S_0\),
- last-date predicted \(S_g\),
- representative pixel time series.

Purpose:

- give the paper-facing visualization of the frozen Punjab baseline inversion.

Recommended file:

- `outputs/punjab_prior/figures/punjab_paper_figure8_baseline_export_panel.png`

## Suggested Section Structure

The paper can be organized after the introduction as:

1. Synthetic Experimental Design
2. Synthetic Experiment Ladder
3. Synthetic Results
4. Punjab Real-Data Inversion Setup
5. Punjab Baseline Results
6. External-Prior Ablations
7. Discussion
8. Conclusion

## Recommended Conclusions

The conclusions should stay tight and aligned with what the experiments actually support.

### Synthetic Conclusion

\[
\text{The physics-aware inversion is feasible in controlled synthetic settings.}
\]

Balancing the elastic and poroelastic responses was necessary to make the two-signal inversion identifiable in practice. The best clean-case architecture was a dual-decoder frequency-separated SWIN-style model. The best robustness branch, based on hybrid direct conditioning with noisy-stage \(S_g\) emphasis, improved the synthetic benchmark relative to earlier variants, but multi-seed validation showed that noisy recovery remains fragile. Therefore, the synthetic stage establishes a strong proof of concept and a defensible model-selection result, but not a fully robust final solution.

### Punjab Conclusion

\[
\text{The real-data inversion pipeline works end-to-end, and the internal-prior baseline is the main Punjab result.}
\]

The expanded grouped-support baseline is the best real-data configuration obtained so far. It provides a stable, interpretable Phase 1 inversion under internal priors consisting of forward consistency, spatial regularization, and temporal regularization. The tested `W3RA` and `GRACE` formulations were integrated successfully, but they did not materially improve the main deformation-fit metrics under the formulations considered here.

### Overall Conclusion

\[
\text{The paper supports a viable inversion framework, not a fully solved final deployment.}
\]

The synthetic experiments identify the best current model class and clarify the conditioning problem. The Punjab experiments demonstrate that the framework can be carried into real-data inversion and produce a stable baseline result. At the same time, the external-prior ablations show that simply attaching weak regional `W3RA` or `GRACE` priors is not sufficient to move the solution meaningfully. Accordingly, the strongest paper claim is that the work establishes a validated methodology, a defensible synthetic benchmark ladder, and a real-data baseline inversion, while also clearly defining the remaining limitations and next methodological targets.

## How To Interpret The Noisy Synthetic Limitation

The paper should argue the noisy synthetic result in a precise way.

The clean synthetic success means:

\[
\text{the inverse map is learnable and the problem is not fundamentally impossible.}
\]

The best single-run noisy improvements mean:

\[
\text{the chosen architecture and conditioning strategy are directionally correct.}
\]

The multi-seed deterioration means:

\[
\text{the noisy inverse problem remains weakly conditioned and sensitive to optimization.}
\]

This is the key interpretation. The noisy synthetic limitation does not mean the method is useless. It means that once realistic noise is introduced, the inverse problem is still underconstrained enough that different random seeds can lead to meaningfully different solutions.

This should be argued in the paper as a conditioning result rather than a failure result. A suitable interpretation is:

> The synthetic experiments demonstrate that two-signal inversion is feasible in controlled conditions and that balancing plus decoder specialization substantially improve recoverability. However, multi-seed validation shows that recovery under noise remains unstable, indicating that the inverse problem is still weakly conditioned in noisy settings.

## What This Means For The Real-Data Punjab Inversion

Yes, the noisy synthetic limitation can affect the confidence we place in the real-data inversion.

The reason is straightforward. Real data are noisier, less controlled, and do not provide direct ground truth for the latent storage components. Therefore, any instability observed in noisy synthetic experiments is a warning that the real-data inversion may also contain:

- sensitivity to initialization or optimization,
- partial leakage between \(S_0\) and \(S_g\),
- amplitude uncertainty,
- or solutions that are deformation-consistent but not uniquely determined.

That does **not** invalidate the Punjab result. It changes how the Punjab result should be presented.

The Punjab inversion should therefore be described as:

\[
\text{a deformation-consistent baseline solution, not a uniquely validated truth field.}
\]

This is the most defensible wording for the paper. It allows us to make a strong claim about feasibility and implementation while avoiding overstatement about certainty of the recovered latent fields.

## Recommended Discussion Statement

If a short discussion paragraph is needed, the following wording is appropriate:

> The synthetic stage established that the inversion is feasible and identified the best-performing model family, but it also showed that recovery under noise remains weakly conditioned. This limitation matters for the Punjab application because real data are noisier and lack direct ground truth. Accordingly, the Punjab inversion should be interpreted as the best current deformation-consistent baseline rather than as a uniquely validated storage reconstruction.

## Writing Guidance

The paper should explicitly avoid the following claims:

- that the Punjab inversion is completely solved,
- that `W3RA` improved the inversion,
- that `GRACE` improved the inversion,
- that the latent storage outputs are already externally calibrated physical products.

The paper can safely claim:

- that the synthetic model-selection phase is complete,
- that the balanced two-signal inversion is learnable in synthetic data,
- that the Punjab inversion runs end-to-end on real data,
- that the internal-prior baseline is the best current Punjab result,
- that `W3RA` and `GRACE` are informative null or near-null ablations under the tested formulations.

## Diagram-Ready Model Description

The model used in this work is a compact 3D Swin-Transformer U-Net variant with a shared encoder and bottleneck, followed by two decoder branches, one for \(S_0\) and one for \(S_g\). It is shallower than the large 3D Swin U-Net shown in the reference image, but the design logic is similar:

- patch-based 3D embedding,
- Swin 3D transformer stages,
- spatial patch merging in the encoder,
- spatial patch expansion in the decoder,
- skip fusion from encoder to decoder,
- separate decoder heads for the two target fields.

## Backbone Used For The Punjab Baseline

The Punjab baseline uses the plain dual-decoder frequency-separated model:

\[
(\hat S_0,\hat S_g)=f_\theta(d).
\]

The synthetic robustness branch uses the noise-conditioned variant:

\[
(\hat S_0,\hat S_g)=f_\theta(d,\nu),
\]

but the encoder-decoder geometry is the same. The only difference is that the conditioned model includes a second input channel carrying the noise-level map.

## Input Representation

The raw deformation input is first split into low- and high-frequency components:

\[
d_{\mathrm{low}} = \mathcal{L}(d), \qquad d_{\mathrm{high}} = d - d_{\mathrm{low}}.
\]

These are processed by two separate patch-embedding branches:

- low-frequency branch,
- high-frequency branch.

For the plain Punjab baseline, each branch receives one channel:

- low branch input: \(d_{\mathrm{low}}\),
- high branch input: \(d_{\mathrm{high}}\).

For the noise-conditioned synthetic variant, each branch receives two channels:

- low branch input: \([d_{\mathrm{low}}, \nu]\),
- high branch input: \([d_{\mathrm{high}}, \nu]\).

## Patching

The 3D patch size is:

\[
(t,h,w) = (2,4,4).
\]

Thus, each patch embedding is a `Conv3d` with:

- kernel size `(2, 4, 4)`,
- stride `(2, 4, 4)`.

If the input window is

\[
T \times H \times W = 12 \times 64 \times 64,
\]

then after patch embedding, each branch becomes:

\[
6 \times 16 \times 16
\]

tokens with channel dimension

\[
C = 32.
\]

## Encoder

The two embedded branches are concatenated in channel dimension:

\[
[x_{\mathrm{low}}, x_{\mathrm{high}}] \in \mathbb{R}^{64 \times 6 \times 16 \times 16}.
\]

The shared encoder then consists of:

### Encoder Stage E1

- one `SwinStage3D`,
- embedding dimension:
  \[
  C = 64,
  \]
- depth:
  \[
  2 \text{ Swin blocks},
  \]
- number of heads:
  \[
  4,
  \]
- window size:
  \[
  (3,4,4).
  \]

The two Swin blocks alternate between:

- no shift,
- shifted windows by half-window:
  \[
  (1,2,2).
  \]

This stage preserves the token grid:

\[
64 \times 6 \times 16 \times 16.
\]

### Patch Merging

After the encoder stage, the model applies spatial patch merging:

\[
(1,2,2),
\]

implemented as a `Conv3d` with stride `(1,2,2)`.

This means:

- time resolution is preserved,
- spatial resolution is halved,
- channel dimension is doubled.

So the representation becomes:

\[
128 \times 6 \times 8 \times 8.
\]

## Bottleneck

The bottleneck is another `SwinStage3D` with:

- dimension:
  \[
  C = 128,
  \]
- depth:
  \[
  2 \text{ Swin blocks},
  \]
- number of heads:
  \[
  4,
  \]
- window size:
  \[
  (3,4,4).
  \]

The bottleneck therefore operates on:

\[
128 \times 6 \times 8 \times 8.
\]

## Dual Decoder

After the bottleneck, the network splits into two decoders:

- decoder for \(S_0\),
- decoder for \(S_g\).

Each decoder has the same geometry.

### Decoder Expansion

Each branch first applies `PatchExpand3D` using a transposed convolution with scale:

\[
(1,2,2),
\]

so the bottleneck feature map expands from:

\[
128 \times 6 \times 8 \times 8
\]

to:

\[
64 \times 6 \times 16 \times 16.
\]

### Skip Fusion

The expanded decoder feature is concatenated with the encoder feature from `E1`:

\[
\text{concat}(x_{\mathrm{dec}}, x_{\mathrm{enc}}),
\]

so the temporary fused tensor has channel size:

\[
128.
\]

This is then reduced by a `1 \times 1 \times 1` convolution back to:

\[
64 \times 6 \times 16 \times 16.
\]

### Decoder Stage D1

Each branch then applies another `SwinStage3D` with:

- dimension:
  \[
  C = 64,
  \]
- depth:
  \[
  2 \text{ Swin blocks},
  \]
- number of heads:
  \[
  4,
  \]
- window size:
  \[
  (3,4,4).
  \]

### Final Patch Expansion To Output Resolution

Each decoder branch then applies a final `ConvTranspose3d` with kernel and stride equal to the original patch size:

\[
(2,4,4).
\]

So the feature map expands from:

\[
64 \times 6 \times 16 \times 16
\]

to:

\[
32 \times 12 \times 64 \times 64.
\]

Finally, a `1 \times 1 \times 1` convolution maps each branch to one output channel:

\[
1 \times 12 \times 64 \times 64.
\]

The model then takes the final time slice only:

\[
\hat S_0 = y_{S_0}[:,:, -1, :, :], \qquad
\hat S_g = y_{S_g}[:,:, -1, :, :].
\]

Thus each forward pass predicts two 2D fields:

\[
\hat S_0, \hat S_g \in \mathbb{R}^{64 \times 64}.
\]

## Diagram Blocks You Can Draw

If you want to make a diagram similar to the example image, the cleanest block layout is:

1. **Input window**
   - `1 x 12 x 64 x 64` for the Punjab baseline
   - or `2 x 12 x 64 x 64` per branch in the noise-conditioned synthetic variant

2. **Frequency split**
   - low-pass deformation branch
   - high-pass residual branch

3. **Patch embedding**
   - two separate `Conv3d` patch embeddings
   - each produces `32 x 6 x 16 x 16`

4. **Concatenation**
   - produces `64 x 6 x 16 x 16`

5. **Encoder Stage E1**
   - `2 x SwinBlock3D`
   - window size `(3,4,4)`

6. **Patch merging**
   - scale `(1,2,2)`
   - output `128 x 6 x 8 x 8`

7. **Bottleneck**
   - `2 x SwinBlock3D`
   - window size `(3,4,4)`

8. **Split into two decoders**
   - \(S_0\) branch
   - \(S_g\) branch

9. **Decoder expansion**
   - transposed conv `(1,2,2)`
   - output `64 x 6 x 16 x 16`

10. **Skip concatenation from E1**
    - concatenate with encoder feature
    - fuse with `1x1x1` convolution

11. **Decoder Stage D1**
    - `2 x SwinBlock3D`

12. **Final patch expansion**
    - transposed conv `(2,4,4)`
    - output `32 x 12 x 64 x 64`

13. **Prediction head**
    - `1x1x1` conv to one channel
    - keep final time slice only

14. **Outputs**
    - \(\hat S_0\) map
    - \(\hat S_g\) map

## Short Figure Caption Style Description

If you want one compact paragraph for a figure caption, you can use:

> Overview of the compact 3D Swin-Transformer U-Net used in this study. The model first separates the deformation input into low-frequency and residual branches, embeds each branch using \(2 \times 4 \times 4\) spatiotemporal patches, and concatenates the resulting tokens before a shared encoder and bottleneck. The encoder contains one Swin 3D stage followed by spatial patch merging, and the bottleneck contains a second Swin 3D stage at lower spatial resolution. The decoder then splits into two branches, one for \(S_0\) and one for \(S_g\), each using patch expansion, skip fusion with the encoder, a Swin 3D decoder stage, and a final transposed-convolution upsampling step back to the original spatiotemporal grid. A \(1 \times 1 \times 1\) head is applied to each branch, and the final time slice is retained as the predicted 2D storage field.
