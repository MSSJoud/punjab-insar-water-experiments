# GRACE Prior Discussion

## Why GRACE Did Not Materially Improve the Punjab Inversion

The Punjab experiments showed that GRACE can be integrated into the inversion pipeline in a technically stable way, but the GRACE priors tested so far did not materially improve the main inversion metrics. This outcome should not be interpreted as evidence that GRACE is useless in principle. Rather, it indicates that the specific GRACE formulations used here were too weak and too indirect relative to the structure of the inverse problem.

The current Punjab inversion seeks to recover two latent storage components from deformation:
\[
(\hat S_0,\hat S_g)=f_\theta(d),
\]
with forward consistency enforced through
\[
\hat d = G_{\mathrm{load}}(\hat S_0)+G_{\mathrm{poro}}(\hat S_g).
\]
The main optimization target is therefore the deformation-space consistency term, together with internal spatial and temporal regularization:
\[
L_{phase1}
=
\lambda_f L_{fwd}
+ \lambda_{s0}L_{spatial}^{S_0}
+ \lambda_{sg}L_{spatial}^{S_g}
+ \lambda_{t0}L_{temporal}^{S_0}
+ \lambda_{tg}L_{temporal}^{S_g}.
\]

In contrast, GRACE only constrains large-scale total storage. In the current two-signal Punjab setting, that means GRACE can only inform the combined quantity
\[
\hat S_{\mathrm{tot}}=\hat S_0+\hat S_g,
\]
not the partition between the two components. This is a fundamental limitation: even a perfectly enforced GRACE prior would not, by itself, resolve the ambiguity between \(\hat S_0\) and \(\hat S_g\).

The first GRACE formulations used here were regional summary constraints of the form
\[
L_{\mathrm{GRACE}}
=
\left\|
\mathcal{A}(\hat S_0+\hat S_g)-g_{\mathrm{GRACE}}
\right\|^2,
\]
where \(\mathcal{A}(\cdot)\) reduced the model state to a Punjab-scale regional statistic and \(g_{\mathrm{GRACE}}\) was the aligned GRACE regional mean or anomaly series. These formulations were too coarse relative to the tile-based deformation objective. The deformation loss acts locally and strongly, while the GRACE prior acts only on a single low-dimensional regional summary. As a result, the model can satisfy the GRACE term with only minor adjustments while leaving the deformation-fit solution effectively unchanged.

This explains why the GRACE branches were stable but near-ties relative to the frozen Phase~1 baseline. The GRACE prior remained active in the optimization, but its practical influence was too small to shift the solution in a meaningful way. The issue is therefore not numerical failure, but limited identifiability leverage.

## Relation to Carlson et al. (2022)

This behavior is not inconsistent with the claims in Carlson et al.~(2022) \cite{carlson2022}. Their inversion framework is not solving the same problem as the present Punjab notebook. In that study, GRACE and GNSS are used jointly to recover a bulk total-water-storage signal:
\[
\Delta TWS \leftarrow \{\mathrm{GNSS},\mathrm{GRACE}\}.
\]
GRACE contributes regional or long-wavelength mass constraints, while GNSS contributes finer spatial detail. The joint inversion is therefore designed so that GRACE is one of the principal observation systems.

By contrast, the present Punjab formulation is attempting to recover two latent storage components from deformation:
\[
(\hat S_0,\hat S_g)\leftarrow d.
\]
Here, GRACE was only added as a soft auxiliary prior, not as a primary observation equation. This difference matters. Carlson et al.~(2022) show that GRACE can improve a joint \(\Delta TWS\) inversion by supplying large-scale mass information, but that does not imply that a weak regional GRACE prior will significantly improve a local two-component latent-state inversion.

Accordingly, the correct interpretation is not that GRACE ``should have worked but failed,'' but rather that the current GRACE prior family is not sufficiently aligned with the dominant ambiguity in the Punjab problem.

## Why the Current GRACE Priors Were Too Weak

There are three main reasons.

First, GRACE is intrinsically coarse. It is well suited to constraining low-frequency, basin-scale storage change, but not local tile-scale spatial structure. The current Punjab inversion, however, is optimized primarily against local deformation residuals.

Second, GRACE constrains total storage, not the separation between components:
\[
\hat S_0+\hat S_g.
\]
The hard part of the Punjab inversion is not only reconstructing total storage, but separating load-like and groundwater-like contributions. GRACE does not directly resolve that decomposition.

Third, the current GRACE priors were implemented as weak penalties. Even the reformulated uncertainty-weighted anomaly prior,
\[
L_{\mathrm{GRACE,uw}}
=
\mathrm{mean}\!\left[
\left(
\frac{
(\overline{\hat S_0+\hat S_g}-\mathrm{mean}(\overline{\hat S_0+\hat S_g}))-g_{\mathrm{GRACE}}
}{\sigma_{\mathrm{GRACE}}}
\right)^2
\right],
\]
still acted only on a regional statistic. This is a better formulation than the purely batch-standardized matching term because it preserves GRACE anomaly amplitude and incorporates uncertainty, but it remains a low-dimensional constraint compared with the full deformation objective.

The empirical result is therefore consistent with theory: the current GRACE term is informative, but not informative enough to materially change the inversion.

## Most Defensible Next GRACE Formulations

If GRACE is revisited in a future stage, the two most defensible options are more structural than the regional-mean penalties tested here.

### 1. Mascon-Operator GRACE Constraint

Instead of reducing the predicted storage field to a single regional mean, define an explicit GRACE observation operator:
\[
L_{\mathrm{GRACE}}
=
\left\|
\mathcal{H}_{\mathrm{GRACE}}(\hat S_0+\hat S_g)-y_{\mathrm{GRACE}}
\right\|^2,
\]
where \(\mathcal{H}_{\mathrm{GRACE}}\) maps the predicted total storage field to GRACE mascon space and \(y_{\mathrm{GRACE}}\) is the observed mascon product. This would treat GRACE more like a true observation equation rather than a weak scalar regularizer.

### 2. Low-Frequency GRACE Constraint

A simpler and more physically transparent alternative is to only constrain the low-frequency part of the total storage field:
\[
\hat S_{\mathrm{tot}}=\hat S_0+\hat S_g,
\]
\[
L_{\mathrm{GRACE,low}}
=
\left\|
\mathcal{L}(\hat S_{\mathrm{tot}})
-\mathcal{L}(S_{\mathrm{GRACE}})
\right\|^2,
\]
where \(\mathcal{L}(\cdot)\) is a low-pass or coarse-grid operator. This is attractive because it matches the native information content of GRACE more honestly: GRACE constrains large-scale storage variability, not fine local detail.

Among these two options, the low-frequency GRACE formulation is likely the better next experiment if one more GRACE test is desired, because it is simpler to implement while remaining physically well matched to the data.

## Conclusion

The GRACE experiments completed so far support a clear conclusion. GRACE was successfully integrated into the Punjab inversion pipeline, but the specific regional-prior formulations tested here did not produce a meaningful improvement in inversion quality. The likely reason is that these priors were too coarse, too weak, and too indirect relative to the dominant ambiguity of the problem, namely the separation of latent storage components under a strongly local deformation objective.

Therefore, the present evidence does not show that GRACE is useless. It shows that the current GRACE prior family is not sufficient. Future GRACE use should either be formulated through a more structural observation operator or restricted explicitly to low-frequency total-storage constraints.
