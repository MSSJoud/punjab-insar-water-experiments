# Synthetic Phase Implementation Record

## Purpose
This file records the completed synthetic benchmark phase used to support the unconstrained synthetic inversion section of the paper.

## Primary notebook
- [punjab_synthetic_to_real_pipeline.ipynb](/home/ubuntu/work/punjab/punjab_synthetic_to_real_pipeline.ipynb)

## Scope
- Synthetic forward model development
- Identifiability tests
- Balanced elastic/poroelastic benchmarking
- Architecture progression
- Noise-conditioned refinements
- Multi-seed validation
- Paper-ready tables and figures

## Final synthetic status
- This phase is complete for the current paper scope.
- The synthetic notebook is now the experiment record and paper-output source.
- No further model branching is planned here unless we need extra export figures or tables for writing.

## Main scientific conclusion
- Unconstrained inversion is feasible in controlled synthetic settings.
- Balancing the elastic and poroelastic forward contributions was necessary.
- The best clean-case architecture was a dual-decoder frequency-separated SWIN-style model.
- The best proof-of-concept robustness branch was the hybrid direct-conditioned model with noisy-stage `Sg` emphasis.
- Multi-seed validation showed the noisy stages remain fragile, so the unconstrained method is informative but not yet robust enough for direct real-data deployment.

## Current best synthetic references
- Clean anchor:
  - dual-decoder frequency-separated model
- Best robustness branch:
  - hybrid direct conditioning + noisy-stage `Sg` emphasis

## Key artifacts
- [synthetic_two_layer_balanced_dualdec_frequency_metrics.csv](/home/ubuntu/work/punjab/outputs/synthetic_two_layer_balanced_dualdec_frequency_metrics.csv)
- [synthetic_two_layer_balanced_dualdec_frequency_summary.json](/home/ubuntu/work/punjab/outputs/synthetic_two_layer_balanced_dualdec_frequency_summary.json)
- [synthetic_two_layer_balanced_conditioned_dualdec_hybrid_sg_metrics.csv](/home/ubuntu/work/punjab/outputs/synthetic_two_layer_balanced_conditioned_dualdec_hybrid_sg_metrics.csv)
- [synthetic_two_layer_balanced_conditioned_dualdec_hybrid_sg_summary.json](/home/ubuntu/work/punjab/outputs/synthetic_two_layer_balanced_conditioned_dualdec_hybrid_sg_summary.json)
- [synthetic_two_layer_balanced_conditioned_dualdec_hybrid_sg_multiseed_summary.csv](/home/ubuntu/work/punjab/outputs/synthetic_two_layer_balanced_conditioned_dualdec_hybrid_sg_multiseed_summary.csv)
- [synthetic_two_layer_balanced_conditioned_dualdec_hybrid_sg_ema_multiseed_summary.csv](/home/ubuntu/work/punjab/outputs/synthetic_two_layer_balanced_conditioned_dualdec_hybrid_sg_ema_multiseed_summary.csv)

## Key figures
- [synthetic_elastic_vs_poro_magnitude.png](/home/ubuntu/work/punjab/outputs/figures/synthetic_elastic_vs_poro_magnitude.png)
- [synthetic_two_layer_balanced_dualdec_frequency_vs_baselines.png](/home/ubuntu/work/punjab/outputs/figures/synthetic_two_layer_balanced_dualdec_frequency_vs_baselines.png)
- [synthetic_two_layer_balanced_conditioned_dualdec_hybrid_sg_vs_baselines.png](/home/ubuntu/work/punjab/outputs/figures/synthetic_two_layer_balanced_conditioned_dualdec_hybrid_sg_vs_baselines.png)
- [synthetic_two_layer_balanced_conditioned_dualdec_hybrid_sg_multiseed.png](/home/ubuntu/work/punjab/outputs/figures/synthetic_two_layer_balanced_conditioned_dualdec_hybrid_sg_multiseed.png)

## Paper-facing role
- This phase supports:
  - unconstrained synthetic inversion experiments
  - motivation for prior-constrained final inversion

## Next action from the synthetic phase
- Use the notebook outputs for paper writing.
- Do not resume synthetic branching unless we explicitly choose to test constrained synthetic inversion later.
