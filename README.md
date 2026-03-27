# Punjab InSAR Water Experiments

This repository is the backup-style working record for the Punjab InSAR and hydrologic inversion experiments. It keeps the project close to the current workspace state, including notebooks, helper code, paper figures, implementation notes, and generated summaries that are useful for reconstruction and review.

## Scope

This backup repository is meant to preserve:

- experiment notebooks
- Python helpers and plotting utilities
- implementation records and planning notes
- paper-facing figures and tables
- lightweight outputs such as `csv`, `json`, and `png`

It intentionally excludes large binary artifacts that are not practical to push to standard GitHub repositories, such as multi-gigabyte `NetCDF` and `HDF5` exports, model checkpoints, and bulky array dumps.

## Main Project Files

- `punjab_prior_constrained_inversion.ipynb`
- `punjab_source_comparison_panels.ipynb`
- `punjab_paper_figures.ipynb`
- `punjab_inversion/`
- `tools/`
- `paper_figures/`
- `outputs/` (lightweight summaries and figures only, as allowed by `.gitignore`)

## Notes

- This repository is a backup-oriented project record, not yet the final reusable package.
- The cleaner reusable package is planned as a separate repository centered on the SWIN3D dual-decoder inversion framework.
- If we later want true raw archival backup including the large binary outputs, the right path is either Git LFS, GitHub Releases, or a storage bucket/archive outside normal git history.
