import base64
import io
import json
from pathlib import Path

import pandas as pd


NOTEBOOK = Path("/home/ubuntu/work/punjab/punjab_synthetic_to_real_pipeline.ipynb")
ROOT = NOTEBOOK.parent
OUT = ROOT / "outputs"
FIG = OUT / "figures"


def md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line if line.endswith("\n") else line + "\n" for line in text.splitlines()],
    }


def code_cell(source: str, outputs: list[dict]) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": 1,
        "outputs": outputs,
        "source": [line if line.endswith("\n") else line + "\n" for line in source.splitlines()],
    }


def display_html_output(html: str, text: str) -> dict:
    return {
        "output_type": "display_data",
        "data": {
            "text/html": html,
            "text/plain": text,
        },
        "metadata": {},
    }


def display_png_output(path: Path, label: str) -> dict:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "output_type": "display_data",
        "data": {
            "image/png": encoded,
            "text/plain": label,
        },
        "metadata": {},
    }


def df_to_output(df: pd.DataFrame, title: str) -> dict:
    html = f"<h4>{title}</h4>" + df.to_html(index=False, border=1, classes=["dataframe"], float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x))
    text = f"{title}\n{df.to_string(index=False)}"
    return display_html_output(html, text)


def nice_model_name(mode: str) -> str:
    mapping = {
        "dual_decoder_frequency_curriculum_balanced": "Dual decoder baseline",
        "noise_conditioned_dual_decoder_balanced": "Direct conditioned",
        "hybrid_clean_then_original_conditioned_dual_decoder_balanced": "Hybrid conditioned",
        "hybrid_sg_emphasis_conditioned_dual_decoder_balanced": "Hybrid + Sg emphasis",
    }
    return mapping.get(mode, mode)


def build_tables():
    key = pd.concat(
        [
            pd.read_csv(OUT / "synthetic_two_layer_balanced_dualdec_frequency_metrics.csv"),
            pd.read_csv(OUT / "synthetic_two_layer_balanced_conditioned_dualdec_metrics.csv"),
            pd.read_csv(OUT / "synthetic_two_layer_balanced_conditioned_dualdec_hybrid_metrics.csv"),
            pd.read_csv(OUT / "synthetic_two_layer_balanced_conditioned_dualdec_hybrid_sg_metrics.csv"),
        ],
        ignore_index=True,
        sort=False,
    )
    key = key[key["noise_scale"].isin([0.00, 0.02, 0.05])].copy()
    key["model"] = key["training_mode"].map(nice_model_name)
    key = key[key["model"].notna()]
    key = key[["model", "noise_scale", "layer", "rmse", "r2", "corr", "fit_slope"]]
    key = key.sort_values(["model", "noise_scale", "layer"]).reset_index(drop=True)

    ms = pd.read_csv(OUT / "synthetic_two_layer_balanced_conditioned_dualdec_hybrid_sg_multiseed_summary.csv").copy()
    ms["rmse_mean±std"] = ms.apply(lambda r: f"{r['rmse_mean']:.3f} ± {r['rmse_std']:.3f}", axis=1)
    ms["r2_mean±std"] = ms.apply(lambda r: f"{r['r2_mean']:.3f} ± {r['r2_std']:.3f}", axis=1)
    ms["corr_mean±std"] = ms.apply(lambda r: f"{r['corr_mean']:.3f} ± {r['corr_std']:.3f}", axis=1)
    ms["slope_mean±std"] = ms.apply(lambda r: f"{r['fit_slope_mean']:.3f} ± {r['fit_slope_std']:.3f}", axis=1)
    ms = ms[["noise_scale", "layer", "rmse_mean±std", "r2_mean±std", "corr_mean±std", "slope_mean±std"]]
    ms = ms.sort_values(["noise_scale", "layer"]).reset_index(drop=True)

    return key, ms


def main():
    nb = json.loads(NOTEBOOK.read_text())
    marker = "## Paper-Ready Results Package"
    if any(marker in "".join(cell.get("source", [])) for cell in nb["cells"]):
        return

    key_table, ms_table = build_tables()

    cells = [
        md_cell(
            """## Paper-Ready Results Package

This section embeds the key tables and figures needed for the paper/report directly inside the notebook. It is intended as the compact results package for writing:

- the main benchmark progression
- the best candidate branch
- the multi-seed stability check
- the most important diagnostic figure explaining why balancing was needed"""
        ),
        md_cell(
            """### Recommended Results To Show In The Paper

Main tables:
- Selected benchmark metrics for the main model progression
- Multi-seed summary for the final candidate branch

Main figures:
- Elastic vs poroelastic magnitude mismatch diagnostic
- Architecture/model progression toward the dual-decoder baseline
- Best robustness branch comparison
- Multi-seed validation of the final candidate branch

Optional appendix:
- EMA final-round comparison, to document why EMA was not adopted"""
        ),
        code_cell(
            "# Embedded paper-ready result tables",
            [
                df_to_output(key_table, "Table 1. Selected benchmark metrics for the main model progression"),
                df_to_output(ms_table, "Table 2. Multi-seed summary for the final candidate branch"),
            ],
        ),
        md_cell(
            """### Figure 1. Elastic vs Poroelastic Magnitude Mismatch

This diagnostic justifies why the balanced synthetic benchmark was needed before evaluating identifiability."""
        ),
        code_cell(
            "# Embedded figure: elastic vs poroelastic magnitude mismatch",
            [display_png_output(FIG / "synthetic_elastic_vs_poro_magnitude.png", "synthetic_elastic_vs_poro_magnitude")],
        ),
        md_cell(
            """### Figure 2. Architecture Progression Toward The Clean Candidate

This figure shows the move from earlier branch-aware/frequency-separated baselines to the dual-decoder clean anchor."""
        ),
        code_cell(
            "# Embedded figure: dual-decoder progression",
            [display_png_output(FIG / "synthetic_two_layer_balanced_dualdec_frequency_vs_baselines.png", "synthetic_two_layer_balanced_dualdec_frequency_vs_baselines")],
        ),
        md_cell(
            """### Figure 3. Best Robustness Branch Comparison

This figure shows the current best direct-conditioned refinement path, ending in the hybrid noisy-stage `Sg`-emphasis model."""
        ),
        code_cell(
            "# Embedded figure: best robustness branch comparison",
            [display_png_output(FIG / "synthetic_two_layer_balanced_conditioned_dualdec_hybrid_sg_vs_baselines.png", "synthetic_two_layer_balanced_conditioned_dualdec_hybrid_sg_vs_baselines")],
        ),
        md_cell(
            """### Figure 4. Multi-Seed Stability Of The Final Candidate Branch

This figure shows the mean and spread across seeds for the final candidate branch. It is the key figure for the current limitation statement."""
        ),
        code_cell(
            "# Embedded figure: multi-seed validation of the final candidate branch",
            [display_png_output(FIG / "synthetic_two_layer_balanced_conditioned_dualdec_hybrid_sg_multiseed.png", "synthetic_two_layer_balanced_conditioned_dualdec_hybrid_sg_multiseed")],
        ),
        md_cell(
            """### Appendix Figure. EMA Final Round

Optional appendix figure documenting the final stabilization attempt and why it was not adopted."""
        ),
        code_cell(
            "# Embedded figure: EMA final-round comparison",
            [display_png_output(FIG / "synthetic_two_layer_balanced_conditioned_dualdec_hybrid_sg_ema_multiseed.png", "synthetic_two_layer_balanced_conditioned_dualdec_hybrid_sg_ema_multiseed")],
        ),
    ]

    nb["cells"].extend(cells)
    NOTEBOOK.write_text(json.dumps(nb, indent=1))


if __name__ == "__main__":
    main()
