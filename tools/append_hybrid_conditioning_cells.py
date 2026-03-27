import json
from pathlib import Path


NOTEBOOK = Path("/home/ubuntu/work/punjab/punjab_synthetic_to_real_pipeline.ipynb")


def md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line if line.endswith("\n") else line + "\n" for line in text.splitlines()],
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [line if line.endswith("\n") else line + "\n" for line in text.splitlines()],
    }


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text())
    marker = "## Hybrid Clean-Then-Original Direct Conditioning Benchmark"
    if any(marker in "".join(cell.get("source", [])) for cell in nb["cells"]):
        return

    nb["cells"].extend(
        [
            md_cell(
                """## Hybrid Clean-Then-Original Direct Conditioning Benchmark

This refinement stays on the same direct input-conditioned dual-decoder branch, but explicitly combines the strongest pieces of the two existing direct-conditioning variants: it trains the clean stage with the corrected conditioning setup, then switches to the original stagewise noisy adaptation for the later noise levels. The goal is to recover cleaner stage-0 behavior without giving up the noisy `Sg` gains that the original stagewise adaptation showed."""
            ),
            code_cell(
                """# Hybrid clean-then-original direct conditioning benchmark
import json

HYBRID_COND_STAGE_EPOCHS = [24, 12, 12, 12]


def train_hybrid_conditioned_dualdec():
    hybrid_rows = []
    hybrid_history = []
    hybrid_state = None

    for stage_idx, (stage_noise, stage_epochs) in enumerate(zip(BALANCED_NOISES, HYBRID_COND_STAGE_EPOCHS)):
        if stage_idx == 0:
            rows_case, hist_case, hybrid_state = train_refined_conditioned_dualdec_case(stage_noise, stage_epochs, init_state=hybrid_state)
        else:
            rows_case, hist_case, hybrid_state = train_conditioned_dualdec_case(stage_noise, stage_epochs, init_state=hybrid_state)
            for row in rows_case:
                row['training_mode'] = 'hybrid_clean_then_original_conditioned_dual_decoder_balanced'
            for item in hist_case:
                item['training_mode'] = 'hybrid_clean_then_original_conditioned_dual_decoder_balanced'
        if stage_idx == 0:
            for row in rows_case:
                row['training_mode'] = 'hybrid_clean_then_original_conditioned_dual_decoder_balanced'
            for item in hist_case:
                item['training_mode'] = 'hybrid_clean_then_original_conditioned_dual_decoder_balanced'
        hybrid_rows.extend(rows_case)
        hybrid_history.extend(hist_case)

    return hybrid_rows, hybrid_history


hybrid_cond_rows, hybrid_cond_history = train_hybrid_conditioned_dualdec()
hybrid_cond_metrics_df = pd.DataFrame(hybrid_cond_rows)
hybrid_cond_history_df = pd.DataFrame(hybrid_cond_history)
display(hybrid_cond_metrics_df)
hybrid_cond_metrics_csv = OUT_DIR / 'synthetic_two_layer_balanced_conditioned_dualdec_hybrid_metrics.csv'
hybrid_cond_history_csv = OUT_DIR / 'synthetic_two_layer_balanced_conditioned_dualdec_hybrid_history.csv'
hybrid_cond_json = OUT_DIR / 'synthetic_two_layer_balanced_conditioned_dualdec_hybrid_summary.json'
hybrid_cond_metrics_df.to_csv(hybrid_cond_metrics_csv, index=False)
hybrid_cond_history_df.to_csv(hybrid_cond_history_csv, index=False)
with hybrid_cond_json.open('w') as f:
    json.dump({'metrics': hybrid_cond_rows}, f, indent=2)
print('Saved hybrid conditioned dual-decoder metrics:', hybrid_cond_metrics_csv)
print('Saved hybrid conditioned dual-decoder history:', hybrid_cond_history_csv)
print('Saved hybrid conditioned dual-decoder summary:', hybrid_cond_json)"""
            ),
            md_cell(
                """## Hybrid Direct Conditioning Results Summary

This compares the plain dual-decoder baseline, the original direct-conditioned model, the corrected direct-conditioned model, and the hybrid refinement that uses corrected conditioning for the clean stage and original stagewise adaptation for the noisy stages."""
            ),
            code_cell(
                """# Compare hybrid direct-conditioned results against prior direct-conditioning variants
dual_df = pd.read_csv(OUT_DIR / 'synthetic_two_layer_balanced_dualdec_frequency_metrics.csv')
cond_df = pd.read_csv(OUT_DIR / 'synthetic_two_layer_balanced_conditioned_dualdec_metrics.csv')
refined_cond_df = pd.read_csv(OUT_DIR / 'synthetic_two_layer_balanced_conditioned_dualdec_refined_metrics.csv')
hybrid_cond_df = pd.read_csv(OUT_DIR / 'synthetic_two_layer_balanced_conditioned_dualdec_hybrid_metrics.csv')
compare_df = pd.concat([dual_df, cond_df, refined_cond_df, hybrid_cond_df], ignore_index=True, sort=False)
display(compare_df)

fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex='col')
metric_specs = [('rmse','RMSE','tab:red'),('r2','R^2','tab:blue'),('corr','Correlation','tab:green'),('fit_slope','Slope','tab:purple')]
mode_styles = {
    'dual_decoder_frequency_curriculum_balanced': {'ls':'--','marker':'o', 'label':'dual decoder'},
    'noise_conditioned_dual_decoder_balanced': {'ls':'-.','marker':'^', 'label':'original conditioned'},
    'refined_noise_conditioned_dual_decoder_balanced': {'ls':':','marker':'s', 'label':'corrected conditioned'},
    'hybrid_clean_then_original_conditioned_dual_decoder_balanced': {'ls':'-','marker':'D', 'label':'hybrid conditioned'},
}
for row_idx, layer_name in enumerate(['S0','Sg']):
    layer_df = compare_df[compare_df['layer'] == layer_name]
    for col_idx, (col, title, color) in enumerate(metric_specs):
        ax = axes[row_idx, col_idx]
        for mode_name, style in mode_styles.items():
            dfp = layer_df[layer_df['training_mode'] == mode_name].sort_values('noise_scale')
            x = dfp['noise_scale'].to_numpy()
            y = dfp[col].to_numpy()
            ax.plot(x, y, color=color, lw=2.1, ls=style['ls'], marker=style['marker'], label=style['label'])
            ax.step(x, y, where='post', color=color, lw=1.0, ls=style['ls'], alpha=0.3)
        ax.set_title(f'{layer_name} {title}')
        ax.set_xlabel('Noise scale')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(sorted(compare_df['noise_scale'].unique()))
        if col in {'r2','corr','fit_slope'}:
            ax.axhline(0.0, color='black', lw=1, ls=':')
        if col == 'fit_slope':
            ax.axhline(1.0, color='black', lw=1, ls='--', alpha=0.8)
handles, labels = axes[0,0].get_legend_handles_labels()
if handles:
    fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False)
fig.suptitle('Balanced two-layer inversion: hybrid direct-conditioning refinement', fontsize=14)
fig.tight_layout(rect=[0,0.02,1,0.94])
plt.show()
fig_path = FIG_DIR / 'synthetic_two_layer_balanced_conditioned_dualdec_hybrid_vs_baselines.png'
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
print('Saved hybrid conditioned comparison figure:', fig_path)"""
            ),
        ]
    )
    NOTEBOOK.write_text(json.dumps(nb, indent=1))


if __name__ == "__main__":
    main()
