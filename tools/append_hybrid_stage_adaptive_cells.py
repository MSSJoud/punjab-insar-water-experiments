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
    marker = "## Hybrid Stage-Adaptive Sg Emphasis Benchmark"
    if any(marker in "".join(cell.get("source", [])) for cell in nb["cells"]):
        return

    nb["cells"].extend(
        [
            md_cell(
                """## Hybrid Stage-Adaptive Sg Emphasis Benchmark

This refinement stays on the hybrid direct-conditioned branch and keeps the clean stage unchanged. The only change is that later noisy stages receive stage-specific optimization and loss balance:

- mild `Sg` emphasis at `noise=0.01`
- moderate `Sg` emphasis at `noise=0.02`
- stronger `Sg` emphasis plus lower learning rate and longer training at `noise=0.05`

The goal is to improve the higher-noise groundwater recovery without disturbing the cleaner stages."""
            ),
            code_cell(
                """# Hybrid stage-adaptive Sg emphasis benchmark
import json

HYBRID_ADAPTIVE_STAGE_EPOCHS = {0.00: 24, 0.01: 12, 0.02: 14, 0.05: 18}
HYBRID_ADAPTIVE_LR_SCALE = {0.00: 1.00, 0.01: 1.00, 0.02: 0.90, 0.05: 0.70}
HYBRID_ADAPTIVE_STAGE_WEIGHTS = {
    0.01: {
        'state_s0': 1.10,
        'state_sg': 2.05,
        'forward_max': DUALDEC_WEIGHTS['forward_max'],
        'corr_s0': 0.09,
        'corr_sg': 0.38,
        'amp_s0': 0.07,
        'amp_sg': 0.32,
    },
    0.02: {
        'state_s0': 1.05,
        'state_sg': 2.20,
        'forward_max': DUALDEC_WEIGHTS['forward_max'],
        'corr_s0': 0.08,
        'corr_sg': 0.42,
        'amp_s0': 0.06,
        'amp_sg': 0.34,
    },
    0.05: {
        'state_s0': 0.98,
        'state_sg': 2.35,
        'forward_max': DUALDEC_WEIGHTS['forward_max'] * 0.9,
        'corr_s0': 0.07,
        'corr_sg': 0.48,
        'amp_s0': 0.05,
        'amp_sg': 0.38,
    },
}


def train_conditioned_dualdec_case_weighted_lr(stage_noise, stage_epochs, loss_weights, lr_scale, init_state=None):
    train_ds, val_ds, test_payload = build_conditioned_dualdec_stage(stage_noise)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    g_load_fft_case, g_poro_fft_case = build_fft_kernels(H_SYN, W_SYN, PHYSICS, DEVICE)
    g_load_fft_case = g_load_fft_case * ALPHA_BAL
    g_poro_fft_case = g_poro_fft_case * BETA_BAL

    model = NoiseConditionedDualDecoderSwinUNet3D().to(DEVICE)
    if init_state is not None:
        model.load_state_dict(copy.deepcopy(init_state))
    lr_case = LR * lr_scale
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_case, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, stage_epochs), eta_min=lr_case * 0.1)

    best_state = copy.deepcopy(model.state_dict())
    best_val = float('inf')
    patience = min(PATIENCE + 3, stage_epochs)
    wait = 0
    history = []

    for epoch in range(stage_epochs):
        tr = run_conditioned_dualdec_epoch_weighted(
            model, train_loader, optimizer, training=True, epoch_idx=epoch, total_epochs=stage_epochs,
            y_mean=test_payload['y_mean'], y_std=test_payload['y_std'], u_mean=test_payload['u_mean'], u_std=test_payload['u_std'],
            g_load_fft=g_load_fft_case, g_poro_fft=g_poro_fft_case, loss_weights=loss_weights,
        )
        va = run_conditioned_dualdec_epoch_weighted(
            model, val_loader, optimizer, training=False, epoch_idx=epoch, total_epochs=stage_epochs,
            y_mean=test_payload['y_mean'], y_std=test_payload['y_std'], u_mean=test_payload['u_mean'], u_std=test_payload['u_std'],
            g_load_fft=g_load_fft_case, g_poro_fft=g_poro_fft_case, loss_weights=loss_weights,
        )
        scheduler.step()
        history.append({'noise_scale': float(stage_noise), 'epoch': epoch + 1, 'train_loss': tr['loss'], 'val_loss': va['loss'], 'lr': scheduler.get_last_lr()[0]})
        if va['loss'] < best_val - 1e-5:
            best_val = va['loss']
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        x_test = test_payload['x_test']
        y_true = test_payload['y_true']
        d_true = test_payload['u_true']
        pred_norm = model(x_test)
        pred = denormalize_targets(pred_norm, test_payload['y_mean'], test_payload['y_std']).cpu().numpy()
        d_hat = forward_physics_two_layer_torch(torch.tensor(pred, dtype=torch.float32, device=DEVICE), g_load_fft_case, g_poro_fft_case, PHYSICS).cpu().numpy()[:, 0]

    rows = []
    for layer_idx, layer_name in enumerate(['S0', 'Sg']):
        yt = y_true[:, layer_idx]
        yp = pred[:, layer_idx]
        rows.append({
            'training_mode': 'hybrid_stage_adaptive_conditioned_dual_decoder_balanced',
            'noise_scale': float(stage_noise),
            'layer': layer_name,
            'rmse': rmse(yt, yp),
            'mae': mae(yt, yp),
            'bias': bias_np(yt, yp),
            'nrmse': nrmse_np(yt, yp),
            'r2': r2_score_np(yt, yp),
            'corr': corr_np(yt, yp),
            'fit_slope': fit_slope_np(yt, yp),
            'fit_intercept': fit_intercept_np(yt, yp),
            'forward_residual_rmse': rmse(d_true, d_hat),
            'best_val_loss': best_val,
        })
    return rows, history, best_state


def train_hybrid_stage_adaptive_conditioned_dualdec():
    rows_all = []
    history_all = []
    state = None
    for stage_noise in BALANCED_NOISES:
        stage_epochs = HYBRID_ADAPTIVE_STAGE_EPOCHS[round(float(stage_noise), 2)]
        if stage_noise == 0.0:
            rows_case, hist_case, state = train_refined_conditioned_dualdec_case(stage_noise, stage_epochs, init_state=state)
            for row in rows_case:
                row['training_mode'] = 'hybrid_stage_adaptive_conditioned_dual_decoder_balanced'
            for item in hist_case:
                item['training_mode'] = 'hybrid_stage_adaptive_conditioned_dual_decoder_balanced'
        else:
            stage_weights = dict(DUALDEC_WEIGHTS)
            stage_weights.update(HYBRID_ADAPTIVE_STAGE_WEIGHTS[round(float(stage_noise), 2)])
            rows_case, hist_case, state = train_conditioned_dualdec_case_weighted_lr(
                stage_noise,
                stage_epochs,
                stage_weights,
                HYBRID_ADAPTIVE_LR_SCALE[round(float(stage_noise), 2)],
                init_state=state,
            )
            for item in hist_case:
                item['training_mode'] = 'hybrid_stage_adaptive_conditioned_dual_decoder_balanced'
        rows_all.extend(rows_case)
        history_all.extend(hist_case)
    return rows_all, history_all


hybrid_adapt_rows, hybrid_adapt_history = train_hybrid_stage_adaptive_conditioned_dualdec()
hybrid_adapt_metrics_df = pd.DataFrame(hybrid_adapt_rows)
hybrid_adapt_history_df = pd.DataFrame(hybrid_adapt_history)
display(hybrid_adapt_metrics_df)
hybrid_adapt_metrics_csv = OUT_DIR / 'synthetic_two_layer_balanced_conditioned_dualdec_hybrid_adaptive_metrics.csv'
hybrid_adapt_history_csv = OUT_DIR / 'synthetic_two_layer_balanced_conditioned_dualdec_hybrid_adaptive_history.csv'
hybrid_adapt_json = OUT_DIR / 'synthetic_two_layer_balanced_conditioned_dualdec_hybrid_adaptive_summary.json'
hybrid_adapt_metrics_df.to_csv(hybrid_adapt_metrics_csv, index=False)
hybrid_adapt_history_df.to_csv(hybrid_adapt_history_csv, index=False)
with hybrid_adapt_json.open('w') as f:
    json.dump({'metrics': hybrid_adapt_rows}, f, indent=2)
print('Saved hybrid adaptive conditioned metrics:', hybrid_adapt_metrics_csv)
print('Saved hybrid adaptive conditioned history:', hybrid_adapt_history_csv)
print('Saved hybrid adaptive conditioned summary:', hybrid_adapt_json)"""
            ),
            md_cell(
                """## Hybrid Stage-Adaptive Results Summary

This compares the hybrid noisy-stage `Sg`-emphasis refinement against the new stage-adaptive version that increases `Sg` emphasis and reduces learning rate only as noise becomes larger."""
            ),
            code_cell(
                """# Compare hybrid stage-adaptive refinement against the hybrid Sg-emphasis baseline
hybrid_df = pd.read_csv(OUT_DIR / 'synthetic_two_layer_balanced_conditioned_dualdec_hybrid_metrics.csv')
hybrid_sg_df = pd.read_csv(OUT_DIR / 'synthetic_two_layer_balanced_conditioned_dualdec_hybrid_sg_metrics.csv')
hybrid_adapt_df = pd.read_csv(OUT_DIR / 'synthetic_two_layer_balanced_conditioned_dualdec_hybrid_adaptive_metrics.csv')
compare_df = pd.concat([hybrid_df, hybrid_sg_df, hybrid_adapt_df], ignore_index=True, sort=False)
display(compare_df)

fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex='col')
metric_specs = [('rmse','RMSE','tab:red'),('r2','R^2','tab:blue'),('corr','Correlation','tab:green'),('fit_slope','Slope','tab:purple')]
mode_styles = {
    'hybrid_clean_then_original_conditioned_dual_decoder_balanced': {'ls':'--','marker':'o', 'label':'hybrid'},
    'hybrid_sg_emphasis_conditioned_dual_decoder_balanced': {'ls':'-.','marker':'^', 'label':'hybrid + Sg emphasis'},
    'hybrid_stage_adaptive_conditioned_dual_decoder_balanced': {'ls':'-','marker':'D', 'label':'hybrid stage-adaptive'},
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
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False)
fig.suptitle('Balanced two-layer inversion: hybrid stage-adaptive noisy-stage refinement', fontsize=14)
fig.tight_layout(rect=[0,0.02,1,0.94])
plt.show()
fig_path = FIG_DIR / 'synthetic_two_layer_balanced_conditioned_dualdec_hybrid_adaptive_vs_hybrid.png'
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
print('Saved hybrid stage-adaptive comparison figure:', fig_path)"""
            ),
        ]
    )

    NOTEBOOK.write_text(json.dumps(nb, indent=1))


if __name__ == "__main__":
    main()
