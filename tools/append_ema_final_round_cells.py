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
    marker = "## Final Round: EMA-Stabilized Hybrid Validation"
    if any(marker in "".join(cell.get("source", [])) for cell in nb["cells"]):
        return

    nb["cells"].extend(
        [
            md_cell(
                """## Final Round: EMA-Stabilized Hybrid Validation

This final round keeps the current best branch fixed:

- hybrid direct conditioning
- corrected clean stage
- noisy-stage `Sg` emphasis

and adds one training stabilizer only: an exponential moving average (EMA) of the model weights during training.

If \\(\\theta_t\\) are the instantaneous parameters and \\(\\bar\\theta_t\\) are the EMA parameters, then

\\[
\\bar\\theta_t = \\beta \\bar\\theta_{t-1} + (1-\\beta)\\theta_t
\\]

with \\(\\beta\\) close to 1. The intent is to reduce seed sensitivity without changing the architecture or the synthetic formulation."""
            ),
            code_cell(
                """# Final round: EMA-stabilized hybrid validation
import json
import random

EMA_BETA = 0.995
FINAL_SEEDS = [7, 21, 42]


def set_all_seeds_final(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clone_state_dict(state_dict):
    return {k: v.detach().clone() for k, v in state_dict.items()}


def ema_update_(ema_state, model, beta=EMA_BETA):
    with torch.no_grad():
        model_state = model.state_dict()
        for key, value in model_state.items():
            ema_state[key].mul_(beta).add_(value.detach(), alpha=(1.0 - beta))


def load_state_dict_from_tensors(model, state_dict):
    model.load_state_dict({k: v.clone() for k, v in state_dict.items()})


def train_conditioned_case_weighted_ema(stage_noise, stage_epochs, loss_weights, seed, *, init_state=None, clean_stage=False):
    set_all_seeds_final(seed)
    if clean_stage:
        train_ds, val_ds, test_payload = build_refined_conditioned_dualdec_stage(stage_noise)
        lr_case = LR * REFINED_COND_LR_SCALE.get(round(float(stage_noise), 2), 1.0)
    else:
        train_ds, val_ds, test_payload = build_conditioned_dualdec_stage_seeded(stage_noise, seed)
        lr_case = LR

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    g_load_fft_case, g_poro_fft_case = build_fft_kernels(H_SYN, W_SYN, PHYSICS, DEVICE)
    g_load_fft_case = g_load_fft_case * ALPHA_BAL
    g_poro_fft_case = g_poro_fft_case * BETA_BAL

    model = NoiseConditionedDualDecoderSwinUNet3D().to(DEVICE)
    if init_state is not None:
        model.load_state_dict(copy.deepcopy(init_state))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_case, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, stage_epochs), eta_min=lr_case * 0.1)

    ema_state = clone_state_dict(model.state_dict())
    best_eval_state = clone_state_dict(model.state_dict())
    best_val = float('inf')
    patience = min(PATIENCE + 2, stage_epochs)
    wait = 0
    history = []

    for epoch in range(stage_epochs):
        tr = run_conditioned_dualdec_epoch_weighted(
            model, train_loader, optimizer, training=True, epoch_idx=epoch, total_epochs=stage_epochs,
            y_mean=test_payload['y_mean'], y_std=test_payload['y_std'],
            u_mean=test_payload['u_mean'], u_std=test_payload['u_std'],
            g_load_fft=g_load_fft_case, g_poro_fft=g_poro_fft_case, loss_weights=loss_weights,
        )
        ema_update_(ema_state, model, beta=EMA_BETA)

        eval_model = NoiseConditionedDualDecoderSwinUNet3D().to(DEVICE)
        load_state_dict_from_tensors(eval_model, ema_state)
        va = run_conditioned_dualdec_epoch_weighted(
            eval_model, val_loader, optimizer=None, training=False, epoch_idx=epoch, total_epochs=stage_epochs,
            y_mean=test_payload['y_mean'], y_std=test_payload['y_std'],
            u_mean=test_payload['u_mean'], u_std=test_payload['u_std'],
            g_load_fft=g_load_fft_case, g_poro_fft=g_poro_fft_case, loss_weights=loss_weights,
        )
        scheduler.step()
        history.append({'seed': int(seed), 'noise_scale': float(stage_noise), 'epoch': epoch + 1, 'train_loss': tr['loss'], 'val_loss': va['loss'], 'lr': scheduler.get_last_lr()[0]})
        if va['loss'] < best_val - 1e-5:
            best_val = va['loss']
            best_eval_state = clone_state_dict(ema_state)
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    final_model = NoiseConditionedDualDecoderSwinUNet3D().to(DEVICE)
    load_state_dict_from_tensors(final_model, best_eval_state)
    final_model.eval()
    with torch.no_grad():
        x_test = test_payload['x_test']
        y_true = test_payload['y_true']
        d_true = test_payload['u_true']
        pred_norm = final_model(x_test)
        pred = denormalize_targets(pred_norm, test_payload['y_mean'], test_payload['y_std']).cpu().numpy()
        d_hat = forward_physics_two_layer_torch(torch.tensor(pred, dtype=torch.float32, device=DEVICE), g_load_fft_case, g_poro_fft_case, PHYSICS).cpu().numpy()[:, 0]

    rows = []
    for layer_idx, layer_name in enumerate(['S0', 'Sg']):
        yt = y_true[:, layer_idx]
        yp = pred[:, layer_idx]
        rows.append({
            'training_mode': 'hybrid_sg_emphasis_ema_multiseed_validation',
            'seed': int(seed),
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
    return rows, history, best_eval_state


final_rows = []
final_history = []
for seed in FINAL_SEEDS:
    stage_state = None
    rows_case, hist_case, stage_state = train_conditioned_case_weighted_ema(
        0.0, 24, HYBRID_SG_WEIGHTS_CLEAN, seed, init_state=stage_state, clean_stage=True
    )
    final_rows.extend(rows_case)
    final_history.extend(hist_case)
    for stage_noise in [0.01, 0.02, 0.05]:
        rows_case, hist_case, stage_state = train_conditioned_case_weighted_ema(
            stage_noise, 12, HYBRID_SG_WEIGHTS_NOISY, seed, init_state=stage_state, clean_stage=False
        )
        final_rows.extend(rows_case)
        final_history.extend(hist_case)

final_metrics_df = pd.DataFrame(final_rows)
final_history_df = pd.DataFrame(final_history)
final_summary_df = (
    final_metrics_df
    .groupby(['noise_scale', 'layer'], as_index=False)
    .agg(
        rmse_mean=('rmse', 'mean'),
        rmse_std=('rmse', 'std'),
        r2_mean=('r2', 'mean'),
        r2_std=('r2', 'std'),
        corr_mean=('corr', 'mean'),
        corr_std=('corr', 'std'),
        fit_slope_mean=('fit_slope', 'mean'),
        fit_slope_std=('fit_slope', 'std'),
    )
)
display(final_metrics_df)
display(final_summary_df)
final_metrics_csv = OUT_DIR / 'synthetic_two_layer_balanced_conditioned_dualdec_hybrid_sg_ema_multiseed_metrics.csv'
final_history_csv = OUT_DIR / 'synthetic_two_layer_balanced_conditioned_dualdec_hybrid_sg_ema_multiseed_history.csv'
final_summary_csv = OUT_DIR / 'synthetic_two_layer_balanced_conditioned_dualdec_hybrid_sg_ema_multiseed_summary.csv'
final_json = OUT_DIR / 'synthetic_two_layer_balanced_conditioned_dualdec_hybrid_sg_ema_multiseed_summary.json'
final_metrics_df.to_csv(final_metrics_csv, index=False)
final_history_df.to_csv(final_history_csv, index=False)
final_summary_df.to_csv(final_summary_csv, index=False)
with final_json.open('w') as f:
    json.dump({'metrics': final_rows}, f, indent=2)
print('Saved EMA final-round metrics:', final_metrics_csv)
print('Saved EMA final-round history:', final_history_csv)
print('Saved EMA final-round summary:', final_summary_csv)
print('Saved EMA final-round json:', final_json)"""
            ),
            md_cell(
                """## Final Round Results Summary

This compares the previous best-branch multi-seed summary against the EMA-stabilized final round. The question is whether EMA improves cross-seed stability enough to justify stopping the model refinement loop."""
            ),
            code_cell(
                """# Compare the previous best-branch multi-seed summary against the EMA final round
prev_df = pd.read_csv(OUT_DIR / 'synthetic_two_layer_balanced_conditioned_dualdec_hybrid_sg_multiseed_summary.csv')
ema_df = pd.read_csv(OUT_DIR / 'synthetic_two_layer_balanced_conditioned_dualdec_hybrid_sg_ema_multiseed_summary.csv')
prev_df['summary_mode'] = 'hybrid_sg_multiseed'
ema_df['summary_mode'] = 'hybrid_sg_ema_multiseed'
compare_df = pd.concat([prev_df, ema_df], ignore_index=True, sort=False)
display(compare_df)

fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex='col')
metric_specs = [
    ('rmse_mean', 'rmse_std', 'RMSE', 'tab:red'),
    ('r2_mean', 'r2_std', 'R^2', 'tab:blue'),
    ('corr_mean', 'corr_std', 'Correlation', 'tab:green'),
    ('fit_slope_mean', 'fit_slope_std', 'Slope', 'tab:purple'),
]
styles = {
    'hybrid_sg_multiseed': {'ls':'--', 'marker':'o', 'label':'best branch'},
    'hybrid_sg_ema_multiseed': {'ls':'-', 'marker':'D', 'label':'EMA final round'},
}
for row_idx, layer_name in enumerate(['S0', 'Sg']):
    layer_df = compare_df[compare_df['layer'] == layer_name]
    for col_idx, (mean_col, std_col, title, color) in enumerate(metric_specs):
        ax = axes[row_idx, col_idx]
        for mode_name, style in styles.items():
            dfp = layer_df[layer_df['summary_mode'] == mode_name].sort_values('noise_scale')
            x = dfp['noise_scale'].to_numpy()
            y = dfp[mean_col].to_numpy()
            s = dfp[std_col].fillna(0.0).to_numpy()
            ax.plot(x, y, color=color, lw=2.1, ls=style['ls'], marker=style['marker'], label=style['label'])
            ax.fill_between(x, y - s, y + s, color=color, alpha=0.12)
        ax.set_title(f'{layer_name} {title}')
        ax.set_xlabel('Noise scale')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(sorted(compare_df['noise_scale'].unique()))
        if 'r2' in mean_col or 'corr' in mean_col or 'slope' in mean_col:
            ax.axhline(0.0, color='black', lw=1, ls=':')
        if 'slope' in mean_col:
            ax.axhline(1.0, color='black', lw=1, ls='--', alpha=0.8)
handles, labels = axes[0,0].get_legend_handles_labels()
if handles:
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False)
fig.suptitle('Final round: multi-seed mean ± std before vs after EMA stabilization', fontsize=14)
fig.tight_layout(rect=[0,0.02,1,0.94])
plt.show()
fig_path = FIG_DIR / 'synthetic_two_layer_balanced_conditioned_dualdec_hybrid_sg_ema_multiseed.png'
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
print('Saved EMA final-round figure:', fig_path)"""
            ),
        ]
    )
    NOTEBOOK.write_text(json.dumps(nb, indent=1))


if __name__ == "__main__":
    main()
