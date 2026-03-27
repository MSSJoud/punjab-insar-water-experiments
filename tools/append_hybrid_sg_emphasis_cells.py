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
    marker = "## Hybrid Direct Conditioning With Noisy-Stage Sg Emphasis"
    if any(marker in "".join(cell.get("source", [])) for cell in nb["cells"]):
        return

    nb["cells"].extend(
        [
            md_cell(
                """## Hybrid Direct Conditioning With Noisy-Stage Sg Emphasis

This refinement stays on the hybrid direct-conditioned branch and changes only the noisy-stage loss balance. The clean stage is left unchanged, while later noisy stages place slightly more weight on the `Sg` state/correlation terms and slightly less on `S0`. The goal is to keep the improved clean behavior from the hybrid schedule while pushing the noisy groundwater recovery a bit further."""
            ),
            code_cell(
                """# Hybrid direct conditioning with noisy-stage Sg emphasis
import json

HYBRID_SG_STAGE_EPOCHS = [24, 12, 12, 12]
HYBRID_SG_WEIGHTS_CLEAN = dict(DUALDEC_WEIGHTS)
HYBRID_SG_WEIGHTS_NOISY = dict(DUALDEC_WEIGHTS)
HYBRID_SG_WEIGHTS_NOISY.update({
    'state_s0': 1.05,
    'state_sg': 2.20,
    'corr_s0': 0.08,
    'corr_sg': 0.42,
    'amp_s0': 0.06,
    'amp_sg': 0.34,
})


def run_conditioned_dualdec_epoch_weighted(model, loader, optimizer, *, training, epoch_idx, total_epochs, y_mean, y_std, u_mean, u_std, g_load_fft, g_poro_fft, loss_weights):
    model.train(mode=training)
    stats = {'loss': 0.0, 'state_s0': 0.0, 'state_sg': 0.0, 'forward': 0.0, 'corr_s0': 0.0, 'corr_sg': 0.0, 'amp_s0': 0.0, 'amp_sg': 0.0, 'n': 0}
    ramp = min(1.0, float(epoch_idx + 1) / max(1, total_epochs // 2))
    lambda_forward = loss_weights['forward_max'] * ramp

    for xb, yb, ub in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        ub = ub.to(DEVICE)
        with torch.set_grad_enabled(training):
            pred_norm = model(xb)
            pred_denorm = denormalize_targets(pred_norm, y_mean, y_std)
            y_true_denorm = denormalize_targets(yb, y_mean, y_std)
            state_s0 = F.mse_loss(pred_norm[:, 0:1], yb[:, 0:1])
            state_sg = F.mse_loss(pred_norm[:, 1:2], yb[:, 1:2])
            corr_s0 = 1.0 - batch_correlation_torch(pred_denorm[:, 0:1], y_true_denorm[:, 0:1])
            corr_sg = 1.0 - batch_correlation_torch(pred_denorm[:, 1:2], y_true_denorm[:, 1:2])
            amp_s0 = amplitude_penalty_torch(pred_denorm[:, 0:1], y_true_denorm[:, 0:1])
            amp_sg = amplitude_penalty_torch(pred_denorm[:, 1:2], y_true_denorm[:, 1:2])
            d_hat = forward_physics_two_layer_torch(pred_denorm, g_load_fft, g_poro_fft, PHYSICS)
            d_hat_norm = normalize_field(d_hat, u_mean, u_std)
            forward_loss = F.mse_loss(d_hat_norm, ub)
            loss = (
                loss_weights['state_s0'] * state_s0
                + loss_weights['state_sg'] * state_sg
                + lambda_forward * forward_loss
                + loss_weights['corr_s0'] * corr_s0
                + loss_weights['corr_sg'] * corr_sg
                + loss_weights['amp_s0'] * amp_s0
                + loss_weights['amp_sg'] * amp_sg
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        bs = xb.size(0)
        stats['loss'] += loss.item() * bs
        stats['state_s0'] += state_s0.item() * bs
        stats['state_sg'] += state_sg.item() * bs
        stats['forward'] += forward_loss.item() * bs
        stats['corr_s0'] += corr_s0.item() * bs
        stats['corr_sg'] += corr_sg.item() * bs
        stats['amp_s0'] += amp_s0.item() * bs
        stats['amp_sg'] += amp_sg.item() * bs
        stats['n'] += bs

    n = max(1, stats['n'])
    return {k: (v / n if k != 'n' else v) for k, v in stats.items()}


def train_conditioned_dualdec_case_weighted(stage_noise, stage_epochs, loss_weights, init_state=None):
    train_ds, val_ds, test_payload = build_conditioned_dualdec_stage(stage_noise)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    g_load_fft_case, g_poro_fft_case = build_fft_kernels(H_SYN, W_SYN, PHYSICS, DEVICE)
    g_load_fft_case = g_load_fft_case * ALPHA_BAL
    g_poro_fft_case = g_poro_fft_case * BETA_BAL

    model = NoiseConditionedDualDecoderSwinUNet3D().to(DEVICE)
    if init_state is not None:
        model.load_state_dict(copy.deepcopy(init_state))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, stage_epochs), eta_min=LR * 0.1)

    best_state = copy.deepcopy(model.state_dict())
    best_val = float('inf')
    patience = min(PATIENCE + 2, stage_epochs)
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
            'training_mode': 'hybrid_sg_emphasis_conditioned_dual_decoder_balanced',
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


def train_hybrid_sg_emphasis_conditioned_dualdec():
    rows_all = []
    history_all = []
    state = None
    for stage_idx, (stage_noise, stage_epochs) in enumerate(zip(BALANCED_NOISES, HYBRID_SG_STAGE_EPOCHS)):
        if stage_idx == 0:
            rows_case, hist_case, state = train_refined_conditioned_dualdec_case(stage_noise, stage_epochs, init_state=state)
            for row in rows_case:
                row['training_mode'] = 'hybrid_sg_emphasis_conditioned_dual_decoder_balanced'
            for item in hist_case:
                item['training_mode'] = 'hybrid_sg_emphasis_conditioned_dual_decoder_balanced'
        else:
            rows_case, hist_case, state = train_conditioned_dualdec_case_weighted(stage_noise, stage_epochs, HYBRID_SG_WEIGHTS_NOISY, init_state=state)
            for item in hist_case:
                item['training_mode'] = 'hybrid_sg_emphasis_conditioned_dual_decoder_balanced'
        rows_all.extend(rows_case)
        history_all.extend(hist_case)
    return rows_all, history_all


hybrid_sg_rows, hybrid_sg_history = train_hybrid_sg_emphasis_conditioned_dualdec()
hybrid_sg_metrics_df = pd.DataFrame(hybrid_sg_rows)
hybrid_sg_history_df = pd.DataFrame(hybrid_sg_history)
display(hybrid_sg_metrics_df)
hybrid_sg_metrics_csv = OUT_DIR / 'synthetic_two_layer_balanced_conditioned_dualdec_hybrid_sg_metrics.csv'
hybrid_sg_history_csv = OUT_DIR / 'synthetic_two_layer_balanced_conditioned_dualdec_hybrid_sg_history.csv'
hybrid_sg_json = OUT_DIR / 'synthetic_two_layer_balanced_conditioned_dualdec_hybrid_sg_summary.json'
hybrid_sg_metrics_df.to_csv(hybrid_sg_metrics_csv, index=False)
hybrid_sg_history_df.to_csv(hybrid_sg_history_csv, index=False)
with hybrid_sg_json.open('w') as f:
    json.dump({'metrics': hybrid_sg_rows}, f, indent=2)
print('Saved hybrid Sg-emphasis conditioned metrics:', hybrid_sg_metrics_csv)
print('Saved hybrid Sg-emphasis conditioned history:', hybrid_sg_history_csv)
print('Saved hybrid Sg-emphasis conditioned summary:', hybrid_sg_json)"""
            ),
            md_cell(
                """## Hybrid Sg-Emphasis Results Summary

This compares the plain dual-decoder baseline, the original direct-conditioned model, the hybrid direct-conditioned refinement, and the hybrid noisy-stage `Sg`-emphasis refinement."""
            ),
            code_cell(
                """# Compare hybrid Sg-emphasis refinement against the hybrid direct-conditioning baseline
dual_df = pd.read_csv(OUT_DIR / 'synthetic_two_layer_balanced_dualdec_frequency_metrics.csv')
cond_df = pd.read_csv(OUT_DIR / 'synthetic_two_layer_balanced_conditioned_dualdec_metrics.csv')
hybrid_df = pd.read_csv(OUT_DIR / 'synthetic_two_layer_balanced_conditioned_dualdec_hybrid_metrics.csv')
hybrid_sg_df = pd.read_csv(OUT_DIR / 'synthetic_two_layer_balanced_conditioned_dualdec_hybrid_sg_metrics.csv')
compare_df = pd.concat([dual_df, cond_df, hybrid_df, hybrid_sg_df], ignore_index=True, sort=False)
display(compare_df)

fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex='col')
metric_specs = [('rmse','RMSE','tab:red'),('r2','R^2','tab:blue'),('corr','Correlation','tab:green'),('fit_slope','Slope','tab:purple')]
mode_styles = {
    'dual_decoder_frequency_curriculum_balanced': {'ls':'--','marker':'o', 'label':'dual decoder'},
    'noise_conditioned_dual_decoder_balanced': {'ls':'-.','marker':'^', 'label':'original conditioned'},
    'hybrid_clean_then_original_conditioned_dual_decoder_balanced': {'ls':':','marker':'s', 'label':'hybrid conditioned'},
    'hybrid_sg_emphasis_conditioned_dual_decoder_balanced': {'ls':'-','marker':'D', 'label':'hybrid + Sg emphasis'},
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
fig.suptitle('Balanced two-layer inversion: hybrid direct-conditioning with noisy-stage Sg emphasis', fontsize=14)
fig.tight_layout(rect=[0,0.02,1,0.94])
plt.show()
fig_path = FIG_DIR / 'synthetic_two_layer_balanced_conditioned_dualdec_hybrid_sg_vs_baselines.png'
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
print('Saved hybrid Sg-emphasis comparison figure:', fig_path)"""
            ),
        ]
    )
    NOTEBOOK.write_text(json.dumps(nb, indent=1))


if __name__ == "__main__":
    main()
