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

    marker = "## Corrected Direct Noise-Conditioned Benchmark"
    if any(marker in "".join(cell.get("source", [])) for cell in nb["cells"]):
        return

    nb["cells"].extend(
        [
            md_cell(
                """## Corrected Direct Noise-Conditioned Benchmark

This benchmark keeps the direct input-conditioned dual-decoder branch, but fixes a conditioning bug from the earlier stage-wise implementation: the noise channel is now preserved as an explicit unnormalized signal instead of being normalized away inside each single-noise stage. The goal is to test the same direct-conditioning idea with a conditioning signal that the network can actually use."""
            ),
            code_cell(
                """# Corrected direct noise-conditioned benchmark
import json

REFINED_COND_STAGE_EPOCHS = [24, 14, 14, 14]
REFINED_COND_LR_SCALE = {0.00: 1.00, 0.01: 0.90, 0.02: 0.85, 0.05: 0.80}
MAX_BALANCED_NOISE = max(BALANCED_NOISES) if max(BALANCED_NOISES) > 0 else 1.0


def build_refined_conditioned_dualdec_stage(noise_scale):
    two_layer, clean_disp, _, _ = make_balanced_two_layer_deformation_local(layers_syn, PHYSICS, s0_scale=1.0, sg_scale=1.0)
    noisy_disp = add_two_layer_noise(clean_disp, noise_scale=noise_scale, seed_offset=int(noise_scale * 1000) + 1701)

    x_disp = []
    x_noise = []
    ys = []
    us = []
    noise_value = float(noise_scale) / float(MAX_BALANCED_NOISE)
    for end_idx in range(WINDOW_SIZE - 1, noisy_disp.shape[0]):
        start_idx = end_idx - WINDOW_SIZE + 1
        d_window = noisy_disp[start_idx:end_idx + 1]
        x_disp.append(d_window[None, ...])
        x_noise.append(np.full((1,) + d_window.shape, fill_value=noise_value, dtype=np.float32))
        ys.append(np.stack([two_layer[end_idx, 0], two_layer[end_idx, 1]], axis=0))
        us.append(noisy_disp[end_idx][None, ...])

    x_disp = torch.tensor(np.stack(x_disp), dtype=torch.float32)
    x_noise = torch.tensor(np.stack(x_noise), dtype=torch.float32)
    y = torch.tensor(np.stack(ys), dtype=torch.float32)
    u = torch.tensor(np.stack(us), dtype=torch.float32)

    n_total = x_disp.shape[0]
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)
    train_idx = np.arange(0, n_train)
    val_idx = np.arange(n_train, n_train + n_val)
    test_idx = np.arange(n_train + n_val, n_total)

    x_mean = x_disp[train_idx].mean(dim=(0, 2, 3, 4), keepdim=True)
    x_std = x_disp[train_idx].std(dim=(0, 2, 3, 4), keepdim=True).clamp_min(1e-6)
    x_disp_norm = (x_disp - x_mean) / x_std
    x_stage = torch.cat([x_disp_norm, x_noise], dim=1)

    y_mean = y[train_idx].mean(dim=(0, 2, 3), keepdim=True)
    y_std = y[train_idx].std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)
    y_norm = (y - y_mean) / y_std

    u_mean = u[train_idx].mean(dim=(0, 2, 3), keepdim=True)
    u_std = u[train_idx].std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)
    u_norm = normalize_field(u, u_mean, u_std)

    train_ds = TensorDataset(x_stage[train_idx], y_norm[train_idx], u_norm[train_idx])
    val_ds = TensorDataset(x_stage[val_idx], y_norm[val_idx], u_norm[val_idx])
    test_payload = {
        'x_test': x_stage[test_idx].to(DEVICE),
        'y_true': y[test_idx].cpu().numpy(),
        'u_true': u[test_idx].cpu().numpy()[:, 0],
        'y_mean': y_mean.to(DEVICE),
        'y_std': y_std.to(DEVICE),
        'u_mean': u_mean.to(DEVICE),
        'u_std': u_std.to(DEVICE),
        'noise_scale': float(noise_scale),
        'noise_condition_value': noise_value,
        'num_train_windows': int(len(train_idx)),
        'num_test_windows': int(len(test_idx)),
    }
    return train_ds, val_ds, test_payload


def train_refined_conditioned_dualdec_case(stage_noise, stage_epochs, init_state=None):
    train_ds, val_ds, test_payload = build_refined_conditioned_dualdec_stage(stage_noise)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    g_load_fft_case, g_poro_fft_case = build_fft_kernels(H_SYN, W_SYN, PHYSICS, DEVICE)
    g_load_fft_case = g_load_fft_case * ALPHA_BAL
    g_poro_fft_case = g_poro_fft_case * BETA_BAL

    model = NoiseConditionedDualDecoderSwinUNet3D().to(DEVICE)
    if init_state is not None:
        model.load_state_dict(copy.deepcopy(init_state))
    lr_case = LR * REFINED_COND_LR_SCALE.get(round(float(stage_noise), 2), 1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_case, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, stage_epochs), eta_min=lr_case * 0.1)

    best_state = copy.deepcopy(model.state_dict())
    best_val = float('inf')
    patience = min(PATIENCE + 3, stage_epochs)
    wait = 0
    history = []

    for epoch in range(stage_epochs):
        tr = run_conditioned_dualdec_epoch(
            model, train_loader, optimizer, training=True, epoch_idx=epoch, total_epochs=stage_epochs,
            y_mean=test_payload['y_mean'], y_std=test_payload['y_std'],
            u_mean=test_payload['u_mean'], u_std=test_payload['u_std'],
            g_load_fft=g_load_fft_case, g_poro_fft=g_poro_fft_case,
        )
        va = run_conditioned_dualdec_epoch(
            model, val_loader, optimizer, training=False, epoch_idx=epoch, total_epochs=stage_epochs,
            y_mean=test_payload['y_mean'], y_std=test_payload['y_std'],
            u_mean=test_payload['u_mean'], u_std=test_payload['u_std'],
            g_load_fft=g_load_fft_case, g_poro_fft=g_poro_fft_case,
        )
        scheduler.step()
        history.append({
            'noise_scale': float(stage_noise),
            'epoch': epoch + 1,
            'train_loss': tr['loss'],
            'val_loss': va['loss'],
            'lr': scheduler.get_last_lr()[0],
            'noise_condition_value': test_payload['noise_condition_value'],
        })
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
        pred_norm = model(test_payload['x_test'])
        pred = denormalize_targets(pred_norm, test_payload['y_mean'], test_payload['y_std']).cpu().numpy()
        d_hat = forward_physics_two_layer_torch(
            torch.tensor(pred, dtype=torch.float32, device=DEVICE),
            g_load_fft_case,
            g_poro_fft_case,
            PHYSICS,
        ).cpu().numpy()[:, 0]

    rows = []
    for layer_idx, layer_name in enumerate(['S0', 'Sg']):
        yt = test_payload['y_true'][:, layer_idx]
        yp = pred[:, layer_idx]
        rows.append({
            'training_mode': 'refined_noise_conditioned_dual_decoder_balanced',
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
            'forward_residual_rmse': rmse(test_payload['u_true'], d_hat),
            'noise_level_condition': test_payload['noise_scale'],
            'noise_condition_value': test_payload['noise_condition_value'],
            'best_val_loss': best_val,
            'num_train_windows': test_payload['num_train_windows'],
            'num_test_windows': test_payload['num_test_windows'],
        })
    return rows, history, best_state


refined_cond_rows = []
refined_cond_history = []
refined_cond_state = None
for stage_noise, stage_epochs in zip(BALANCED_NOISES, REFINED_COND_STAGE_EPOCHS):
    rows_case, hist_case, refined_cond_state = train_refined_conditioned_dualdec_case(stage_noise, stage_epochs, init_state=refined_cond_state)
    refined_cond_rows.extend(rows_case)
    refined_cond_history.extend(hist_case)

refined_cond_metrics_df = pd.DataFrame(refined_cond_rows)
refined_cond_history_df = pd.DataFrame(refined_cond_history)
display(refined_cond_metrics_df)
refined_cond_metrics_csv = OUT_DIR / 'synthetic_two_layer_balanced_conditioned_dualdec_refined_metrics.csv'
refined_cond_history_csv = OUT_DIR / 'synthetic_two_layer_balanced_conditioned_dualdec_refined_history.csv'
refined_cond_json = OUT_DIR / 'synthetic_two_layer_balanced_conditioned_dualdec_refined_summary.json'
refined_cond_metrics_df.to_csv(refined_cond_metrics_csv, index=False)
refined_cond_history_df.to_csv(refined_cond_history_csv, index=False)
with refined_cond_json.open('w') as f:
    json.dump({'metrics': refined_cond_rows}, f, indent=2)
print('Saved refined conditioned dual-decoder metrics:', refined_cond_metrics_csv)
print('Saved refined conditioned dual-decoder history:', refined_cond_history_csv)
print('Saved refined conditioned dual-decoder summary:', refined_cond_json)"""
            ),
            md_cell(
                """## Corrected Direct Conditioning Results Summary

This compares the plain dual-decoder baseline, the original direct noise-conditioned model, and the corrected direct-conditioned variant that preserves the conditioning signal instead of normalizing it away inside each single-noise stage."""
            ),
            code_cell(
                """# Compare corrected direct-conditioned results against the baseline and original conditioned model
dual_df = pd.read_csv(OUT_DIR / 'synthetic_two_layer_balanced_dualdec_frequency_metrics.csv')
cond_df = pd.read_csv(OUT_DIR / 'synthetic_two_layer_balanced_conditioned_dualdec_metrics.csv')
refined_cond_df = pd.read_csv(OUT_DIR / 'synthetic_two_layer_balanced_conditioned_dualdec_refined_metrics.csv')
compare_df = pd.concat([dual_df, cond_df, refined_cond_df], ignore_index=True, sort=False)
display(compare_df)

fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex='col')
metric_specs = [('rmse','RMSE','tab:red'),('r2','R^2','tab:blue'),('corr','Correlation','tab:green'),('fit_slope','Slope','tab:purple')]
mode_styles = {
    'dual_decoder_frequency_curriculum_balanced': {'ls':'--','marker':'o', 'label':'dual decoder'},
    'noise_conditioned_dual_decoder_balanced': {'ls':'-.','marker':'^', 'label':'original conditioned'},
    'refined_noise_conditioned_dual_decoder_balanced': {'ls':'-','marker':'D', 'label':'corrected conditioned'},
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
fig.suptitle('Balanced two-layer inversion: baseline vs original direct conditioning vs corrected direct conditioning', fontsize=14)
fig.tight_layout(rect=[0,0.02,1,0.94])
plt.show()
fig_path = FIG_DIR / 'synthetic_two_layer_balanced_conditioned_dualdec_refined_vs_baselines.png'
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
print('Saved corrected conditioned comparison figure:', fig_path)"""
            ),
        ]
    )

    NOTEBOOK.write_text(json.dumps(nb, indent=1))


if __name__ == "__main__":
    main()
