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
    marker = "## Cumulative Direct Noise-Conditioned Benchmark"
    if any(marker in "".join(cell.get("source", [])) for cell in nb["cells"]):
        return

    nb["cells"].extend(
        [
            md_cell(
                """## Cumulative Direct Noise-Conditioned Benchmark

This refinement stays on the direct input-conditioned dual-decoder branch, but changes the stage curriculum: each nominal stage is trained on a cumulative mixture of all lower noise levels up to that stage. This gives the conditioning channel real variation within a stage while preserving access to cleaner examples during noisy-stage adaptation."""
            ),
            code_cell(
                """# Cumulative direct noise-conditioned benchmark
import json

CUMCOND_STAGE_EPOCHS = [24, 14, 14, 14]
CUMCOND_LR_SCALE = {0.00: 1.00, 0.01: 0.92, 0.02: 0.88, 0.05: 0.82}


def build_single_conditioned_arrays(noise_scale, seed_offset):
    two_layer, clean_disp, _, _ = make_balanced_two_layer_deformation_local(layers_syn, PHYSICS, s0_scale=1.0, sg_scale=1.0)
    noisy_disp = add_two_layer_noise(clean_disp, noise_scale=noise_scale, seed_offset=seed_offset)

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
    return (
        torch.tensor(np.stack(x_disp), dtype=torch.float32),
        torch.tensor(np.stack(x_noise), dtype=torch.float32),
        torch.tensor(np.stack(ys), dtype=torch.float32),
        torch.tensor(np.stack(us), dtype=torch.float32),
    )


def build_cumulative_conditioned_stage(stage_noise):
    allowed_noises = [n for n in BALANCED_NOISES if n <= stage_noise + 1e-9]
    x_disp_parts = []
    x_noise_parts = []
    y_parts = []
    u_parts = []
    split_train = []
    split_val = []
    split_test = []

    for i, noise_scale in enumerate(allowed_noises):
        x_disp_i, x_noise_i, y_i, u_i = build_single_conditioned_arrays(noise_scale, seed_offset=int(noise_scale * 1000) + 3100 + i * 97)
        n_total = x_disp_i.shape[0]
        n_train = int(0.70 * n_total)
        n_val = int(0.15 * n_total)
        idx = np.arange(n_total)
        split_train.append(idx[:n_train] + (0 if not x_disp_parts else sum(p.shape[0] for p in x_disp_parts)))
        split_val.append(idx[n_train:n_train + n_val] + (0 if not x_disp_parts else sum(p.shape[0] for p in x_disp_parts)))
        split_test.append(idx[n_train + n_val:] + (0 if not x_disp_parts else sum(p.shape[0] for p in x_disp_parts)))
        x_disp_parts.append(x_disp_i)
        x_noise_parts.append(x_noise_i)
        y_parts.append(y_i)
        u_parts.append(u_i)

    x_disp = torch.cat(x_disp_parts, dim=0)
    x_noise = torch.cat(x_noise_parts, dim=0)
    y = torch.cat(y_parts, dim=0)
    u = torch.cat(u_parts, dim=0)
    train_idx = np.concatenate(split_train)
    val_idx = np.concatenate(split_val)
    test_idx = np.concatenate(split_test)

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

    target_mask = np.isclose(x_noise[test_idx, 0, 0, 0, 0].cpu().numpy() * float(MAX_BALANCED_NOISE), stage_noise, atol=1e-6)
    target_test_idx = test_idx[target_mask]
    test_payload = {
        'x_test': x_stage[target_test_idx].to(DEVICE),
        'y_true': y[target_test_idx].cpu().numpy(),
        'u_true': u[target_test_idx].cpu().numpy()[:, 0],
        'y_mean': y_mean.to(DEVICE),
        'y_std': y_std.to(DEVICE),
        'u_mean': u_mean.to(DEVICE),
        'u_std': u_std.to(DEVICE),
        'noise_scale': float(stage_noise),
        'noise_condition_value_mean': float(x_noise[train_idx].mean().cpu()),
        'num_train_windows': int(len(train_idx)),
        'num_test_windows': int(len(target_test_idx)),
        'mixture_noises': [float(n) for n in allowed_noises],
    }
    return train_ds, val_ds, test_payload


def train_cumulative_conditioned_dualdec_case(stage_noise, stage_epochs, init_state=None):
    train_ds, val_ds, test_payload = build_cumulative_conditioned_stage(stage_noise)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    g_load_fft_case, g_poro_fft_case = build_fft_kernels(H_SYN, W_SYN, PHYSICS, DEVICE)
    g_load_fft_case = g_load_fft_case * ALPHA_BAL
    g_poro_fft_case = g_poro_fft_case * BETA_BAL

    model = NoiseConditionedDualDecoderSwinUNet3D().to(DEVICE)
    if init_state is not None:
        model.load_state_dict(copy.deepcopy(init_state))
    lr_case = LR * CUMCOND_LR_SCALE.get(round(float(stage_noise), 2), 1.0)
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
            'noise_condition_value_mean': test_payload['noise_condition_value_mean'],
            'mixture_noises': ','.join(f'{n:.2f}' for n in test_payload['mixture_noises']),
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
            'training_mode': 'cumulative_noise_conditioned_dual_decoder_balanced',
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
            'noise_condition_value_mean': test_payload['noise_condition_value_mean'],
            'best_val_loss': best_val,
            'num_train_windows': test_payload['num_train_windows'],
            'num_test_windows': test_payload['num_test_windows'],
            'mixture_noises': ','.join(f'{n:.2f}' for n in test_payload['mixture_noises']),
        })
    return rows, history, best_state


cumcond_rows = []
cumcond_history = []
cumcond_state = None
for stage_noise, stage_epochs in zip(BALANCED_NOISES, CUMCOND_STAGE_EPOCHS):
    rows_case, hist_case, cumcond_state = train_cumulative_conditioned_dualdec_case(stage_noise, stage_epochs, init_state=cumcond_state)
    cumcond_rows.extend(rows_case)
    cumcond_history.extend(hist_case)

cumcond_metrics_df = pd.DataFrame(cumcond_rows)
cumcond_history_df = pd.DataFrame(cumcond_history)
display(cumcond_metrics_df)
cumcond_metrics_csv = OUT_DIR / 'synthetic_two_layer_balanced_conditioned_dualdec_cumulative_metrics.csv'
cumcond_history_csv = OUT_DIR / 'synthetic_two_layer_balanced_conditioned_dualdec_cumulative_history.csv'
cumcond_json = OUT_DIR / 'synthetic_two_layer_balanced_conditioned_dualdec_cumulative_summary.json'
cumcond_metrics_df.to_csv(cumcond_metrics_csv, index=False)
cumcond_history_df.to_csv(cumcond_history_csv, index=False)
with cumcond_json.open('w') as f:
    json.dump({'metrics': cumcond_rows}, f, indent=2)
print('Saved cumulative conditioned dual-decoder metrics:', cumcond_metrics_csv)
print('Saved cumulative conditioned dual-decoder history:', cumcond_history_csv)
print('Saved cumulative conditioned dual-decoder summary:', cumcond_json)"""
            ),
            md_cell(
                """## Cumulative Direct Conditioning Results Summary

This compares the plain dual-decoder baseline, the original direct-conditioned model, the corrected direct-conditioned model, and the cumulative direct-conditioned refinement that mixes lower noise levels into each later stage."""
            ),
            code_cell(
                """# Compare cumulative direct-conditioned results against prior direct-conditioning variants
dual_df = pd.read_csv(OUT_DIR / 'synthetic_two_layer_balanced_dualdec_frequency_metrics.csv')
cond_df = pd.read_csv(OUT_DIR / 'synthetic_two_layer_balanced_conditioned_dualdec_metrics.csv')
refined_cond_df = pd.read_csv(OUT_DIR / 'synthetic_two_layer_balanced_conditioned_dualdec_refined_metrics.csv')
cumcond_df = pd.read_csv(OUT_DIR / 'synthetic_two_layer_balanced_conditioned_dualdec_cumulative_metrics.csv')
compare_df = pd.concat([dual_df, cond_df, refined_cond_df, cumcond_df], ignore_index=True, sort=False)
display(compare_df)

fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex='col')
metric_specs = [('rmse','RMSE','tab:red'),('r2','R^2','tab:blue'),('corr','Correlation','tab:green'),('fit_slope','Slope','tab:purple')]
mode_styles = {
    'dual_decoder_frequency_curriculum_balanced': {'ls':'--','marker':'o', 'label':'dual decoder'},
    'noise_conditioned_dual_decoder_balanced': {'ls':'-.','marker':'^', 'label':'original conditioned'},
    'refined_noise_conditioned_dual_decoder_balanced': {'ls':':','marker':'s', 'label':'corrected conditioned'},
    'cumulative_noise_conditioned_dual_decoder_balanced': {'ls':'-','marker':'D', 'label':'cumulative conditioned'},
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
fig.suptitle('Balanced two-layer inversion: cumulative direct-conditioning refinement', fontsize=14)
fig.tight_layout(rect=[0,0.02,1,0.94])
plt.show()
fig_path = FIG_DIR / 'synthetic_two_layer_balanced_conditioned_dualdec_cumulative_vs_baselines.png'
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
print('Saved cumulative conditioned comparison figure:', fig_path)"""
            ),
        ]
    )
    NOTEBOOK.write_text(json.dumps(nb, indent=1))


if __name__ == "__main__":
    main()
