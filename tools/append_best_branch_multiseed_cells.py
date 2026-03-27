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
    marker = "## Best-Branch Multi-Seed Validation"
    if any(marker in "".join(cell.get("source", [])) for cell in nb["cells"]):
        return

    nb["cells"].extend(
        [
            md_cell(
                """## Best-Branch Multi-Seed Validation

This final synthetic validation step keeps the current best branch fixed:

- hybrid direct conditioning
- corrected clean stage
- noisy-stage `Sg` emphasis

and evaluates it across multiple random seeds. The objective is to test whether the observed gains are stable enough to support reporting this branch as the current candidate model."""
            ),
            code_cell(
                """# Best-branch multi-seed validation
import json
import random

BEST_BRANCH_SEEDS = [7, 21, 42]


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_conditioned_dualdec_stage_seeded(noise_scale, seed):
    two_layer, clean_disp, _, _ = make_balanced_two_layer_deformation_local(layers_syn, PHYSICS, s0_scale=1.0, sg_scale=1.0)
    noisy_disp = add_two_layer_noise(clean_disp, noise_scale=noise_scale, seed_offset=int(noise_scale * 1000) + 711 + seed * 1000)

    xs = []
    ys = []
    us = []
    for end_idx in range(WINDOW_SIZE - 1, noisy_disp.shape[0]):
        start_idx = end_idx - WINDOW_SIZE + 1
        d_window = noisy_disp[start_idx:end_idx + 1]
        noise_window = np.full_like(d_window, fill_value=noise_scale, dtype=np.float32)
        xs.append(np.stack([d_window, noise_window], axis=0))
        ys.append(np.stack([two_layer[end_idx, 0], two_layer[end_idx, 1]], axis=0))
        us.append(noisy_disp[end_idx][None, ...])
    x = torch.tensor(np.stack(xs), dtype=torch.float32)
    y = torch.tensor(np.stack(ys), dtype=torch.float32)
    u = torch.tensor(np.stack(us), dtype=torch.float32)

    n_total = x.shape[0]
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)
    train_idx = np.arange(0, n_train)
    val_idx = np.arange(n_train, n_train + n_val)
    test_idx = np.arange(n_train + n_val, n_total)

    x_mean = x[train_idx].mean(dim=(0, 2, 3, 4), keepdim=True)
    x_std = x[train_idx].std(dim=(0, 2, 3, 4), keepdim=True).clamp_min(1e-6)
    x_norm = (x - x_mean) / x_std

    y_mean = y[train_idx].mean(dim=(0, 2, 3), keepdim=True)
    y_std = y[train_idx].std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)
    y_norm = (y - y_mean) / y_std

    u_mean = u[train_idx].mean(dim=(0, 2, 3), keepdim=True)
    u_std = u[train_idx].std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)
    u_norm = normalize_field(u, u_mean, u_std)

    train_ds = TensorDataset(x_norm[train_idx], y_norm[train_idx], u_norm[train_idx])
    val_ds = TensorDataset(x_norm[val_idx], y_norm[val_idx], u_norm[val_idx])
    test_payload = {
        'x_test': x_norm[test_idx].to(DEVICE),
        'y_true': y[test_idx].cpu().numpy(),
        'u_true': u[test_idx].cpu().numpy()[:, 0],
        'y_mean': y_mean.to(DEVICE),
        'y_std': y_std.to(DEVICE),
        'u_mean': u_mean.to(DEVICE),
        'u_std': u_std.to(DEVICE),
        'noise_scale': float(noise_scale),
        'num_train_windows': int(len(train_idx)),
        'num_test_windows': int(len(test_idx)),
    }
    return train_ds, val_ds, test_payload


def train_refined_conditioned_dualdec_case_seeded(stage_noise, stage_epochs, seed, init_state=None):
    set_all_seeds(seed)
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
        history.append({'seed': int(seed), 'noise_scale': float(stage_noise), 'epoch': epoch + 1, 'train_loss': tr['loss'], 'val_loss': va['loss'], 'lr': scheduler.get_last_lr()[0]})
        if va['loss'] < best_val - 1e-5:
            best_val = va['loss']
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return best_state, best_val, history


def train_conditioned_dualdec_case_weighted_seeded(stage_noise, stage_epochs, loss_weights, seed, init_state=None):
    set_all_seeds(seed)
    train_ds, val_ds, test_payload = build_conditioned_dualdec_stage_seeded(stage_noise, seed)
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
        history.append({'seed': int(seed), 'noise_scale': float(stage_noise), 'epoch': epoch + 1, 'train_loss': tr['loss'], 'val_loss': va['loss'], 'lr': scheduler.get_last_lr()[0]})
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
            'training_mode': 'hybrid_sg_emphasis_multiseed_validation',
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
    return rows, history, best_state


multiseed_rows = []
multiseed_history = []
for seed in BEST_BRANCH_SEEDS:
    branch_state = None
    clean_state, clean_best, clean_history = train_refined_conditioned_dualdec_case_seeded(0.0, 24, seed, init_state=branch_state)
    for layer_name, layer_idx in [('S0', 0), ('Sg', 1)]:
        pass
    branch_state = clean_state
    multiseed_history.extend(clean_history)

    # Evaluate clean stage explicitly
    _, _, clean_payload = build_refined_conditioned_dualdec_stage(0.0)
    g_load_fft_case, g_poro_fft_case = build_fft_kernels(H_SYN, W_SYN, PHYSICS, DEVICE)
    g_load_fft_case = g_load_fft_case * ALPHA_BAL
    g_poro_fft_case = g_poro_fft_case * BETA_BAL
    model = NoiseConditionedDualDecoderSwinUNet3D().to(DEVICE)
    model.load_state_dict(copy.deepcopy(branch_state))
    model.eval()
    with torch.no_grad():
        pred_norm = model(clean_payload['x_test'])
        pred = denormalize_targets(pred_norm, clean_payload['y_mean'], clean_payload['y_std']).cpu().numpy()
        d_hat = forward_physics_two_layer_torch(torch.tensor(pred, dtype=torch.float32, device=DEVICE), g_load_fft_case, g_poro_fft_case, PHYSICS).cpu().numpy()[:, 0]
    for layer_idx, layer_name in enumerate(['S0', 'Sg']):
        yt = clean_payload['y_true'][:, layer_idx]
        yp = pred[:, layer_idx]
        multiseed_rows.append({
            'training_mode': 'hybrid_sg_emphasis_multiseed_validation',
            'seed': int(seed),
            'noise_scale': 0.0,
            'layer': layer_name,
            'rmse': rmse(yt, yp),
            'mae': mae(yt, yp),
            'bias': bias_np(yt, yp),
            'nrmse': nrmse_np(yt, yp),
            'r2': r2_score_np(yt, yp),
            'corr': corr_np(yt, yp),
            'fit_slope': fit_slope_np(yt, yp),
            'fit_intercept': fit_intercept_np(yt, yp),
            'forward_residual_rmse': rmse(clean_payload['u_true'], d_hat),
            'best_val_loss': clean_best,
        })

    for stage_noise, stage_epochs in zip([0.01, 0.02, 0.05], [12, 12, 12]):
        rows_case, hist_case, branch_state = train_conditioned_dualdec_case_weighted_seeded(
            stage_noise,
            stage_epochs,
            HYBRID_SG_WEIGHTS_NOISY,
            seed,
            init_state=branch_state,
        )
        multiseed_rows.extend(rows_case)
        multiseed_history.extend(hist_case)

multiseed_metrics_df = pd.DataFrame(multiseed_rows)
multiseed_history_df = pd.DataFrame(multiseed_history)
multiseed_summary_df = (
    multiseed_metrics_df
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
display(multiseed_metrics_df)
display(multiseed_summary_df)
multiseed_metrics_csv = OUT_DIR / 'synthetic_two_layer_balanced_conditioned_dualdec_hybrid_sg_multiseed_metrics.csv'
multiseed_history_csv = OUT_DIR / 'synthetic_two_layer_balanced_conditioned_dualdec_hybrid_sg_multiseed_history.csv'
multiseed_summary_csv = OUT_DIR / 'synthetic_two_layer_balanced_conditioned_dualdec_hybrid_sg_multiseed_summary.csv'
multiseed_json = OUT_DIR / 'synthetic_two_layer_balanced_conditioned_dualdec_hybrid_sg_multiseed_summary.json'
multiseed_metrics_df.to_csv(multiseed_metrics_csv, index=False)
multiseed_history_df.to_csv(multiseed_history_csv, index=False)
multiseed_summary_df.to_csv(multiseed_summary_csv, index=False)
with multiseed_json.open('w') as f:
    json.dump({'metrics': multiseed_rows}, f, indent=2)
print('Saved best-branch multi-seed metrics:', multiseed_metrics_csv)
print('Saved best-branch multi-seed history:', multiseed_history_csv)
print('Saved best-branch multi-seed summary:', multiseed_summary_csv)
print('Saved best-branch multi-seed json:', multiseed_json)"""
            ),
            md_cell(
                """## Best-Branch Multi-Seed Results Summary

This summarizes the current best branch across multiple random seeds. The goal is to assess stability before deciding whether the current synthetic benchmark is mature enough to stop model-side refinement and move into reporting mode."""
            ),
            code_cell(
                """# Plot best-branch multi-seed mean and spread
ms_df = pd.read_csv(OUT_DIR / 'synthetic_two_layer_balanced_conditioned_dualdec_hybrid_sg_multiseed_summary.csv')
display(ms_df)

fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex='col')
metric_specs = [
    ('rmse_mean', 'rmse_std', 'RMSE', 'tab:red'),
    ('r2_mean', 'r2_std', 'R^2', 'tab:blue'),
    ('corr_mean', 'corr_std', 'Correlation', 'tab:green'),
    ('fit_slope_mean', 'fit_slope_std', 'Slope', 'tab:purple'),
]
for row_idx, layer_name in enumerate(['S0', 'Sg']):
    layer_df = ms_df[ms_df['layer'] == layer_name].sort_values('noise_scale')
    x = layer_df['noise_scale'].to_numpy()
    for col_idx, (mean_col, std_col, title, color) in enumerate(metric_specs):
        ax = axes[row_idx, col_idx]
        y = layer_df[mean_col].to_numpy()
        s = layer_df[std_col].fillna(0.0).to_numpy()
        ax.plot(x, y, color=color, lw=2.2, marker='o')
        ax.step(x, y, where='post', color=color, lw=1.0, alpha=0.35)
        ax.fill_between(x, y - s, y + s, color=color, alpha=0.18)
        ax.set_title(f'{layer_name} {title}')
        ax.set_xlabel('Noise scale')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(sorted(ms_df['noise_scale'].unique()))
        if 'r2' in mean_col or 'corr' in mean_col or 'slope' in mean_col:
            ax.axhline(0.0, color='black', lw=1, ls=':')
        if 'slope' in mean_col:
            ax.axhline(1.0, color='black', lw=1, ls='--', alpha=0.8)
fig.suptitle('Best-branch multi-seed validation: mean ± std across random seeds', fontsize=14)
fig.tight_layout(rect=[0,0.02,1,0.94])
plt.show()
fig_path = FIG_DIR / 'synthetic_two_layer_balanced_conditioned_dualdec_hybrid_sg_multiseed.png'
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
print('Saved best-branch multi-seed figure:', fig_path)"""
            ),
        ]
    )
    NOTEBOOK.write_text(json.dumps(nb, indent=1))


if __name__ == "__main__":
    main()
