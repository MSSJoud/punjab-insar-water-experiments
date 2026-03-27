from __future__ import annotations

import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, Dataset


OUT_DIR = Path("/home/ubuntu/work/punjab/outputs")
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
WINDOW_SIZE = 12
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
MAX_EPOCHS = 25
PATIENCE = 6
LAMBDA_PHYS = 0.5
LAMBDA_TV = 1e-4
LAYER_WEIGHTS = (0.05, 0.10, 0.25, 0.60)
GREEN_KERNEL_SIZE = 9
GREEN_SIGMA = 2.0


def build_green_kernel(kernel_size=9, sigma=2.0, device=None, dtype=torch.float32):
    coords = torch.arange(kernel_size, dtype=dtype, device=device) - (kernel_size - 1) / 2
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, kernel_size, kernel_size)


def anisotropic_total_variation(y_pred):
    dx = torch.abs(y_pred[..., :, 1:] - y_pred[..., :, :-1]).mean()
    dy = torch.abs(y_pred[..., 1:, :] - y_pred[..., :-1, :]).mean()
    return dx + dy


class SyntheticSgDataset(Dataset):
    def __init__(self, disp, S0, Sg, indices, x_mean, x_std, y_mean, y_std, u_mean, u_std, window_size=12):
        disp_t = torch.tensor(disp, dtype=torch.float32)
        s0_t = torch.tensor(S0, dtype=torch.float32)
        sg_t = torch.tensor(Sg[:, None, :, :], dtype=torch.float32)
        self.samples = []
        for end_idx in indices:
            start_idx = end_idx - window_size + 1
            x = torch.stack([disp_t[start_idx : end_idx + 1], s0_t[start_idx : end_idx + 1]], dim=0)
            y = sg_t[end_idx]
            u = disp_t[end_idx].unsqueeze(0)
            x = (x - x_mean) / x_std
            y = (y - y_mean) / y_std
            u = (u - u_mean) / u_std
            self.samples.append((x, y, u))
        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std
        self.u_mean = u_mean
        self.u_std = u_std

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class CNN3DBaseline(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(32, out_channels, kernel_size=1),
        )

    def forward(self, x):
        y = self.net(x)
        return y[:, :, -1]


def denormalize(y_norm, mean, std):
    return y_norm * std + mean


def forward_poroelastic_torch(sg_pred, s0_last_raw, green_kernel=None):
    disp = LAYER_WEIGHTS[0] * s0_last_raw.unsqueeze(1) + LAYER_WEIGHTS[3] * sg_pred
    return F.conv2d(disp, green_kernel, padding=green_kernel.shape[-1] // 2)


def rmse(a, b):
    return float(np.sqrt(np.nanmean((a - b) ** 2)))


def r2_score_np(y_true, y_pred):
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    m = np.isfinite(yt) & np.isfinite(yp)
    if m.sum() < 2:
        return np.nan
    yt, yp = yt[m], yp[m]
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    if ss_tot == 0:
        return np.nan
    return float(1.0 - ss_res / ss_tot)


def corr_np(a, b):
    aa = np.asarray(a).ravel()
    bb = np.asarray(b).ravel()
    m = np.isfinite(aa) & np.isfinite(bb)
    if m.sum() < 2:
        return np.nan
    return float(np.corrcoef(aa[m], bb[m])[0, 1])


def make_synthetic():
    rng = np.random.default_rng(SEED)
    t = np.arange(36)
    h = 64
    w = 64
    S0 = 10 * np.sin(2 * np.pi * t[:, None, None] / 12) + gaussian_filter(rng.normal(size=(36, h, w)), sigma=(0, 3, 3))
    Ss = 6 * np.sin(2 * np.pi * (t[:, None, None] - 1) / 12) + gaussian_filter(rng.normal(size=(36, h, w)), sigma=(0, 4, 4))
    Sd = 3 * np.sin(2 * np.pi * (t[:, None, None] - 2) / 12) + gaussian_filter(rng.normal(size=(36, h, w)), sigma=(0, 5, 5))
    Sg = 0.4 * t[:, None, None] + 2 * np.sin(2 * np.pi * (t[:, None, None] - 4) / 12) + gaussian_filter(rng.normal(size=(36, h, w)), sigma=(0, 6, 6))
    disp = (LAYER_WEIGHTS[0] * S0 + LAYER_WEIGHTS[1] * Ss + LAYER_WEIGHTS[2] * Sd + LAYER_WEIGHTS[3] * Sg)
    for i in range(disp.shape[0]):
        disp[i] = gaussian_filter(disp[i], sigma=2.0)
    disp_noisy = disp + rng.normal(scale=0.3, size=disp.shape)
    return S0, Sg, disp_noisy


def compute_stats(disp, S0, Sg, train_indices):
    disp_t = torch.tensor(disp, dtype=torch.float32)
    s0_t = torch.tensor(S0, dtype=torch.float32)
    sg_t = torch.tensor(Sg[:, None, :, :], dtype=torch.float32)
    x_list, y_list, u_list = [], [], []
    for end_idx in train_indices:
        start_idx = end_idx - WINDOW_SIZE + 1
        x_list.append(torch.stack([disp_t[start_idx : end_idx + 1], s0_t[start_idx : end_idx + 1]], dim=0))
        y_list.append(sg_t[end_idx])
        u_list.append(disp_t[end_idx].unsqueeze(0))
    x_train = torch.stack(x_list, dim=0)
    y_train = torch.stack(y_list, dim=0)
    u_train = torch.stack(u_list, dim=0)
    x_mean = x_train.mean(dim=0)
    x_std = x_train.std(dim=0).clamp_min(1e-6)
    y_mean = y_train.mean(dim=0)
    y_std = y_train.std(dim=0).clamp_min(1e-6)
    u_mean = u_train.mean(dim=0)
    u_std = u_train.std(dim=0).clamp_min(1e-6)
    return x_mean, x_std, y_mean, y_std, u_mean, u_std


def save_plot(y_true, y_pred, out_path):
    pixel = (32, 32)
    x = np.arange(y_true.shape[0])
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(x, y_true[:, 0, pixel[0], pixel[1]], label="true Sg", lw=2)
    ax.plot(x, y_pred[:, 0, pixel[0], pixel[1]], label="pred Sg", lw=2, ls="--")
    ax.set_title(f"CNN baseline Sg recovery @ pixel {pixel}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print("Device:", DEVICE)

    S0, Sg, disp = make_synthetic()
    valid_end = np.arange(WINDOW_SIZE - 1, disp.shape[0])
    n = len(valid_end)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    train_ends = valid_end[:n_train]
    val_ends = valid_end[n_train : n_train + n_val]
    test_ends = valid_end[n_train + n_val :]

    x_mean, x_std, y_mean, y_std, u_mean, u_std = compute_stats(disp, S0, Sg, train_ends)
    train_ds = SyntheticSgDataset(disp, S0, Sg, train_ends, x_mean, x_std, y_mean, y_std, u_mean, u_std, WINDOW_SIZE)
    val_ds = SyntheticSgDataset(disp, S0, Sg, val_ends, x_mean, x_std, y_mean, y_std, u_mean, u_std, WINDOW_SIZE)
    test_ds = SyntheticSgDataset(disp, S0, Sg, test_ends, x_mean, x_std, y_mean, y_std, u_mean, u_std, WINDOW_SIZE)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    model = CNN3DBaseline().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    green_kernel = build_green_kernel(kernel_size=GREEN_KERNEL_SIZE, sigma=GREEN_SIGMA, device=DEVICE)

    best_val = np.inf
    wait = 0
    best_state = None
    history = []

    for epoch in range(MAX_EPOCHS):
        def step(loader, train=False):
            model.train() if train else model.eval()
            stats = {"loss": 0.0, "layer": 0.0, "phys": 0.0}
            n_seen = 0
            for xb, yb, ub in loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                ub = ub.to(DEVICE)
                with torch.set_grad_enabled(train):
                    yp_norm = model(xb)
                    yp_raw = denormalize(yp_norm, y_mean.to(DEVICE), y_std.to(DEVICE))
                    s0_last_raw = xb[:, 1, -1] * x_std[1, -1].to(DEVICE) + x_mean[1, -1].to(DEVICE)
                    d_hat_raw = forward_poroelastic_torch(yp_raw, s0_last_raw, green_kernel)
                    d_hat_norm = (d_hat_raw - u_mean.to(DEVICE)) / u_std.to(DEVICE)
                    loss_layer = F.mse_loss(yp_norm, yb)
                    loss_phys = F.mse_loss(d_hat_norm, ub)
                    loss = loss_layer + LAMBDA_PHYS * loss_phys + LAMBDA_TV * anisotropic_total_variation(yp_raw)
                    if train:
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                b = xb.shape[0]
                n_seen += b
                stats["loss"] += float(loss.item()) * b
                stats["layer"] += float(loss_layer.item()) * b
                stats["phys"] += float(loss_phys.item()) * b
            return {k: v / max(n_seen, 1) for k, v in stats.items()}

        tr = step(train_dl, train=True)
        va = step(val_dl, train=False)
        scheduler.step()
        history.append({"epoch": epoch + 1, "lr": float(opt.param_groups[0]["lr"]), "train_loss": tr["loss"], "val_loss": va["loss"]})
        print(f"Epoch {epoch + 1:02d} | lr={opt.param_groups[0]['lr']:.2e} | train={tr['loss']:.4f} | val={va['loss']:.4f}")
        if va["loss"] < best_val:
            best_val = va["loss"]
            wait = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"Early stopping at epoch {epoch + 1}; best val={best_val:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    history_path = OUT_DIR / "synthetic_training_history_cnn3d.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    ckpt_path = OUT_DIR / "cnn3d_sg_baseline_best.pt"
    torch.save(model.state_dict(), ckpt_path)

    model.eval()
    x_test = torch.stack([s[0] for s in test_ds], dim=0).to(DEVICE)
    y_true = np.stack([Sg[end_idx][None, :, :] for end_idx in test_ends], axis=0)
    u_true = torch.stack([s[2] for s in test_ds], dim=0).numpy()[:, 0] * u_std.numpy()[0] + u_mean.numpy()[0]
    with torch.no_grad():
        y_pred_norm = model(x_test)
        y_pred = denormalize(y_pred_norm, y_mean.to(DEVICE), y_std.to(DEVICE)).cpu().numpy()
        s0_last_raw = x_test[:, 1, -1] * x_std[1, -1].to(DEVICE) + x_mean[1, -1].to(DEVICE)
        u_hat = forward_poroelastic_torch(torch.tensor(y_pred, dtype=torch.float32, device=DEVICE), s0_last_raw, green_kernel).cpu().numpy()[:, 0]

    metrics_df = pd.DataFrame(
        [
            {
                "split": "test",
                "layer": "Sg",
                "rmse": rmse(y_true[:, 0], y_pred[:, 0]),
                "r2": r2_score_np(y_true[:, 0], y_pred[:, 0]),
                "corr": corr_np(y_true[:, 0], y_pred[:, 0]),
                "forward_residual_rmse": rmse(u_true, u_hat),
                "model_name": "CNN3DBaseline",
            }
        ]
    )
    metrics_path = OUT_DIR / "synthetic_gate_metrics_cnn3d.csv"
    metrics_df.to_csv(metrics_path, index=False)
    fig_path = FIG_DIR / "synthetic_timeseries_layers_test_cnn3d.png"
    save_plot(y_true, y_pred, fig_path)

    summary = {
        "device": DEVICE,
        "best_val_loss": float(best_val),
        "num_train_windows": int(len(train_ends)),
        "num_val_windows": int(len(val_ends)),
        "num_test_windows": int(len(test_ends)),
        "history_csv": str(history_path),
        "metrics_csv": str(metrics_path),
        "timeseries_plot": str(fig_path),
        "checkpoint": str(ckpt_path),
    }
    (OUT_DIR / "synthetic_run_summary_cnn3d.json").write_text(json.dumps(summary, indent=2))

    print("\nTest metrics")
    print(metrics_df.to_string(index=False))
    print("\nArtifacts")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
