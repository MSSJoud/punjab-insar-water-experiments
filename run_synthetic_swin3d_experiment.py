from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, Dataset, Subset


OUT_DIR = Path("/home/ubuntu/work/punjab/outputs")
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
T_SYN = 108
H_SYN = 64
W_SYN = 64
WINDOW_SIZE = 12
BATCH_SIZE = 4
EMBED_DIM = 48
NUM_HEADS = 4
STAGE_DEPTHS = (2, 2, 2)
PATCH_SIZE = (2, 4, 4)
MERGE_SCALE = (1, 2, 2)
WINDOW_ATTN_SIZE = (3, 4, 4)
LEARNING_RATE = 1e-4
MAX_EPOCHS = 20
PATIENCE = 5
LAMBDA_PHYS = 0.5
LAMBDA_TV = 1e-4
USE_GRACE_LOSS = False
LAMBDA_GRACE = 0.0
LAYER_NAMES = ("S0", "Ss", "Sd", "Sg")
Sg_INDEX = LAYER_NAMES.index("Sg")


@dataclass(frozen=True)
class PhysicsConfig:
    E: float = 1e9
    nu: float = 0.25
    rho_w: float = 1000.0
    g: float = 9.81
    alpha: float = 0.8
    Hg: float = 150.0
    Seff: float = 0.2
    dx: float = 10000.0
    dy: float = 10000.0
    a_load: float = 3000.0
    a_poro: float = 3000.0


PHYSICS = PhysicsConfig()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def build_elastic_kernel(E, nu, dx, dy, a, nx, ny):
    xgrid = (np.arange(nx) - nx / 2) * dx
    ygrid = (np.arange(ny) - ny / 2) * dy
    xx, yy = np.meshgrid(xgrid, ygrid)
    r = np.sqrt(xx**2 + yy**2)
    r[r < 1e-6] = 1e-6
    return (1 + nu) / (np.pi * E * (1 - nu)) * (1 - np.exp(-r / a)) / r


def build_poroelastic_kernel(E, nu, alpha, hg, dx, dy, a, nx, ny):
    xgrid = (np.arange(nx) - nx / 2) * dx
    ygrid = (np.arange(ny) - ny / 2) * dy
    xx, yy = np.meshgrid(xgrid, ygrid)
    r = np.sqrt(xx**2 + yy**2)
    r[r < 1e-6] = 1e-6
    return alpha * (1 + nu) * hg * 9.81 / (np.pi * E * (1 - nu)) * (1 - np.exp(-r / a)) / r


def build_fft_kernels(ny, nx, physics: PhysicsConfig, device: str):
    g_load = build_elastic_kernel(physics.E, physics.nu, physics.dx, physics.dy, physics.a_load, nx, ny)
    g_poro = build_poroelastic_kernel(
        physics.E,
        physics.nu,
        physics.alpha,
        physics.Hg,
        physics.dx,
        physics.dy,
        physics.a_poro,
        nx,
        ny,
    )
    g_load_fft = torch.fft.fft2(torch.fft.ifftshift(torch.tensor(g_load, dtype=torch.float32, device=device)))
    g_poro_fft = torch.fft.fft2(torch.fft.ifftshift(torch.tensor(g_poro, dtype=torch.float32, device=device)))
    return g_load_fft, g_poro_fft


def fft_convolve2d(field, kernel_fft):
    return torch.fft.ifft2(torch.fft.fft2(field) * kernel_fft).real


def forward_physics_torch(y_pred, g_load_fft, g_poro_fft, physics: PhysicsConfig):
    # y_pred: [B,4,H,W] ordered as S0,Ss,Sd,Sg
    delta_l = physics.rho_w * y_pred[:, :3].sum(dim=1)
    delta_p = physics.rho_w * physics.g * (y_pred[:, Sg_INDEX] / physics.Seff)
    uz_load = fft_convolve2d(delta_l, g_load_fft)
    uz_poro = fft_convolve2d(delta_p, g_poro_fft)
    return (uz_load + uz_poro).unsqueeze(1)


def anisotropic_total_variation(y_pred):
    dx = torch.abs(y_pred[..., :, 1:] - y_pred[..., :, :-1]).mean()
    dy = torch.abs(y_pred[..., 1:, :] - y_pred[..., :-1, :]).mean()
    return dx + dy


def rmse(a, b):
    return float(np.sqrt(np.nanmean((a - b) ** 2)))


def r2_score_np(y_true, y_pred):
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    m = np.isfinite(yt) & np.isfinite(yp)
    if m.sum() < 2:
        return np.nan
    yt = yt[m]
    yp = yp[m]
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


def make_spatial_pattern(rng, sigma_large, sigma_small, amplitude=1.0):
    field = 0.7 * gaussian_filter(rng.normal(size=(H_SYN, W_SYN)), sigma=sigma_large)
    field += 0.3 * gaussian_filter(rng.normal(size=(H_SYN, W_SYN)), sigma=sigma_small)
    field = field - field.mean()
    field = field / (field.std() + 1e-6)
    return amplitude * field


def make_synthetic_layers():
    rng = np.random.default_rng(SEED)
    t = np.arange(T_SYN, dtype=np.float32)

    p_s0 = make_spatial_pattern(rng, sigma_large=8, sigma_small=3, amplitude=1.2)
    p_ss = make_spatial_pattern(rng, sigma_large=10, sigma_small=4, amplitude=0.9)
    p_sd = make_spatial_pattern(rng, sigma_large=12, sigma_small=5, amplitude=0.8)
    p_sg = make_spatial_pattern(rng, sigma_large=14, sigma_small=6, amplitude=1.5)

    annual = np.sin(2 * np.pi * t / 12.0)
    semi = np.sin(4 * np.pi * t / 12.0)
    interannual = np.sin(2 * np.pi * t / 36.0)
    trend = (t - t.mean()) / t.max()

    S0 = 8.0 * annual[:, None, None] * p_s0 + 1.5 * semi[:, None, None] * p_s0
    Ss = 5.0 * np.sin(2 * np.pi * (t - 1.0) / 12.0)[:, None, None] * p_ss + 1.2 * interannual[:, None, None] * p_ss
    Sd = 3.5 * np.sin(2 * np.pi * (t - 2.0) / 12.0)[:, None, None] * p_sd + 1.8 * trend[:, None, None] * p_sd
    Sg = 6.0 * np.sin(2 * np.pi * (t - 4.0) / 24.0)[:, None, None] * p_sg + 5.0 * trend[:, None, None] * p_sg

    layers = np.stack([S0, Ss, Sd, Sg], axis=1)
    layers += 0.25 * gaussian_filter(rng.normal(size=layers.shape), sigma=(0, 0, 2, 2))
    return layers.astype(np.float32)


def make_synthetic_deformation(layers, physics: PhysicsConfig):
    s0 = layers[:, 0]
    ss = layers[:, 1]
    sd = layers[:, 2]
    sg = layers[:, 3]

    g_load = build_elastic_kernel(physics.E, physics.nu, physics.dx, physics.dy, physics.a_load, W_SYN, H_SYN)
    g_poro = build_poroelastic_kernel(
        physics.E, physics.nu, physics.alpha, physics.Hg, physics.dx, physics.dy, physics.a_poro, W_SYN, H_SYN
    )
    g_load_fft = np.fft.fft2(np.fft.ifftshift(g_load))
    g_poro_fft = np.fft.fft2(np.fft.ifftshift(g_poro))

    delta_l = physics.rho_w * (s0 + ss + sd)
    delta_p = physics.rho_w * physics.g * (sg / physics.Seff)

    uz_load = np.zeros((T_SYN, H_SYN, W_SYN), dtype=np.float32)
    uz_poro = np.zeros_like(uz_load)
    for i in range(T_SYN):
        uz_load[i] = np.fft.ifft2(np.fft.fft2(delta_l[i]) * g_load_fft).real.astype(np.float32)
        uz_poro[i] = np.fft.ifft2(np.fft.fft2(delta_p[i]) * g_poro_fft).real.astype(np.float32)

    uz_total = uz_load + uz_poro

    rng = np.random.default_rng(SEED + 1)
    white_noise = 0.02 * rng.normal(size=uz_total.shape)
    corr_noise = np.zeros_like(uz_total)
    for i in range(T_SYN):
        corr_noise[i] = 0.05 * make_spatial_pattern(rng, sigma_large=10, sigma_small=4, amplitude=1.0)
    seasonal_noise = 0.03 * np.sin(2 * np.pi * np.arange(T_SYN) / 12.0)[:, None, None]
    uz_noisy = uz_total + white_noise + corr_noise + seasonal_noise

    return uz_total.astype(np.float32), uz_noisy.astype(np.float32)


class WindowedSyntheticHydroDataset(Dataset):
    def __init__(self, disp, layers, window_size=12):
        disp_t = torch.tensor(disp, dtype=torch.float32)
        layers_t = torch.tensor(layers, dtype=torch.float32)

        xs, ys, us = [], [], []
        for end_idx in range(window_size - 1, disp_t.shape[0]):
            start_idx = end_idx - window_size + 1
            x_window = torch.stack(
                [disp_t[start_idx : end_idx + 1], layers_t[start_idx : end_idx + 1, 0]],
                dim=0,
            )
            xs.append(x_window)
            ys.append(layers_t[end_idx])
            us.append(disp_t[end_idx].unsqueeze(0))

        self.x = torch.stack(xs, dim=0)      # [N,2,T,H,W]
        self.y = torch.stack(ys, dim=0)      # [N,4,H,W]
        self.u_last = torch.stack(us, dim=0) # [N,1,H,W]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.u_last[idx]


class NormalizedWindowedDataset(Dataset):
    def __init__(self, base_dataset, indices, x_mean, x_std, y_mean, y_std, u_mean, u_std):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std
        self.u_mean = u_mean
        self.u_std = u_std

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x, y, u = self.base_dataset[self.indices[idx]]
        x = (x - self.x_mean) / self.x_std
        y = (y - self.y_mean) / self.y_std
        u = (u - self.u_mean) / self.u_std
        return x, y, u


def compute_stats(ds, train_idx):
    x_train = ds.x[train_idx]
    y_train = ds.y[train_idx]
    u_train = ds.u_last[train_idx]
    x_mean = x_train.mean(dim=0)
    x_std = x_train.std(dim=0).clamp_min(1e-6)
    y_mean = y_train.mean(dim=0)
    y_std = y_train.std(dim=0).clamp_min(1e-6)
    u_mean = u_train.mean(dim=0)
    u_std = u_train.std(dim=0).clamp_min(1e-6)
    return x_mean, x_std, y_mean, y_std, u_mean, u_std


def window_partition_3d(x, window_size):
    b, t, h, w, c = x.shape
    wt, wh, ww = window_size
    x = x.view(b, t // wt, wt, h // wh, wh, w // ww, ww, c)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    return windows.view(-1, wt * wh * ww, c)


def window_reverse_3d(windows, window_size, b, t, h, w, c):
    wt, wh, ww = window_size
    x = windows.view(b, t // wt, h // wh, w // ww, wt, wh, ww, c)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    return x.view(b, t, h, w, c)


class WindowAttention3D(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        b_windows, n_tokens, dim = x.shape
        qkv = self.qkv(x).reshape(b_windows, n_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            n_w = mask.shape[0]
            attn = attn.view(b_windows // n_w, n_w, self.num_heads, n_tokens, n_tokens)
            attn = attn + mask.unsqueeze(0).unsqueeze(2)
            attn = attn.view(-1, self.num_heads, n_tokens, n_tokens)
        attn = torch.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b_windows, n_tokens, dim)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class SwinBlock3D(nn.Module):
    def __init__(self, dim, num_heads, window_size=(3, 4, 4), shift_size=(0, 0, 0)):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim)

    def _attn_mask(self, t, h, w, device):
        if all(s == 0 for s in self.shift_size):
            return None
        wt, wh, ww = self.window_size
        st, sh, sw = self.shift_size
        img_mask = torch.zeros((1, t, h, w, 1), device=device)
        cnt = 0
        t_slices = (slice(0, -wt), slice(-wt, -st), slice(-st, None))
        h_slices = (slice(0, -wh), slice(-wh, -sh), slice(-sh, None))
        w_slices = (slice(0, -ww), slice(-ww, -sw), slice(-sw, None))
        for ts in t_slices:
            for hs in h_slices:
                for ws in w_slices:
                    img_mask[:, ts, hs, ws, :] = cnt
                    cnt += 1
        mask_windows = window_partition_3d(img_mask, self.window_size).view(-1, wt * wh * ww)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
        return attn_mask

    def forward(self, x):
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 3, 4, 1).contiguous()

        wt, wh, ww = self.window_size
        pad_t = (wt - t % wt) % wt
        pad_h = (wh - h % wh) % wh
        pad_w = (ww - w % ww) % ww
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_t))
        _, tp, hp, wp, _ = x.shape

        shortcut = x
        if any(s > 0 for s in self.shift_size):
            shifted = torch.roll(
                x,
                shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]),
                dims=(1, 2, 3),
            )
        else:
            shifted = x

        x_windows = window_partition_3d(shifted, self.window_size)
        x_windows = self.norm1(x_windows)
        attn_mask = self._attn_mask(tp, hp, wp, x.device)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        shifted = window_reverse_3d(attn_windows, self.window_size, b, tp, hp, wp, c)

        if any(s > 0 for s in self.shift_size):
            x = torch.roll(
                shifted,
                shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]),
                dims=(1, 2, 3),
            )
        else:
            x = shifted

        x = shortcut + x
        x_flat = x.view(b, tp * hp * wp, c)
        x = x_flat + self.mlp(self.norm2(x_flat))
        x = x.view(b, tp, hp, wp, c)
        x = x[:, :t, :h, :w, :].permute(0, 4, 1, 2, 3).contiguous()
        return x


class SwinStage3D(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size):
        super().__init__()
        shift = tuple(s // 2 for s in window_size)
        blocks = []
        for idx in range(depth):
            blocks.append(
                SwinBlock3D(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=(0, 0, 0) if idx % 2 == 0 else shift,
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class PatchEmbed3D(nn.Module):
    def __init__(self, in_channels=2, embed_dim=48, patch_size=(2, 4, 4)):
        super().__init__()
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x)


class PatchMerging3D(nn.Module):
    def __init__(self, in_dim, out_dim, scale=(1, 2, 2)):
        super().__init__()
        self.reduction = nn.Conv3d(in_dim, out_dim, kernel_size=scale, stride=scale)

    def forward(self, x):
        return self.reduction(x)


class PatchExpand3D(nn.Module):
    def __init__(self, in_dim, out_dim, scale=(1, 2, 2)):
        super().__init__()
        self.expand = nn.ConvTranspose3d(in_dim, out_dim, kernel_size=scale, stride=scale)

    def forward(self, x):
        return self.expand(x)


class SwinUNetPhysicsNet3D(nn.Module):
    def __init__(self, in_channels=2, out_channels=4):
        super().__init__()
        bottleneck_dim = EMBED_DIM * 2
        self.patch_embed = PatchEmbed3D(in_channels=in_channels, embed_dim=EMBED_DIM, patch_size=PATCH_SIZE)
        self.encoder = SwinStage3D(EMBED_DIM, STAGE_DEPTHS[0], NUM_HEADS, WINDOW_ATTN_SIZE)
        self.patch_merge = PatchMerging3D(EMBED_DIM, bottleneck_dim, scale=MERGE_SCALE)
        self.bottleneck = SwinStage3D(bottleneck_dim, STAGE_DEPTHS[1], NUM_HEADS, WINDOW_ATTN_SIZE)
        self.patch_expand = PatchExpand3D(bottleneck_dim, EMBED_DIM, scale=MERGE_SCALE)
        self.skip_fuse = nn.Conv3d(EMBED_DIM * 2, EMBED_DIM, kernel_size=1)
        self.decoder = SwinStage3D(EMBED_DIM, STAGE_DEPTHS[2], NUM_HEADS, WINDOW_ATTN_SIZE)
        self.output_upsample = nn.ConvTranspose3d(EMBED_DIM, EMBED_DIM // 2, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
        self.output_head = nn.Conv3d(EMBED_DIM // 2, out_channels, kernel_size=1)

    def forward(self, x):
        x0 = self.patch_embed(x)
        skip = self.encoder(x0)
        z = self.patch_merge(skip)
        z = self.bottleneck(z)
        z = self.patch_expand(z)
        if z.shape[2:] != skip.shape[2:]:
            z = F.interpolate(z, size=skip.shape[2:], mode="trilinear", align_corners=False)
        z = self.skip_fuse(torch.cat([z, skip], dim=1))
        z = self.decoder(z)
        z = self.output_upsample(z)
        y = self.output_head(z)
        return y[:, :, -1]


def denormalize_targets(y_pred_norm, y_mean, y_std):
    return y_pred_norm * y_std + y_mean


def run_epoch(
    model,
    loader,
    g_load_fft,
    g_poro_fft,
    optimizer=None,
    scheduler=None,
    y_mean=None,
    y_std=None,
    u_mean=None,
    u_std=None,
):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    running = {"loss": 0.0, "loss_layers": 0.0, "loss_phys": 0.0, "loss_tv": 0.0}
    n = 0

    for xb, yb, ub in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        ub = ub.to(DEVICE)

        with torch.set_grad_enabled(train_mode):
            yp_norm = model(xb)
            layer_loss = F.mse_loss(yp_norm[:, :3], yb[:, :3]) + 2.0 * F.mse_loss(yp_norm[:, Sg_INDEX : Sg_INDEX + 1], yb[:, Sg_INDEX : Sg_INDEX + 1])
            yp_raw = denormalize_targets(yp_norm, y_mean, y_std)
            d_hat_raw = forward_physics_torch(yp_raw, g_load_fft, g_poro_fft, PHYSICS)
            d_hat_norm = (d_hat_raw - u_mean) / u_std
            loss_phys = F.mse_loss(d_hat_norm, ub)
            loss_tv = anisotropic_total_variation(yp_raw)
            loss = layer_loss + LAMBDA_PHYS * loss_phys + LAMBDA_TV * loss_tv

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        bsz = xb.shape[0]
        n += bsz
        running["loss"] += float(loss.item()) * bsz
        running["loss_layers"] += float(layer_loss.item()) * bsz
        running["loss_phys"] += float(loss_phys.item()) * bsz
        running["loss_tv"] += float(loss_tv.item()) * bsz

    if train_mode and scheduler is not None:
        scheduler.step()

    return {k: v / max(n, 1) for k, v in running.items()}


def save_timeseries_plot(y_true, y_pred, out_path):
    pixel = (H_SYN // 2, W_SYN // 2)
    x = np.arange(y_true.shape[0])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, idx, name in zip(axes.ravel(), range(4), LAYER_NAMES):
        ax.plot(x, y_true[:, idx, pixel[0], pixel[1]], label="true", lw=2)
        ax.plot(x, y_pred[:, idx, pixel[0], pixel[1]], label="pred", lw=2, ls="--")
        ax.set_title(f"{name} @ pixel {pixel}")
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    set_seed(SEED)
    print("Device:", DEVICE)

    layers = make_synthetic_layers()
    _, disp_noisy = make_synthetic_deformation(layers, PHYSICS)
    ds = WindowedSyntheticHydroDataset(disp=disp_noisy, layers=layers, window_size=WINDOW_SIZE)
    n = len(ds)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    train_idx = np.arange(0, n_train)
    val_idx = np.arange(n_train, n_train + n_val)
    test_idx = np.arange(n_train + n_val, n)

    x_mean, x_std, y_mean, y_std, u_mean, u_std = compute_stats(ds, train_idx)
    train_ds = NormalizedWindowedDataset(ds, train_idx, x_mean, x_std, y_mean, y_std, u_mean, u_std)
    val_ds = NormalizedWindowedDataset(ds, val_idx, x_mean, x_std, y_mean, y_std, u_mean, u_std)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = SwinUNetPhysicsNet3D(in_channels=2, out_channels=4).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    g_load_fft, g_poro_fft = build_fft_kernels(H_SYN, W_SYN, PHYSICS, DEVICE)

    y_mean_d = y_mean.to(DEVICE)
    y_std_d = y_std.to(DEVICE)
    u_mean_d = u_mean.to(DEVICE)
    u_std_d = u_std.to(DEVICE)

    best_val = np.inf
    wait = 0
    best_state = None
    history = []

    for epoch in range(MAX_EPOCHS):
        tr = run_epoch(
            model,
            train_dl,
            g_load_fft,
            g_poro_fft,
            optimizer=opt,
            scheduler=scheduler,
            y_mean=y_mean_d,
            y_std=y_std_d,
            u_mean=u_mean_d,
            u_std=u_std_d,
        )
        va = run_epoch(
            model,
            val_dl,
            g_load_fft,
            g_poro_fft,
            optimizer=None,
            scheduler=None,
            y_mean=y_mean_d,
            y_std=y_std_d,
            u_mean=u_mean_d,
            u_std=u_std_d,
        )
        history.append(
            {
                "epoch": epoch + 1,
                "lr": float(opt.param_groups[0]["lr"]),
                "train_loss": tr["loss"],
                "train_layers": tr["loss_layers"],
                "train_phys": tr["loss_phys"],
                "train_tv": tr["loss_tv"],
                "val_loss": va["loss"],
                "val_layers": va["loss_layers"],
                "val_phys": va["loss_phys"],
                "val_tv": va["loss_tv"],
            }
        )
        print(
            f"Epoch {epoch + 1:02d} | lr={opt.param_groups[0]['lr']:.2e} | "
            f"train={tr['loss']:.4f} (L={tr['loss_layers']:.4f}, P={tr['loss_phys']:.4f}) | "
            f"val={va['loss']:.4f}"
        )
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

    model.eval()
    with torch.no_grad():
        x_test = ((ds.x[test_idx] - x_mean) / x_std).to(DEVICE)
        y_true = ds.y[test_idx].cpu().numpy()
        u_true = ds.u_last[test_idx].cpu().numpy()[:, 0]
        y_pred_norm = model(x_test)
        y_pred = denormalize_targets(y_pred_norm, y_mean_d, y_std_d).cpu().numpy()
        u_hat = forward_physics_torch(
            torch.tensor(y_pred, dtype=torch.float32, device=DEVICE),
            g_load_fft,
            g_poro_fft,
            PHYSICS,
        ).cpu().numpy()[:, 0]

    rows = []
    for idx, name in enumerate(LAYER_NAMES):
        rows.append(
            {
                "split": "test",
                "layer": name,
                "rmse": rmse(y_true[:, idx], y_pred[:, idx]),
                "r2": r2_score_np(y_true[:, idx], y_pred[:, idx]),
                "corr": corr_np(y_true[:, idx], y_pred[:, idx]),
            }
        )
    metrics_df = pd.DataFrame(rows)
    metrics_df["forward_residual_rmse"] = rmse(u_true, u_hat)
    metrics_df["model_name"] = "SwinUNetPhysicsNet3D_Windowed"
    metrics_df["use_grace_loss"] = USE_GRACE_LOSS

    history_path = OUT_DIR / "synthetic_training_history.csv"
    metrics_path = OUT_DIR / "synthetic_gate_metrics.csv"
    ckpt_path = OUT_DIR / "swin3d_physics_best.pt"
    figure_path = FIG_DIR / "synthetic_timeseries_layers_test.png"
    summary_path = OUT_DIR / "synthetic_run_summary.json"

    pd.DataFrame(history).to_csv(history_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    torch.save(model.state_dict(), ckpt_path)
    save_timeseries_plot(y_true, y_pred, figure_path)

    summary = {
        "device": DEVICE,
        "best_val_loss": float(best_val),
        "num_train_windows": int(len(train_idx)),
        "num_val_windows": int(len(val_idx)),
        "num_test_windows": int(len(test_idx)),
        "history_csv": str(history_path),
        "metrics_csv": str(metrics_path),
        "timeseries_plot": str(figure_path),
        "checkpoint": str(ckpt_path),
        "normalization": {
            "inputs": "train-split per-pixel/per-window-position z-score",
            "targets": "train-split per-pixel z-score",
            "physics_loss": "denormalized layer predictions -> elastic+poroelastic forward model -> renormalized deformation",
        },
        "forward_model": {
            "layers": list(LAYER_NAMES),
            "elastic_load_layers": ["S0", "Ss", "Sd"],
            "poroelastic_layers": ["Sg"],
            "implementation": "explicit elastic and poroelastic Green's functions with FFT convolution",
        },
        "architecture": {
            "type": "windowed and shifted-attention Swin-U-Net-style 3D model",
            "window_size": WINDOW_ATTN_SIZE,
            "patch_size": PATCH_SIZE,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\nTest metrics")
    print(metrics_df.to_string(index=False))
    print("\nArtifacts")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
