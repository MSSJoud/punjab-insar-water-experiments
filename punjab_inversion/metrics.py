import numpy as np
import torch


def normalize_field(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


def batch_correlation_torch(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred_f = pred.reshape(pred.shape[0], -1)
    tgt_f = target.reshape(target.shape[0], -1)
    pred_c = pred_f - pred_f.mean(dim=1, keepdim=True)
    tgt_c = tgt_f - tgt_f.mean(dim=1, keepdim=True)
    num = (pred_c * tgt_c).sum(dim=1)
    den = torch.sqrt((pred_c.square().sum(dim=1) + eps) * (tgt_c.square().sum(dim=1) + eps))
    return (num / den).mean()


def amplitude_penalty_torch(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred_flat = pred.reshape(pred.shape[0], -1)
    tgt_flat = target.reshape(target.shape[0], -1)
    pred_std = pred_flat.std(dim=1, unbiased=False)
    tgt_std = tgt_flat.std(dim=1, unbiased=False)
    pred_rng = pred_flat.amax(dim=1) - pred_flat.amin(dim=1)
    tgt_rng = tgt_flat.amax(dim=1) - tgt_flat.amin(dim=1)
    std_term = (torch.log(pred_std + eps) - torch.log(tgt_std + eps)) ** 2
    rng_term = (torch.log(pred_rng + eps) - torch.log(tgt_rng + eps)) ** 2
    return (std_term + rng_term).mean()


def anisotropic_total_variation(y_pred: torch.Tensor) -> torch.Tensor:
    dx = torch.abs(y_pred[..., :, 1:] - y_pred[..., :, :-1]).mean()
    dy = torch.abs(y_pred[..., 1:, :] - y_pred[..., :-1, :]).mean()
    return dx + dy


def rmse(a, b) -> float:
    aa = np.asarray(a)
    bb = np.asarray(b)
    return float(np.sqrt(np.nanmean((aa - bb) ** 2)))


def r2_score_np(y_true, y_pred) -> float:
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    m = np.isfinite(yt) & np.isfinite(yp)
    if m.sum() < 2:
        return np.nan
    yt = yt[m]
    yp = yp[m]
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    return np.nan if ss_tot == 0 else float(1.0 - ss_res / ss_tot)


def corr_np(a, b) -> float:
    aa = np.asarray(a).ravel()
    bb = np.asarray(b).ravel()
    m = np.isfinite(aa) & np.isfinite(bb)
    if m.sum() < 2:
        return np.nan
    return float(np.corrcoef(aa[m], bb[m])[0, 1])


def mae(a, b) -> float:
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def bias_np(a, b) -> float:
    return float(np.mean(np.asarray(b) - np.asarray(a)))


def nrmse_np(a, b) -> float:
    aa = np.asarray(a)
    span = float(np.nanmax(aa) - np.nanmin(aa))
    return np.nan if span == 0 else rmse(a, b) / span


def fit_slope_np(a, b) -> float:
    aa = np.asarray(a).ravel()
    bb = np.asarray(b).ravel()
    m = np.isfinite(aa) & np.isfinite(bb)
    if m.sum() < 2:
        return np.nan
    slope, _ = np.polyfit(aa[m], bb[m], deg=1)
    return float(slope)


def fit_intercept_np(a, b) -> float:
    aa = np.asarray(a).ravel()
    bb = np.asarray(b).ravel()
    m = np.isfinite(aa) & np.isfinite(bb)
    if m.sum() < 2:
        return np.nan
    _, intercept = np.polyfit(aa[m], bb[m], deg=1)
    return float(intercept)
