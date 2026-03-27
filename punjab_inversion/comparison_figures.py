from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_simple_h5_grid(path: str | Path, dataset: str = "z") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        lat = handle["lat"][:]
        lon = handle["lon"][:]
        grid = handle[dataset][:]
    return lat, lon, grid


def decode_netcdf_time(path: str | Path, dataset_name: str = "time") -> pd.DatetimeIndex:
    with h5py.File(path, "r") as handle:
        values = handle[dataset_name][:]
        units = handle[dataset_name].attrs.get("units", "")
        if isinstance(units, bytes):
            units = units.decode("utf-8")
    if isinstance(units, str) and units.startswith("days since "):
        base = pd.Timestamp(units.replace("days since ", ""))
        return pd.DatetimeIndex(base + pd.to_timedelta(values.astype(float), unit="D"))
    raise ValueError(f"Unsupported time units in {path}: {units}")


def load_h5_netcdf_variable(
    path: str | Path,
    variable: str,
    *,
    lat_name: str = "lat",
    lon_name: str = "lon",
    time_name: str = "time",
) -> dict[str, object]:
    with h5py.File(path, "r") as handle:
        data = handle[variable][:]
        lat = handle[lat_name][:]
        lon = handle[lon_name][:]
        attrs = {k: (v.decode("utf-8") if isinstance(v, bytes) else v) for k, v in handle[variable].attrs.items()}
        if time_name in handle:
            time = decode_netcdf_time(path, dataset_name=time_name)
        else:
            time = None
    return {"data": data, "lat": lat, "lon": lon, "time": time, "attrs": attrs}


def robust_limits(
    array: np.ndarray,
    *,
    low: float = 2.0,
    high: float = 98.0,
    symmetric: bool = False,
    fallback: float = 1.0,
) -> tuple[float, float]:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return (-fallback, fallback) if symmetric else (0.0, fallback)
    if symmetric:
        bound = float(np.nanpercentile(np.abs(finite), high))
        if not np.isfinite(bound) or bound <= 0:
            bound = fallback
        return -bound, bound
    lo, hi = np.nanpercentile(finite, [low, high])
    if not np.isfinite(lo):
        lo = 0.0
    if not np.isfinite(hi) or hi == lo:
        hi = lo + fallback
    return float(lo), float(hi)


def latest_valid_slice(data: np.ndarray) -> tuple[int, np.ndarray]:
    if data.ndim == 2:
        return 0, data
    for idx in range(data.shape[0] - 1, -1, -1):
        slc = data[idx]
        if np.isfinite(slc).any():
            return idx, slc
    raise ValueError("No finite slice found.")


def anomaly_relative_to_time_mean(data: np.ndarray) -> np.ndarray:
    return data - np.nanmean(data, axis=0, keepdims=True)


def basin_mean_timeseries(data: np.ndarray) -> np.ndarray:
    reshaped = data.reshape(data.shape[0], -1)
    return np.nanmean(reshaped, axis=1)


def map_extent(lat: np.ndarray, lon: np.ndarray) -> list[float]:
    return [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]


def select_active_bbox(mask: np.ndarray, margin_pixels: int = 64) -> tuple[int, int, int, int]:
    rows, cols = np.where(mask)
    r0 = max(int(rows.min()) - margin_pixels, 0)
    r1 = min(int(rows.max()) + margin_pixels + 1, mask.shape[0])
    c0 = max(int(cols.min()) - margin_pixels, 0)
    c1 = min(int(cols.max()) + margin_pixels + 1, mask.shape[1])
    return r0, r1, c0, c1


def select_finite_bbox(array: np.ndarray, margin_pixels: int = 64) -> tuple[int, int, int, int]:
    finite = np.isfinite(array)
    if not finite.any():
        raise ValueError("No finite values found for bbox selection.")
    return select_active_bbox(finite, margin_pixels=margin_pixels)


def _masked(array: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.masked_invalid(array)


def _comparison_panel_payload(
    *,
    vel_path: str | Path,
    grace_aligned_path: str | Path,
    w3ra_aligned_anom_path: str | Path,
    s0_pred_path: str | Path,
    sg_pred_path: str | Path,
    velocity_percentile_high: float = 95.0,
    inversion_crop_margin_pixels: int = 64,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    vel_lat, vel_lon, velocity = load_simple_h5_grid(vel_path)
    vel_extent = map_extent(vel_lat, vel_lon)

    grace = load_h5_netcdf_variable(grace_aligned_path, "lwe_thickness")
    grace_anom = anomaly_relative_to_time_mean(grace["data"])
    grace_idx, grace_map = latest_valid_slice(grace_anom)

    w3ra_s0 = load_h5_netcdf_variable(w3ra_aligned_anom_path, "S0")
    w3ra_sg = load_h5_netcdf_variable(w3ra_aligned_anom_path, "Sg")
    w3ra_s0_idx, w3ra_s0_map = latest_valid_slice(w3ra_s0["data"])
    w3ra_sg_idx, w3ra_sg_map = latest_valid_slice(w3ra_sg["data"])

    s0_pred = load_h5_netcdf_variable(s0_pred_path, "S0_pred")
    sg_pred = load_h5_netcdf_variable(sg_pred_path, "Sg_pred")
    s0_idx, s0_map = latest_valid_slice(s0_pred["data"])
    sg_idx, sg_map = latest_valid_slice(sg_pred["data"])

    finite_union = np.isfinite(s0_map) | np.isfinite(sg_map)
    r0, r1, c0, c1 = select_active_bbox(finite_union, margin_pixels=inversion_crop_margin_pixels)
    inv_lat = s0_pred["lat"][r0:r1]
    inv_lon = s0_pred["lon"][c0:c1]
    inv_extent = map_extent(inv_lat, inv_lon)
    s0_crop = s0_map[r0:r1, c0:c1]
    sg_crop = sg_map[r0:r1, c0:c1]

    vel_vmin, vel_vmax = robust_limits(
        velocity,
        low=100.0 - velocity_percentile_high,
        high=velocity_percentile_high,
        symmetric=True,
        fallback=10.0,
    )
    grace_vmin, grace_vmax = robust_limits(grace_map, symmetric=True, fallback=1.0)
    w3ra_s0_vmin, w3ra_s0_vmax = robust_limits(w3ra_s0_map, symmetric=True, fallback=1.0)
    w3ra_sg_vmin, w3ra_sg_vmax = robust_limits(w3ra_sg_map, symmetric=True, fallback=1.0)
    s0_vmin, s0_vmax = robust_limits(s0_crop, symmetric=True, fallback=1.0)
    sg_vmin, sg_vmax = robust_limits(sg_crop, symmetric=True, fallback=1.0)

    panels = [
        {
            "key": "velocity",
            "title": "A. InSAR velocity",
            "array": velocity,
            "extent": vel_extent,
            "vmin": vel_vmin,
            "vmax": vel_vmax,
            "time": None,
        },
        {
            "key": "grace_tws",
            "title": "B. GRACE TWS anomaly\n(latest aligned date)",
            "array": grace_map,
            "extent": map_extent(grace["lat"], grace["lon"]),
            "vmin": grace_vmin,
            "vmax": grace_vmax,
            "time": str(grace["time"][grace_idx].date()) if grace["time"] is not None else None,
        },
        {
            "key": "w3ra_s0",
            "title": "C. W3RA $S_0$ anomaly\n(latest aligned date)",
            "array": w3ra_s0_map,
            "extent": map_extent(w3ra_s0["lat"], w3ra_s0["lon"]),
            "vmin": w3ra_s0_vmin,
            "vmax": w3ra_s0_vmax,
            "time": str(w3ra_s0["time"][w3ra_s0_idx].date()) if w3ra_s0["time"] is not None else None,
        },
        {
            "key": "w3ra_sg",
            "title": "D. W3RA $S_g$ anomaly\n(latest aligned date)",
            "array": w3ra_sg_map,
            "extent": map_extent(w3ra_sg["lat"], w3ra_sg["lon"]),
            "vmin": w3ra_sg_vmin,
            "vmax": w3ra_sg_vmax,
            "time": str(w3ra_sg["time"][w3ra_sg_idx].date()) if w3ra_sg["time"] is not None else None,
        },
        {
            "key": "inversion_s0",
            "title": "E. Inversion $S_0$\n(latest available date)",
            "array": s0_crop,
            "extent": inv_extent,
            "vmin": s0_vmin,
            "vmax": s0_vmax,
            "time": str(s0_pred["time"][s0_idx].date()) if s0_pred["time"] is not None else None,
        },
        {
            "key": "inversion_sg",
            "title": "F. Inversion $S_g$\n(latest available date)",
            "array": sg_crop,
            "extent": inv_extent,
            "vmin": sg_vmin,
            "vmax": sg_vmax,
            "time": str(sg_pred["time"][sg_idx].date()) if sg_pred["time"] is not None else None,
        },
    ]

    summary = {
        "velocity_limits": [float(vel_vmin), float(vel_vmax)],
        "grace_limits": [float(grace_vmin), float(grace_vmax)],
        "w3ra_s0_limits": [float(w3ra_s0_vmin), float(w3ra_s0_vmax)],
        "w3ra_sg_limits": [float(w3ra_sg_vmin), float(w3ra_sg_vmax)],
        "inversion_s0_limits": [float(s0_vmin), float(s0_vmax)],
        "inversion_sg_limits": [float(sg_vmin), float(sg_vmax)],
        "grace_time": str(grace["time"][grace_idx].date()) if grace["time"] is not None else None,
        "w3ra_time": str(w3ra_s0["time"][w3ra_s0_idx].date()) if w3ra_s0["time"] is not None else None,
        "inversion_time": str(s0_pred["time"][s0_idx].date()) if s0_pred["time"] is not None else None,
        "inversion_bbox_rows": [int(r0), int(r1)],
        "inversion_bbox_cols": [int(c0), int(c1)],
    }

    return panels, summary


def make_punjab_comparison_maps(
    *,
    vel_path: str | Path,
    grace_aligned_path: str | Path,
    w3ra_aligned_anom_path: str | Path,
    s0_pred_path: str | Path,
    sg_pred_path: str | Path,
    output_path: str | Path,
    velocity_unit_label: str,
    grace_unit_label: str,
    w3ra_unit_label: str,
    inversion_unit_label: str,
    velocity_percentile_high: float = 95.0,
    inversion_crop_margin_pixels: int = 64,
) -> dict[str, object]:
    panels, summary = _comparison_panel_payload(
        vel_path=vel_path,
        grace_aligned_path=grace_aligned_path,
        w3ra_aligned_anom_path=w3ra_aligned_anom_path,
        s0_pred_path=s0_pred_path,
        sg_pred_path=sg_pred_path,
        velocity_percentile_high=velocity_percentile_high,
        inversion_crop_margin_pixels=inversion_crop_margin_pixels,
    )

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    cmap_div = plt.get_cmap("RdBu_r").copy()
    cmap_div.set_bad(color="white")

    unit_labels = {
        "velocity": velocity_unit_label,
        "grace_tws": grace_unit_label,
        "w3ra_s0": w3ra_unit_label,
        "w3ra_sg": w3ra_unit_label,
        "inversion_s0": inversion_unit_label,
        "inversion_sg": inversion_unit_label,
    }

    for ax, panel in zip(axes.ravel(), panels, strict=False):
        im = ax.imshow(
            _masked(panel["array"]),
            origin="lower",
            extent=panel["extent"],
            cmap=cmap_div,
            vmin=panel["vmin"],
            vmax=panel["vmax"],
            aspect="auto",
        )
        ax.set_title(panel["title"])
        ax.set_xlabel("Longitude [degrees_east]")
        ax.set_ylabel("Latitude [degrees_north]")
        fig.colorbar(im, ax=ax, shrink=0.80, label=unit_labels[panel["key"]])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    summary["output_path"] = str(output_path)
    return summary


def make_punjab_comparison_individual_panels(
    *,
    vel_path: str | Path,
    grace_aligned_path: str | Path,
    w3ra_aligned_anom_path: str | Path,
    s0_pred_path: str | Path,
    sg_pred_path: str | Path,
    output_dir: str | Path,
    velocity_unit_label: str,
    grace_unit_label: str,
    w3ra_unit_label: str,
    inversion_unit_label: str,
    velocity_percentile_high: float = 95.0,
    inversion_crop_margin_pixels: int = 64,
) -> list[dict[str, object]]:
    panels, summary = _comparison_panel_payload(
        vel_path=vel_path,
        grace_aligned_path=grace_aligned_path,
        w3ra_aligned_anom_path=w3ra_aligned_anom_path,
        s0_pred_path=s0_pred_path,
        sg_pred_path=sg_pred_path,
        velocity_percentile_high=velocity_percentile_high,
        inversion_crop_margin_pixels=inversion_crop_margin_pixels,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmap_div = plt.get_cmap("RdBu_r").copy()
    cmap_div.set_bad(color="white")
    unit_labels = {
        "velocity": velocity_unit_label,
        "grace_tws": grace_unit_label,
        "w3ra_s0": w3ra_unit_label,
        "w3ra_sg": w3ra_unit_label,
        "inversion_s0": inversion_unit_label,
        "inversion_sg": inversion_unit_label,
    }
    saved = []
    for panel in panels:
        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
        im = ax.imshow(
            _masked(panel["array"]),
            origin="lower",
            extent=panel["extent"],
            cmap=cmap_div,
            vmin=panel["vmin"],
            vmax=panel["vmax"],
            aspect="auto",
        )
        ax.set_title(panel["title"])
        ax.set_xlabel("Longitude [degrees_east]")
        ax.set_ylabel("Latitude [degrees_north]")
        fig.colorbar(im, ax=ax, shrink=0.85, label=unit_labels[panel["key"]])
        output_path = output_dir / f"{panel['key']}.png"
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        saved.append(
            {
                "key": panel["key"],
                "title": panel["title"],
                "time": panel["time"],
                "output_path": str(output_path),
            }
        )
    return [{"summary": summary}] + saved


def make_punjab_comparison_timeseries(
    *,
    grace_aligned_path: str | Path,
    w3ra_aligned_anom_path: str | Path,
    s0_pred_path: str | Path,
    sg_pred_path: str | Path,
    output_path: str | Path,
    grace_unit_label: str,
    w3ra_unit_label: str,
    inversion_unit_label: str,
) -> dict[str, object]:
    grace = load_h5_netcdf_variable(grace_aligned_path, "lwe_thickness")
    grace_anom = anomaly_relative_to_time_mean(grace["data"])
    grace_series = basin_mean_timeseries(grace_anom)

    w3ra_s0 = load_h5_netcdf_variable(w3ra_aligned_anom_path, "S0")
    w3ra_sg = load_h5_netcdf_variable(w3ra_aligned_anom_path, "Sg")
    w3ra_s0_series = basin_mean_timeseries(w3ra_s0["data"])
    w3ra_sg_series = basin_mean_timeseries(w3ra_sg["data"])

    s0_pred = load_h5_netcdf_variable(s0_pred_path, "S0_pred")
    sg_pred = load_h5_netcdf_variable(sg_pred_path, "Sg_pred")
    inv_s0_series = basin_mean_timeseries(s0_pred["data"])
    inv_sg_series = basin_mean_timeseries(sg_pred["data"])

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), constrained_layout=True, sharex=False)

    axes[0].plot(grace["time"], grace_series, color="tab:purple", linewidth=2.0)
    axes[0].set_title("A. GRACE basin-mean TWS anomaly over Punjab")
    axes[0].set_ylabel(grace_unit_label)
    axes[0].grid(alpha=0.25)

    axes[1].plot(w3ra_s0["time"], w3ra_s0_series, label="W3RA $S_0$", color="tab:blue")
    axes[1].plot(w3ra_sg["time"], w3ra_sg_series, label="W3RA $S_g$", color="tab:green")
    axes[1].set_title("B. W3RA basin-mean storage anomalies")
    axes[1].set_ylabel(w3ra_unit_label)
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)

    axes[2].plot(s0_pred["time"], inv_s0_series, label="Inversion $S_0$", color="tab:blue")
    axes[2].plot(sg_pred["time"], inv_sg_series, label="Inversion $S_g$", color="tab:red")
    axes[2].set_title("C. Inversion support-area mean latent outputs")
    axes[2].set_ylabel(inversion_unit_label)
    axes[2].set_xlabel("Time")
    axes[2].legend(frameon=False)
    axes[2].grid(alpha=0.25)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return {
        "grace_time_start": str(grace["time"][0].date()),
        "grace_time_end": str(grace["time"][-1].date()),
        "w3ra_time_start": str(w3ra_s0["time"][0].date()),
        "w3ra_time_end": str(w3ra_s0["time"][-1].date()),
        "inversion_time_start": str(s0_pred["time"][0].date()),
        "inversion_time_end": str(s0_pred["time"][-1].date()),
        "output_path": str(output_path),
    }
