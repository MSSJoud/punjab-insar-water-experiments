from __future__ import annotations

import json
import shutil
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import xarray as xr


def load_simple_h5_grid(path: str | Path, dataset: str = "z") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        lat = handle["lat"][:]
        lon = handle["lon"][:]
        grid = handle[dataset][:]
    return lat, lon, grid


def compute_time_valid_fraction(
    disp_path: str | Path,
    dataset: str = "z",
    chunk_size: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(disp_path, "r") as handle:
        lat = handle["lat"][:]
        lon = handle["lon"][:]
        disp = handle[dataset]
        n_time = disp.shape[0]
        valid_sum = np.zeros(disp.shape[1:], dtype=np.float32)
        for start in range(0, n_time, chunk_size):
            stop = min(start + chunk_size, n_time)
            valid_sum += np.isfinite(disp[start:stop]).sum(axis=0, dtype=np.float32)
    return lat, lon, valid_sum / float(n_time)


def build_support_mask(
    coherence: np.ndarray,
    time_valid_fraction: np.ndarray,
    coherence_threshold: float = 0.20,
    time_valid_fraction_threshold: float = 0.05,
) -> np.ndarray:
    return (
        np.isfinite(coherence)
        & np.isfinite(time_valid_fraction)
        & (coherence >= coherence_threshold)
        & (time_valid_fraction >= time_valid_fraction_threshold)
    )


def _map_extent(lat: np.ndarray, lon: np.ndarray) -> list[float]:
    return [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]


def _stretch_rgb(rgb: np.ndarray, lower: float = 2.0, upper: float = 98.0, gamma: float = 0.95) -> np.ndarray:
    rgb = np.moveaxis(np.asarray(rgb, dtype=np.float32), 0, -1)
    finite = np.isfinite(rgb)
    if not finite.any():
        return np.zeros_like(rgb, dtype=np.float32)
    lo = np.nanpercentile(rgb[finite], lower)
    hi = np.nanpercentile(rgb[finite], upper)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = 0.0, 1.0
    stretched = np.clip((rgb - lo) / (hi - lo), 0.0, 1.0)
    if gamma != 1.0:
        stretched = np.power(stretched, gamma)
    return stretched


def make_study_area_figure(
    *,
    sentinel_tiff_path: str | Path,
    output_path: str | Path,
    paper_output_path: str | Path | None = None,
    study_bbox: tuple[float, float, float, float] = (75.5972, 76.8819, 31.1222, 31.7500),
    regional_bbox: tuple[float, float, float, float] = (73.0, 77.0, 29.0, 33.0),
    borders_path: str | Path = "/home/ubuntu/anaconda3/envs/swin_env/lib/python3.10/site-packages/pyogrio/tests/fixtures/naturalearth_lowres/naturalearth_lowres.shp",
) -> dict:
    """Create a paper-ready study-region figure with a locator and Sentinel-2 zoom."""

    west, east, south, north = study_bbox
    reg_west, reg_east, reg_south, reg_north = regional_bbox

    with rasterio.open(sentinel_tiff_path) as ds:
        rgb = ds.read([1, 2, 3])
        img_bounds = ds.bounds
    rgb = _stretch_rgb(rgb)

    fig = plt.figure(figsize=(12.8, 6.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.8])
    ax_loc = fig.add_subplot(gs[0, 0])
    ax_zoom = fig.add_subplot(gs[0, 1])

    ax_loc.set_facecolor("#eef3f7")
    borders = gpd.read_file(borders_path)
    region = borders.cx[reg_west:reg_east, reg_south:reg_north].copy()
    region.plot(
        ax=ax_loc,
        facecolor="#f4efe3",
        edgecolor="#6f6a62",
        linewidth=0.7,
    )
    study_rect = Rectangle(
        (west, south),
        east - west,
        north - south,
        facecolor="none",
        edgecolor="#c62828",
        linewidth=2.2,
    )
    ax_loc.add_patch(study_rect)
    label_points = {
        "Pakistan": (73.7, 30.8),
        "India": (75.9, 30.7),
    }
    for label, (x, y) in label_points.items():
        ax_loc.text(x, y, label, color="#4d4d4d", fontsize=10, ha="center", va="center")
    ax_loc.text(
        west + 0.02,
        north + 0.05,
        "Punjab test bed",
        color="#8b1e1e",
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="bottom",
        bbox={"facecolor": (1, 1, 1, 0.72), "edgecolor": "none", "pad": 2.0},
    )
    ax_loc.set_xlim(reg_west, reg_east)
    ax_loc.set_ylim(reg_south, reg_north)
    ax_loc.set_title("A. Regional locator", fontsize=13)
    ax_loc.set_xlabel("Longitude [degrees east]")
    ax_loc.set_ylabel("Latitude [degrees north]")
    ax_loc.grid(color="#bdb7a8", linewidth=0.6, alpha=0.5)

    extent = [img_bounds.left, img_bounds.right, img_bounds.bottom, img_bounds.top]
    ax_zoom.imshow(rgb, origin="upper", extent=extent, aspect="auto")
    ax_zoom.add_patch(
        Rectangle(
            (west, south),
            east - west,
            north - south,
            facecolor="none",
            edgecolor="#ffd54f",
            linewidth=2.0,
            linestyle="-",
        )
    )
    ax_zoom.text(
        west + 0.015,
        north - 0.02,
        "InSAR / inversion extent",
        color="white",
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="top",
        bbox={"facecolor": (0, 0, 0, 0.35), "edgecolor": "none", "pad": 2.5},
    )
    ax_zoom.set_xlim(west, east)
    ax_zoom.set_ylim(south, north)
    ax_zoom.set_title("B. Sentinel-2 true-color zoom", fontsize=13)
    ax_zoom.set_xlabel("Longitude [degrees east]")
    ax_zoom.set_ylabel("Latitude [degrees north]")

    for x1, y1, x2, y2 in [
        (west, south, west, south),
        (east, south, east, south),
        (east, north, east, north),
        (west, north, west, north),
    ]:
        fig.add_artist(
            ConnectionPatch(
                xyA=(x1, y1),
                coordsA=ax_loc.transData,
                xyB=(x2, y2),
                coordsB=ax_zoom.transData,
                color="#8b1e1e",
                linewidth=0.9,
                alpha=0.65,
            )
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    if paper_output_path is not None:
        paper_output_path = Path(paper_output_path)
        paper_output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(paper_output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return {
        "output_path": str(output_path),
        "paper_output_path": str(paper_output_path) if paper_output_path is not None else None,
        "study_bbox": [west, east, south, north],
        "regional_bbox": [reg_west, reg_east, reg_south, reg_north],
        "sentinel_bounds": [float(img_bounds.left), float(img_bounds.right), float(img_bounds.bottom), float(img_bounds.top)],
    }


def make_support_mask_figure(
    *,
    vel_path: str | Path,
    coh_path: str | Path,
    disp_path: str | Path,
    output_path: str | Path,
    coherence_threshold: float = 0.20,
    time_valid_fraction_threshold: float = 0.05,
    velocity_percentiles: tuple[float, float] = (1.0, 99.0),
    coherence_display_percentiles: tuple[float, float] = (5.0, 99.0),
) -> dict:
    lat, lon, velocity = load_simple_h5_grid(vel_path)
    _, _, coherence = load_simple_h5_grid(coh_path)
    _, _, time_valid_fraction = compute_time_valid_fraction(disp_path)
    support_mask = build_support_mask(
        coherence,
        time_valid_fraction,
        coherence_threshold=coherence_threshold,
        time_valid_fraction_threshold=time_valid_fraction_threshold,
    )
    extent = _map_extent(lat, lon)

    finite_velocity = velocity[np.isfinite(velocity)]
    vlim = np.nanpercentile(finite_velocity, list(velocity_percentiles)) if finite_velocity.size else (-1.0, 1.0)
    finite_coh = coherence[np.isfinite(coherence)]
    coh_vmin, coh_vmax = (
        np.nanpercentile(finite_coh, list(coherence_display_percentiles))
        if finite_coh.size
        else (0.0, 1.0)
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    im0 = axes[0, 0].imshow(
        velocity,
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=float(vlim[0]),
        vmax=float(vlim[1]),
        aspect="auto",
    )
    axes[0, 0].set_title("A. InSAR Linear Rate")
    fig.colorbar(im0, ax=axes[0, 0], shrink=0.82, label="Linear rate [mm yr$^{-1}$]")

    im1 = axes[0, 1].imshow(
        coherence,
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=float(coh_vmin),
        vmax=float(coh_vmax),
        aspect="auto",
    )
    axes[0, 1].contour(
        lon,
        lat,
        support_mask.astype(float),
        levels=[0.5],
        colors="w",
        linewidths=0.8,
    )
    axes[0, 1].set_title("B. Static Coherence Raster")
    fig.colorbar(im1, ax=axes[0, 1], shrink=0.82, label="Coherence [0-1]")

    im2 = axes[1, 0].imshow(
        time_valid_fraction,
        origin="lower",
        extent=extent,
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )
    axes[1, 0].set_title("C. Temporal Observation Fraction")
    fig.colorbar(im2, ax=axes[1, 0], shrink=0.82, label="Valid fraction [0-1]")

    im3 = axes[1, 1].imshow(
        support_mask.astype(np.uint8),
        origin="lower",
        extent=extent,
        cmap="gray_r",
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    axes[1, 1].set_title("D. Support Mask Used In Training")
    fig.colorbar(im3, ax=axes[1, 1], shrink=0.82, ticks=[0, 1], label="Selected [0/1]")

    for ax in axes.ravel():
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "support_fraction": float(np.mean(support_mask)),
        "velocity_vmin": float(vlim[0]),
        "velocity_vmax": float(vlim[1]),
        "coherence_vmin": float(coh_vmin),
        "coherence_vmax": float(coh_vmax),
        "coherence_threshold": float(coherence_threshold),
        "time_valid_fraction_threshold": float(time_valid_fraction_threshold),
        "output_path": str(output_path),
    }


def copy_existing_figure(source_path: str | Path, output_path: str | Path) -> dict:
    source = Path(source_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    return {"source_path": str(source), "output_path": str(output)}


def make_prior_ablation_figure(
    *,
    summary_json_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    summary = pd.DataFrame(json.load(open(summary_json_path)))
    summary = summary.copy()
    summary["delta_forward"] = summary["val_forward"] - summary.loc[summary["label"] == "Phase 1 Baseline", "val_forward"].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    x = np.arange(len(summary))

    axes[0].bar(x, summary["val_forward"], color=["#2f4b7c"] + ["#8da0cb"] * (len(summary) - 1))
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(summary["label"], rotation=30, ha="right")
    axes[0].set_ylabel("Validation Forward Loss")
    axes[0].set_title("A. Punjab Baseline And Prior Ablations")

    axes[1].bar(x, summary["forward_rmse_norm_mean"], color=["#2f4b7c"] + ["#66c2a5"] * (len(summary) - 1))
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(summary["label"], rotation=30, ha="right")
    axes[1].set_ylabel("Normalized Forward RMSE")
    axes[1].set_title("B. Validation Residual Scale")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return summary


def load_prediction_archive_metadata(archive_path: str | Path) -> dict:
    with h5py.File(archive_path, "r") as handle:
        return {
            "S0_pred_shape": tuple(handle["S0_pred"].shape),
            "Sg_pred_shape": tuple(handle["Sg_pred"].shape),
            "time_start": handle["dates"][0].decode("utf-8"),
            "time_end": handle["dates"][-1].decode("utf-8"),
            "n_supported_pixels": int(handle["pixel_rows"].shape[0]),
            "support_fraction": float(handle["support_mask"][:].mean()),
        }


def load_netcdf_h5_metadata(path: str | Path) -> dict:
    meta: dict[str, object] = {}
    with h5py.File(path, "r") as handle:
        for key in ("time", "lat", "lon", "support_mask"):
            if key in handle:
                meta[f"{key}_shape"] = tuple(handle[key].shape)
        data_vars = [name for name in handle.keys() if name.endswith("_pred")]
        meta["data_vars"] = data_vars
        if "time" in handle:
            time_values = handle["time"][:]
            units = handle["time"].attrs.get("units", b"")
            if isinstance(units, bytes):
                units = units.decode("utf-8")
            if isinstance(units, str) and units.startswith("days since "):
                base_time = pd.Timestamp(units.replace("days since ", ""))
                times = base_time + pd.to_timedelta(time_values.astype(float), unit="D")
            else:
                times = pd.to_datetime(time_values, unit="ns", utc=True).tz_convert(None)
            meta["time_start"] = str(times[0].date())
            meta["time_end"] = str(times[-1].date())
    return meta


def make_baseline_export_panel(
    *,
    archive_path: str | Path,
    sample_pixels_csv: str | Path,
    output_path: str | Path,
    map_s0_path: str | Path | None = None,
    map_sg_path: str | Path | None = None,
) -> pd.DataFrame:
    sample_pixels = pd.read_csv(sample_pixels_csv)
    with h5py.File(archive_path, "r") as handle:
        lat = handle["lat"][:]
        lon = handle["lon"][:]
        s0_series = handle["S0_pred"][:]
        sg_series = handle["Sg_pred"][:]
        rows = handle["pixel_rows"][:]
        cols = handle["pixel_cols"][:]
        dates = pd.to_datetime([d.decode("utf-8") for d in handle["dates"][:]])

    if map_s0_path is not None and map_sg_path is not None:
        s0_ds = xr.open_dataset(map_s0_path)
        sg_ds = xr.open_dataset(map_sg_path)
        try:
            s0_map = np.asarray(s0_ds["S0_pred"].isel(time=-1))
            sg_map = np.asarray(sg_ds["Sg_pred"].isel(time=-1))
            map_lat = np.asarray(s0_ds["lat"])
            map_lon = np.asarray(s0_ds["lon"])
        finally:
            s0_ds.close()
            sg_ds.close()
    else:
        with h5py.File(archive_path, "r") as handle:
            s0_map = handle["S0_last_map"][:]
            sg_map = handle["Sg_last_map"][:]
            map_lat = lat
            map_lon = lon

    pixel_lookup = {(int(r), int(c)): idx for idx, (r, c) in enumerate(zip(rows, cols, strict=False))}
    extent = _map_extent(map_lat, map_lon)

    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 4)
    map_axes = [
        fig.add_subplot(grid[0, 0:2]),
        fig.add_subplot(grid[0, 2:4]),
    ]
    ts_axes = [
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[1, 2]),
    ]
    legend_ax = fig.add_subplot(grid[1, 3])

    maps = [
        (s0_map, "A. Latest $S_0$ Over Full Scene", "coolwarm"),
        (sg_map, "B. Latest $S_g$ Over Full Scene", "coolwarm"),
    ]
    for ax, (data, title, cmap) in zip(map_axes, maps, strict=False):
        finite = data[np.isfinite(data)]
        vmin, vmax = np.nanpercentile(finite, [2.0, 98.0]) if finite.size else (-1.0, 1.0)
        image = ax.imshow(
            data,
            origin="lower",
            extent=extent,
            cmap=cmap,
            vmin=float(vmin),
            vmax=float(vmax),
            aspect="auto",
        )
        ax.set_title(title)
        ax.set_xlabel("Longitude [degrees_east]")
        ax.set_ylabel("Latitude [degrees_north]")
        fig.colorbar(image, ax=ax, shrink=0.82, label="Inversion latent output [model units]")

    for idx, row in sample_pixels.iterrows():
        pixel_index = pixel_lookup.get((int(row["row"]), int(row["col"])))
        if pixel_index is None:
            continue
        color = f"C{idx}"
        for ax in map_axes:
            ax.plot(float(row["lon"]), float(row["lat"]), marker="o", markersize=4, color=color)
        ts_ax = ts_axes[idx]
        ts_ax.plot(dates, s0_series[:, pixel_index], color="#1f77b4", label="$S_0$")
        ts_ax.plot(dates, sg_series[:, pixel_index], color="#d62728", label="$S_g$")
        ts_ax.set_title(f"{chr(67 + idx)}. Pixel {idx + 1}")
        ts_ax.set_xlabel("Time")
        ts_ax.set_ylabel("Latent output [model units]")
        ts_ax.tick_params(axis="x", rotation=30)
        ts_ax.text(
            0.02,
            0.98,
            f"lat={row['lat']:.3f}\nlon={row['lon']:.3f}",
            transform=ts_ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )

    legend_ax.axis("off")
    legend_ax.plot([], [], color="#1f77b4", label="$S_0$")
    legend_ax.plot([], [], color="#d62728", label="$S_g$")
    legend_ax.legend(loc="center", frameon=False)
    legend_ax.set_title("Series Legend")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return sample_pixels


def make_synthetic_conditioning_figure(*, output_path: str | Path) -> dict[str, float]:
    from run_synthetic_swin3d_experiment import PHYSICS, build_elastic_kernel, build_poroelastic_kernel, make_synthetic_layers

    layers = make_synthetic_layers()
    s0, ss, sd, sg = layers[:, 0], layers[:, 1], layers[:, 2], layers[:, 3]
    g_load = build_elastic_kernel(PHYSICS.E, PHYSICS.nu, PHYSICS.dx, PHYSICS.dy, PHYSICS.a_load, s0.shape[-1], s0.shape[-2])
    g_poro = build_poroelastic_kernel(
        PHYSICS.E, PHYSICS.nu, PHYSICS.alpha, PHYSICS.Hg, PHYSICS.dx, PHYSICS.dy, PHYSICS.a_poro, s0.shape[-1], s0.shape[-2]
    )
    g_load_fft = np.fft.fft2(np.fft.ifftshift(g_load))
    g_poro_fft = np.fft.fft2(np.fft.ifftshift(g_poro))

    delta_l = PHYSICS.rho_w * (s0 + ss + sd)
    delta_p = PHYSICS.rho_w * PHYSICS.g * (sg / PHYSICS.Seff)

    uz_load = np.zeros_like(delta_l, dtype=np.float32)
    uz_poro = np.zeros_like(delta_p, dtype=np.float32)
    for i in range(delta_l.shape[0]):
        uz_load[i] = np.fft.ifft2(np.fft.fft2(delta_l[i]) * g_load_fft).real.astype(np.float32)
        uz_poro[i] = np.fft.ifft2(np.fft.fft2(delta_p[i]) * g_poro_fft).real.astype(np.float32)
    uz_total = uz_load + uz_poro

    rms_load_mm = np.sqrt(np.mean(uz_load**2, axis=(1, 2))) * 1e3
    rms_poro_mm = np.sqrt(np.mean(uz_poro**2, axis=(1, 2))) * 1e3
    rms_total_mm = np.sqrt(np.mean(uz_total**2, axis=(1, 2))) * 1e3
    ratio = float(np.median(rms_poro_mm / np.maximum(rms_load_mm, 1e-12)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    t = np.arange(len(rms_total_mm))
    axes[0].semilogy(t, rms_load_mm, label="Elastic", linewidth=2)
    axes[0].semilogy(t, rms_poro_mm, label="Poroelastic", linewidth=2)
    axes[0].semilogy(t, rms_total_mm, label="Total", linewidth=2, color="black", alpha=0.8)
    axes[0].set_title("A. RMS displacement by time step")
    axes[0].set_xlabel("Synthetic month")
    axes[0].set_ylabel("RMS displacement [mm]")
    axes[0].grid(alpha=0.25, which="both")
    axes[0].legend(frameon=False)

    labels = ["Elastic", "Poroelastic", "Total"]
    medians = [float(np.median(rms_load_mm)), float(np.median(rms_poro_mm)), float(np.median(rms_total_mm))]
    colors = ["#4c78a8", "#f58518", "#333333"]
    axes[1].bar(labels, medians, color=colors)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Median RMS displacement [mm]")
    axes[1].set_title("B. Median magnitude contrast")
    axes[1].grid(axis="y", alpha=0.25, which="both")
    axes[1].text(
        0.98,
        0.95,
        f"median poro/elastic ratio = {ratio:.1f}x",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {
        "median_rms_elastic_mm": medians[0],
        "median_rms_poroelastic_mm": medians[1],
        "median_rms_total_mm": medians[2],
        "median_poro_to_elastic_ratio": ratio,
        "output_path": str(output_path),
    }
