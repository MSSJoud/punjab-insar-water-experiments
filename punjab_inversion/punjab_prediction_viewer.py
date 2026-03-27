from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset

from .metrics import normalize_field
from .models import DualDecoderFrequencySeparatedSwinUNet3D
from .physics import PhysicsConfig, build_fft_kernels, set_seed


WINDOW_SIZE = 12
DEFAULT_SUPPORT_CONFIG = {
    "coherence_threshold": 0.20,
    "time_valid_fraction_threshold": 0.05,
    "tile_size": (64, 64),
    "tile_stride": (64, 64),
    "tile_min_valid_fraction": 0.20,
}


@dataclass(frozen=True)
class PredictionArchiveSummary:
    output_path: str
    summary_path: str
    preview_path: str
    n_times: int
    n_tiles: int
    n_supported_pixels: int
    date_start: str
    date_end: str
    grid_shape: tuple[int, int]
    checkpoint_path: str


def parse_acquisition_dates(date_path: str | Path) -> pd.DatetimeIndex:
    with h5py.File(date_path, "r") as f:
        raw = f["acquisition_dates"][0]
    return pd.DatetimeIndex(
        pd.to_datetime([v.decode() if isinstance(v, bytes) else str(v) for v in raw], format="%d-%b-%Y")
    )


def load_punjab_meta(data_root: str | Path) -> dict:
    root = Path(data_root)
    disp_path = root / "disp_all_ll.h5"
    coh_path = root / "coh_ll.h5"
    date_path = root / "aquisition_dates_ll.h5"
    dates = parse_acquisition_dates(date_path)
    with h5py.File(disp_path, "r") as f:
        lat = np.asarray(f["lat"])
        lon = np.asarray(f["lon"])
        disp_shape = tuple(f["z"].shape)
    return {
        "root": str(root),
        "disp_path": str(disp_path),
        "coh_path": str(coh_path),
        "date_path": str(date_path),
        "lat": lat,
        "lon": lon,
        "dates": dates,
        "disp_shape": disp_shape,
    }


def compute_time_valid_fraction(disp_path: str | Path, chunk_size: int = 25) -> np.ndarray:
    with h5py.File(disp_path, "r") as f:
        z = f["z"]
        counts = np.zeros(z.shape[1:], dtype=np.uint16)
        n_time = z.shape[0]
        for t0 in range(0, n_time, chunk_size):
            chunk = np.asarray(z[t0 : min(t0 + chunk_size, n_time)])
            counts += np.isfinite(chunk).sum(axis=0).astype(np.uint16)
    return counts.astype(np.float32) / max(n_time, 1)


def build_support_mask(meta: dict, support_config: dict | None = None) -> np.ndarray:
    cfg = DEFAULT_SUPPORT_CONFIG if support_config is None else support_config
    with h5py.File(meta["coh_path"], "r") as f:
        coh = np.asarray(f["z"])
    time_valid_fraction = compute_time_valid_fraction(meta["disp_path"])
    temporal_support = time_valid_fraction >= cfg["time_valid_fraction_threshold"]
    coh_valid = np.isfinite(coh)
    return coh_valid & temporal_support & (coh >= cfg["coherence_threshold"])


def build_valid_tiles(valid_mask: np.ndarray, support_config: dict | None = None) -> list[dict]:
    cfg = DEFAULT_SUPPORT_CONFIG if support_config is None else support_config
    tile_h, tile_w = cfg["tile_size"]
    stride_h, stride_w = cfg["tile_stride"]
    h, w = valid_mask.shape
    tiles: list[dict] = []
    for row in range(0, h - tile_h + 1, stride_h):
        for col in range(0, w - tile_w + 1, stride_w):
            tile = valid_mask[row : row + tile_h, col : col + tile_w]
            frac = float(tile.mean())
            if frac >= cfg["tile_min_valid_fraction"]:
                tiles.append({"row": row, "col": col, "valid_fraction": frac})
    return tiles


def build_full_scene_tiles(grid_shape: tuple[int, int], support_config: dict | None = None) -> list[dict]:
    cfg = DEFAULT_SUPPORT_CONFIG if support_config is None else support_config
    tile_h, tile_w = cfg["tile_size"]
    stride_h, stride_w = cfg["tile_stride"]
    h, w = grid_shape

    row_starts = list(range(0, max(h - tile_h + 1, 1), stride_h))
    col_starts = list(range(0, max(w - tile_w + 1, 1), stride_w))
    last_row = max(h - tile_h, 0)
    last_col = max(w - tile_w, 0)
    if not row_starts or row_starts[-1] != last_row:
        row_starts.append(last_row)
    if not col_starts or col_starts[-1] != last_col:
        col_starts.append(last_col)

    tiles: list[dict] = []
    for row in row_starts:
        for col in col_starts:
            tiles.append({"row": int(row), "col": int(col), "valid_fraction": 1.0})
    return tiles


def make_all_end_indices(dates: pd.DatetimeIndex, window_size: int = WINDOW_SIZE) -> np.ndarray:
    return np.arange(window_size - 1, len(dates))


class PunjabWindowedTileDataset(Dataset):
    def __init__(
        self,
        disp_path: str | Path,
        end_indices,
        tiles: list[dict],
        support_mask: np.ndarray,
        window_size: int = WINDOW_SIZE,
        fill_value: float = 0.0,
    ) -> None:
        self.disp_path = str(disp_path)
        self.end_indices = list(end_indices)
        self.tiles = list(tiles)
        self.support_mask = support_mask.astype(np.float32)
        self.window_size = int(window_size)
        self.fill_value = float(fill_value)
        self.tile_h = 64
        self.tile_w = 64
        self._disp_file = None

    def _disp_dataset(self):
        if self._disp_file is None:
            self._disp_file = h5py.File(self.disp_path, "r")
        return self._disp_file["z"]

    def __len__(self) -> int:
        return len(self.end_indices) * len(self.tiles)

    def __getitem__(self, idx: int) -> dict:
        time_idx = idx // len(self.tiles)
        tile_idx = idx % len(self.tiles)
        end_idx = self.end_indices[time_idx]
        start_idx = end_idx - self.window_size + 1
        tile = self.tiles[tile_idx]
        r0, c0 = tile["row"], tile["col"]
        r1, c1 = r0 + self.tile_h, c0 + self.tile_w
        x = np.asarray(self._disp_dataset()[start_idx : end_idx + 1, r0:r1, c0:c1], dtype=np.float32)
        mask = self.support_mask[r0:r1, c0:c1]
        x = np.where(np.isfinite(x), x, self.fill_value)
        x = x * mask[None, ...]
        return {
            "x": torch.tensor(x[None, ...], dtype=torch.float32),
            "mask": torch.tensor(mask[None, ...], dtype=torch.float32),
            "end_idx": int(end_idx),
            "tile_row": int(r0),
            "tile_col": int(c0),
        }


def compute_scalar_stats(dataset: Dataset, key: str, batch_size: int = 4) -> tuple[float, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    total_sum = 0.0
    total_sq = 0.0
    total_count = 0
    for batch in loader:
        x = batch[key].float()
        total_sum += x.sum().item()
        total_sq += (x * x).sum().item()
        total_count += x.numel()
    mean = total_sum / max(total_count, 1)
    var = max(total_sq / max(total_count, 1) - mean**2, 1e-8)
    return mean, float(np.sqrt(var))


class NormalizedPunjabTileDataset(Dataset):
    def __init__(self, base_dataset: Dataset, x_mean: float, x_std: float) -> None:
        self.base_dataset = base_dataset
        self.x_mean = float(x_mean)
        self.x_std = float(max(x_std, 1e-6))

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.base_dataset[idx]
        return {
            **item,
            "x": (item["x"] - self.x_mean) / self.x_std,
        }


def build_baseline_prediction_archive(
    output_path: str | Path,
    checkpoint_path: str | Path,
    *,
    data_root: str | Path = "/mnt/data/aoi_punjab",
    support_config: dict | None = None,
    end_indices_override=None,
    tiles_override: list[dict] | None = None,
    batch_size: int = 8,
    device: str | None = None,
    seed: int = 42,
    progress_every: int = 100,
    keep_all_tile_values: bool = False,
    support_mask_override: np.ndarray | None = None,
) -> PredictionArchiveSummary:
    import matplotlib.pyplot as plt

    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    preview_path = output_path.with_suffix(".png")

    meta = load_punjab_meta(data_root)
    support_mask = (
        np.asarray(support_mask_override, dtype=bool)
        if support_mask_override is not None
        else build_support_mask(meta, support_config=support_config)
    )
    tiles = list(tiles_override) if tiles_override is not None else build_valid_tiles(support_mask, support_config=support_config)
    end_indices = np.asarray(end_indices_override if end_indices_override is not None else make_all_end_indices(meta["dates"]))

    base_ds = PunjabWindowedTileDataset(meta["disp_path"], end_indices, tiles, support_mask, window_size=WINDOW_SIZE)
    x_mean, x_std = compute_scalar_stats(base_ds, key="x")
    ds = NormalizedPunjabTileDataset(base_ds, x_mean, x_std)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = DualDecoderFrequencySeparatedSwinUNet3D(
        base_dim=32,
        time_patch=2,
        spatial_patch=4,
        num_heads=4,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    physics = PhysicsConfig()
    tile_h, tile_w = 64, 64
    build_fft_kernels(tile_h, tile_w, physics, device)

    h, w = support_mask.shape
    archive_mask = np.zeros_like(support_mask, dtype=bool)
    for tile in tiles:
        r0, c0 = tile["row"], tile["col"]
        if keep_all_tile_values:
            archive_mask[r0 : r0 + tile_h, c0 : c0 + tile_w] = True
        else:
            archive_mask[r0 : r0 + tile_h, c0 : c0 + tile_w] = support_mask[r0 : r0 + tile_h, c0 : c0 + tile_w]
    pixel_rows, pixel_cols = np.where(archive_mask)
    n_pixels = len(pixel_rows)
    n_times = len(end_indices)
    time_lookup = {int(v): i for i, v in enumerate(end_indices)}
    index_map = np.full((h, w), -1, dtype=np.int32)
    index_map[pixel_rows, pixel_cols] = np.arange(n_pixels, dtype=np.int32)

    s0 = np.full((n_times, n_pixels), np.nan, dtype=np.float16)
    sg = np.full((n_times, n_pixels), np.nan, dtype=np.float16)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            xb = batch["x"].to(device)
            mask = batch["mask"].to(device)
            pred_state = model(xb).detach().cpu().numpy()
            mask_np = batch["mask"].cpu().numpy().astype(bool)
            end_idx_np = batch["end_idx"].numpy()
            tile_row_np = batch["tile_row"].numpy()
            tile_col_np = batch["tile_col"].numpy()
            for i in range(pred_state.shape[0]):
                t_pos = time_lookup[int(end_idx_np[i])]
                r0 = int(tile_row_np[i])
                c0 = int(tile_col_np[i])
                if keep_all_tile_values:
                    local_valid = np.ones((tile_h, tile_w), dtype=bool)
                else:
                    local_valid = mask_np[i, 0]
                local_indices = index_map[r0 : r0 + tile_h, c0 : c0 + tile_w][local_valid]
                s0[t_pos, local_indices] = pred_state[i, 0][local_valid].astype(np.float16)
                sg[t_pos, local_indices] = pred_state[i, 1][local_valid].astype(np.float16)
            if progress_every and batch_idx % progress_every == 0:
                print(f"archive inference batch {batch_idx}/{len(loader)}")

    mean_s0 = np.full((h, w), np.nan, dtype=np.float32)
    mean_sg = np.full((h, w), np.nan, dtype=np.float32)
    last_s0 = np.full((h, w), np.nan, dtype=np.float32)
    last_sg = np.full((h, w), np.nan, dtype=np.float32)
    std_s0 = np.full((h, w), np.nan, dtype=np.float32)
    std_sg = np.full((h, w), np.nan, dtype=np.float32)

    s0_f32 = s0.astype(np.float32)
    sg_f32 = sg.astype(np.float32)
    mean_s0[pixel_rows, pixel_cols] = np.nanmean(s0_f32, axis=0)
    mean_sg[pixel_rows, pixel_cols] = np.nanmean(sg_f32, axis=0)
    last_s0[pixel_rows, pixel_cols] = s0_f32[-1]
    last_sg[pixel_rows, pixel_cols] = sg_f32[-1]
    std_s0[pixel_rows, pixel_cols] = np.nanstd(s0_f32, axis=0)
    std_sg[pixel_rows, pixel_cols] = np.nanstd(sg_f32, axis=0)

    str_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(output_path, "w") as f:
        archive_dates = pd.DatetimeIndex(meta["dates"][end_indices])
        f.create_dataset("dates", data=np.array([str(pd.Timestamp(d).date()) for d in archive_dates], dtype=str_dtype))
        f.create_dataset("time_iso", data=np.array([pd.Timestamp(d).isoformat() for d in archive_dates], dtype=str_dtype))
        f.create_dataset("time_unix_ns", data=archive_dates.view("int64"))
        f.create_dataset("end_indices", data=end_indices.astype(np.int32))
        f.create_dataset("lat", data=np.asarray(meta["lat"], dtype=np.float32), compression="gzip")
        f.create_dataset("lon", data=np.asarray(meta["lon"], dtype=np.float32), compression="gzip")
        f.create_dataset("pixel_rows", data=pixel_rows.astype(np.int32), compression="gzip")
        f.create_dataset("pixel_cols", data=pixel_cols.astype(np.int32), compression="gzip")
        f.create_dataset("index_map", data=index_map, compression="gzip")
        f.create_dataset("support_mask", data=archive_mask.astype(np.uint8), compression="gzip")
        f.create_dataset("source_support_mask", data=support_mask.astype(np.uint8), compression="gzip")
        f.create_dataset("s0", data=s0, compression="gzip", shuffle=True)
        f.create_dataset("sg", data=sg, compression="gzip", shuffle=True)
        f["S0_pred"] = h5py.SoftLink("/s0")
        f["Sg_pred"] = h5py.SoftLink("/sg")
        f.create_dataset("s0_mean_map", data=mean_s0, compression="gzip")
        f.create_dataset("sg_mean_map", data=mean_sg, compression="gzip")
        f["S0_mean_map"] = h5py.SoftLink("/s0_mean_map")
        f["Sg_mean_map"] = h5py.SoftLink("/sg_mean_map")
        f.create_dataset("s0_last_map", data=last_s0, compression="gzip")
        f.create_dataset("sg_last_map", data=last_sg, compression="gzip")
        f["S0_last_map"] = h5py.SoftLink("/s0_last_map")
        f["Sg_last_map"] = h5py.SoftLink("/sg_last_map")
        f.create_dataset("s0_std_map", data=std_s0, compression="gzip")
        f.create_dataset("sg_std_map", data=std_sg, compression="gzip")
        f["S0_std_map"] = h5py.SoftLink("/s0_std_map")
        f["Sg_std_map"] = h5py.SoftLink("/sg_std_map")
        f.attrs["grid_shape"] = (h, w)
        f.attrs["n_tiles"] = len(tiles)
        f.attrs["n_supported_pixels"] = n_pixels
        f.attrs["keep_all_tile_values"] = bool(keep_all_tile_values)
        f.attrs["checkpoint_path"] = str(checkpoint_path)
        f.attrs["x_mean"] = x_mean
        f.attrs["x_std"] = x_std
        f.attrs["date_start"] = str(archive_dates[0].date())
        f.attrs["date_end"] = str(archive_dates[-1].date())
        if support_mask_override is not None and keep_all_tile_values:
            note = "Predictions are limited to the selected tiles; all tile values are retained and no support mask is applied."
        elif keep_all_tile_values:
            note = "Predictions are limited to the selected baseline tiles; outside tile coverage is NaN."
        else:
            note = "Predictions are limited to the selected baseline tiles; outside support is NaN."
        f.attrs["note"] = note

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    panels = [
        ("S0 mean", mean_s0, "viridis"),
        ("Sg mean", mean_sg, "viridis"),
        ("S0 last", last_s0, "viridis"),
        ("Sg last", last_sg, "viridis"),
    ]
    for ax, (title, arr, cmap) in zip(axes.ravel(), panels):
        im = ax.imshow(arr, cmap=cmap)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(preview_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary = PredictionArchiveSummary(
        output_path=str(output_path),
        summary_path=str(summary_path),
        preview_path=str(preview_path),
        n_times=n_times,
        n_tiles=len(tiles),
        n_supported_pixels=n_pixels,
        date_start=str(pd.Timestamp(meta["dates"][end_indices][0]).date()),
        date_end=str(pd.Timestamp(meta["dates"][end_indices][-1]).date()),
        grid_shape=(h, w),
        checkpoint_path=str(checkpoint_path),
    )
    with open(summary_path, "w") as f:
        import json

        json.dump(summary.__dict__, f, indent=2)
    return summary


class PredictionArchive:
    def __init__(self, archive_path: str | Path) -> None:
        self.path = Path(archive_path)
        self.h5 = h5py.File(self.path, "r")
        self.dates = pd.DatetimeIndex(pd.to_datetime(self.h5["dates"][:].astype(str)))
        self.index_map = self.h5["index_map"][:]
        self.support_mask = self.h5["support_mask"][:].astype(bool)

    def close(self) -> None:
        self.h5.close()

    def map_data(self, layer: str = "sg", mode: str = "mean", time_idx: int | None = None) -> np.ndarray:
        layer = layer.lower()
        if mode == "time":
            if time_idx is None:
                raise ValueError("time_idx is required when mode='time'")
            vec = self.h5[layer][time_idx].astype(np.float32)
            arr = np.full(self.index_map.shape, np.nan, dtype=np.float32)
            valid = self.index_map >= 0
            arr[valid] = vec[self.index_map[valid]]
            return arr
        ds_name = f"{layer}_{mode}_map"
        return self.h5[ds_name][:].astype(np.float32)

    def pixel_series(self, row: int, col: int) -> dict | None:
        if row < 0 or col < 0 or row >= self.index_map.shape[0] or col >= self.index_map.shape[1]:
            return None
        idx = int(self.index_map[row, col])
        if idx < 0:
            return None
        return {
            "dates": self.dates,
            "s0": self.h5["s0"][:, idx].astype(np.float32),
            "sg": self.h5["sg"][:, idx].astype(np.float32),
            "row": row,
            "col": col,
        }


def export_prediction_archive_to_netcdf(
    archive_path: str | Path,
    *,
    s0_output_path: str | Path | None = None,
    sg_output_path: str | Path | None = None,
) -> tuple[Path, Path]:
    archive = PredictionArchive(archive_path)
    try:
        h, w = archive.index_map.shape
        n_times = len(archive.dates)
        row_idx, col_idx = np.where(archive.support_mask)
        s0_cube = np.full((n_times, h, w), np.nan, dtype=np.float32)
        sg_cube = np.full((n_times, h, w), np.nan, dtype=np.float32)
        s0_vals = archive.h5["s0"][:].astype(np.float32)
        sg_vals = archive.h5["sg"][:].astype(np.float32)
        s0_cube[:, row_idx, col_idx] = s0_vals
        sg_cube[:, row_idx, col_idx] = sg_vals

        lat = archive.h5["lat"][:] if "lat" in archive.h5 else np.arange(h, dtype=np.int32)
        lon = archive.h5["lon"][:] if "lon" in archive.h5 else np.arange(w, dtype=np.int32)
        y = np.arange(h, dtype=np.int32)
        x = np.arange(w, dtype=np.int32)
        coords = {
            "time": archive.dates,
            "y": y,
            "x": x,
            "lat": ("y", lat),
            "lon": ("x", lon),
        }
        common_attrs = {
            "source_archive": str(archive.path),
            "note": archive.h5.attrs.get("note", "Predictions are limited to the selected archive footprint; outside support is NaN."),
        }
        data_vars_s0 = {
            "S0_pred": (("time", "y", "x"), s0_cube),
            "support_mask": (("y", "x"), archive.support_mask.astype(np.uint8)),
        }
        data_vars_sg = {
            "Sg_pred": (("time", "y", "x"), sg_cube),
            "support_mask": (("y", "x"), archive.support_mask.astype(np.uint8)),
        }
        if "source_support_mask" in archive.h5:
            source_support = archive.h5["source_support_mask"][:].astype(np.uint8)
            data_vars_s0["source_support_mask"] = (("y", "x"), source_support)
            data_vars_sg["source_support_mask"] = (("y", "x"), source_support)
        s0_ds = xr.Dataset(
            data_vars_s0,
            coords=coords,
            attrs=common_attrs,
        )
        sg_ds = xr.Dataset(
            data_vars_sg,
            coords=coords,
            attrs=common_attrs,
        )

        archive_path = Path(archive_path)
        s0_output = Path(s0_output_path) if s0_output_path is not None else archive_path.with_name(archive_path.stem + "_s0.nc")
        sg_output = Path(sg_output_path) if sg_output_path is not None else archive_path.with_name(archive_path.stem + "_sg.nc")
        s0_ds.to_netcdf(s0_output)
        sg_ds.to_netcdf(sg_output)
        return s0_output, sg_output
    finally:
        archive.close()


def launch_notebook_prediction_viewer(archive_path: str | Path):
    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    from IPython.display import display

    archive = PredictionArchive(archive_path)
    layer = widgets.ToggleButtons(options=[("Sg", "sg"), ("S0", "s0")], value="sg", description="Map layer")
    mode = widgets.ToggleButtons(options=[("Mean", "mean"), ("Last", "last"), ("Std", "std"), ("Time", "time")], value="mean", description="Map mode")
    time_slider = widgets.IntSlider(min=0, max=len(archive.dates) - 1, step=1, value=len(archive.dates) - 1, description="Time")
    message = widgets.HTML(value="Click a supported pixel in the map to plot its S0/Sg time series.")

    fig, (ax_map, ax_ts) = plt.subplots(1, 2, figsize=(14, 6))
    marker = {"artist": None}

    def current_map() -> np.ndarray:
        if mode.value == "time":
            time_slider.disabled = False
            return archive.map_data(layer=layer.value, mode="time", time_idx=time_slider.value)
        time_slider.disabled = True
        return archive.map_data(layer=layer.value, mode=mode.value)

    img = ax_map.imshow(current_map(), cmap="viridis")
    ax_map.set_title("Prediction map")
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    fig.colorbar(img, ax=ax_map, fraction=0.046, pad=0.04)
    ax_ts.set_title("Click a supported pixel")
    ax_ts.set_xlabel("Date")
    ax_ts.set_ylabel("Predicted state")
    fig.tight_layout()

    def redraw_map(*_args) -> None:
        arr = current_map()
        img.set_data(arr)
        if mode.value == "time":
            title = f"{layer.label} at {archive.dates[time_slider.value].date()}"
        else:
            title = f"{layer.label} {mode.label}"
        ax_map.set_title(title)
        finite = np.isfinite(arr)
        if finite.any():
            img.set_clim(float(np.nanmin(arr)), float(np.nanmax(arr)))
        fig.canvas.draw_idle()

    def on_click(event) -> None:
        if event.inaxes != ax_map or event.xdata is None or event.ydata is None:
            return
        col = int(round(event.xdata))
        row = int(round(event.ydata))
        series = archive.pixel_series(row, col)
        if series is None:
            message.value = f"<b>No supported prediction at row={row}, col={col}.</b>"
            return
        ax_ts.clear()
        ax_ts.plot(series["dates"], series["s0"], label="S0", color="tab:blue")
        ax_ts.plot(series["dates"], series["sg"], label="Sg", color="tab:green")
        ax_ts.set_title(f"Predicted S0/Sg at row={row}, col={col}")
        ax_ts.set_xlabel("Date")
        ax_ts.set_ylabel("Predicted state")
        ax_ts.legend(loc="upper right")
        if marker["artist"] is not None:
            marker["artist"].remove()
        marker["artist"] = ax_map.scatter([col], [row], s=28, c="red")
        fig.canvas.draw_idle()
        message.value = f"<b>Selected row={row}, col={col}</b>"

    layer.observe(redraw_map, names="value")
    mode.observe(redraw_map, names="value")
    time_slider.observe(redraw_map, names="value")
    fig.canvas.mpl_connect("button_press_event", on_click)

    display(widgets.VBox([widgets.HBox([layer, mode]), time_slider, message]))
    plt.show()
    return archive
