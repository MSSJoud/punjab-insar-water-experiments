from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


@dataclass(frozen=True)
class PriorAlignmentConfig:
    baseline_mode: str = "overlap_mean"
    resample_rule: str = "MS"


GRACE_CANDIDATE_PATTERNS = (
    "*GRACE*",
    "*grace*",
    "*mascon*",
    "*mascons*",
    "*JPL*",
    "*CSR*",
)


def punjab_month_index(dates) -> pd.DatetimeIndex:
    dt = pd.to_datetime(dates)
    return pd.DatetimeIndex(dt.to_period("M").to_timestamp())


def align_w3ra_to_punjab_dates(
    w3ra_ds: xr.Dataset,
    punjab_dates,
    variables: tuple[str, ...] = ("S0", "Sg"),
) -> xr.Dataset:
    month_index = punjab_month_index(punjab_dates)
    aligned = w3ra_ds[list(variables)].sel(time=month_index)
    aligned = aligned.assign_coords(time=pd.DatetimeIndex(pd.to_datetime(punjab_dates)))
    aligned = aligned.assign_coords(source_month=("time", month_index.values))
    return aligned


def compute_w3ra_anomalies(
    aligned_ds: xr.Dataset,
    baseline_start=None,
    baseline_end=None,
    variables: tuple[str, ...] | None = None,
) -> xr.Dataset:
    vars_to_use = list(variables) if variables is not None else list(aligned_ds.data_vars)
    ds = aligned_ds[vars_to_use]
    if baseline_start is None:
        baseline_start = pd.to_datetime(ds.time.values[0])
    if baseline_end is None:
        baseline_end = pd.to_datetime(ds.time.values[-1])
    baseline = ds.sel(time=slice(pd.to_datetime(baseline_start), pd.to_datetime(baseline_end))).mean("time")
    anomaly = ds - baseline
    anomaly.attrs["baseline_start"] = str(pd.to_datetime(baseline_start).date())
    anomaly.attrs["baseline_end"] = str(pd.to_datetime(baseline_end).date())
    return anomaly


def basin_mean_timeseries(ds: xr.Dataset, variables: tuple[str, ...] | None = None) -> pd.DataFrame:
    vars_to_use = list(variables) if variables is not None else list(ds.data_vars)
    data = {"time": pd.to_datetime(ds.time.values)}
    if "source_month" in ds.coords:
        data["source_month"] = pd.to_datetime(ds["source_month"].values)
    for var in vars_to_use:
        data[f"{var}_mean"] = ds[var].mean(dim=("lat", "lon")).values
    return pd.DataFrame(data)


def summarize_w3ra_alignment(
    aligned_ds: xr.Dataset,
    anomaly_ds: xr.Dataset | None = None,
    variables: tuple[str, ...] | None = None,
) -> dict:
    vars_to_use = list(variables) if variables is not None else list(aligned_ds.data_vars)
    summary = {
        "n_times": int(aligned_ds.sizes["time"]),
        "date_start": str(pd.to_datetime(aligned_ds.time.values[0]).date()),
        "date_end": str(pd.to_datetime(aligned_ds.time.values[-1]).date()),
        "source_month_start": str(pd.to_datetime(aligned_ds["source_month"].values[0]).date())
        if "source_month" in aligned_ds.coords
        else None,
        "source_month_end": str(pd.to_datetime(aligned_ds["source_month"].values[-1]).date())
        if "source_month" in aligned_ds.coords
        else None,
        "variables": vars_to_use,
    }
    for var in vars_to_use:
        summary[f"{var}_mean"] = float(aligned_ds[var].mean().item())
        summary[f"{var}_std"] = float(aligned_ds[var].std().item())
        if anomaly_ds is not None:
            summary[f"{var}_anom_mean"] = float(anomaly_ds[var].mean().item())
            summary[f"{var}_anom_std"] = float(anomaly_ds[var].std().item())
    return summary


def interpolate_w3ra_tile(
    aligned_ds: xr.Dataset,
    target_time,
    lat_values,
    lon_values,
    variables: tuple[str, ...] = ("S0", "Sg"),
    method: str = "linear",
) -> np.ndarray:
    tile = aligned_ds[list(variables)].sel(time=pd.to_datetime(target_time)).interp(
        lat=np.asarray(lat_values),
        lon=np.asarray(lon_values),
        method=method,
    )
    stacked = np.stack([tile[var].values for var in variables], axis=0).astype(np.float32)
    return np.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0)


def discover_grace_candidates(search_roots, patterns: tuple[str, ...] = GRACE_CANDIDATE_PATTERNS) -> list[str]:
    matches: list[str] = []
    for root in search_roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for pattern in patterns:
            for path in root_path.rglob(pattern):
                if path.is_file():
                    matches.append(str(path))
    return sorted(set(matches))


def align_grace_to_punjab_dates(
    grace_ds: xr.Dataset,
    punjab_dates,
    variable: str,
) -> xr.Dataset:
    target_dates = pd.DatetimeIndex(pd.to_datetime(punjab_dates))
    target_months = pd.DatetimeIndex(target_dates.to_period("M").to_timestamp(), name="month")
    unique_target_months = pd.DatetimeIndex(target_months.unique(), name="month")
    source_dates = pd.DatetimeIndex(pd.to_datetime(grace_ds.time.values))
    source_months = pd.DatetimeIndex(source_dates.to_period("M").to_timestamp(), name="month")

    # Some mascon products contain duplicate timestamps within the same month.
    # Collapse them to a unique monthly series before matching Punjab dates.
    monthly_source = grace_ds[[variable]].assign_coords(month=("time", source_months))
    monthly = (
        monthly_source.groupby("month").mean("time").reindex(month=unique_target_months).interpolate_na(dim="month", method="linear")
    )
    source_time_lookup = pd.Series(source_dates.values, index=source_months).groupby(level=0).first()
    source_time = source_time_lookup.reindex(unique_target_months)
    source_available = source_time.notna().to_numpy()
    source_time_fill = pd.Series(unique_target_months.values, index=unique_target_months)
    source_time = pd.DatetimeIndex(pd.to_datetime(source_time.fillna(source_time_fill).values))
    aligned = monthly.rename({"month": "aligned_month"})
    aligned = aligned.sel(aligned_month=xr.DataArray(target_months, dims="time"))
    aligned = aligned.drop_vars("aligned_month")
    aligned = aligned.assign_coords(time=target_dates)
    source_time_repeated = pd.Series(source_time.values, index=unique_target_months).reindex(target_months)
    source_available_repeated = pd.Series(source_available, index=unique_target_months).reindex(target_months).to_numpy()
    aligned = aligned.assign_coords(source_time=("time", pd.DatetimeIndex(pd.to_datetime(source_time_repeated.values)).values))
    aligned = aligned.assign_coords(source_available=("time", source_available_repeated))
    aligned = aligned.assign_coords(
        source_month=("time", pd.DatetimeIndex(pd.to_datetime(source_time_repeated.values)).to_period("M").to_timestamp().values)
    )
    return aligned


def compute_grace_anomalies(
    aligned_ds: xr.Dataset,
    variable: str,
    baseline_start=None,
    baseline_end=None,
) -> xr.Dataset:
    return compute_w3ra_anomalies(
        aligned_ds,
        baseline_start=baseline_start,
        baseline_end=baseline_end,
        variables=(variable,),
    )


def summarize_grace_alignment(
    aligned_ds: xr.Dataset,
    variable: str,
    anomaly_ds: xr.Dataset | None = None,
) -> dict:
    summary = {
        "n_times": int(aligned_ds.sizes["time"]),
        "date_start": str(pd.to_datetime(aligned_ds.time.values[0]).date()),
        "date_end": str(pd.to_datetime(aligned_ds.time.values[-1]).date()),
        "source_month_start": str(pd.to_datetime(aligned_ds["source_month"].values[0]).date())
        if "source_month" in aligned_ds.coords
        else None,
        "source_month_end": str(pd.to_datetime(aligned_ds["source_month"].values[-1]).date())
        if "source_month" in aligned_ds.coords
        else None,
        "variable": variable,
        f"{variable}_mean": float(aligned_ds[variable].mean().item()),
        f"{variable}_std": float(aligned_ds[variable].std().item()),
    }
    if anomaly_ds is not None:
        summary[f"{variable}_anom_mean"] = float(anomaly_ds[variable].mean().item())
        summary[f"{variable}_anom_std"] = float(anomaly_ds[variable].std().item())
    return summary
