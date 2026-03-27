from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path("/home/ubuntu/work/punjab")
NOTEBOOK_PATH = ROOT / "punjab_source_comparison_panels.ipynb"


def main() -> None:
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            """# Punjab Source Comparison Panels

This notebook generates paper-facing comparison figures for Punjab using:

- InSAR velocity from the GMTSAR `vel_ll.h5` product
- GRACE aligned TWS fields
- W3RA aligned `S0` and `Sg` anomaly fields
- baseline inversion `S0` and `Sg` outputs re-exported over the full tile scene without applying the coherence-derived support mask

The unit labels and color-stretch settings are editable in the configuration cell below.
The inversion maps are displayed without the coherence-derived support mask. They are cropped only to the finite prediction footprint present in the exported inversion files.
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """from pathlib import Path
import json
import shutil

from IPython.display import Image, Markdown, display
from IPython import get_ipython
import h5py
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from punjab_inversion.comparison_figures import (
    make_punjab_comparison_individual_panels,
    make_punjab_comparison_maps,
    make_punjab_comparison_timeseries,
)

ROOT = Path('/home/ubuntu/work/punjab')
DATA_DIR = Path('/mnt/data/aoi_punjab')
OUT_DIR = ROOT / 'outputs' / 'punjab_prior'
FIG_DIR = OUT_DIR / 'figures'
PAPER_FIG_DIR = ROOT / 'paper_figures'

VEL_PATH = DATA_DIR / 'vel_ll.h5'
GRACE_ALIGNED_PATH = OUT_DIR / 'punjab_phase2_grace_aligned.nc'
W3RA_ALIGNED_ANOM_PATH = OUT_DIR / 'punjab_phase2_w3ra_aligned_anomalies.nc'
MAP_S0_PRED_PATH = OUT_DIR / 'punjab_phase1_baseline_prediction_full_scene_no_mask_latest_only_S0.nc'
MAP_SG_PRED_PATH = OUT_DIR / 'punjab_phase1_baseline_prediction_full_scene_no_mask_latest_only_Sg.nc'
TS_S0_PRED_PATH = OUT_DIR / 'punjab_phase1_baseline_prediction_all_tiles_all_values_S0.nc'
TS_SG_PRED_PATH = OUT_DIR / 'punjab_phase1_baseline_prediction_all_tiles_all_values_Sg.nc'
FULL_TS_ARCHIVE_PATH = OUT_DIR / 'punjab_phase1_baseline_prediction_full_scene_no_mask_all_values.h5'
FALLBACK_TS_ARCHIVE_PATH = OUT_DIR / 'punjab_phase1_baseline_prediction_archive_all_tiles_all_values.h5'
SEASONAL_S0_PATH = OUT_DIR / 'punjab_phase1_baseline_prediction_full_scene_no_mask_seasonal_dates_S0.nc'
SEASONAL_SG_PATH = OUT_DIR / 'punjab_phase1_baseline_prediction_full_scene_no_mask_seasonal_dates_Sg.nc'
TWO_PIXEL_TS_CSV = OUT_DIR / 'punjab_phase1_two_pixel_timeseries_no_mask.csv'
TWO_PIXEL_TS_JSON = OUT_DIR / 'punjab_phase1_two_pixel_timeseries_no_mask_summary.json'

MAP_FIG_PATH = FIG_DIR / 'punjab_source_comparison_maps.png'
TS_FIG_PATH = FIG_DIR / 'punjab_source_comparison_timeseries.png'
PANEL_DIR = FIG_DIR / 'punjab_source_comparison_panels'
MAP_SUMMARY_PATH = OUT_DIR / 'punjab_source_comparison_maps_summary.json'
TS_SUMMARY_PATH = OUT_DIR / 'punjab_source_comparison_timeseries_summary.json'
PANEL_SUMMARY_PATH = OUT_DIR / 'punjab_source_comparison_panels_summary.json'
SEASONAL_PANEL_FIG_PATH = FIG_DIR / 'punjab_paper_figure8_baseline_export_panel.png'
PAPER_FIG09_PATH = PAPER_FIG_DIR / 'Fig09_punjab_baseline_export_panel.png'

# Edit these labels if you want different wording in the paper figures.
VELOCITY_UNIT_LABEL = 'InSAR linear rate [mm yr$^{-1}$]'
GRACE_UNIT_LABEL = 'GRACE TWS anomaly [cm LWE]'
W3RA_UNIT_LABEL = 'W3RA storage anomaly [m]'
INVERSION_UNIT_LABEL = 'Inversion latent output [model units]'

# Edit this if you want a wider or narrower velocity stretch.
VELOCITY_PERCENTILE_HIGH = 95.0
INVERSION_CROP_MARGIN_PIXELS = 64

FIG_DIR.mkdir(parents=True, exist_ok=True)
PANEL_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Unit Provenance

- **InSAR velocity**: displayed from `vel_ll.h5`, treated as the GMTSAR linear-rate product and labeled in `mm yr$^{-1}$`.
- **GRACE**: `lwe_thickness` comes from the aligned GRACE file and retains `cm` units, interpreted here as `cm LWE`.
- **W3RA**: the Punjab `.mat`-derived storage fields are now exported with `units = m`, so the default label here is meters of storage thickness.
- **Inversion**: the exported `S0` and `Sg` maps are latent model outputs and are labeled as `model units`.

If you want different wording in the paper, update the labels in the configuration cell and rerun the figure cells.
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """map_summary = make_punjab_comparison_maps(
    vel_path=VEL_PATH,
    grace_aligned_path=GRACE_ALIGNED_PATH,
    w3ra_aligned_anom_path=W3RA_ALIGNED_ANOM_PATH,
    s0_pred_path=MAP_S0_PRED_PATH,
    sg_pred_path=MAP_SG_PRED_PATH,
    output_path=MAP_FIG_PATH,
    velocity_unit_label=VELOCITY_UNIT_LABEL,
    grace_unit_label=GRACE_UNIT_LABEL,
    w3ra_unit_label=W3RA_UNIT_LABEL,
    inversion_unit_label=INVERSION_UNIT_LABEL,
    velocity_percentile_high=VELOCITY_PERCENTILE_HIGH,
    inversion_crop_margin_pixels=INVERSION_CROP_MARGIN_PIXELS,
)

MAP_SUMMARY_PATH.write_text(json.dumps(map_summary, indent=2))
shutil.copy2(MAP_FIG_PATH, PAPER_FIG_DIR / 'Fig10_punjab_source_comparison_maps.png')

display(Image(filename=str(MAP_FIG_PATH)))
display(
    Markdown(
        'Figure draft: Punjab source comparison maps. '
        'Panel A shows the InSAR velocity with a readable robust symmetric stretch; '
        'Panel B shows the latest aligned GRACE TWS anomaly; '
        'Panels C-D show the latest aligned W3RA `S_0` and `S_g` anomalies; '
        'Panels E-F show the latest available baseline inversion `S_0` and `S_g` outputs using the full-scene no-mask latest-date re-export, cropped only to the finite prediction footprint.'
    )
)

map_summary
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Individual Panels

The next cell writes each panel as its own PNG and shows them one by one in the notebook so we can inspect and adjust them independently.
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """panel_summary = make_punjab_comparison_individual_panels(
    vel_path=VEL_PATH,
    grace_aligned_path=GRACE_ALIGNED_PATH,
    w3ra_aligned_anom_path=W3RA_ALIGNED_ANOM_PATH,
    s0_pred_path=MAP_S0_PRED_PATH,
    sg_pred_path=MAP_SG_PRED_PATH,
    output_dir=PANEL_DIR,
    velocity_unit_label=VELOCITY_UNIT_LABEL,
    grace_unit_label=GRACE_UNIT_LABEL,
    w3ra_unit_label=W3RA_UNIT_LABEL,
    inversion_unit_label=INVERSION_UNIT_LABEL,
    velocity_percentile_high=VELOCITY_PERCENTILE_HIGH,
    inversion_crop_margin_pixels=INVERSION_CROP_MARGIN_PIXELS,
)

PANEL_SUMMARY_PATH.write_text(json.dumps(panel_summary, indent=2))

for item in panel_summary[1:]:
    src = Path(item['output_path'])
    shutil.copy2(src, PAPER_FIG_DIR / f\"{src.stem}.png\")
    display(Image(filename=str(src)))
    display(Markdown(f\"**{item['title']}**  \\nSaved to `{src}`\"))

panel_summary
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """ts_summary = make_punjab_comparison_timeseries(
    grace_aligned_path=GRACE_ALIGNED_PATH,
    w3ra_aligned_anom_path=W3RA_ALIGNED_ANOM_PATH,
    s0_pred_path=TS_S0_PRED_PATH,
    sg_pred_path=TS_SG_PRED_PATH,
    output_path=TS_FIG_PATH,
    grace_unit_label=GRACE_UNIT_LABEL,
    w3ra_unit_label=W3RA_UNIT_LABEL,
    inversion_unit_label=INVERSION_UNIT_LABEL,
)

TS_SUMMARY_PATH.write_text(json.dumps(ts_summary, indent=2))
shutil.copy2(TS_FIG_PATH, PAPER_FIG_DIR / 'Fig11_punjab_source_comparison_timeseries.png')

display(Image(filename=str(TS_FIG_PATH)))
display(
    Markdown(
        'Figure draft: Punjab comparison time series. '
        'The three panels show basin-mean GRACE TWS anomaly, basin-mean W3RA `S_0` and `S_g` anomalies, '
        'and support-area mean baseline inversion `S_0` and `S_g` outputs over their aligned time ranges.'
    )
)

ts_summary
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Seasonal Maps And Two Pixel Time Series

This section shows:

- `S0` and `Sg` over three seasonal dates from the full-scene no-mask export
- one pixel from the highest-coherence area in the static coherence raster
- one informative river-adjacent pixel nearest to `N31.229126°, E76.564970°`
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """seasonal_s0 = xr.open_dataset(SEASONAL_S0_PATH)
seasonal_sg = xr.open_dataset(SEASONAL_SG_PATH)
pixel_ts = pd.read_csv(TWO_PIXEL_TS_CSV)
pixel_meta = json.loads(TWO_PIXEL_TS_JSON.read_text())

dates = pd.to_datetime(seasonal_s0['time'].values)
lat = seasonal_s0['lat'].values
lon = seasonal_s0['lon'].values
extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]

s0_stack = seasonal_s0['S0_pred'].values
sg_stack = seasonal_sg['Sg_pred'].values
s0_lim = np.nanpercentile(s0_stack[np.isfinite(s0_stack)], [2, 98])
sg_lim = np.nanpercentile(sg_stack[np.isfinite(sg_stack)], [2, 98])

fig = plt.figure(figsize=(16, 14), constrained_layout=True)
grid = fig.add_gridspec(4, 3)
map_axes = [fig.add_subplot(grid[0, i]) for i in range(3)] + [fig.add_subplot(grid[1, i]) for i in range(3)]
ts1 = fig.add_subplot(grid[2, :])
ts2 = fig.add_subplot(grid[3, :])

last_s0_im = None
for i, ax in enumerate(map_axes[:3]):
    im = ax.imshow(s0_stack[i], origin='lower', extent=extent, cmap='coolwarm', vmin=float(s0_lim[0]), vmax=float(s0_lim[1]), aspect='auto')
    ax.set_title(f\"A{i+1}. $S_0$ on {dates[i].date()}\")
    ax.set_xlabel('Longitude [degrees_east]')
    ax.set_ylabel('Latitude [degrees_north]')
    last_s0_im = im

last_sg_im = None
for i, ax in enumerate(map_axes[3:]):
    im = ax.imshow(sg_stack[i], origin='lower', extent=extent, cmap='coolwarm', vmin=float(sg_lim[0]), vmax=float(sg_lim[1]), aspect='auto')
    ax.set_title(f\"B{i+1}. $S_g$ on {dates[i].date()}\")
    ax.set_xlabel('Longitude [degrees_east]')
    ax.set_ylabel('Latitude [degrees_north]')
    last_sg_im = im

fig.colorbar(last_s0_im, ax=map_axes[:3], shrink=0.90, location='right', pad=0.02, label='Predicted storage [m EWH]')
fig.colorbar(last_sg_im, ax=map_axes[3:], shrink=0.90, location='right', pad=0.02, label='Predicted storage [m EWH]')

color_map = {'high_coherence': '#1f77b4', 'river_reference': '#d62728'}
for label, meta in pixel_meta['pixels'].items():
    for ax in map_axes:
        ax.plot(meta['lon'], meta['lat'], marker='o', markersize=4, color=color_map[label])

for prefix, (label, ax) in zip(['C', 'D'], [('high_coherence', ts1), ('river_reference', ts2)], strict=False):
    sub = pixel_ts[pixel_ts['pixel_label'] == label].copy()
    sub['date'] = pd.to_datetime(sub['date'])
    meta = pixel_meta['pixels'][label]
    ax.plot(sub['date'], sub['S0_pred'], color='#1f77b4', label='$S_0$')
    ax.plot(sub['date'], sub['Sg_pred'], color='#d62728', label='$S_g$')
    ax.set_title(f\"{prefix}. {label.replace('_', ' ').title()}\\nlat={meta['lat']:.3f}, lon={meta['lon']:.3f}\")
    ax.set_xlabel('Time')
    ax.set_ylabel('Predicted storage [m EWH]')
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

fig.savefig(SEASONAL_PANEL_FIG_PATH, dpi=220, bbox_inches='tight')
shutil.copy2(SEASONAL_PANEL_FIG_PATH, PAPER_FIG09_PATH)

display(fig)
plt.close(fig)

display(
    Markdown(
        'Seasonal full-scene maps at three dates (2023-03-30, 2023-07-16, 2023-11-13) together with two pixel time series. '
        'The blue marker is the highest-coherence pixel in the static coherence raster; '
        'the red marker is the nearest river-adjacent pixel with non-trivial temporal variation to `N31.229126°, E76.564970°`. '
        f'Saved to `{SEASONAL_PANEL_FIG_PATH}` and copied to `{PAPER_FIG09_PATH}`.'
    )
)
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Interactive Velocity Map

This cell opens an interactive InSAR-velocity map. Click anywhere on the map to update:

- predicted `S_0` time series
- predicted `S_g` time series
- observed deformation at the clicked pixel

Notes:

- For reliable clicks in Jupyter, run `%matplotlib widget` first if your notebook supports it.
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
"""archive_path = FULL_TS_ARCHIVE_PATH if FULL_TS_ARCHIVE_PATH.exists() else FALLBACK_TS_ARCHIVE_PATH
if not archive_path.exists():
    raise FileNotFoundError(
        'Missing both interactive archives. Expected one of:\\n'
        f'  - {FULL_TS_ARCHIVE_PATH}\\n'
        f'  - {FALLBACK_TS_ARCHIVE_PATH}'
    )

try:
    ip = get_ipython()
    if ip is not None:
        ip.run_line_magic('matplotlib', 'widget')
except Exception:
    pass

with h5py.File(VEL_PATH, 'r') as f:
    vel_lat = np.asarray(f['lat'], dtype=np.float32)
    vel_lon = np.asarray(f['lon'], dtype=np.float32)
    vel_map = np.asarray(f['z'], dtype=np.float32)

disp_h5 = h5py.File(DATA_DIR / 'disp_all_ll.h5', 'r')
disp_ds = disp_h5['z']

archive = h5py.File(archive_path, 'r')
archive_dates = pd.to_datetime(archive['time_iso'][:].astype(str))
archive_index_map = archive['index_map'][:].astype(np.int32)
archive_end_indices = archive['end_indices'][:].astype(np.int32)
archive_lat = archive['lat'][:].astype(np.float32)
archive_lon = archive['lon'][:].astype(np.float32)
s0_store = archive['s0']
sg_store = archive['sg']
archive_available = archive_index_map >= 0
available_rows, available_cols = np.where(archive_available)

vel_vmin, vel_vmax = np.nanpercentile(
    np.abs(vel_map[np.isfinite(vel_map)]),
    [VELOCITY_PERCENTILE_HIGH],
)[0], np.nanpercentile(
    np.abs(vel_map[np.isfinite(vel_map)]),
    [VELOCITY_PERCENTILE_HIGH],
)[0]
vel_vmin = -float(vel_vmin)
vel_vmax = float(vel_vmax)

fig = plt.figure(figsize=(10.5, 6.4), constrained_layout=True)
grid = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.0, 1.0])
ax_map = fig.add_subplot(grid[:, 0])
ax_storage = fig.add_subplot(grid[0, 1])
ax_def = fig.add_subplot(grid[1, 1], sharex=ax_storage)

im = ax_map.imshow(
    np.ma.masked_invalid(vel_map),
    origin='lower',
    extent=[float(vel_lon.min()), float(vel_lon.max()), float(vel_lat.min()), float(vel_lat.max())],
    cmap='RdBu_r',
    vmin=vel_vmin,
    vmax=vel_vmax,
    aspect='auto',
)
ax_map.set_title('Interactive InSAR velocity map')
ax_map.set_xlabel('Longitude [degrees_east]')
ax_map.set_ylabel('Latitude [degrees_north]')
fig.colorbar(im, ax=ax_map, shrink=0.82, label=VELOCITY_UNIT_LABEL)

marker, = ax_map.plot([], [], marker='o', markersize=7, color='black', markerfacecolor='gold', linestyle='None')
s0_line, = ax_storage.plot([], [], color='#1f77b4', label='$S_0$')
sg_line, = ax_storage.plot([], [], color='#d62728', label='$S_g$')
obs_def_line, = ax_def.plot([], [], color='0.3', alpha=0.85, label='Observed deformation')

ax_storage.set_title('Predicted storage signals')
ax_storage.set_ylabel('Predicted storage [m EWH]')
ax_storage.grid(alpha=0.25)
ax_storage.legend(frameon=False, loc='upper left')

ax_def.set_title('Observed deformation')
ax_def.set_ylabel('Deformation [cm]')
ax_def.set_xlabel('Time')
ax_def.grid(alpha=0.25)
ax_def.legend(frameon=False, loc='upper left')

status_html = widgets.HTML(value='Click the map or enter coordinates to load time series.')
lat_box = widgets.FloatText(value=float(archive_lat[len(archive_lat) // 2]), description='Lat', layout=widgets.Layout(width='220px'))
lon_box = widgets.FloatText(value=float(archive_lon[len(archive_lon) // 2]), description='Lon', layout=widgets.Layout(width='220px'))
load_button = widgets.Button(description='Load nearest pixel', button_style='primary')

def update_click(row: int, col: int) -> None:
    idx = int(archive_index_map[row, col])
    requested_row, requested_col = row, col
    snapped = False
    if idx < 0:
        distances = (available_rows - row) ** 2 + (available_cols - col) ** 2
        nearest = int(np.argmin(distances))
        row = int(available_rows[nearest])
        col = int(available_cols[nearest])
        idx = int(archive_index_map[row, col])
        snapped = True

    s0_series = s0_store[:, idx].astype(np.float32)
    sg_series = sg_store[:, idx].astype(np.float32)
    observed = np.asarray(disp_ds[archive_end_indices, row, col], dtype=np.float32)

    marker.set_data([archive_lon[col]], [archive_lat[row]])
    s0_line.set_data(archive_dates, s0_series)
    sg_line.set_data(archive_dates, sg_series)
    obs_def_line.set_data(archive_dates, observed)

    ax_storage.relim()
    ax_storage.autoscale_view()
    ax_def.relim()
    ax_def.autoscale_view()

    ax_storage.set_title(
        f'Predicted storage signals\\nlat={archive_lat[row]:.3f}, lon={archive_lon[col]:.3f}'
    )
    ax_def.set_title(
        f'Observed deformation\\nrow={row}, col={col}'
    )
    if snapped:
        status_html.value = (
            f'Nearest archived pixel used: requested ({requested_row}, {requested_col}), '
            f'loaded ({row}, {col}), lat={archive_lat[row]:.4f}, lon={archive_lon[col]:.4f}'
        )
    else:
        status_html.value = (
            f'Loaded row={row}, col={col}, lat={archive_lat[row]:.4f}, lon={archive_lon[col]:.4f}'
        )
    lat_box.value = float(archive_lat[row])
    lon_box.value = float(archive_lon[col])
    fig.canvas.draw_idle()

def on_click(event) -> None:
    if event.inaxes != ax_map or event.xdata is None or event.ydata is None:
        return
    col = int(np.argmin(np.abs(archive_lon - event.xdata)))
    row = int(np.argmin(np.abs(archive_lat - event.ydata)))
    update_click(row, col)

fig.canvas.mpl_connect('button_press_event', on_click)

def on_load_button(_btn) -> None:
    col = int(np.argmin(np.abs(archive_lon - lon_box.value)))
    row = int(np.argmin(np.abs(archive_lat - lat_box.value)))
    update_click(row, col)

load_button.on_click(on_load_button)

center_row = int(len(archive_lat) // 2)
center_col = int(len(archive_lon) // 2)
update_click(center_row, center_col)

controls = widgets.HBox([lat_box, lon_box, load_button])
display(widgets.VBox([controls, fig.canvas, status_html]))

display(
    Markdown(
        f'Using interactive archive: `{archive_path.name}`. '
        + (
            'This is the full-scene multi-date archive.'
            if archive_path == FULL_TS_ARCHIVE_PATH
            else 'This is the current all-tiles multi-date fallback archive; clicks outside stored coverage snap to the nearest archived pixel.'
        )
        + ' If direct clicking is still flaky in your frontend, use the lat/lon boxes and `Load nearest pixel`.'
    )
)
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """print('Saved map figure to:', MAP_FIG_PATH)
print('Saved time-series figure to:', TS_FIG_PATH)
print('Saved seasonal export panel to:', SEASONAL_PANEL_FIG_PATH)
print('Saved individual panel figures under:', PANEL_DIR)
print('Copied map figure to:', PAPER_FIG_DIR / 'Fig10_punjab_source_comparison_maps.png')
print('Copied time-series figure to:', PAPER_FIG_DIR / 'Fig11_punjab_source_comparison_timeseries.png')
print('Copied Figure 9 export panel to:', PAPER_FIG09_PATH)
print('Saved map summary to:', MAP_SUMMARY_PATH)
print('Saved time-series summary to:', TS_SUMMARY_PATH)
print('Saved panel summary to:', PANEL_SUMMARY_PATH)
"""
        )
    )

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python (swin_env)",
            "language": "python",
            "name": "swin_env",
        },
        "language_info": {
            "name": "python",
            "version": "3.12",
        },
    }

    NOTEBOOK_PATH.write_text(nbf.writes(nb))


if __name__ == "__main__":
    main()
