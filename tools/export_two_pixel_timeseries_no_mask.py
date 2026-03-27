from __future__ import annotations

from pathlib import Path

import json
import numpy as np
import pandas as pd
import h5py

from punjab_inversion.punjab_prediction_viewer import (
    DEFAULT_SUPPORT_CONFIG,
    PredictionArchive,
    build_baseline_prediction_archive,
    load_punjab_meta,
    make_all_end_indices,
)


ROOT = Path("/home/ubuntu/work/punjab")
OUT_DIR = ROOT / "outputs" / "punjab_prior"
DATA_ROOT = Path("/mnt/data/aoi_punjab")
CHECKPOINT_PATH = OUT_DIR / "punjab_phase1_pilot_best_grouped_support_expanded.pt"
ARCHIVE_PATH = OUT_DIR / "punjab_phase1_two_pixel_timeseries_no_mask.h5"
CSV_PATH = OUT_DIR / "punjab_phase1_two_pixel_timeseries_no_mask.csv"
JSON_PATH = OUT_DIR / "punjab_phase1_two_pixel_timeseries_no_mask_summary.json"

SUPPORT_CONFIG = {
    **DEFAULT_SUPPORT_CONFIG,
    "tile_size": (64, 64),
    "tile_stride": (64, 64),
}

RIVER_LAT = 31.229126
RIVER_LON = 76.564970
RIVER_SEARCH_RADIUS_DEG = 0.05
RIVER_MIN_COMBINED_STD = 0.02


def tile_origin(index: int, tile_size: int, upper: int) -> int:
    origin = (index // tile_size) * tile_size
    return min(max(origin, 0), upper - tile_size)


def main() -> None:
    meta = load_punjab_meta(DATA_ROOT)
    h, w = meta["disp_shape"][1:]
    tile_h, tile_w = SUPPORT_CONFIG["tile_size"]
    last_row = h - tile_h
    last_col = w - tile_w
    dates = pd.DatetimeIndex(meta["dates"])
    no_mask = np.ones((h, w), dtype=bool)

    with h5py.File(meta["coh_path"], "r") as f:
        lat = f["lat"][:]
        lon = f["lon"][:]
        coh = f["z"][:]

    finite = np.isfinite(coh)
    idx = np.nanargmax(np.where(finite, coh, -np.inf))
    high_r, high_c = np.unravel_index(idx, coh.shape)
    river_r = int(np.argmin(np.abs(lat - RIVER_LAT)))
    river_c = int(np.argmin(np.abs(lon - RIVER_LON)))

    targets = {
        "high_coherence": {"row": int(high_r), "col": int(high_c)},
        "river_reference": {"row": int(river_r), "col": int(river_c)},
    }

    seen = set()
    tiles = []
    for info in targets.values():
        r0 = tile_origin(info["row"], tile_h, h)
        c0 = tile_origin(info["col"], tile_w, w)
        key = (r0, c0)
        if key in seen:
            continue
        seen.add(key)
        tiles.append({"row": r0, "col": c0, "valid_fraction": 1.0})

    summary = build_baseline_prediction_archive(
        ARCHIVE_PATH,
        CHECKPOINT_PATH,
        data_root=DATA_ROOT,
        support_config=SUPPORT_CONFIG,
        end_indices_override=make_all_end_indices(dates),
        tiles_override=tiles,
        keep_all_tile_values=True,
        support_mask_override=no_mask,
        progress_every=50,
    )

    archive = PredictionArchive(ARCHIVE_PATH)
    try:
        rows = []
        summary_payload = {"archive_summary": summary.__dict__, "pixels": {}}

        # Replace the exact river-reference pixel if it is effectively flat by selecting
        # the nearest informative pixel within the exported river tile neighborhood.
        river_info = targets["river_reference"]
        LAT = lat[:, None]
        LON = lon[None, :]
        dist = np.sqrt((LAT - RIVER_LAT) ** 2 + (LON - RIVER_LON) ** 2)
        candidate_rows, candidate_cols = np.where((archive.index_map >= 0) & (dist <= RIVER_SEARCH_RADIUS_DEG))
        informative_candidates = []
        for r, c in zip(candidate_rows, candidate_cols, strict=False):
            series = archive.pixel_series(int(r), int(c))
            s0_std = float(np.nanstd(series["s0"]))
            sg_std = float(np.nanstd(series["sg"]))
            combined_std = s0_std + sg_std
            if combined_std >= RIVER_MIN_COMBINED_STD:
                informative_candidates.append((float(dist[r, c]), -combined_std, int(r), int(c), s0_std, sg_std))
        if informative_candidates:
            informative_candidates.sort()
            _, _, best_r, best_c, best_s0_std, best_sg_std = informative_candidates[0]
            targets["river_reference"] = {"row": best_r, "col": best_c}
            summary_payload["river_reference_selection"] = {
                "requested_lat": RIVER_LAT,
                "requested_lon": RIVER_LON,
                "original_nearest_row": river_info["row"],
                "original_nearest_col": river_info["col"],
                "selected_row": best_r,
                "selected_col": best_c,
                "selected_lat": float(lat[best_r]),
                "selected_lon": float(lon[best_c]),
                "selected_distance_deg": float(dist[best_r, best_c]),
                "selected_s0_std": best_s0_std,
                "selected_sg_std": best_sg_std,
            }

        for label, info in targets.items():
            series = archive.pixel_series(info["row"], info["col"])
            summary_payload["pixels"][label] = {
                "row": info["row"],
                "col": info["col"],
                "lat": float(lat[info["row"]]),
                "lon": float(lon[info["col"]]),
                "coherence": None if not np.isfinite(coh[info["row"], info["col"]]) else float(coh[info["row"], info["col"]]),
            }
            for dt, s0, sg in zip(series["dates"], series["s0"], series["sg"], strict=False):
                rows.append(
                    {
                        "pixel_label": label,
                        "date": str(pd.Timestamp(dt).date()),
                        "row": info["row"],
                        "col": info["col"],
                        "lat": float(lat[info["row"]]),
                        "lon": float(lon[info["col"]]),
                        "S0_pred": float(s0),
                        "Sg_pred": float(sg),
                    }
                )
        pd.DataFrame(rows).to_csv(CSV_PATH, index=False)
        JSON_PATH.write_text(json.dumps(summary_payload, indent=2))
    finally:
        archive.close()

    print("archive:", ARCHIVE_PATH)
    print("csv:", CSV_PATH)
    print("json:", JSON_PATH)


if __name__ == "__main__":
    main()
