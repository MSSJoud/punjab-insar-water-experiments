from __future__ import annotations

from pathlib import Path

import numpy as np

from punjab_inversion.punjab_prediction_viewer import (
    DEFAULT_SUPPORT_CONFIG,
    build_baseline_prediction_archive,
    build_full_scene_tiles,
    export_prediction_archive_to_netcdf,
    load_punjab_meta,
)


ROOT = Path("/home/ubuntu/work/punjab")
OUT_DIR = ROOT / "outputs" / "punjab_prior"
DATA_ROOT = Path("/mnt/data/aoi_punjab")
CHECKPOINT_PATH = OUT_DIR / "punjab_phase1_pilot_best_grouped_support_expanded.pt"
ARCHIVE_PATH = OUT_DIR / "punjab_phase1_baseline_prediction_full_scene_no_mask_latest_only.h5"
S0_OUTPUT_PATH = OUT_DIR / "punjab_phase1_baseline_prediction_full_scene_no_mask_latest_only_S0.nc"
SG_OUTPUT_PATH = OUT_DIR / "punjab_phase1_baseline_prediction_full_scene_no_mask_latest_only_Sg.nc"

SUPPORT_CONFIG = {
    **DEFAULT_SUPPORT_CONFIG,
    "tile_size": (64, 64),
    "tile_stride": (64, 64),
}


def main() -> None:
    meta = load_punjab_meta(DATA_ROOT)
    h, w = meta["disp_shape"][1:]
    tiles = build_full_scene_tiles((h, w), support_config=SUPPORT_CONFIG)
    last_end_index = np.array([len(meta["dates"]) - 1], dtype=np.int32)
    no_mask = np.ones((h, w), dtype=bool)

    summary = build_baseline_prediction_archive(
        ARCHIVE_PATH,
        CHECKPOINT_PATH,
        data_root=DATA_ROOT,
        support_config=SUPPORT_CONFIG,
        end_indices_override=last_end_index,
        tiles_override=tiles,
        keep_all_tile_values=True,
        support_mask_override=no_mask,
        progress_every=20,
    )

    export_prediction_archive_to_netcdf(
        ARCHIVE_PATH,
        s0_output_path=S0_OUTPUT_PATH,
        sg_output_path=SG_OUTPUT_PATH,
    )

    print("archive:", ARCHIVE_PATH)
    print("summary:", summary.summary_path)
    print("s0_nc:", S0_OUTPUT_PATH)
    print("sg_nc:", SG_OUTPUT_PATH)
    print("n_tiles:", len(tiles))


if __name__ == "__main__":
    main()
