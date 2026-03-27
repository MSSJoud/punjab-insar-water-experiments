from __future__ import annotations

from pathlib import Path

from punjab_inversion.punjab_prediction_viewer import (
    DEFAULT_SUPPORT_CONFIG,
    build_baseline_prediction_archive,
    build_valid_tiles,
    export_prediction_archive_to_netcdf,
    load_punjab_meta,
    build_support_mask,
    make_all_end_indices,
)


ROOT = Path("/home/ubuntu/work/punjab")
OUT_DIR = ROOT / "outputs" / "punjab_prior"
DATA_ROOT = Path("/mnt/data/aoi_punjab")
CHECKPOINT_PATH = OUT_DIR / "punjab_phase1_pilot_best_grouped_support_expanded.pt"
ARCHIVE_PATH = OUT_DIR / "punjab_phase1_baseline_prediction_archive_expanded_tiles_all_values.h5"
S0_OUTPUT_PATH = OUT_DIR / "punjab_phase1_baseline_prediction_expanded_tiles_all_values_S0.nc"
SG_OUTPUT_PATH = OUT_DIR / "punjab_phase1_baseline_prediction_expanded_tiles_all_values_Sg.nc"

SUPPORT_CONFIG = {
    **DEFAULT_SUPPORT_CONFIG,
    "coherence_threshold": 0.20,
    "time_valid_fraction_threshold": 0.05,
    "tile_size": (64, 64),
    "tile_stride": (64, 64),
    "tile_min_valid_fraction": 0.20,
}


def select_evenly_spaced_tile_indices(n_tiles: int, tile_count: int) -> list[int]:
    if n_tiles <= 0 or tile_count <= 0:
        return []
    tile_count = min(int(tile_count), int(n_tiles))
    if tile_count == 1:
        return [0]
    import numpy as np

    indices = np.linspace(0, n_tiles - 1, tile_count)
    return sorted({int(round(v)) for v in indices})


def main() -> None:
    meta = load_punjab_meta(DATA_ROOT)
    support_mask = build_support_mask(meta, support_config=SUPPORT_CONFIG)
    all_tiles = build_valid_tiles(support_mask, support_config=SUPPORT_CONFIG)
    chosen_tile_indices = select_evenly_spaced_tile_indices(len(all_tiles), 6)
    chosen_tiles = [all_tiles[i] for i in chosen_tile_indices]
    end_indices = make_all_end_indices(meta["dates"])

    summary = build_baseline_prediction_archive(
        ARCHIVE_PATH,
        CHECKPOINT_PATH,
        data_root=DATA_ROOT,
        support_config=SUPPORT_CONFIG,
        end_indices_override=end_indices,
        tiles_override=chosen_tiles,
        keep_all_tile_values=True,
        progress_every=200,
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
    print("tile_indices:", chosen_tile_indices)
    print("n_tiles:", len(chosen_tiles))


if __name__ == "__main__":
    main()
