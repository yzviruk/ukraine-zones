"""
Stage 3b: drainage network derived from the DEM, as a uniform alternative to
OSM streams (whose coverage has large gaps in the south of the oblast).

Input:  data/processed/env/elevation.tif   (stage 3)
Output: data/processed/env/
    flow_acc_km2     upstream catchment area, km^2
    dist_dstream     m to the nearest DEM channel (catchment >= CHANNEL_KM2)
    hand_dstream     elevation minus elevation of that nearest channel cell
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from pysheds.grid import Grid
from scipy import ndimage

if not hasattr(np, "in1d"):  # pysheds 0.4 still calls np.in1d, removed in NumPy 2.4
    np.in1d = np.isin

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV = PROJECT_ROOT / "data" / "processed" / "env"
CHANNEL_KM2 = 0.5  # channel initiation; small enough to include balky (dry valleys)
NODATA = -9999.0


def save(name, array, profile, inside):
    out = np.where(inside, array, NODATA).astype("float32")
    with rasterio.open(ENV / f"{name}.tif", "w", **profile) as dst:
        dst.write(out, 1)
    print(f"  {name}.tif")


def main() -> None:
    src = ENV / "elevation.tif"
    with rasterio.open(src) as ds:
        profile = ds.profile
        elevation = ds.read(1)
        cell = ds.res[0]
    inside = elevation != NODATA

    print("Conditioning DEM ...")
    grid = Grid.from_raster(str(src))
    dem = grid.read_raster(str(src))
    dem = grid.fill_pits(dem)
    dem = grid.fill_depressions(dem)
    dem = grid.resolve_flats(dem)

    print("Flow direction / accumulation ...")
    fdir = grid.flowdir(dem)
    acc = np.asarray(grid.accumulation(fdir), dtype="float64")
    acc_km2 = acc * cell * cell / 1e6
    save("flow_acc_km2", acc_km2, profile, inside)

    channels = (acc_km2 >= CHANNEL_KM2) & inside
    dist, (iy, ix) = ndimage.distance_transform_edt(~channels, sampling=cell, return_indices=True)
    save("dist_dstream", dist, profile, inside)
    save("hand_dstream", elevation - elevation[iy, ix], profile, inside)
    print(f"channel cells: {channels.sum():,} ({100 * channels.sum() / inside.sum():.1f}%)")


if __name__ == "__main__":
    main()
