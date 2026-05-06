"""
Stage 2 (raster): compute "far zones" via distance transform.

Why raster instead of vector buffer:
  - Vector buffer on 200k+ polygons O(N^2) -> tens of GB RAM, hours.
  - Raster distance transform O(pixels) -> ~500 MB RAM, ~1 minute.

Resolution 200 m is invisible at country-scale zoom but cuts the
raster down to ~7500x4000 pixels = manageable.

Inputs:  data/processed/stage1.gpkg
Outputs: data/processed/stage2.gpkg (10 layers in EPSG:6381)
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize, shapes
from rasterio.transform import from_bounds
from scipy.ndimage import distance_transform_edt
from shapely.geometry import shape
from shapely.ops import unary_union

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
SRC = PROJECT_ROOT / "data" / "processed" / "stage1.gpkg"
OUT = PROJECT_ROOT / "data" / "processed" / "stage2.gpkg"

DISTANCES_KM = list(range(1, 11))
TARGET_CRS = 6381
PIXEL_SIZE_M = 200  # 200 m grid


def main() -> None:
    if not SRC.exists():
        raise SystemExit("Missing %s. Run 01_extract_settlements.py first." % SRC)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    print("Loading stage1 ...")
    settlement_areas = gpd.read_file(SRC, layer="settlement_areas")
    ukraine = gpd.read_file(SRC, layer="ukraine")

    if settlement_areas.crs.to_epsg() != TARGET_CRS:
        raise SystemExit("settlement_areas wrong CRS")
    if ukraine.crs.to_epsg() != TARGET_CRS:
        raise SystemExit("ukraine wrong CRS")

    xmin, ymin, xmax, ymax = ukraine.total_bounds
    xmin = (xmin // PIXEL_SIZE_M) * PIXEL_SIZE_M
    ymin = (ymin // PIXEL_SIZE_M) * PIXEL_SIZE_M
    xmax = ((xmax // PIXEL_SIZE_M) + 1) * PIXEL_SIZE_M
    ymax = ((ymax // PIXEL_SIZE_M) + 1) * PIXEL_SIZE_M

    width = int((xmax - xmin) / PIXEL_SIZE_M)
    height = int((ymax - ymin) / PIXEL_SIZE_M)
    print("Raster grid: %d x %d pixels at %d m" % (width, height, PIXEL_SIZE_M))

    transform = from_bounds(xmin, ymin, xmax, ymax, width, height)

    print("Rasterising Ukraine boundary ...")
    ukraine_mask = rasterize(
        [(g, 1) for g in ukraine.geometry if g is not None],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
    ).astype(bool)
    print("  -> ukraine pixels: %d" % ukraine_mask.sum())

    print("Rasterising settlement_areas ...")
    settlement_mask = rasterize(
        [(g, 1) for g in settlement_areas.geometry if g is not None],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
    ).astype(bool)
    print("  -> settlement pixels: %d" % settlement_mask.sum())

    # Free vector data we don't need anymore.
    del settlement_areas, ukraine

    print("Computing Euclidean distance transform ...")
    # distance_transform_edt(input) computes distance to NEAREST ZERO pixel.
    # We want distance to nearest SETTLEMENT pixel, so input = ~settlement_mask
    # (zero where settlement is, non-zero elsewhere).
    distance_pixels = distance_transform_edt(~settlement_mask)
    print("  -> distance map computed (%s)" % str(distance_pixels.dtype))

    for km in DISTANCES_KM:
        threshold_pixels = km * 1000 / PIXEL_SIZE_M
        far_mask = (distance_pixels > threshold_pixels) & ukraine_mask
        print("  zone %d km -> %d pixels" % (km, far_mask.sum()))

        # Vectorise mask back into polygons.
        polys = []
        for geom_dict, val in shapes(
            far_mask.astype("uint8"),
            mask=far_mask,
            transform=transform,
        ):
            if val == 1:
                polys.append(shape(geom_dict))

        if polys:
            far_geom = unary_union(polys)
        else:
            from shapely.geometry import MultiPolygon
            far_geom = MultiPolygon()

        gdf = gpd.GeoDataFrame(
            {"distance_km": [km]},
            geometry=[far_geom],
            crs="EPSG:%d" % TARGET_CRS,
        )
        gdf.to_file(OUT, layer="zone_%dkm" % km, driver="GPKG")

    print("Stage 2 done.")


if __name__ == "__main__":
    main()