"""
Stage 3: environmental rasters for Vinnytsia oblast (30 m, EPSG:6381).

Inputs:
    Copernicus GLO-30 DEM tiles (downloaded to data/raw/dem/; free, no auth)
    data/raw/ua-border.geojson                      (oblast polygon)
    docs/data/rivers.geojson, streams.geojson       (OSM waterways)
    docs/data/{chernozems,podzolized_chernozems,grey_forest}.geojson  (HWSD soils)
    data/processed/stage1.gpkg                      (modern built-up areas)
Output: data/processed/env/<name>.tif (gitignored), one band each:
    elevation      m (Copernicus DSM: includes forest canopy/buildings)
    slope          degrees
    aspect         degrees clockwise from north, downslope direction; -1 = flat
    tpi300         elevation minus mean within ~300 m  (+ ridge/plateau edge, - valley)
    tpi1000        same, ~1 km
    rel_valley     elevation minus minimum within ~500 m (height above local valley floor)
    dist_water     m to the nearest OSM river or stream
    dist_river     m to the nearest OSM river (waterway=river)
    hand           elevation minus elevation of the nearest waterway cell
    soil           0 other, 1 chernozems, 2 podzolized chernozems, 3 grey forest
    dist_builtup   m to modern built-up area (survey-bias covariate, not environment)
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask, rasterize
from rasterio.merge import merge
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW = PROJECT_ROOT / "data" / "raw"
DEM_DIR = RAW / "dem"
DOCS_DATA = PROJECT_ROOT / "docs" / "data"
STAGE1 = PROJECT_ROOT / "data" / "processed" / "stage1.gpkg"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "env"

OBLAST = "Вінницька область"
CRS = "EPSG:6381"
CELL = 30.0
BUFFER_M = 3000  # keeps edge statistics valid near the oblast border
NODATA = -9999.0
DEM_URL = "https://copernicus-dem-30m.s3.amazonaws.com/{0}/{0}.tif"
SOIL_LAYERS = {"chernozems": 1, "podzolized_chernozems": 2, "grey_forest": 3}


def oblast_polygon():
    border = gpd.read_file(RAW / "ua-border.geojson")
    border = border[border.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    return border[border["name"] == OBLAST].to_crs(CRS).geometry.iloc[0]


def dem_tiles(bounds_4326) -> list[Path]:
    west, south, east, north = bounds_4326
    DEM_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for lat in range(int(np.floor(south)), int(np.floor(north)) + 1):
        for lon in range(int(np.floor(west)), int(np.floor(east)) + 1):
            name = f"Copernicus_DSM_COG_10_N{lat:02d}_00_E{lon:03d}_00_DEM"
            path = DEM_DIR / f"{name}.tif"
            if not path.exists():
                print(f"  downloading {name} ...")
                urllib.request.urlretrieve(DEM_URL.format(name), path)
            paths.append(path)
    return paths


def target_grid(area):
    west, south, east, north = area.bounds
    west, south = np.floor(west / CELL) * CELL, np.floor(south / CELL) * CELL
    east, north = np.ceil(east / CELL) * CELL, np.ceil(north / CELL) * CELL
    shape = (int((north - south) / CELL), int((east - west) / CELL))
    return from_origin(west, north, CELL, CELL), shape


def load_elevation(tiles, transform, shape) -> np.ndarray:
    sources = [rasterio.open(p) for p in tiles]
    mosaic, src_transform = merge(sources)
    for src in sources:
        src.close()
    elevation = np.full(shape, np.nan, dtype="float32")
    reproject(
        mosaic[0].astype("float32"),
        elevation,
        src_transform=src_transform,
        src_crs="EPSG:4326",
        dst_transform=transform,
        dst_crs=CRS,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    return elevation


def slope_aspect(elevation):
    d_row, d_col = np.gradient(elevation, CELL)
    gx, gy = d_col, -d_row  # east and north components (rows grow southwards)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    aspect = (np.degrees(np.arctan2(-gx, -gy)) + 360) % 360
    aspect[slope < 0.5] = -1
    return slope.astype("float32"), aspect.astype("float32")


def disk(radius_m: float) -> np.ndarray:
    r = int(round(radius_m / CELL))
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    return x * x + y * y <= r * r


def tpi(elevation, radius_m: float):
    size = 2 * int(round(radius_m / CELL)) + 1
    return (elevation - ndimage.uniform_filter(elevation, size=size)).astype("float32")


def lines_mask(gdf, transform, shape) -> np.ndarray:
    return rasterize(
        ((g, 1) for g in gdf.geometry),
        out_shape=shape,
        transform=transform,
        all_touched=True,
        dtype="uint8",
    ).astype(bool)


def distance(mask, return_indices=False):
    return ndimage.distance_transform_edt(~mask, sampling=CELL, return_indices=return_indices)


def save(name, array, transform, inside, dtype="float32", nodata=NODATA):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = np.where(inside, np.nan_to_num(array, nan=nodata), nodata).astype(dtype)
    profile = dict(
        driver="GTiff",
        height=out.shape[0],
        width=out.shape[1],
        count=1,
        dtype=dtype,
        crs=CRS,
        transform=transform,
        nodata=nodata,
        compress="deflate",
        predictor=2 if dtype != "float32" else 3,
        tiled=True,
    )
    with rasterio.open(OUT_DIR / f"{name}.tif", "w", **profile) as dst:
        dst.write(out, 1)
    print(f"  {name}.tif")


def main() -> None:
    oblast = oblast_polygon()
    area = oblast.buffer(BUFFER_M)
    transform, shape = target_grid(area)
    inside = geometry_mask([area], out_shape=shape, transform=transform, invert=True)
    print(f"grid {shape} cells of {CELL:.0f} m")

    print("DEM ...")
    bounds_4326 = gpd.GeoSeries([area], crs=CRS).to_crs(4326).total_bounds
    elevation = load_elevation(dem_tiles(bounds_4326), transform, shape)
    filled = np.where(np.isnan(elevation), np.nanmean(elevation), elevation)
    slope, aspect = slope_aspect(filled)
    save("elevation", elevation, transform, inside)
    save("slope", slope, transform, inside)
    save("aspect", aspect, transform, inside)
    save("tpi300", tpi(filled, 300), transform, inside)
    save("tpi1000", tpi(filled, 1000), transform, inside)
    valley_floor = ndimage.minimum_filter(filled, footprint=disk(500))
    save("rel_valley", filled - valley_floor, transform, inside)

    print("Waterways ...")
    bbox = tuple(bounds_4326)
    ways = pd.concat(
        [gpd.read_file(DOCS_DATA / f, bbox=bbox) for f in ("rivers.geojson", "streams.geojson")],
        ignore_index=True,
    ).to_crs(CRS)
    any_water = lines_mask(ways, transform, shape)
    rivers = lines_mask(ways[ways["waterway"] == "river"], transform, shape)
    dist_water, (iy, ix) = distance(any_water, return_indices=True)
    save("dist_water", dist_water, transform, inside)
    save("dist_river", distance(rivers), transform, inside)
    save("hand", filled - filled[iy, ix], transform, inside)

    print("Soils ...")
    soil = np.zeros(shape, dtype="uint8")
    for name, code in SOIL_LAYERS.items():
        gdf = gpd.read_file(DOCS_DATA / f"{name}.geojson").to_crs(CRS)
        layer = rasterize(
            ((g, code) for g in gdf.geometry), out_shape=shape, transform=transform, dtype="uint8"
        )
        soil = np.maximum(soil, layer)
    save("soil", soil, transform, inside, dtype="uint8", nodata=255)

    print("Built-up ...")
    built = gpd.read_file(STAGE1, layer="settlement_areas").explode(index_parts=False)
    built = built[built.intersects(area)]
    save("dist_builtup", distance(lines_mask(built, transform, shape)), transform, inside)


if __name__ == "__main__":
    main()
