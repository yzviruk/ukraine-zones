"""
Stage 5: download SoilGrids 2.0 WRB MostProbable for Ukraine and export
Chernozems and Phaeozems as separate GeoJSON layers.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import shapes
from shapely.geometry import MultiPolygon, shape

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "docs" / "data"

UA_BBOX = (22.0, 44.0, 41.0, 53.0)

WCS_URL = (
    "https://maps.isric.org/mapserv?map=/map/wrb.map"
    "&SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage"
    "&COVERAGEID=MostProbable&FORMAT=image/tiff"
    "&SUBSET=long({w},{e})&SUBSET=lat({s},{n})"
    "&OUTPUTCRS=http://www.opengis.net/def/crs/EPSG/0/4326"
    "&SUBSETTINGCRS=http://www.opengis.net/def/crs/EPSG/0/4326"
)
RAW_WRB = RAW_DIR / "ua-wrb.tif"

WRB_CODES = {"Chernozems": 7, "Phaeozems": 20}

DOWNSCALE_FACTOR = 4       # 250 m -> 1 km. Drops vertex count ~16x.
MIN_POLY_AREA_DEG = 1e-4   # ~1 km^2; drops noise specks
SIMPLIFY_DEG = 0.01        # ~1 km
COORD_PRECISION = 4


def download_if_missing(url: str, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 1024:
        print("  %s exists, skip" % dst.name)
        return
    print("Downloading %s ..." % dst.name)
    urllib.request.urlretrieve(url, dst)


def downscale(arr: np.ndarray, factor: int) -> np.ndarray:
    """Block-mode downsample: take dominant code in each factor x factor block."""
    h, w = arr.shape
    h2 = (h // factor) * factor
    w2 = (w // factor) * factor
    cropped = arr[:h2, :w2]
    blocks = cropped.reshape(h2 // factor, factor, w2 // factor, factor)
    # Mode along the two block axes: collapse to (h2//factor, w2//factor).
    # For categorical data simple max isn't right, but for our purpose
    # picking the upper-left pixel of each block is fast and good enough.
    return blocks[:, 0, :, 0]


def vectorise(raster: np.ndarray, transform, code: int):
    mask = (raster == code)
    n = int(mask.sum())
    if n == 0:
        return None, 0
    polys = []
    for geom_dict, val in shapes(mask.astype("uint8"), mask=mask, transform=transform):
        if val == 1:
            poly = shape(geom_dict)
            if poly.area >= MIN_POLY_AREA_DEG:
                polys.append(poly)
    if not polys:
        return None, n

    # shapes() already returns disjoint polygons -> wrap as MultiPolygon
    # (NO unary_union: it's O(N^2) for N polygons and we don't need it).
    geom = MultiPolygon(polys) if len(polys) > 1 else polys[0]
    geom = geom.simplify(SIMPLIFY_DEG, preserve_topology=True)
    return geom, n


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    w, s, e, n = UA_BBOX
    download_if_missing(WCS_URL.format(w=w, s=s, e=e, n=n), RAW_WRB)

    print("Loading raster ...")
    with rasterio.open(RAW_WRB) as src:
        wrb = src.read(1)
        transform = src.transform

    print("  original shape %s" % (wrb.shape,))
    if DOWNSCALE_FACTOR > 1:
        wrb = downscale(wrb, DOWNSCALE_FACTOR)
        # Adjust transform: pixel size grows by factor.
        transform = transform * transform.scale(DOWNSCALE_FACTOR, DOWNSCALE_FACTOR)
        print("  downscaled shape %s" % (wrb.shape,))

    for class_name, code in WRB_CODES.items():
        print("Vectorising %s (code %d) ..." % (class_name, code))
        geom, n_pixels = vectorise(wrb, transform, code)
        if geom is None:
            print("  -> 0 pixels")
            continue
        gdf = gpd.GeoDataFrame(
            {"class": [class_name]}, geometry=[geom], crs="EPSG:4326",
        )
        out_path = OUT_DIR / ("%s.geojson" % class_name.lower())
        if out_path.exists():
            out_path.unlink()
        gdf.to_file(out_path, driver="GeoJSON", COORDINATE_PRECISION=COORD_PRECISION)
        print("  -> %s: %d px, %d KB" %
              (out_path.name, n_pixels, out_path.stat().st_size // 1024))


if __name__ == "__main__":
    main()