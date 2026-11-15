"""
Stage 6: SoilGrids 2.0 chernozems and phaeozems as a second, more detailed soil
layer next to HWSD (stage 5).

SoilGrids (ISRIC, 250 m) is a MACHINE-LEARNING PREDICTION of the most probable
WRB reference soil group, not a surveyed map: it draws finer patterns than HWSD
(~1 km raster of the 1:1M European Soil Database), but its class boundaries are
modelled, and for Ukraine the two disagree a lot (SoilGrids calls ~70 % of
Vinnytsia oblast chernozem, HWSD ~20 %). Both are shown; the site labels this
one as a model prediction.

    soilgrids_chernozems   MostProbable == Chernozems
    soilgrids_phaeozems    MostProbable == Phaeozems  (опідзолені, темно-сірі)

The class mask is smoothed (majority in a 3 x 3 window) and small patches are
dropped, so the layer stays light enough for the browser: SoilGrids is noisy at
the pixel level anyway.

Output: docs/data/soilgrids_<class>.geojson in EPSG:4326, clipped to the border.

Licence: SoilGrids 2.0 is CC BY 4.0; cite Poggio et al. (2021), doi:10.5194/soil-7-217-2021.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from scipy import ndimage
from shapely.geometry import MultiPolygon, shape
from shapely.ops import unary_union

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
BORDER = RAW_DIR / "ua-border.geojson"
OUT_DIR = PROJECT_ROOT / "docs" / "data"
RAW_WRB = RAW_DIR / "soilgrids" / "ua-wrb-mostprobable.tif"

UA_BBOX = (22.0, 44.0, 41.0, 53.0)
WCS_URL = (
    "https://maps.isric.org/mapserv?map=/map/wrb.map"
    "&SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage"
    "&COVERAGEID=MostProbable&FORMAT=image/tiff"
    "&SUBSET=long({w},{e})&SUBSET=lat({s},{n})"
    "&OUTPUTCRS=http://www.opengis.net/def/crs/EPSG/0/4326"
    "&SUBSETTINGCRS=http://www.opengis.net/def/crs/EPSG/0/4326"
)
# Codes of the MostProbable coverage: WRB groups in alphabetical order from 0.
WRB_CODES = {"chernozems": 7, "phaeozems": 20}
SOURCE = "SoilGrids 2.0, ISRIC 2020 (model prediction, 250 m), CC BY 4.0"

SMOOTH_CELLS = 3  # majority window, ~750 m
BORDER_SIMPLIFY_DEG = 0.002
SIMPLIFY_DEG = 0.0015  # ~150 m, under one pixel
MIN_POLY_AREA_DEG = 1e-4  # ~0.8 km2: drops specks and tiny holes
COORD_PRECISION = 4


def download() -> None:
    if RAW_WRB.exists() and RAW_WRB.stat().st_size > 1024:
        print(f"  {RAW_WRB.name} exists, skip")
        return
    RAW_WRB.parent.mkdir(parents=True, exist_ok=True)
    w, s, e, n = UA_BBOX
    print(f"Downloading {RAW_WRB.name} ...")
    urllib.request.urlretrieve(WCS_URL.format(w=w, s=s, e=e, n=n), RAW_WRB)


def smooth(mask: np.ndarray, cells: int = SMOOTH_CELLS) -> np.ndarray:
    """Majority of the class in a cells x cells window."""
    return ndimage.uniform_filter(mask.astype("float32"), cells, mode="nearest") > 0.5


def without_small_holes(poly, min_area: float):
    keep = [
        ring
        for ring in poly.interiors
        if shape({"type": "Polygon", "coordinates": [ring.coords]}).area >= min_area
    ]
    return type(poly)(poly.exterior, keep)


def vectorise(mask: np.ndarray, transform, min_area: float = MIN_POLY_AREA_DEG) -> list:
    polys = []
    for geom_dict, val in shapes(mask.astype("uint8"), mask=mask, transform=transform):
        if val == 1:
            poly = shape(geom_dict)
            if poly.area >= min_area:
                polys.append(without_small_holes(poly, min_area))
    return polys


def as_multipolygon(geom, min_area: float = MIN_POLY_AREA_DEG) -> MultiPolygon:
    parts = getattr(geom, "geoms", [geom])
    return MultiPolygon([p for p in parts if p.geom_type == "Polygon" and p.area >= min_area])


def load_border():
    border = gpd.read_file(BORDER)
    border = border[border.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    if border.empty:
        raise SystemExit(f"No polygon geometry in {BORDER.name}")
    return unary_union(border.geometry.values).simplify(BORDER_SIMPLIFY_DEG)


def main() -> None:
    if not BORDER.exists():
        raise SystemExit(f"Missing {BORDER}.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    download()
    with rasterio.open(RAW_WRB) as src:
        wrb, transform = src.read(1), src.transform
    print(f"  raster {wrb.shape}, pixel {transform.a:.5f} deg")
    print("Loading Ukraine border ...")
    border = load_border()

    for layer, code in WRB_CODES.items():
        print(f"Vectorising {layer} (code {code}) ...")
        polys = vectorise(smooth(wrb == code), transform)
        if not polys:
            print("  -> 0 polygons, skip")
            continue
        geom = MultiPolygon(polys).simplify(SIMPLIFY_DEG, preserve_topology=True)
        geom = as_multipolygon(geom.buffer(0).intersection(border))
        gdf = gpd.GeoDataFrame(
            {"class": [layer], "source": [SOURCE]}, geometry=[geom], crs="EPSG:4326"
        )
        out_path = OUT_DIR / f"soilgrids_{layer}.geojson"
        out_path.unlink(missing_ok=True)
        gdf.to_file(out_path, driver="GeoJSON", COORDINATE_PRECISION=COORD_PRECISION)
        size_kb = out_path.stat().st_size // 1024
        print(f"  -> {out_path.name}: {len(geom.geoms)} polygons, {size_kb} KB")


if __name__ == "__main__":
    main()
