"""
Stage 3c: SoilGrids 2.0 (ISRIC, 250 m) soil layers for Vinnytsia oblast, as
candidate predictors next to HWSD (which is too coarse at the site level).

SoilGrids is a machine-learning prediction, and today's soil is only a proxy for
the soil of the 3rd-4th centuries; see 05e_soils.py for what it is worth here.

Downloaded through the ISRIC WCS in EPSG:4326 (native pixels ~230 x 155 m here):
    wrb_<Group>      probability of the WRB group, %  (Chernozems, Phaeozems,
                     Luvisols, Fluvisols, Gleysols)
    clay, sand       g/kg      (15-30 cm, mean)
    soc              dg/kg     (15-30 cm, mean)
    phh2o            pH x 10   (15-30 cm, mean)
Output: data/processed/env/soilgrids/*.tif  (gitignored; ~6 MB)

sample() gives two kinds of features at metric coordinates (EPSG:6381):
    point            the pixel under the point
    zone             mean within ZONE_M: the arable land around a settlement
                     ("resource zone"), which is what the literature ties the
                     settlements to - not the soil under the houses.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "data" / "processed" / "env" / "soilgrids"

BBOX = (27.2, 47.8, 30.4, 50.1)  # oblast + margin for the zone window
WCS_URL = (
    "https://maps.isric.org/mapserv?map=/map/{map}.map"
    "&SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage"
    "&COVERAGEID={coverage}&FORMAT=image/tiff"
    "&SUBSET=long({w},{e})&SUBSET=lat({s},{n})"
    "&OUTPUTCRS=http://www.opengis.net/def/crs/EPSG/0/4326"
    "&SUBSETTINGCRS=http://www.opengis.net/def/crs/EPSG/0/4326"
)
WRB_GROUPS = ["Chernozems", "Phaeozems", "Luvisols", "Fluvisols", "Gleysols"]
PROPERTIES = ["clay", "sand", "soc", "phh2o"]
DEPTH = "15-30cm_mean"
LAYERS = {f"wrb_{g}": ("wrb", g) for g in WRB_GROUPS} | {p: (p, f"{p}_{DEPTH}") for p in PROPERTIES}
ZONE_M = 1500
METRIC_CRS = 6381


def download() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    w, s, e, n = BBOX
    for name, (map_name, coverage) in LAYERS.items():
        path = OUT / f"{name}.tif"
        if path.exists() and path.stat().st_size > 1024:
            continue
        print(f"Downloading {name} ...")
        url = WCS_URL.format(map=map_name, coverage=coverage, w=w, s=s, e=e, n=n)
        urllib.request.urlretrieve(url, path)


def zone_mean(a: np.ndarray, pixel_m: tuple[float, float], radius_m: float) -> np.ndarray:
    """Mean of the valid cells in a (2 * radius) box around every cell."""
    size = tuple(max(1, int(round(2 * radius_m / p))) | 1 for p in pixel_m)  # odd, (rows, cols)
    valid = ~np.isnan(a)
    num = ndimage.uniform_filter(np.where(valid, a, 0.0), size, mode="nearest")
    den = ndimage.uniform_filter(valid.astype("float32"), size, mode="nearest")
    return np.where(den > 0.2, num / np.maximum(den, 1e-6), np.nan)


def read(name: str):
    """Layer as float array with NaN for no data (0 in the WCS output), and its transform."""
    with rasterio.open(OUT / f"{name}.tif") as src:
        a = src.read(1).astype("float32")
        transform = src.transform
    if not name.startswith("wrb_"):
        a[a == 0] = np.nan  # properties: 0 = water / no data
    return a, transform


def sample(xy: np.ndarray, names=None, zone_m: float = ZONE_M) -> dict[str, np.ndarray]:
    """Soil features at metric points: '<name>' (point) and '<name>_zone'."""
    lon, lat = Transformer.from_crs(METRIC_CRS, 4326, always_xy=True).transform(xy[:, 0], xy[:, 1])
    out = {}
    for name in names or LAYERS:
        a, t = read(name)
        cols = np.clip(((lon - t.c) / t.a).astype(int), 0, a.shape[1] - 1)
        rows = np.clip(((lat - t.f) / t.e).astype(int), 0, a.shape[0] - 1)
        mid_lat = np.radians((BBOX[1] + BBOX[3]) / 2)
        pixel_m = (abs(t.e) * 111_320, t.a * 111_320 * np.cos(mid_lat))
        out[name] = a[rows, cols]
        out[f"{name}_zone"] = zone_mean(a, pixel_m, zone_m)[rows, cols]
    return out


def main() -> None:
    download()
    for name in LAYERS:
        a, t = read(name)
        print(f"  {name}: {a.shape}, median {np.nanmedian(a):.0f}, nodata {np.isnan(a).mean():.1%}")


if __name__ == "__main__":
    main()
