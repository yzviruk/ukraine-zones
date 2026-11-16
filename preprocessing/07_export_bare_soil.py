"""
Stage 07: "soil darkness of ploughed land" as a picture layer for the web map.

Input: bare-soil composites of Sentinel-2 made by
research/chernyakhiv/03d_bare_soil.py (data/processed/env/bare_soil/<tile>_bare.tif,
band 7 `vis_norm`: brightness of bare soil relative to the median bare field,
median over 2018-2026; lower = darker = more humus).

Output (docs/data/):
    bare_soil_darkness.png    RGBA picture in Web Mercator, ~100 m, clipped to the
                              oblast; transparent where there is no ploughed land
    bare_soil_darkness.json   corner coordinates for L.imageOverlay, class breaks,
                              colours, attribution

This is a MEASURED relative index (not a soil class, not a humus percentage) and
it exists only for arable land. Classes are quantiles of the oblast's arable land.

Data: Contains modified Copernicus Sentinel data 2018-2026; cropland mask
(c) ESA WorldCover project 2021.

Run: python 07_export_bare_soil.py --tiles 35UPQ 35UNQ 35UPP 35UNP
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import Affine, from_bounds
from rasterio.warp import Resampling, reproject, transform_bounds

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BARE = PROJECT_ROOT / "data" / "processed" / "env" / "bare_soil"
BORDER = PROJECT_ROOT / "data" / "raw" / "ua-border.geojson"
OUT = PROJECT_ROOT / "docs" / "data"

OBLAST = "Вінницька область"
PX = 20.0
AGG = 5  # 20 m -> 100 m before warping
MIN_SHARE = 0.3  # of a 100 m cell that must be bare-soil data
WEB_PX = 150.0  # Web Mercator metres per pixel (~100 m on the ground at 49 N)
BREAKS_Q = (10, 25, 45, 65, 80, 92)  # quantiles of arable land -> 7 classes
# dark humus-rich soil -> light eroded / grey forest soil
COLORS = ["#2b1a12", "#4a2f1d", "#6d4a2c", "#93703f", "#b8975c", "#d8c285", "#f0e6b8"]
ALPHA = 215


def calibrated_mosaic(tiles: list[str]) -> tuple[np.ndarray, Affine, object]:
    """Mean `vis_norm` of the tiles on one UTM grid. Each tile is normalised by its
    own scenes, so every new tile is scaled to the mosaic on their overlap."""
    srcs = [rasterio.open(BARE / f"{t}_bare.tif") for t in tiles]
    crs = srcs[0].crs
    left = min(s.bounds.left for s in srcs)
    top = max(s.bounds.top for s in srcs)
    w = round((max(s.bounds.right for s in srcs) - left) / PX)
    h = round((top - min(s.bounds.bottom for s in srcs)) / PX)
    total = np.zeros((h, w), np.float32)
    count = np.zeros((h, w), np.float32)
    for tile, s in zip(tiles, srcs, strict=True):
        r0, c0 = round((top - s.bounds.top) / PX), round((s.bounds.left - left) / PX)
        sl = (slice(r0, r0 + s.height), slice(c0, c0 + s.width))
        a = s.read(7)
        ok = np.isfinite(a)
        both = ok & (count[sl] > 0)
        if both.sum() > 10_000:
            ratio = float(np.median(total[sl][both] / count[sl][both] / a[both]))
            print(
                f"  {tile}: scaled by {ratio:.3f} on {both.sum() * PX * PX / 1e6:.0f} km2 overlap"
            )
            a = a * ratio
        total[sl] += np.where(ok, a, 0)
        count[sl] += ok
        s.close()
    mosaic = np.where(count > 0, total / np.maximum(count, 1), np.nan).astype(np.float32)
    return mosaic, Affine(PX, 0, left, 0, -PX, top), crs


def block_mean(a: np.ndarray, k: int, min_share: float) -> np.ndarray:
    """k x k block mean of the finite cells; NaN where fewer than min_share are finite."""
    h, w = (s // k * k for s in a.shape)
    blocks = a[:h, :w].reshape(h // k, k, w // k, k)
    ok = np.isfinite(blocks)
    num = np.where(ok, blocks, 0).sum(axis=(1, 3))
    den = ok.sum(axis=(1, 3))
    return np.where(den >= min_share * k * k, num / np.maximum(den, 1), np.nan).astype(np.float32)


def classify(values: np.ndarray, breaks: list[float]) -> np.ndarray:
    """Class 1..len(breaks)+1 by brightness (1 = darkest); 0 = no data."""
    out = np.zeros(values.shape, np.uint8)
    ok = np.isfinite(values)
    out[ok] = np.digitize(values[ok], breaks) + 1
    return out


def colorize(classes: np.ndarray) -> np.ndarray:
    """(4, h, w) RGBA; class 0 is transparent."""
    table = np.zeros((len(COLORS) + 1, 4), np.uint8)
    for i, c in enumerate(COLORS, 1):
        table[i] = [int(c[j : j + 2], 16) for j in (1, 3, 5)] + [ALPHA]
    return np.moveaxis(table[classes], -1, 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", nargs="+", default=["35UPQ", "35UNQ", "35UPP", "35UNP"])
    args = ap.parse_args()

    mosaic, transform, crs = calibrated_mosaic(args.tiles)
    coarse = block_mean(mosaic, AGG, MIN_SHARE)
    coarse_transform = transform * Affine.scale(AGG)
    del mosaic

    border = gpd.read_file(BORDER)
    border = border[border.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    oblast = border[border["name"] == OBLAST]
    west, south, east, north = transform_bounds(
        "EPSG:4326", "EPSG:3857", *oblast.total_bounds, densify_pts=21
    )
    width, height = round((east - west) / WEB_PX), round((north - south) / WEB_PX)
    web_transform = from_bounds(west, south, east, north, width, height)
    web = np.full((height, width), np.nan, np.float32)
    reproject(
        coarse,
        web,
        src_transform=coarse_transform,
        src_crs=crs,
        src_nodata=np.nan,
        dst_transform=web_transform,
        dst_crs="EPSG:3857",
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    outside = geometry_mask(oblast.to_crs(3857).geometry, (height, width), web_transform)
    web[outside] = np.nan

    values = web[np.isfinite(web)]
    breaks = [round(float(np.percentile(values, q)), 3) for q in BREAKS_Q]
    classes = classify(web, breaks)
    rgba = colorize(classes)
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "bare_soil_darkness.png"
    with rasterio.open(
        png, "w", driver="PNG", dtype="uint8", count=4, width=width, height=height
    ) as dst:
        dst.write(rgba)
    for aux in OUT.glob("bare_soil_darkness.png.aux.xml"):
        aux.unlink()
    w4, s4, e4, n4 = (round(float(v), 5) for v in oblast.total_bounds)
    meta = {
        "bounds": [[s4, w4], [n4, e4]],  # Leaflet: [[south, west], [north, east]]
        "size": [width, height],
        "index": "vis_norm: brightness of bare ploughed soil relative to the median field",
        "breaks": breaks,
        "colors": COLORS,
        "share_of_oblast_with_data": round(float(values.size) / float((~outside).sum()), 3),
        "tiles": args.tiles,
        "attribution": (
            "Contains modified Copernicus Sentinel data 2018–2026; cropland: © ESA WorldCover 2021"
        ),
    }
    (OUT / "bare_soil_darkness.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"{png.name}: {width} x {height}, {png.stat().st_size / 1e6:.1f} MB; breaks {breaks}")
    print(f"data on {meta['share_of_oblast_with_data']:.0%} of the oblast")


if __name__ == "__main__":
    main()
