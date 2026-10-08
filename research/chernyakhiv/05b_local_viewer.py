"""
Stage 5b: LOCAL map viewer for the stage-5.1 refinement (exact site positions,
so it is written under data/processed/ and never published; decision 1 in PLAN.md).

Input:
    data/processed/chernyakhiv_sites.gpkg   layers "sites" and "sites_refined"
    data/processed/env/flow_acc_km2.tif, elevation.tif
    docs/data/rivers.geojson, streams.geojson
Output: data/processed/chernyakhiv_viewer/
    index.html, sites.geojson, waterways.geojson, channels.png, hillshade.webp

View:
    cd data/processed/chernyakhiv_viewer && python -m http.server 8001
    -> http://localhost:8001
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.warp import Resampling, calculate_default_transform, reproject
from shapely.geometry import LineString

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
ENV = PROCESSED / "env"
SITES = PROCESSED / "chernyakhiv_sites.gpkg"
WATERWAYS = [PROJECT_ROOT / "docs" / "data" / f for f in ("rivers.geojson", "streams.geojson")]
OUT = PROCESSED / "chernyakhiv_viewer"

WEB_CRS = "EPSG:3857"
CHANNEL_RES_M = 30  # keep channels at DEM resolution, they are one cell wide
HILLSHADE_RES_M = 45
# catchment classes (km2) -> RGBA, matching the snapping thresholds of 05a
CHANNEL_CLASSES = [(1, (110, 170, 230, 200)), (10, (40, 110, 210, 230)), (100, (10, 50, 150, 255))]
KEEP = ["no", "village", "district", "bank", "water_name", "direction", "desc"]


def to_web(src_path: Path, res: float, resampling) -> tuple[np.ndarray, list]:
    """Reproject a raster to Web Mercator; return array and Leaflet bounds."""
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, WEB_CRS, src.width, src.height, *src.bounds, resolution=res
        )
        out = np.full((height, width), np.nan, dtype="float32")
        reproject(
            source=rasterio.band(src, 1),
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=transform,
            dst_crs=WEB_CRS,
            dst_nodata=np.nan,
            resampling=resampling,
        )
    x0, y1 = transform * (0, 0)
    x1, y0 = transform * (width, height)
    corners = gpd.GeoSeries.from_xy([x0, x1], [y0, y1], crs=WEB_CRS).to_crs(4326)
    bounds = [[corners.y[0], corners.x[0]], [corners.y[1], corners.x[1]]]
    return out, bounds


def channels_png() -> list:
    acc, bounds = to_web(ENV / "flow_acc_km2.tif", CHANNEL_RES_M, Resampling.max)
    rgba = np.zeros(acc.shape + (4,), dtype="uint8")
    for threshold, colour in CHANNEL_CLASSES:
        rgba[np.nan_to_num(acc) >= threshold] = colour
    Image.fromarray(rgba, "RGBA").save(OUT / "channels.png", optimize=True)
    return bounds


def hillshade_image() -> list:
    dem, bounds = to_web(ENV / "elevation.tif", HILLSHADE_RES_M, Resampling.bilinear)
    dy, dx = np.gradient(np.nan_to_num(dem, nan=np.nanmean(dem)), HILLSHADE_RES_M)
    dx, dy = dx * 3, -dy * 3  # vertical exaggeration; rows grow southward
    azimuth, altitude = np.radians(315), np.radians(45)
    slope = np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    shade = np.sin(altitude) * np.cos(slope) + np.cos(altitude) * np.sin(slope) * np.cos(
        azimuth - aspect
    )
    grey = (np.clip(shade, 0, 1) * 255).astype("uint8")
    alpha = np.where(np.isnan(dem), 0, 255).astype("uint8")
    Image.fromarray(np.dstack([grey, grey, grey, alpha]), "RGBA").save(
        OUT / "hillshade.webp", quality=75, method=4
    )
    return bounds


def sites_geojson() -> None:
    before = gpd.read_file(SITES, layer="sites")
    before = before[before["type"] == "Поселення"].reset_index(drop=True)
    after = gpd.read_file(SITES, layer="sites_refined")
    assert (before["no"].to_numpy() == after["no"].to_numpy()).all()
    props = after[KEEP + ["water_class", "bank_used", "snap_how", "channel_km2", "shift_m"]]
    props = props.astype(object).where(props.notna(), None)
    feats = []
    for i, rec in enumerate(props.to_dict(orient="records")):
        b, a = before.geometry[i], after.geometry[i]
        feats += [
            {**rec, "role": "before", "geometry": b},
            {**rec, "role": "after", "geometry": a},
        ]
        if rec["snap_how"] != "none":
            feats.append({**rec, "role": "shift", "geometry": LineString([b, a])})
    gdf = gpd.GeoDataFrame(feats, geometry="geometry", crs=before.crs).to_crs(4326)
    gdf.to_file(OUT / "sites.geojson", driver="GeoJSON", COORDINATE_PRECISION=6)


def waterways_geojson(area) -> None:
    parts = [gpd.read_file(f, bbox=area.bounds) for f in WATERWAYS]
    ways = pd.concat(parts, ignore_index=True)[["name", "geometry"]]
    ways = gpd.GeoDataFrame(ways, crs=4326).clip(area)
    ways.to_file(OUT / "waterways.geojson", driver="GeoJSON", COORDINATE_PRECISION=5)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("sites ...")
    sites_geojson()
    area = gpd.read_file(OUT / "sites.geojson").total_bounds
    pad = 0.05
    box = gpd.GeoSeries.from_xy([area[0] - pad, area[2] + pad], [area[1] - pad, area[3] + pad])
    print("waterways ...")
    waterways_geojson(box.union_all().envelope)
    print("channels ...")
    overlays = {"channels": channels_png()}
    print("hillshade ...")
    overlays["hillshade"] = hillshade_image()
    html = (HERE / "viewer_template.html").read_text(encoding="utf-8")
    html = html.replace("/*OVERLAYS*/null", json.dumps(overlays))
    (OUT / "index.html").write_text(html, encoding="utf-8")
    for f in sorted(OUT.iterdir()):
        print(f"  {f.name}: {f.stat().st_size // 1024} KB")
    print(f"cd {OUT.relative_to(PROJECT_ROOT)} && python -m http.server 8001")


if __name__ == "__main__":
    main()
