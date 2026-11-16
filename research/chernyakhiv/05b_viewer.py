"""
Stage 5b: map viewer for the exact site positions (stage 5.1) and full catalogue
texts. Two modes:

    local (default)  data/processed/chernyakhiv_viewer/  (gitignored, plain)
    --private        docs/private/  (published on GitHub Pages, ENCRYPTED)

Decision 1 in PLAN.md: exact positions and catalogue texts are never published
in the clear. In --private mode everything about the sites (positions, shift
lines, descriptions, full catalogue entries) goes into sites.enc.json:
gzip -> AES-256-GCM with a key from PBKDF2-SHA256 (600k iterations, random salt).
The page asks for the password and decrypts in the browser (WebCrypto). Only the
ciphertext is committed; the background layers (hillshade, DEM channels, OSM
waterways, 150 m suitability) say nothing about the sites and stay plain.
The candidate zones (high suitability with no known site nearby) are cut out
around the known sites, so their holes give the sites away: they go into the
encrypted payload too, never into a plain file.

Password: env CHERNYAKHIV_PASSWORD, otherwise asked twice in the terminal.
Use a long passphrase (>= 14 characters; the ciphertext is public, so a weak
password can be brute-forced offline). Changing it = re-run with a new one.

Input:
    data/processed/chernyakhiv_sites.gpkg   layers "sites", "sites_refined"
    data/processed/env/flow_acc_km2.tif, elevation.tif
    data/processed/chernyakhiv_model/suitability_150m_v1.tif, _v2.tif (05d; optional)
    docs/data/rivers.geojson, streams.geojson

View:
    local:   cd data/processed/chernyakhiv_viewer && python -m http.server 8001
    private: https://yzviruk.github.io/ukraine-zones/private/ (after push)
"""

from __future__ import annotations

import argparse
import base64
import getpass
import gzip
import json
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.features import shapes
from rasterio.warp import Resampling, calculate_default_transform, reproject
from scipy.spatial import cKDTree
from shapely import set_precision
from shapely.geometry import LineString, shape
from shapely.ops import unary_union

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
ENV = PROCESSED / "env"
SITES = PROCESSED / "chernyakhiv_sites.gpkg"
# Both models are shown, switched in the panel (05d_map.py: v1 = catchment only).
MODELS = ("v1", "v2")
SUITABILITY = {m: PROCESSED / "chernyakhiv_model" / f"suitability_150m_{m}.tif" for m in MODELS}
WATERWAYS = [PROJECT_ROOT / "docs" / "data" / f for f in ("rivers.geojson", "streams.geojson")]
OUT_LOCAL = PROCESSED / "chernyakhiv_viewer"
OUT_PRIVATE = PROJECT_ROOT / "docs" / "private"

WEB_CRS = "EPSG:3857"
CHANNEL_RES_M = 30  # keep channels at DEM resolution, they are one cell wide
HILLSHADE_RES_M = 45
SUITABILITY_RES_M = 150
# catchment classes (km2) -> RGBA, matching the snapping thresholds of 05a
CHANNEL_CLASSES = [(1, (110, 170, 230, 200)), (10, (40, 110, 210, 230)), (100, (10, 50, 150, 255))]
# suitability percentile -> RGBA (05d: high = top 10 % of cells)
SUITABILITY_CLASSES = [
    (50, (231, 212, 232, 110)),
    (75, (194, 165, 207, 160)),
    (90, (153, 112, 171, 200)),
    (97, (118, 42, 131, 230)),
]
KEEP = ["no", "village", "district", "bank", "water_name", "direction", "desc", "text"]
# Candidate zones: suitability percentile >= CANDIDATE_RANK (TOP_RANK = the best
# 3 %) and no known site within CANDIDATE_GAP_M. Patches under two cells dropped.
CANDIDATE_RANK, TOP_RANK = 90, 97
CANDIDATE_GAP_M = 1000
MIN_PATCH_M2 = 2 * SUITABILITY_RES_M**2
# Position accuracy: class -> radius (m). Estimates from how the point was placed
# (refine.md: median snap shift 290 m, 90 % < 830 m), NOT validated on the ground.
ACCURACY_M = {"A": 300, "B": 500, "C": 1000, "D": 1500}
KDF_ITERATIONS = 600_000
MIN_PASSWORD = 14


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


def classes_png(values: np.ndarray, classes, path: Path) -> None:
    rgba = np.zeros(values.shape + (4,), dtype="uint8")
    for threshold, colour in classes:
        rgba[np.nan_to_num(values, nan=-1) >= threshold] = colour
    Image.fromarray(rgba, "RGBA").save(path, optimize=True)


def channels_png(out: Path) -> list:
    acc, bounds = to_web(ENV / "flow_acc_km2.tif", CHANNEL_RES_M, Resampling.max)
    classes_png(acc, CHANNEL_CLASSES, out / "channels.png")
    return bounds


def bare_soil_png(out: Path) -> list | None:
    """Soil darkness of ploughed land (Sentinel-2; preprocessing/07_export_bare_soil.py).
    The same picture as on the public site: it says nothing about the sites."""
    public = PROJECT_ROOT / "docs" / "data"
    meta = public / "bare_soil_darkness.json"
    if not meta.exists():
        print("  (no soil darkness layer: run preprocessing/07_export_bare_soil.py)")
        return None
    (out / "bare_soil.png").write_bytes((public / "bare_soil_darkness.png").read_bytes())
    return json.loads(meta.read_text(encoding="utf-8"))["bounds"]


def suitability_png(out: Path, model: str) -> list | None:
    if not SUITABILITY[model].exists():
        print(f"  (no suitability raster for {model}: run 05d_map.py)")
        return None
    rank, bounds = to_web(SUITABILITY[model], SUITABILITY_RES_M, Resampling.nearest)
    classes_png(rank, SUITABILITY_CLASSES, out / f"suitability_{model}.png")
    return bounds


def hillshade_image(out: Path) -> list:
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
        out / "hillshade.webp", quality=75, method=4
    )
    return bounds


def accuracy(snap_how, placed) -> str:
    """A: snapped to the named water; B: to a channel of the stated type;
    C: placed from the village by the text alone (direction, part of the village,
    distance, second village); D: the village alone."""
    if snap_how == "named":
        return "A"
    if snap_how == "dem":
        return "B"
    return "D" if placed in ("village", None, "") else "C"


def candidate_mask(rank, transform, site_xy, gap_m=CANDIDATE_GAP_M, min_rank=CANDIDATE_RANK):
    """Cells with suitability >= min_rank and no known site within gap_m."""
    rows, cols = np.nonzero(rank >= min_rank)
    x = transform.c + (cols + 0.5) * transform.a
    y = transform.f + (rows + 0.5) * transform.e
    dist, _ = cKDTree(site_xy).query(np.c_[x, y])
    mask = np.zeros(rank.shape, dtype=bool)
    mask[rows[dist > gap_m], cols[dist > gap_m]] = True
    return mask


def candidate_features(site_xy, model: str) -> list[dict]:
    """Candidate zones of one model as two dissolved polygons: the top 3 % and the rest."""
    if not SUITABILITY[model].exists():
        print(f"  (no suitability raster for {model}: run 05d_map.py)")
        return []
    with rasterio.open(SUITABILITY[model]) as src:
        rank, transform = src.read(1), src.transform
    mask = candidate_mask(rank, transform, site_xy)
    feats = []
    for cls, part in (("top", mask & (rank >= TOP_RANK)), ("high", mask & (rank < TOP_RANK))):
        polys = [
            shape(geom) for geom, _ in shapes(part.astype("uint8"), mask=part, transform=transform)
        ]
        polys = [p for p in polys if p.area >= MIN_PATCH_M2]
        if polys:
            area = sum(p.area for p in polys) / 1e6
            print(f"  candidates {model} {cls}: {len(polys)} patches, {area:.0f} km2")
            feats.append(
                {"role": "candidate", "cls": cls, "model": model, "geometry": unary_union(polys)}
            )
    return feats


def sites_geojson() -> bytes:
    stage2 = gpd.read_file(SITES, layer="sites")
    before = stage2[stage2["type"] == "Поселення"].reset_index(drop=True)
    burials = stage2[stage2["type"] == "Могильник"].reset_index(drop=True)
    # The map shows the best estimate (05a: offset by the landform of the text);
    # the fixed-offset "sites_refined" layer is for the analyses only.
    after = gpd.read_file(SITES, layer="sites_best")
    assert (before["no"].to_numpy() == after["no"].to_numpy()).all()
    after["shift_m"] = after.distance(before)
    extra = ["water_class", "bank_used", "snap_how", "channel_km2", "shift_m", "placed", "landform"]
    props = after[KEEP + extra]
    props = props.astype(object).where(props.notna(), None)
    feats = []
    for i, rec in enumerate(props.to_dict(orient="records")):
        b, a = before.geometry[i], after.geometry[i]
        acc = accuracy(rec["snap_how"], rec["placed"])
        # Text only on the main point; the other two link to it by "no".
        short = {k: rec[k] for k in ("no", "village", "snap_how", "shift_m")}
        feats += [
            {**short, "role": "before", "geometry": b},
            {**rec, "role": "after", "acc": acc, "acc_m": ACCURACY_M[acc], "geometry": a},
        ]
        if rec["snap_how"] != "none":
            feats.append({**short, "role": "shift", "geometry": LineString([b, a])})
    # Cemeteries are not refined (stage 5.1 covers settlements): stage-2 positions.
    burial_props = burials[KEEP + ["placed"]]
    burial_props = burial_props.astype(object).where(burial_props.notna(), None)
    for i, rec in enumerate(burial_props.to_dict(orient="records")):
        acc = accuracy("none", rec["placed"])
        feats.append(
            {
                **rec,
                "role": "burial",
                "snap_how": "none",
                "acc": acc,
                "acc_m": ACCURACY_M[acc],
                "geometry": burials.geometry[i],
            }
        )
    known = pd.concat([after.geometry, burials.geometry])
    for model in MODELS:
        feats += candidate_features(np.c_[known.x, known.y], model)
    gdf = gpd.GeoDataFrame(feats, geometry="geometry", crs=before.crs).to_crs(4326)
    gdf["geometry"] = set_precision(gdf.geometry.values, 1e-5)  # ~1 m; keeps the payload small
    return gdf.to_json(drop_id=True, to_wgs84=True, ensure_ascii=False).encode("utf-8")


def waterways_geojson(area, out: Path) -> None:
    parts = [gpd.read_file(f, bbox=area.bounds) for f in WATERWAYS]
    ways = pd.concat(parts, ignore_index=True)[["name", "geometry"]]
    ways = gpd.GeoDataFrame(ways, crs=4326).clip(area)
    path = out / "waterways.geojson"
    path.unlink(missing_ok=True)
    ways.to_file(path, driver="GeoJSON", COORDINATE_PRECISION=5)


def ask_password() -> str:
    password = os.environ.get("CHERNYAKHIV_PASSWORD")
    if not password:
        password = getpass.getpass("Пароль для docs/private: ")
        if getpass.getpass("Ще раз: ") != password:
            raise SystemExit("Паролі не збігаються.")
    if len(password) < MIN_PASSWORD:
        raise SystemExit(f"Пароль закороткий: потрібно щонайменше {MIN_PASSWORD} символів.")
    return password


def encrypt(data: bytes, password: str) -> dict:
    """gzip -> AES-256-GCM; key = PBKDF2-SHA256(password). WebCrypto-compatible."""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    salt, iv = os.urandom(16), os.urandom(12)
    key = PBKDF2HMAC(hashes.SHA256(), 32, salt, KDF_ITERATIONS).derive(password.encode("utf-8"))
    ct = AESGCM(key).encrypt(iv, gzip.compress(data, mtime=0), None)
    b64 = lambda b: base64.b64encode(b).decode("ascii")  # noqa: E731
    return {
        "kdf": "PBKDF2-SHA256",
        "iterations": KDF_ITERATIONS,
        "cipher": "AES-256-GCM",
        "compression": "gzip",
        "salt": b64(salt),
        "iv": b64(iv),
        "ct": b64(ct),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private", action="store_true", help="encrypted, into docs/private")
    parser.add_argument("--out", type=Path, help="output folder (default by mode)")
    args = parser.parse_args()
    out = args.out or (OUT_PRIVATE if args.private else OUT_LOCAL)
    password = ask_password() if args.private else None
    out.mkdir(parents=True, exist_ok=True)

    print("sites ...")
    sites = sites_geojson()
    if args.private:
        (out / "sites.geojson").unlink(missing_ok=True)  # never leave plain data there
        enc = encrypt(sites, password)
        (out / "sites.enc.json").write_text(json.dumps(enc), encoding="ascii")
    else:
        (out / "sites.geojson").write_bytes(sites)
    area = gpd.read_file(SITES, layer="sites").to_crs(4326).total_bounds
    pad = 0.05
    box = gpd.GeoSeries.from_xy([area[0] - pad, area[2] + pad], [area[1] - pad, area[3] + pad])
    print("waterways ...")
    waterways_geojson(box.union_all().envelope, out)
    print("channels ...")
    overlays = {"channels": channels_png(out)}
    print("hillshade ...")
    overlays["hillshade"] = hillshade_image(out)
    print("suitability ...")
    (out / "suitability.png").unlink(missing_ok=True)  # the single-model file of older runs
    overlays["suitability"] = {m: suitability_png(out, m) for m in MODELS}
    overlays["bare_soil"] = bare_soil_png(out)
    html = (HERE / "viewer_template.html").read_text(encoding="utf-8")
    html = html.replace("/*OVERLAYS*/null", json.dumps(overlays))
    html = html.replace('/*MODE*/"plain"', json.dumps("encrypted" if args.private else "plain"))
    (out / "index.html").write_text(html, encoding="utf-8")
    for f in sorted(out.iterdir()):
        print(f"  {f.name}: {f.stat().st_size // 1024} KB")
    if not args.private:
        print(f"cd {out} && python -m http.server 8001")


if __name__ == "__main__":
    main()
