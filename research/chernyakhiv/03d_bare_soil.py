"""
Stage S (pilot): bare-soil composite from Sentinel-2 as a MEASURED proxy for
soil darkness (humus), instead of soil maps that contradict each other (soils.md).

Idea: the more humus, the darker a bare ploughed field. From all Sentinel-2 scenes
of several years keep only the moments when a cropland pixel is bare and take the
per-pixel median reflectance ("bare soil composite", cf. SCMaP / GEOS3). The
result is a relative index, not a soil class and not a humus percentage.

Sources (no registration):
    Sentinel-2 L2A, Collection 1: STAC https://earth-search.aws.element84.com/v1,
        collection sentinel-2-c1-l2a (COGs on AWS; DN = reflectance * 10000 + 1000).
        2022 is nearly absent from the collection and is skipped.
    Cropland mask: ESA WorldCover 2021 v200, class 40 (COGs on AWS).

Steps:
    fetch      per scene: 6 bands + SCL read at 20 m (10 m bands through their
               20 m overview), pixels outside a LOOSE bare-cropland mask zeroed,
               written to env/bare_soil/scenes/<tile>/<id>.tif (~15-40 MB each).
               Loose thresholds are kept so the strict ones can be tuned offline.
    composite  strict mask, per-pixel median over bare dates ->
               env/bare_soil/<tile>_bare.tif, native UTM grid, 20 m, bands:
               B02 B03 B04 B08 B11 B12 (reflectance), vis_norm, n_dates.
               vis_norm = median over dates of visible brightness divided by the
               scene's median bare-cropland brightness: removes scene-wide
               differences (moisture after rain, sun, haze) that otherwise print
               the mosaic of acquisition dates onto the map.

Run:  .venv/Scripts/python -X utf8 research/chernyakhiv/03d_bare_soil.py all --tiles 35UPQ
Data: Contains modified Copernicus Sentinel data; (c) ESA WorldCover project 2021.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "data" / "processed" / "env" / "bare_soil"

STAC = "https://earth-search.aws.element84.com/v1/search"
COLLECTION = "sentinel-2-c1-l2a"
WORLDCOVER = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
    "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
)
CROPLAND = 40

YEARS = (2018, 2019, 2020, 2021, 2023, 2024, 2025, 2026)
MONTHS = (3, 4, 5, 8, 9, 10, 11)
MAX_CLOUD = 20
BANDS = {  # name in the composite -> STAC asset
    "B02": "blue",
    "B03": "green",
    "B04": "red",
    "B08": "nir",
    "B11": "swir16",
    "B12": "swir22",
}
SIZE = 5490  # 20 m pixels per MGRS tile side
DN_OFFSET = 1000
DN_SCALE = 10000.0

# Scene classification (SCL): 2 dark area, 4 vegetation, 5 not vegetated,
# 7 unclassified are kept (dark bare chernozem is often labelled 2 or 7);
# 3 cloud shadow, 8-10 clouds/cirrus are dilated and removed; 6 water, 11 snow out.
SCL_KEEP = (2, 4, 5, 7)
SCL_CLOUD = (3, 8, 9, 10)
CLOUD_DILATE_PX = 5  # 100 m
LOOSE = {"ndvi_max": 0.35, "nbr2_max": 0.20}  # stored on disk
STRICT = {"ndvi_max": 0.25, "nbr2_max": 0.10}  # used for the composite
MIN_DATES = 3

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "3")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "120")


# ---------------------------------------------------------------- pure functions
def reflectance(dn: np.ndarray) -> np.ndarray:
    """Collection-1 digital numbers -> surface reflectance; 0 (nodata) -> NaN."""
    out = (dn.astype(np.float32) - DN_OFFSET) / DN_SCALE
    out[dn == 0] = np.nan
    return out


def indices(red, nir, swir1, swir2):
    """NDVI and NBR2 from reflectance (or DN with the offset removed)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (nir - red) / (nir + red)
        nbr2 = (swir1 - swir2) / (swir1 + swir2)
    return ndvi, nbr2


def bare_mask(red, nir, swir1, swir2, ndvi_max: float, nbr2_max: float) -> np.ndarray:
    """Bare, dry, residue-free soil: low NDVI and low NBR2. NaN input -> False.

    NBR2 = (B11 - B12) / (B11 + B12) rises with crop residues and soil moisture,
    so the cap removes stubble and wet fields that NDVI alone lets through.
    """
    ndvi, nbr2 = indices(red, nir, swir1, swir2)
    with np.errstate(invalid="ignore"):
        return (ndvi < ndvi_max) & (ndvi > -0.05) & (nbr2 < nbr2_max) & (nbr2 > -0.15)


def clear_mask(scl: np.ndarray, dilate_px: int = CLOUD_DILATE_PX) -> np.ndarray:
    """Pixels of usable SCL classes, away from clouds and their shadows."""
    cloud = np.isin(scl, SCL_CLOUD)
    if dilate_px and cloud.any():
        cloud = ndimage.binary_dilation(cloud, iterations=dilate_px)
    return np.isin(scl, SCL_KEEP) & ~cloud


def masked_median(stack: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Median along axis 0 over valid entries. Returns (median float32, count).

    stack: (n, h, w) numeric; valid: (n, h, w) bool. Pixels with no valid entry
    get NaN. Even counts average the two middle values.
    """
    n = valid.sum(axis=0)
    data = np.where(valid, stack, np.inf).astype(np.float32)
    data.sort(axis=0)
    lo = np.clip((n - 1) // 2, 0, None)[None]
    hi = (n // 2)[None]
    hi = np.minimum(hi, stack.shape[0] - 1)
    med = (np.take_along_axis(data, lo, 0)[0] + np.take_along_axis(data, hi, 0)[0]) / 2
    med[n == 0] = np.nan
    return med, n.astype(np.int32)


# ---------------------------------------------------------------------- catalogue
def _post(body: dict) -> dict:
    req = urllib.request.Request(
        STAC, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except Exception as exc:  # noqa: BLE001 - network: retry
            if attempt == 4:
                raise
            print(f"  STAC retry {attempt + 1}: {exc}")
            time.sleep(10 * (attempt + 1))
    raise RuntimeError("unreachable")


def search_scenes(tile: str) -> list[dict]:
    """Scenes of one MGRS tile in the bare-soil seasons, sorted by date."""
    body = {
        "collections": [COLLECTION],
        "query": {"grid:code": {"eq": f"MGRS-{tile}"}, "eo:cloud_cover": {"lt": MAX_CLOUD}},
        "datetime": f"{min(YEARS)}-01-01T00:00:00Z/{max(YEARS)}-12-31T23:59:59Z",
        "limit": 200,
    }
    items = []
    while True:
        js = _post(body)
        items += js["features"]
        nxt = [ln for ln in js.get("links", []) if ln.get("rel") == "next"]
        if not nxt or not js["features"]:
            break
        body = nxt[0].get("body") or body
    keep = {}
    for it in items:
        dt = it["properties"]["datetime"]
        if int(dt[:4]) in YEARS and int(dt[5:7]) in MONTHS:
            keep[it["id"]] = it
    return sorted(keep.values(), key=lambda i: i["properties"]["datetime"])


# -------------------------------------------------------------------------- fetch
def tile_profile(item: dict) -> dict:
    with rasterio.open(item["assets"]["scl"]["href"]) as src:
        assert src.width == SIZE and src.height == SIZE, (src.width, src.height)
        return {"crs": src.crs, "transform": src.transform, "width": SIZE, "height": SIZE}


def cropland_mask(tile: str, profile: dict) -> np.ndarray:
    """WorldCover cropland on the tile grid (majority of the 10 m pixels)."""
    path = OUT / f"{tile}_cropland.tif"
    if path.exists():
        with rasterio.open(path) as src:
            return src.read(1).astype(bool)
    left, top = profile["transform"] * (0, 0)
    right, bottom = profile["transform"] * (SIZE, SIZE)
    w, s, e, n = transform_bounds(profile["crs"], "EPSG:4326", left, bottom, right, top)
    out = np.zeros((SIZE, SIZE), np.uint8)
    for lat in range(int(s // 3) * 3, int(n // 3) * 3 + 1, 3):
        for lon in range(int(w // 3) * 3, int(e // 3) * 3 + 1, 3):
            url = WORLDCOVER.format(tile=f"N{lat:02d}E{lon:03d}")
            part = np.zeros((SIZE, SIZE), np.uint8)
            print(f"  WorldCover N{lat:02d}E{lon:03d} ...")
            with rasterio.open(url) as src:
                reproject(
                    rasterio.band(src, 1),
                    part,
                    dst_transform=profile["transform"],
                    dst_crs=profile["crs"],
                    resampling=Resampling.mode,
                )
            out = np.where(part > 0, part, out)
    crop = out == CROPLAND
    OUT.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", dtype="uint8", count=1, compress="deflate", **profile
    ) as dst:
        dst.write(crop.astype(np.uint8), 1)
    return crop


def _read_20m(href: str) -> np.ndarray:
    for attempt in range(4):
        try:
            with rasterio.open(href) as src:
                return src.read(1, out_shape=(SIZE, SIZE))
        except Exception as exc:  # noqa: BLE001 - network: retry
            if attempt == 3:
                raise
            print(f"  read retry {attempt + 1}: {exc}")
            time.sleep(15 * (attempt + 1))
    raise RuntimeError("unreachable")


def fetch_scene(item: dict, crop: np.ndarray, profile: dict, folder: Path) -> str:
    path = folder / f"{item['id']}.tif"
    if path.exists():
        return "cached"
    keys = [*BANDS.values(), "scl"]
    with ThreadPoolExecutor(len(keys)) as pool:
        hrefs = [item["assets"][k]["href"] for k in keys]
        arrays = dict(zip(keys, pool.map(_read_20m, hrefs), strict=True))
    scl = arrays.pop("scl")
    dn = np.stack([arrays[k] for k in BANDS.values()])
    del arrays
    ok = crop & clear_mask(scl) & (dn > 0).all(axis=0)
    r = [dn[i].astype(np.float32) - DN_OFFSET for i in range(6)]
    ok &= bare_mask(r[2], r[3], r[4], r[5], **LOOSE)
    strict = ok & bare_mask(r[2], r[3], r[4], r[5], **STRICT)
    vis = (r[0] + r[1] + r[2]) / 3 / DN_SCALE
    vis_median = float(np.median(vis[strict])) if strict.sum() > 1000 else float("nan")
    del r, vis
    dn[:, ~ok] = 0
    tmp = path.with_suffix(f".{os.getpid()}.part")  # several processes may fetch in parallel
    with rasterio.open(
        tmp,
        "w",
        driver="GTiff",
        dtype="uint16",
        count=6,
        nodata=0,
        compress="deflate",
        predictor=2,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        **profile,
    ) as dst:
        dst.write(dn)
        dst.update_tags(
            datetime=item["properties"]["datetime"],
            vis_median=repr(vis_median),
            loose_px=int(ok.sum()),
            strict_px=int(strict.sum()),
        )
    tmp.replace(path)
    return f"loose {ok.mean():.1%}, strict {strict.mean():.1%}, vis {vis_median:.3f}"


def fetch(tile: str, workers: int = 3, limit: int = 0, reverse: bool = False) -> None:
    folder = OUT / "scenes" / tile
    folder.mkdir(parents=True, exist_ok=True)
    items = search_scenes(tile)
    if limit:
        items = items[:: max(1, len(items) // limit)][:limit]
    print(f"{tile}: {len(items)} scenes", flush=True)
    profile = tile_profile(items[0])
    if reverse:
        items = items[::-1]
    crop = cropland_mask(tile, profile)
    print(f"  cropland {crop.mean():.1%} of the tile")
    t0 = time.time()

    def job(arg):
        i, item = arg
        try:
            msg = fetch_scene(item, crop, profile, folder)
        except Exception as exc:  # noqa: BLE001 - keep going, report at the end
            msg = f"FAILED {exc}"
        print(f"  [{i + 1}/{len(items)}] {item['id']} {msg} ({time.time() - t0:.0f}s)", flush=True)
        return msg

    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(job, enumerate(items)))
    failed = sum(m.startswith("FAILED") for m in results)
    print(f"{tile}: fetched in {time.time() - t0:.0f}s, failed {failed}")


# ---------------------------------------------------------------------- composite
def composite(tile: str, strip: int = 48, strict: dict | None = None, suffix: str = "") -> Path:
    strict = strict or STRICT
    files = sorted((OUT / "scenes" / tile).glob("*.tif"))
    scenes = []
    for f in files:
        with rasterio.open(f) as src:
            tags = src.tags()
            profile = {"crs": src.crs, "transform": src.transform, "width": SIZE, "height": SIZE}
        vm = float(tags.get("vis_median", "nan"))
        if np.isfinite(vm) and int(tags.get("strict_px", 0)) > 20000:
            scenes.append((f, vm))
    print(f"{tile}: compositing {len(scenes)} of {len(files)} scenes")
    names = [*BANDS, "vis_norm", "n_dates"]
    path = OUT / f"{tile}_bare{suffix}.tif"
    t0 = time.time()
    handles = [rasterio.open(f) for f, _ in scenes]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        dtype="float32",
        count=len(names),
        nodata=np.nan,
        compress="deflate",
        predictor=3,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        **profile,
    ) as dst:
        for b, name in enumerate(names, 1):
            dst.set_band_description(b, name)
        for row in range(0, SIZE, strip):
            h = min(strip, SIZE - row)
            win = Window(0, row, SIZE, h)
            stack = np.zeros((len(scenes), 6, h, SIZE), np.uint16)
            for i, src in enumerate(handles):
                stack[i] = src.read(window=win)
            valid = (stack > 0).all(axis=1)
            valid &= bare_mask(
                *(stack[:, k].astype(np.float32) - DN_OFFSET for k in (2, 3, 4, 5)), **strict
            )
            out = np.full((len(names), h, SIZE), np.nan, np.float32)
            n = valid.sum(axis=0)
            for k in range(6):
                med, _ = masked_median(stack[:, k], valid)
                out[k] = (med - DN_OFFSET) / DN_SCALE
            vis = (stack[:, :3].astype(np.float32).mean(axis=1) - DN_OFFSET) / DN_SCALE
            vis /= np.array([vm for _, vm in scenes], np.float32)[:, None, None]
            out[6], _ = masked_median(vis, valid)
            out[:7, n < MIN_DATES] = np.nan
            out[7] = n
            dst.write(out, window=win)
            del stack, valid, vis, out
        dst.update_tags(
            scenes=len(scenes), **{k: str(v) for k, v in strict.items()}, min_dates=MIN_DATES
        )
    for src in handles:
        src.close()
    print(f"{tile}: {path.name} in {time.time() - t0:.0f}s")
    return path


# ------------------------------------------------------------- mosaic and model grid
def mosaic(tiles: list[str], suffix: str = ""):
    """Composites of several tiles on one UTM grid: vis_norm, vis (mean B02-B04),
    swir (mean B11-B12), n_dates, inside. Returns (layers, transform, crs, seams).

    vis_norm is normalised by each tile's own scenes, so every new tile is scaled
    to the mosaic on their ~10 km overlap; `seams` keeps the factors and the rank
    correlation in the overlap (an independent repeatability check).
    """
    from scipy import stats

    px = 20.0
    srcs = [rasterio.open(OUT / f"{t}_bare{suffix}.tif") for t in tiles]
    crs = srcs[0].crs
    assert all(s.crs == crs for s in srcs)
    left = min(s.bounds.left for s in srcs)
    top = max(s.bounds.top for s in srcs)
    w = round((max(s.bounds.right for s in srcs) - left) / px)
    h = round((top - min(s.bounds.bottom for s in srcs)) / px)
    names = ["vis_norm", "vis", "swir"]
    total = {n: np.zeros((h, w), np.float32) for n in names}
    n_dates = np.zeros((h, w), np.float32)
    count = np.zeros((h, w), np.float32)
    inside = np.zeros((h, w), bool)
    seams = {}
    for tile, s in zip(tiles, srcs, strict=True):
        r0, c0 = round((top - s.bounds.top) / px), round((s.bounds.left - left) / px)
        sl = (slice(r0, r0 + s.height), slice(c0, c0 + s.width))
        layers = {
            "vis_norm": s.read(7),
            "vis": s.read([1, 2, 3]).mean(axis=0),
            "swir": s.read([5, 6]).mean(axis=0),
        }
        ok = np.isfinite(layers["vis_norm"])
        both = ok & (count[sl] > 0)
        seam = {"overlap_km2": round(float(both.sum()) * px * px / 1e6)}
        if both.sum() > 10_000:
            for n in names:
                have = total[n][sl][both] / count[sl][both]
                ratio = float(np.median(have / layers[n][both]))
                if n == "vis_norm":
                    pick = slice(None, None, max(1, int(both.sum()) // 200_000))
                    rho = stats.spearmanr(have[pick], layers[n][both][pick]).statistic
                    seam["spearman_vis_norm"] = round(float(rho), 3)
                seam[f"ratio_{n}"] = round(ratio, 3)
                layers[n] = layers[n] * ratio
        seams[tile] = seam
        for n in names:
            total[n][sl] += np.where(ok, layers[n], 0)
        n_dates[sl] = np.maximum(n_dates[sl], np.nan_to_num(s.read(8)))
        count[sl] += ok
        inside[sl] = True
        s.close()
        del layers
    out = {n: np.where(count > 0, total[n] / np.maximum(count, 1), np.nan) for n in names}
    out["n_dates"] = n_dates
    out["inside"] = inside
    return out, rasterio.Affine(px, 0, left, 0, -px, top), crs, seams


def grid(tiles: list[str]) -> None:
    """Keep the measurement for later models: the mosaic on the 30 m EPSG:6381 grid
    of the other environment rasters -> env/bare_<layer>.tif (MODEL.read-compatible)."""
    env = OUT.parent
    with rasterio.open(env / "elevation.tif") as src:
        profile = src.profile
    layers, transform, crs, seams = mosaic(tiles)
    print("seams:", json.dumps(seams))
    for name in ("vis_norm", "vis", "swir", "n_dates"):
        dst = np.full((profile["height"], profile["width"]), np.nan, np.float32)
        reproject(
            layers[name].astype(np.float32),
            dst,
            src_transform=transform,
            src_crs=crs,
            src_nodata=np.nan,
            dst_transform=profile["transform"],
            dst_crs=profile["crs"],
            dst_nodata=np.nan,
            resampling=Resampling.average,
        )
        out = {**profile, "dtype": "float32", "nodata": -9999.0, "compress": "deflate"}
        with rasterio.open(env / f"bare_{name}.tif", "w", **out) as f:
            f.write(np.nan_to_num(dst, nan=-9999.0), 1)
            f.update_tags(tiles=" ".join(tiles), source="03d_bare_soil.py grid")
        print(f"  env/bare_{name}.tif: data on {np.isfinite(dst).mean():.1%} of the grid")
    (OUT / "seams.json").write_text(json.dumps(seams, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["fetch", "composite", "all", "grid"])
    ap.add_argument("--tiles", nargs="+", default=["35UPQ"])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N scenes")
    ap.add_argument("--reverse", action="store_true", help="newest first (a second process)")
    args = ap.parse_args()
    if args.step == "grid":
        grid(args.tiles)
        return
    for tile in args.tiles:
        if args.step in ("fetch", "all"):
            fetch(tile, args.workers, args.limit, args.reverse)
        if args.step in ("composite", "all"):
            composite(tile)


if __name__ == "__main__":
    main()
