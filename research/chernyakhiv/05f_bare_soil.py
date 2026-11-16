"""
Stage S, steps S3-S5: is the Sentinel-2 bare-soil composite (03d) worth anything?

S3  darkness index and sanity checks: coverage, agreement with SoilGrids / HWSD
    at ~1 km, confounders (slope, height above the stream, relief position).
S4  as a model feature: same presence-background data and spatial CV as 05e,
    restricted to the area covered by the composite; plus the village level.
S5  hypothesis: are known settlements visible as local dark / light anomalies
    on ploughed fields?  Compared with background points and with the water
    features, because valley sides are darker than plateaus anyway.

Index: `vis_norm` of 03d = visible brightness of bare soil relative to the
scene's median bare field, median over dates. LOWER = DARKER. Relative only.

Outputs: research/chernyakhiv/results/bare_soil_stats.json, bare_soil_*.png;
local only (exact sites): data/processed/chernyakhiv_model/bare_anomaly_<tiles>.tif

Run: .venv/Scripts/python -X utf8 research/chernyakhiv/05f_bare_soil.py --tiles 35UPQ
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.transform import Affine
from scipy import ndimage, stats
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BARE = load_module("03d_bare_soil")
CHECK = load_module("05e_soils")  # spatial_cv, block_ids, univariate, logistic
MODEL, SOIL, GEO = CHECK.MODEL, CHECK.SOIL, CHECK.GEO

PX = 20.0
POINT_PX = 25  # +-240 m, as the other model features
ZONE_M = 1500
SMALL_PX, LARGE_PX = 11, 61  # anomaly: +-100 m against +-600 m
SEARCH_PX = 51  # strongest anomaly within +-500 m (geocoding error)
BLOCK = 50  # 1 km blocks for the comparison with soil maps
WATER = CHECK.WATER
SEAMS: dict[str, dict] = {}  # filled by mosaic(): calibration of each tile on the overlap
CHIP_SOURCE, CHIP_TAG = None, ""  # set in main(): raster for the local chips
FIELD_EDGE = 3.0  # jump in the number of bare dates that marks a field border
MIN_FIELD_PX = 100  # 4 ha
N_MATCH = 5  # background points matched to each site by the water features


# ------------------------------------------------------------------ raster tools
def mosaic(tiles: list[str]) -> tuple[dict[str, np.ndarray], Affine, object]:
    layers, transform, crs, seams = BARE.mosaic(tiles, ARGS.suffix)
    SEAMS.update(seams)
    return layers, transform, crs


def focal_mean(a: np.ndarray, size: int, min_share: float = 0.2):
    """Mean of the finite cells in a size x size window, and the share of such cells."""
    ok = np.isfinite(a)
    num = ndimage.uniform_filter(np.where(ok, a, 0).astype(np.float32), size, mode="constant")
    den = ndimage.uniform_filter(ok.astype(np.float32), size, mode="constant")
    return np.where(den >= min_share, num / np.maximum(den, 1e-6), np.nan), den


def anomaly(a: np.ndarray) -> np.ndarray:
    """Local brightness minus the surroundings; negative = a dark spot."""
    small, _ = focal_mean(a, SMALL_PX, 0.5)
    large, _ = focal_mean(a, LARGE_PX, 0.3)
    return (small - large).astype(np.float32)


def field_residual(dark: np.ndarray, n_dates: np.ndarray) -> np.ndarray:
    """Brightness minus the median of its own field.

    Neighbouring fields differ in brightness for reasons unrelated to soil (crop,
    tillage, which dates they were bare), and that contrast swamps any spot inside
    a field. A field is approximated as a connected patch with an even number of
    bare dates: the count is the same across one field and jumps at its border.
    Patches under MIN_FIELD_PX are dropped.
    """
    ok = np.isfinite(dark)
    n = np.where(ok, n_dates, -100).astype(np.float32)
    n = ndimage.median_filter(n, 5)  # the count flickers by a date or two inside a field
    edge = np.abs(n - ndimage.uniform_filter(n, 3, mode="nearest")) > FIELD_EDGE
    labels, count = ndimage.label(ok & ~edge)
    flat = labels.ravel()
    inside = flat > 0
    frame = pd.DataFrame({"field": flat[inside], "v": dark.ravel()[inside]})
    grouped = frame.groupby("field")["v"]
    median, size = grouped.transform("median"), grouped.transform("size")
    out = np.full(dark.size, np.nan, np.float32)
    out[np.flatnonzero(inside)] = np.where(size >= MIN_FIELD_PX, frame["v"] - median, np.nan)
    print(f"  fields: {count} patches, {int((size >= MIN_FIELD_PX).sum())} px in large ones")
    return out.reshape(dark.shape)


def to_rc(xy: np.ndarray, transform: Affine, crs, shape) -> tuple[np.ndarray, np.ndarray]:
    """EPSG:6381 coordinates -> row/col on the composite grid, and an in-grid flag."""
    tr = Transformer.from_crs(GEO.METRIC_CRS, crs, always_xy=True)
    x, y = tr.transform(xy[:, 0], xy[:, 1])
    cols = np.floor((x - transform.c) / PX).astype(int)
    rows = np.floor((transform.f - y) / PX).astype(int)
    ok = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
    return np.c_[np.clip(rows, 0, shape[0] - 1), np.clip(cols, 0, shape[1] - 1)], ok


def feature_layers(m: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    dark = m["vis_norm"]
    point, cover_point = focal_mean(dark, POINT_PX, 0.1)
    zone_px = int(2 * ZONE_M / PX) + 1
    zone, cover_zone = focal_mean(dark, zone_px, 0.1)
    swir_point, _ = focal_mean(m["swir"], POINT_PX, 0.1)
    anom = anomaly(dark)
    lo = ndimage.minimum_filter(np.where(np.isfinite(anom), anom, np.inf), SEARCH_PX)
    hi = ndimage.maximum_filter(np.where(np.isfinite(anom), anom, -np.inf), SEARCH_PX)
    resid = field_residual(dark, m["n_dates"])
    spot, _ = focal_mean(resid, 5, 0.6)  # +-40 m: a 100 m wide site
    slo = ndimage.minimum_filter(np.where(np.isfinite(spot), spot, np.inf), SEARCH_PX)
    shi = ndimage.maximum_filter(np.where(np.isfinite(spot), spot, -np.inf), SEARCH_PX)
    return {
        "field_resid": resid,
        "spot_point": spot,
        "spot_min500": np.where(np.isfinite(slo), slo, np.nan),
        "spot_max500": np.where(np.isfinite(shi), shi, np.nan),
        "dark_point": point,
        "dark_zone": zone,
        "dark_rel": point - zone,
        "swir_point": swir_point,
        "cover_point": cover_point,
        "cover_zone": cover_zone,
        "anom_point": anom,
        "anom_min500": np.where(np.isfinite(lo), lo, np.nan),
        "anom_max500": np.where(np.isfinite(hi), hi, np.nan),
    }


def sample_layers(layers, xy, transform, crs) -> tuple[pd.DataFrame, np.ndarray]:
    shape = next(iter(layers.values())).shape
    rc, ok = to_rc(xy, transform, crs, shape)
    return pd.DataFrame({k: v[rc[:, 0], rc[:, 1]] for k, v in layers.items()}), ok


# --------------------------------------------------------------------------- S3
def env_at(xy: np.ndarray, names) -> dict[str, np.ndarray]:
    out = {}
    for name in names:
        arr, profile = MODEL.read(name)
        t = profile["transform"]
        c = np.floor((xy[:, 0] - t.c) / t.a).astype(int)
        r = np.floor((t.f - xy[:, 1]) / -t.e).astype(int)
        ok = (r >= 0) & (r < arr.shape[0]) & (c >= 0) & (c < arr.shape[1])
        v = np.full(len(xy), np.nan, np.float32)
        v[ok] = arr[r[ok], c[ok]]
        out[name] = v
    return out


def spearman(a, b) -> dict:
    ok = np.isfinite(a) & np.isfinite(b)
    r = stats.spearmanr(a[ok], b[ok])
    return {"rho": round(float(r.statistic), 3), "n": int(ok.sum())}


def sanity(m, transform, crs, rng) -> dict:
    dark, n = m["vis_norm"], m["n_dates"]
    crop_px = 0
    for t in ARGS.tiles:
        with rasterio.open(BARE.OUT / f"{t}_cropland.tif") as src:
            crop_px += int(src.read(1).sum())
    have = np.isfinite(dark)
    out = {
        "tiles": ARGS.tiles,
        "pixels_with_data_km2": round(float(have.sum()) * PX * PX / 1e6),
        "share_of_cropland_with_data": round(float(have.sum()) / crop_px, 3),
        "share_of_area_with_data": round(float(have.sum()) / float(m["inside"].sum()), 3),
        "bare_dates_per_pixel": {
            f"p{q}": float(np.percentile(n[have], q)) for q in (10, 25, 50, 75, 90)
        },
        "vis_norm_percentiles": {
            f"p{q}": round(float(np.percentile(dark[have], q)), 3) for q in (2, 10, 50, 90, 98)
        },
        "vis_reflectance_percentiles": {
            f"p{q}": round(float(np.nanpercentile(m["vis"][have], q)), 4)
            for q in (2, 10, 50, 90, 98)
        },
    }
    # 1 km blocks against the soil maps
    h, w = (s // BLOCK * BLOCK for s in dark.shape)
    blocks = dark[:h, :w].reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK)
    share = np.isfinite(blocks).mean(axis=(1, 3))
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(blocks, axis=(1, 3))
    bi, bj = np.nonzero(share >= 0.2)
    x = transform.c + (bj + 0.5) * BLOCK * PX
    y = transform.f - (bi + 0.5) * BLOCK * PX
    bx, by = Transformer.from_crs(crs, GEO.METRIC_CRS, always_xy=True).transform(x, y)
    bxy = np.c_[bx, by]
    val = mean[bi, bj]
    soil = SOIL.sample(bxy)
    env = env_at(bxy, ["soil", "slope", "hand", "elevation", "tpi1000"])
    inside = np.isfinite(env["elevation"])
    out["blocks_1km"] = int(inside.sum())
    out["block_spearman"] = {
        k: spearman(val[inside], np.asarray(v, float)[inside])
        for k, v in {**soil, **env}.items()
        if not k.endswith("_zone") and k != "soil"
    }
    wrb = np.c_[[np.asarray(soil[f"wrb_{g}"], float) for g in SOIL.WRB_GROUPS]].T
    top = np.array(SOIL.WRB_GROUPS)[np.nanargmax(np.nan_to_num(wrb, nan=-1), axis=1)]
    out["block_by_soilgrids_class"] = {
        g: {
            "n": int((inside & (top == g)).sum()),
            "median_vis_norm": round(float(np.median(val[inside & (top == g)])), 3),
        }
        for g in SOIL.WRB_GROUPS
        if (inside & (top == g)).sum() >= 20
    }
    lu, dk = inside & (top == "Luvisols"), inside & np.isin(top, ["Chernozems", "Phaeozems"])
    if lu.sum() >= 20 and dk.sum() >= 20:
        out["auc_luvisol_vs_chernozem_phaeozem"] = round(
            float(
                roc_auc_score(np.r_[np.ones(lu.sum()), np.zeros(dk.sum())], np.r_[val[lu], val[dk]])
            ),
            3,
        )
    hw = {0: "other", 1: "chernozems", 2: "podzolized_chernozems", 3: "grey_forest"}
    out["block_by_hwsd_class"] = {
        name: {
            "n": int((inside & (env["soil"] == k)).sum()),
            "median_vis_norm": round(float(np.median(val[inside & (env["soil"] == k)])), 3),
        }
        for k, name in hw.items()
        if (inside & (env["soil"] == k)).sum() >= 20
    }
    gf, ch = inside & (env["soil"] == 3), inside & (env["soil"] == 1)
    if gf.sum() >= 20 and ch.sum() >= 20:
        out["auc_hwsd_grey_forest_vs_chernozem"] = round(
            float(
                roc_auc_score(np.r_[np.ones(gf.sum()), np.zeros(ch.sum())], np.r_[val[gf], val[ch]])
            ),
            3,
        )
    # pixel level: what else drives the brightness
    pi, pj = np.nonzero(have)
    pick = rng.choice(len(pi), size=min(300_000, len(pi)), replace=False)
    px = transform.c + (pj[pick] + 0.5) * PX
    py = transform.f - (pi[pick] + 0.5) * PX
    qx, qy = Transformer.from_crs(crs, GEO.METRIC_CRS, always_xy=True).transform(px, py)
    env = env_at(np.c_[qx, qy], ["slope", "hand", "tpi300", "tpi1000", "rel_valley"])
    v = dark[pi[pick], pj[pick]]
    out["pixel_spearman"] = {k: spearman(v, e) for k, e in env.items()}
    out["pixel_spearman"]["n_dates"] = spearman(v, n[pi[pick], pj[pick]])
    out["pixel_spearman"]["swir"] = spearman(v, m["swir"][pi[pick], pj[pick]])
    return out


def pictures(m, tag: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    step = 10  # 200 m
    with np.errstate(invalid="ignore"):
        h, w = (s // step * step for s in m["vis_norm"].shape)
        d = np.nanmean(m["vis_norm"][:h, :w].reshape(h // step, step, w // step, step), axis=(1, 3))
        n = m["n_dates"][:h, :w].reshape(h // step, step, w // step, step).mean(axis=(1, 3))
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    lo, hi = np.nanpercentile(d, [2, 98])
    im = axes[0].imshow(d, cmap="cividis", vmin=lo, vmax=hi)
    axes[0].set_title("Яскравість голої ріллі (vis_norm; темніше = темніший ґрунт), 200 м")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    im = axes[1].imshow(n, cmap="viridis", vmin=0, vmax=np.percentile(n, 98))
    axes[1].set_title("Кількість дат з голим ґрунтом")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    for ax in axes:
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(RESULTS / f"bare_soil_{tag}.png", dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- S4
def matched_difference(data: pd.DataFrame, y, col: str, by=None) -> dict:
    """Site value minus the mean of N_MATCH background points with the same water
    setting (catchment size, distance to a channel). Valley sides are darker than
    plateaus anyway; this asks whether sites differ from OTHER valley sides."""
    ok = data[col].notna().to_numpy()
    w = data[by or WATER].to_numpy(dtype=float)
    ok = ok & np.isfinite(w).all(axis=1)
    w = (w - w[y == 0].mean(axis=0)) / w[y == 0].std(axis=0)
    bg, st = ok & (y == 0), ok & (y == 1)
    if bg.sum() < 50 or st.sum() < 20:
        return {"n_sites": int(st.sum())}
    _, idx = cKDTree(w[bg]).query(w[st], k=N_MATCH)
    v = data[col].to_numpy(dtype=float)
    diff = v[st] - v[bg][idx].mean(axis=1)
    test = stats.wilcoxon(diff)
    return {
        "n_sites": int(st.sum()),
        "median_difference": round(float(np.median(diff)), 4),
        "share_sites_lower": round(float((diff < 0).mean()), 3),
        "p_wilcoxon": float(f"{test.pvalue:.2g}"),
        "background_sd": round(float(v[bg].std()), 4),
    }


def chips(data: pd.DataFrame, xy: np.ndarray, y, n: int = 24, half_m: int = 1000) -> None:
    """LOCAL ONLY (exact sites): brightness around the settlements with the strongest
    dark spot nearby, to look at them by eye. Saved next to the model rasters."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dark, transform, crs = CHIP_SOURCE
    sites = data[(y == 1) & data["spot_min500"].notna()].sort_values("spot_min500").head(n)
    rc, _ = to_rc(xy[sites.index], transform, crs, dark.shape)
    half = int(half_m / PX)
    fig, axes = plt.subplots(4, 6, figsize=(18, 12))
    for ax, (r, c), (_, row) in zip(axes.ravel(), rc, sites.iterrows(), strict=False):
        win = dark[max(r - half, 0) : r + half, max(c - half, 0) : c + half]
        lo, hi = -0.15, 0.15
        ax.imshow(win, cmap="gray", vmin=lo, vmax=hi)
        ax.plot(min(half, c), min(half, r), "+", color="red", ms=14)
        ax.add_patch(plt.Circle((min(half, c), min(half, r)), 500 / PX, fill=False, color="red"))
        ax.set_title(f"пляма {row['spot_min500']:.3f}, snap={row['snap']}", fontsize=9)
        ax.set_axis_off()
    fig.suptitle(
        "Відхилення яскравості від медіани свого поля довкола поселень (2 × 2 км; коло 500 м) — НЕ ПУБЛІКУВАТИ"
    )
    fig.tight_layout()
    fig.savefig(MODEL.OUT_LOCAL / f"bare_anomaly_chips_{CHIP_TAG}.png", dpi=100)
    plt.close(fig)


def site_level(layers, transform, crs, rng) -> dict:
    ds = MODEL.build_dataset(rng)
    data, xy = ds["data"], ds["xy"]
    feats, ok = sample_layers(layers, xy, transform, crs)
    data = pd.concat([data, feats], axis=1)[ok].reset_index(drop=True)
    xy = xy[ok]
    y = data["y"].to_numpy()
    cols = [c for c in layers if c != "field_resid"]
    cover = {c: round(float(data.loc[y == 1, c].notna().mean()), 3) for c in cols}
    cover_bg = {c: round(float(data.loc[y == 0, c].notna().mean()), 3) for c in cols}
    uni = CHECK.univariate(data, y, cols)
    filled = data.copy()
    filled[cols] = filled[cols].fillna(filled[cols].median())
    sets = {
        "water_only": WATER,
        "water+dark_point": [*WATER, "dark_point"],
        "water+dark_zone": [*WATER, "dark_zone"],
        "water+dark_point+zone": [*WATER, "dark_point", "dark_zone"],
        "water+dark_rel": [*WATER, "dark_rel"],
        "water+cover": [*WATER, "cover_point", "cover_zone"],
        "water+dark+cover": [*WATER, "dark_point", "dark_zone", "cover_point", "cover_zone"],
        "water+anomaly": [*WATER, "anom_min500", "anom_max500"],
        "dark_only": ["dark_point", "dark_zone"],
    }
    free = (data["snap"] == "none").to_numpy() | (y == 0)
    blocks = CHECK.block_ids(xy)
    out = {
        "n_sites": int(y.sum()),
        "n_background": int((y == 0).sum()),
        "share_with_value_sites": cover,
        "share_with_value_background": cover_bg,
        "univariate": uni,
        "logistic_cv": CHECK.spatial_cv(filled, y, blocks, sets, CHECK.logistic, rng, free),
        "boosting_cv": CHECK.spatial_cv(
            filled,
            y,
            blocks,
            {k: sets[k] for k in ("water_only", "water+dark+cover", "water+anomaly")},
            CHECK.boosting,
            rng,
        ),
    }
    # S5: anomalies only where the point itself is on a field with data, by snap quality
    on_field = data["anom_point"].notna().to_numpy()
    s5 = {}
    for label, sel in {
        "all": np.ones(len(y), bool),
        "snapped_to_bank": (data["snap"] != "none").to_numpy(),
    }.items():
        pick = on_field & (sel | (y == 0))
        s5[label] = {
            "n_sites": int((pick & (y == 1)).sum()),
            "n_background": int((pick & (y == 0)).sum()),
            **CHECK.univariate(
                data[pick].reset_index(drop=True),
                y[pick],
                ["anom_point", "anom_min500", "anom_max500", "dark_rel", "spot_point"],
            ),
        }
    out["anomaly_on_fields"] = s5
    out["matched_by_water"] = {
        c: matched_difference(data, y, c)
        for c in [
            "dark_point",
            "dark_zone",
            "dark_rel",
            "anom_point",
            "anom_min500",
            "anom_max500",
            "spot_point",
            "spot_min500",
            "spot_max500",
        ]
    }
    relief = [*WATER, "hand_ch5", "rel_valley", "slope_mean", "tpi300"]
    out["matched_by_water_and_relief"] = {
        c: matched_difference(data, y, c, relief)
        for c in [
            "dark_point",
            "dark_rel",
            "anom_min500",
            "spot_point",
            "spot_min500",
            "spot_max500",
        ]
    }
    full = [*WATER, "hand_ch5", "rel_valley", "slope_mean", "tpi300", "tpi1000"]
    sets_relief = {
        "water+relief": full,
        "water+relief+dark_rel": [*full, "dark_rel"],
        "water+relief+anomaly": [*full, "anom_min500"],
        "water+relief+dark_rel+anomaly": [*full, "dark_rel", "anom_min500"],
        "water+relief+spot": [*full, "spot_min500", "spot_max500"],
    }
    out["logistic_cv_relief"] = CHECK.spatial_cv(
        filled, y, blocks, sets_relief, CHECK.logistic, rng, free
    )
    chips(data, xy, y)
    return out


def village_level(layers, transform, crs, rng) -> dict:
    with_sites = gpd.read_file(CHECK.PUBLIC_VILLAGES).to_crs(GEO.METRIC_CRS).geometry
    oblast, _ = GEO.load_admin()
    places = GEO.load_places(oblast)
    places = places[places.within(oblast) & places["place"].isin(["village", "hamlet", "town"])]
    pts = places.to_crs(GEO.METRIC_CRS).geometry
    d, _ = cKDTree(np.c_[with_sites.x, with_sites.y]).query(np.c_[pts.x, pts.y])
    other = pts[d > 200]
    xy = np.r_[np.c_[with_sites.x, with_sites.y], np.c_[other.x, other.y]]
    y = np.r_[np.ones(len(with_sites)), np.zeros(len(other))].astype(int)
    zone_px = int(2 * CHECK.VILLAGE_ZONE_M / PX) + 1
    zone, cover = focal_mean(layers["_dark"], zone_px, 0.1)
    data, ok = sample_layers({"dark_zone": zone, "cover_zone": cover}, xy, transform, crs)
    ok &= layers["_inside"][tuple(to_rc(xy, transform, crs, zone.shape)[0].T)]
    data, y, xy = data[ok].reset_index(drop=True), y[ok], xy[ok]
    uni = CHECK.univariate(data, y, ["dark_zone", "cover_zone"])
    data = data.fillna(data.median())
    sets = {"dark_zone": ["dark_zone"], "dark+cover": ["dark_zone", "cover_zone"]}
    return {
        "n_villages_with_sites": int(y.sum()),
        "n_other_villages": int((y == 0).sum()),
        "zone_m": CHECK.VILLAGE_ZONE_M,
        "univariate": uni,
        "logistic_cv": CHECK.spatial_cv(data, y, CHECK.block_ids(xy), sets, CHECK.logistic, rng),
    }


def main() -> None:
    rng = np.random.default_rng(MODEL.SEED)
    tag = "_".join(ARGS.tiles) + ARGS.suffix
    RESULTS.mkdir(exist_ok=True)
    m, transform, crs = mosaic(ARGS.tiles)
    print("mosaic", m["vis_norm"].shape)
    out = {"seams": SEAMS, "sanity": sanity(m, transform, crs, rng)}
    print(json.dumps(out["sanity"], ensure_ascii=False, indent=1))
    pictures(m, tag)
    layers = feature_layers(m)
    MODEL.OUT_LOCAL.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        MODEL.OUT_LOCAL / f"bare_anomaly_{tag}.tif",
        "w",
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=np.nan,
        compress="deflate",
        tiled=True,
        crs=crs,
        transform=transform,
        height=m["vis_norm"].shape[0],
        width=m["vis_norm"].shape[1],
    ) as dst:
        dst.write(layers["anom_point"], 1)
    global CHIP_SOURCE, CHIP_TAG  # noqa: PLW0603
    CHIP_SOURCE, CHIP_TAG = (layers["field_resid"], transform, crs), tag
    out["site_level"] = site_level(layers, transform, crs, rng)
    layers["_dark"], layers["_inside"] = m["vis_norm"], m["inside"]
    out["village_level"] = village_level(layers, transform, crs, rng)
    (RESULTS / f"bare_soil_stats_{tag}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {k: out[k] for k in ("site_level", "village_level")}, ensure_ascii=False, indent=1
        )
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", nargs="+", default=["35UPQ"])
    ap.add_argument("--suffix", default="", help="composite variant, e.g. _pre")
    ARGS = ap.parse_args()
    main()
