"""
Stage D: model v2 - the features of literature.md plus Sentinel-2 soil darkness,
against the catchment-only model of 05c on the same points and spatial CV.

New features (30 m rasters; background points are the same as in 05c):
    sun_ch1, sun_ch5   sunny side of the valley: +1 = the cell lies due north of
                       the nearest channel (>= 1 / 5 km2), so its valley side
                       faces south; -1 = faces north. At the point: the side of
                       the stream is exactly what the catalogue text gives.
    terrace_share      share of the +-240 m window that is a gentle valley side:
                       slope 1-6 deg, 50-400 m from a channel >= 1 km2, 3-20 m above it
    floodplain_share   share of the window that is flat (< 1.5 deg) and < 2.5 m
                       above that channel
    dist_head          min distance in the window to a channel head (where the
                       catchment first reaches 1 km2), capped
    depth1000          how far below the highest ground within 1 km (hiddenness)
    sg_*_zone          SoilGrids in the 1.5 km resource zone: chernozems +
                       phaeozems, luvisols, fluvisols (probability, %), sand
    dark_*, cover_*    bare-soil brightness (03d): window, 1.5 km zone, window
                       minus zone, and the share of cells that have a value
                       (arable land). Gaps (forest, villages) are filled from
                       the zone, then with the oblast median.

Acceptance (PLAN.md): village-level AUC > 0.61 and AUC > 0.5 for settlements
with no water in the description; otherwise the catchment-only model stays.

Outputs: research/chernyakhiv/results/model_v2_stats.json, model_v2_importance.png
Run:     .venv/Scripts/python -X utf8 research/chernyakhiv/05g_model_v2.py
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from rasterio.warp import Resampling, reproject
from scipy import ndimage
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CHECK = load_module("05e_soils")  # univariate, block_ids, boosting
MODEL, SOIL = CHECK.MODEL, CHECK.SOIL

N_REPEATS, N_FOLDS = 5, 5
N_MAP_REPEATS, N_BOOT = 5, 2000
ZONE_CELLS = 101  # +-1.5 km at 30 m
HEAD_KM2 = (1.0, 1.5)  # a channel cell this small is next to the channel's head
TERRACE = {"slope": (1.0, 6.0), "dist": (50.0, 400.0), "hand": (3.0, 20.0)}
FLOODPLAIN = {"slope": 1.5, "hand": 2.5}

WATER = ["facc_max", "dist_ch5"]
GROUPS = {
    "sun": ["sun_ch1", "sun_ch5"],
    "terrace": ["terrace_share", "floodplain_share", "hand_ch5", "slope_mean"],
    "heads": ["dist_head"],
    "hidden": ["depth1000", "tpi1000"],
    "soil": ["sg_dark_zone", "sg_luvisol_zone", "sg_fluvisol_zone", "sg_sand_zone"],
    "dark": ["dark_point", "dark_zone", "dark_rel", "cover_point", "cover_zone"],
}
V2_NO_DARK = WATER + [c for g, cols in GROUPS.items() if g != "dark" for c in cols]
V2 = V2_NO_DARK + GROUPS["dark"]
V1_KEEP = ["facc_max", "dist_ch5", "hand_ch5", "slope_mean", "tpi1000"]
LABELS = {
    **MODEL.LABELS,
    "sun_ch1": "Сонячний бік долини (водотік ≥ 1 км²)",
    "sun_ch5": "Сонячний бік долини (водотік ≥ 5 км²)",
    "terrace_share": "Частка пологого борту долини у вікні",
    "floodplain_share": "Частка заплави у вікні",
    "dist_head": "Відстань до витоку водотоку, м",
    "depth1000": "Глибина відносно найвищого місця в 1 км, м",
    "sg_dark_zone": "SoilGrids: чорноземи + файоземи в зоні 1.5 км, %",
    "sg_luvisol_zone": "SoilGrids: лювісолі в зоні 1.5 км, %",
    "sg_fluvisol_zone": "SoilGrids: флювісолі в зоні 1.5 км, %",
    "sg_sand_zone": "SoilGrids: пісок у зоні 1.5 км",
    "dark_point": "Яскравість ріллі у вікні (менше = темніше)",
    "dark_zone": "Яскравість ріллі в зоні 1.5 км",
    "dark_rel": "Яскравість ріллі: вікно мінус зона",
    "cover_point": "Частка ріллі з даними у вікні",
    "cover_zone": "Частка ріллі з даними в зоні 1.5 км",
}


# ------------------------------------------------------------------ raster features
def nearest_channel(acc: np.ndarray, threshold: float, cell: float):
    """Distance to the nearest channel cell and that cell's row / column."""
    channels = np.nan_to_num(acc) >= threshold
    dist, (iy, ix) = ndimage.distance_transform_edt(~channels, sampling=cell, return_indices=True)
    return dist.astype("float32"), iy, ix


def sunny_side(dist: np.ndarray, iy: np.ndarray, cell: float) -> np.ndarray:
    """Southward component of the unit vector from a cell to its nearest channel.

    +1: the channel is due south, the cell sits on the valley side that faces
    south. Rows grow southward. 0 on the channel itself."""
    rows = np.arange(dist.shape[0], dtype="float32")[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (iy - rows) * cell / dist
    return np.where(dist > 0, out, 0.0).astype("float32")


def terrace_mask(slope, dist, hand) -> np.ndarray:
    t = TERRACE
    return (
        (slope >= t["slope"][0])
        & (slope <= t["slope"][1])
        & (dist >= t["dist"][0])
        & (dist <= t["dist"][1])
        & (hand >= t["hand"][0])
        & (hand <= t["hand"][1])
    )


def floodplain_mask(slope, hand) -> np.ndarray:
    return (slope < FLOODPLAIN["slope"]) & (hand < FLOODPLAIN["hand"])


def focal(a: np.ndarray, size: int, min_share: float = 0.1):
    """Mean of the finite cells in a size x size window, and their share of it."""
    ok = np.isfinite(a)
    num = ndimage.uniform_filter(np.where(ok, a, 0).astype("float32"), size, mode="constant")
    den = ndimage.uniform_filter(ok.astype("float32"), size, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(den >= min_share, num / den, np.nan)
    return mean.astype("float32"), den


def darkness_features(dark: np.ndarray, inside: np.ndarray) -> dict[str, np.ndarray]:
    """Brightness in the window and in the zone, gap-filled, plus coverage.

    No value under forest and villages: the window falls back to the zone, the
    zone to the median; `cover_*` tells the model how much was measured."""
    point, cover_point = focal(dark, MODEL.WINDOW)
    zone, cover_zone = focal(dark, ZONE_CELLS)
    zone = np.where(np.isnan(zone), np.nanmedian(zone[inside]), zone)
    point = np.where(np.isnan(point), zone, point)
    out = {
        "dark_point": point,
        "dark_zone": zone,
        "dark_rel": point - zone,
        "cover_point": cover_point,
        "cover_zone": cover_zone,
    }
    return {k: np.where(inside, v, np.nan).astype("float32") for k, v in out.items()}


def soilgrids_zone(names: list[str], profile: dict, inside: np.ndarray) -> np.ndarray:
    """Sum of SoilGrids layers, averaged over the resource zone, on the model grid."""
    total, t = None, None
    for name in names:
        a, t = SOIL.read(name)
        total = a if total is None else total + a
    mid_lat = np.radians((SOIL.BBOX[1] + SOIL.BBOX[3]) / 2)
    pixel_m = (abs(t.e) * 111_320, t.a * 111_320 * np.cos(mid_lat))
    zone = SOIL.zone_mean(total, pixel_m, SOIL.ZONE_M).astype("float32")
    out = np.full(inside.shape, np.nan, dtype="float32")
    reproject(
        zone,
        out,
        src_transform=t,
        src_crs=4326,
        src_nodata=np.nan,
        dst_transform=profile["transform"],
        dst_crs=profile["crs"],
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    out = np.where(np.isnan(out), np.nanmedian(out[inside]), out)
    return np.where(inside, out, np.nan).astype("float32")


def feature_rasters():
    """The 05c features, then the new ones."""
    yield from MODEL.feature_rasters()
    elevation, profile = MODEL.read("elevation")
    cell = profile["transform"].a
    inside = ~np.isnan(elevation)
    acc, _ = MODEL.read("flow_acc_km2")
    slope, _ = MODEL.read("slope")

    dist, iy, ix = nearest_channel(acc, 1.0, cell)
    yield "sun_ch1", np.where(inside, sunny_side(dist, iy, cell), np.nan)
    hand = elevation - elevation[iy, ix]
    del iy, ix
    yield "terrace_share", MODEL.window_mean(
        np.where(inside, terrace_mask(slope, dist, hand).astype("float32"), np.nan)
    )
    yield "floodplain_share", MODEL.window_mean(
        np.where(inside, floodplain_mask(slope, hand).astype("float32"), np.nan)
    )
    del hand, slope
    dist, iy, _ = nearest_channel(acc, 5.0, cell)
    yield "sun_ch5", np.where(inside, sunny_side(dist, iy, cell), np.nan)
    del iy
    heads = (np.nan_to_num(acc) >= HEAD_KM2[0]) & (np.nan_to_num(acc) < HEAD_KM2[1])
    dist = np.minimum(ndimage.distance_transform_edt(~heads, sampling=cell), 3000)
    yield "dist_head", np.where(inside, MODEL.window_min(dist.astype("float32")), np.nan)
    del acc, dist, heads
    top = ndimage.maximum_filter(np.where(inside, elevation, -np.inf), 67)  # +-1 km
    yield "depth1000", MODEL.window_mean(np.where(inside, top - elevation, np.nan))
    del top, elevation

    yield "sg_dark_zone", soilgrids_zone(["wrb_Chernozems", "wrb_Phaeozems"], profile, inside)
    yield "sg_luvisol_zone", soilgrids_zone(["wrb_Luvisols"], profile, inside)
    yield "sg_fluvisol_zone", soilgrids_zone(["wrb_Fluvisols"], profile, inside)
    yield "sg_sand_zone", soilgrids_zone(["sand"], profile, inside)

    dark, _ = MODEL.read("bare_vis_norm")
    yield from darkness_features(dark, inside).items()


# --------------------------------------------------------------------------- models
def with_squares(x):
    return np.c_[x, np.asarray(x) ** 2]


def logistic_plain():
    return make_pipeline(
        StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)
    )


def logistic_sq():
    return make_pipeline(
        StandardScaler(),
        FunctionTransformer(with_squares),
        StandardScaler(),
        LogisticRegression(class_weight="balanced", C=0.3, max_iter=5000),
    )


def model_sets() -> dict[str, tuple[list[str], object]]:
    sets = {
        "water_only": (WATER, logistic_plain),  # the accepted model of 05c / 05d
        "water_optimum": (WATER, logistic_sq),  # feature 4: an optimum, not "the bigger"
    }
    for group, cols in GROUPS.items():
        sets[f"water+{group}"] = ([*WATER, *[c for c in cols if c not in WATER]], logistic_sq)
    sets["v2_no_dark"] = (V2_NO_DARK, logistic_sq)
    # Darkness split in two: the brightness itself, and how much arable land there
    # is around. The second may be a discovery bias (sites are found on ploughed
    # fields), not a choice made in the 3rd-4th centuries.
    sets["v2_no_dark+brightness"] = ([*V2_NO_DARK, *GROUPS["dark"][:3]], logistic_sq)
    sets["v2_no_dark+cover"] = ([*V2_NO_DARK, *GROUPS["dark"][3:]], logistic_sq)
    sets["v2"] = (V2, logistic_sq)
    sets["gbm_v1"] = (V1_KEEP, CHECK.boosting)
    sets["gbm_v2_no_dark"] = (V2_NO_DARK, CHECK.boosting)
    sets["gbm_v2"] = (V2, CHECK.boosting)
    return sets


MAPPED = [
    "water_only",
    "water+sun",
    "v2_no_dark",
    "v2_no_dark+brightness",
    "v2_no_dark+cover",
    "v2",
    "gbm_v2_no_dark",
    "gbm_v2",
]
MAP_MODEL = "v2"  # 05d_map.py draws this one as "v2" (next to "water_only" as "v1"):
# the logistic model - smoother and easier to explain than the boosted one


def suitability(ds: dict, name: str = MAP_MODEL) -> np.ndarray:
    """Model `name` fitted on all points, on the 150 m grid, as background percentile."""
    cols, make = model_sets()[name]
    data = ds["data"]
    model = make().fit(data[cols], data["y"])
    bg = np.sort(model.predict_proba(data.loc[data["y"] == 0, cols])[:, 1])
    rank = np.full(len(ds["grid_df"]), np.nan, dtype="float32")
    valid = ds["grid_valid"]
    pred = model.predict_proba(ds["grid_df"].loc[valid, cols])[:, 1]
    rank[valid] = 100 * np.searchsorted(bg, pred) / len(bg)
    return rank


def spatial_cv(data: pd.DataFrame, stage2: pd.DataFrame, xy: np.ndarray, rng) -> tuple[dict, dict]:
    """Fold AUCs per set, paired with water_only on the same folds; AUC for the
    settlements with no water in the text; the same settlements at their stage-2
    positions (before they were pulled to the bank named in the text)."""
    sets = model_sets()
    y = data["y"].to_numpy()
    blocks = CHECK.block_ids(xy)
    n_sites = len(stage2)
    s2_ok = stage2.notna().all(axis=1).to_numpy()
    free = (data["snap"] == "none").to_numpy() | (y == 0)
    folds_auc = {n: [] for n in sets}
    extra = {
        n: {"unsnapped": [], "stage2": [], "top10": [], "top20": [], "boyce": []} for n in sets
    }
    importances = {n: [] for n in ("gbm_v2",)}
    for _ in range(N_REPEATS):
        folds = rng.integers(N_FOLDS, size=blocks.max() + 1)[blocks]
        oof = {n: np.full(len(y), np.nan) for n in sets}
        oof2 = {n: np.full(n_sites, np.nan) for n in sets}
        for k in range(N_FOLDS):
            test = folds == k
            if y[test].sum() < 10 or y[~test].sum() < 10:
                continue
            for name, (cols, make) in sets.items():
                model = make().fit(data.loc[~test, cols], y[~test])
                pred = model.predict_proba(data.loc[test, cols])[:, 1]
                oof[name][test] = pred
                folds_auc[name].append(roc_auc_score(y[test], pred))
                t2 = test[:n_sites] & s2_ok
                if t2.any():
                    oof2[name][t2] = model.predict_proba(stage2.loc[t2, cols])[:, 1]
                if name in importances:
                    imp = permutation_importance(
                        model,
                        data.loc[test, cols],
                        y[test],
                        scoring="roc_auc",
                        n_repeats=5,
                        random_state=MODEL.SEED,
                    )
                    importances[name].append(imp.importances_mean)
        for name in sets:
            p = oof[name]
            done = ~np.isnan(p)
            pr, bg = p[done & (y == 1)], p[done & (y == 0)]
            extra[name]["unsnapped"].append(roc_auc_score(y[done & free], p[done & free]))
            p2 = oof2[name][~np.isnan(oof2[name])]
            extra[name]["stage2"].append(
                roc_auc_score(np.r_[np.ones(len(p2)), np.zeros(len(bg))], np.r_[p2, bg])
            )
            extra[name]["top10"].append(MODEL.top_share(pr, bg, 0.1))
            extra[name]["top20"].append(MODEL.top_share(pr, bg, 0.2))
            extra[name]["boyce"].append(MODEL.boyce(pr, bg))
    base = np.array(folds_auc["water_only"])
    summary = {}
    for name in sets:
        a = np.array(folds_auc[name])
        diff = a - base
        summary[name] = {
            "n_features": len(sets[name][0]),
            "auc": round(float(a.mean()), 3),
            "sd": round(float(a.std()), 3),
            "gain_over_water_only": round(float(diff.mean()), 3),
            "gain_sd": round(float(diff.std()), 3),
            "folds_better": f"{int((diff > 0).sum())}/{len(diff)}",
            "auc_unsnapped_sites": round(float(np.mean(extra[name]["unsnapped"])), 3),
            "auc_stage2_positions": round(float(np.mean(extra[name]["stage2"])), 3),
            "sites_in_top10pct": round(float(np.mean(extra[name]["top10"])), 3),
            "sites_in_top20pct": round(float(np.mean(extra[name]["top20"])), 3),
            "boyce": round(float(np.mean(extra[name]["boyce"])), 3),
        }
    imp = {
        n: pd.Series(np.mean(v, axis=0), index=sets[n][0]).sort_values(ascending=False)
        for n, v in importances.items()
    }
    return summary, imp


def cv_maps(ds: dict, rng) -> dict[str, np.ndarray]:
    """Out-of-fold suitability (0-100) on the 150 m grid, as MODEL.cv_map."""
    data, grid_df, valid, xy = ds["data"], ds["grid_df"], ds["grid_valid"], ds["xy"]
    sets = model_sets()
    keys_pts = np.floor(xy / MODEL.BLOCK_M).astype(int)
    keys_pts = keys_pts[:, 0] * 100_000 + keys_pts[:, 1]
    keys_grid = np.floor(ds["grid_xy"] / MODEL.BLOCK_M).astype(int)
    keys_grid = keys_grid[:, 0] * 100_000 + keys_grid[:, 1]
    uniq = np.unique(np.r_[keys_pts, keys_grid])
    fold_of = dict(zip(uniq, rng.integers(N_FOLDS, size=len(uniq)), strict=True))
    f_pts = np.array([fold_of[k] for k in keys_pts])
    f_grid = pd.Series(keys_grid).map(fold_of).to_numpy()
    y = data["y"].to_numpy()
    out = {}
    for name in MAPPED:
        cols, make = sets[name]
        rank = np.full(len(grid_df), np.nan, dtype="float32")
        for k in range(N_FOLDS):
            train = f_pts != k
            model = make().fit(data.loc[train, cols], y[train])
            bg = np.sort(model.predict_proba(data.loc[train & (y == 0), cols])[:, 1])
            cells = valid & (f_grid == k)
            if cells.any():
                pred = model.predict_proba(grid_df.loc[cells, cols])[:, 1]
                rank[cells] = 100 * np.searchsorted(bg, pred) / len(bg)
        out[name] = rank
    return out


def plot_importance(imp: pd.Series, path: Path) -> None:
    imp = imp.iloc[::-1]
    dark = set(GROUPS["dark"])
    colors = ["#eb6834" if n in dark else MODEL.COLORS["gbm"] for n in imp.index]
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.barh([LABELS[n] for n in imp.index], imp.to_numpy(), color=colors)
    ax.axvline(0, color=MODEL.INK_2, lw=0.8)
    ax.set_title(
        "Модель v2 (бустинг): падіння AUC при перемішуванні ознаки, просторова КВ\n"
        "помаранчеві — темнота ріллі із Sentinel-2",
        fontsize=10,
        color=MODEL.INK,
        loc="left",
    )
    MODEL.style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def village_level(ds: dict, rng) -> dict:
    """Village AUC of the out-of-fold maps over N_MAP_REPEATS fold assignments, and
    a paired bootstrap (over villages) of the gain over water_only, on the village
    scores averaged across the repeats."""
    aucs = {n: [] for n in MAPPED}
    mean_score, y = dict.fromkeys(MAPPED, 0.0), None
    for rep in range(N_MAP_REPEATS):
        print(f" maps, repeat {rep + 1}/{N_MAP_REPEATS}")
        maps = cv_maps(ds, rng)
        y, scores = MODEL.village_scores(
            maps, ds["grid"].shape, ds["grid"], ds["grid_profile"]["transform"]
        )
        for n in MAPPED:
            ok = ~np.isnan(scores[n])
            aucs[n].append(roc_auc_score(y[ok], scores[n][ok]))
            mean_score[n] = mean_score[n] + scores[n] / N_MAP_REPEATS
    ok = ~np.isnan(np.c_[[mean_score[n] for n in MAPPED]]).any(axis=0)
    y = y[ok]
    boot = rng.integers(len(y), size=(N_BOOT, len(y)))
    boot = [b for b in boot if 0 < y[b].sum() < len(b)]
    base = np.array([roc_auc_score(y[b], mean_score["water_only"][ok][b]) for b in boot])
    out = {"n_villages": int(len(y)), "n_with_sites": int(y.sum()), "repeats": N_MAP_REPEATS}
    for n in MAPPED:
        s = mean_score[n][ok]
        diff = np.array([roc_auc_score(y[b], s[b]) for b in boot]) - base
        out[n] = {
            "auc": round(float(np.mean(aucs[n])), 3),
            "sd_over_fold_assignments": round(float(np.std(aucs[n])), 3),
            "auc_of_mean_score": round(float(roc_auc_score(y, s)), 3),
            "gain_over_water_only": round(float(np.mean(diff)), 3),
            "gain_ci95": [round(float(v), 3) for v in np.percentile(diff, [2.5, 97.5])],
            "share_of_bootstraps_better": round(float((diff > 0).mean()), 3),
        }
    return out


def accepted(summary: dict, villages: dict) -> dict:
    """PLAN.md: village level > 0.61 and no-water settlements > 0.5."""
    out = {}
    for name in MAPPED:
        v, u = villages[name]["auc"], summary[name]["auc_unsnapped_sites"]
        out[name] = {
            "village_auc": v,
            "auc_unsnapped_sites": u,
            "passes": bool(v > 0.61 and u > 0.5),
            "village_gain_ci95": villages[name]["gain_ci95"],
        }
    return out


def main() -> None:
    rng = np.random.default_rng(MODEL.SEED)
    RESULTS.mkdir(exist_ok=True)
    SOIL.download()
    ds = MODEL.build_dataset(rng, feature_rasters)
    data = ds["data"]
    y = data["y"].to_numpy()
    new = [c for cols in GROUPS.values() for c in cols]
    new = list(dict.fromkeys(new))
    snapped = (data["snap"] != "none").to_numpy()
    uni = {
        "all_sites": CHECK.univariate(data, y, new),
        "snapped_sites": CHECK.univariate(
            data[snapped | (y == 0)].reset_index(drop=True), y[snapped | (y == 0)], new
        ),
        "unsnapped_sites": CHECK.univariate(
            data[~snapped | (y == 0)].reset_index(drop=True), y[~snapped | (y == 0)], new
        ),
    }
    print("spatial CV ...")
    summary, imp = spatial_cv(data, ds["stage2"], ds["xy"], rng)
    print(json.dumps(summary, indent=1))
    print("out-of-fold maps + village check ...")
    villages = village_level(ds, rng)
    print(json.dumps(villages, indent=1))
    plot_importance(imp["gbm_v2"], RESULTS / "model_v2_importance.png")
    out = {
        "n_presence": int(y.sum()),
        "n_background": int((y == 0).sum()),
        "n_unsnapped_sites": int((~snapped & (y == 1)).sum()),
        "params": {"block_km": MODEL.BLOCK_M / 1000, "folds": N_FOLDS, "repeats": N_REPEATS},
        "feature_groups": {"water": WATER, **GROUPS},
        "univariate": uni,
        "cv": summary,
        "gbm_v2_permutation_importance_auc": imp["gbm_v2"].round(4).to_dict(),
        "village_auc_p90_within_2km_oof": villages,
        "acceptance": accepted(summary, villages),
    }
    (RESULTS / "model_v2_stats.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(out["acceptance"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
