"""
Stage 5c: presence-background model of Chernyakhiv settlement locations,
with spatial cross-validation. The map itself is built by 05d_map.py.

Presence:   491 settlements, refined positions (stage 5.1, layer "sites_refined").
Background: random oblast points whose distance to modern built-up areas follows
            the settlements' distribution (they were found, and geocoded, next to
            villages): the model then compares settlements with what was AVAILABLE
            around villages, not with the empty steppe. dist_builtup itself is not
            a predictor.

Features are rasters at 30 m (so points and the map use one definition), all
taken over a +-240 m window, never at the point itself: stage 5.1 puts snapped
sites exactly 150 m from a channel, and point features (distance to the channel
above all) let a flexible model learn that offset instead of the landscape.
    facc_max    log10 of the largest catchment in the window
    dist_ch5    min distance to a channel with catchment >= 5 km2 (capped)
    hand_ch5    mean height above that channel
    dist_ch50   min distance to a channel >= 50 km2 (capped)
    tpi300      max (a raised edge nearby); tpi1000, rel_valley: mean
    slope_mean, northness, eastness   means
    soil_*      HWSD class one-hot at the point (weak; ~1 km anyway)
Absolute elevation is left out: it mostly encodes the region (big valleys,
survey routes), not the choice of a place.

Spatial CV: 25 km blocks randomly assigned to 5 folds, 3 repeats.
Models: logistic regression on catchment only (baseline), logistic regression on
all features (+ squares), gradient boosting.

Outputs:
    research/chernyakhiv/results/model_stats.json, model_*.png    (aggregated)
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage, stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
ENV = PROJECT_ROOT / "data" / "processed" / "env"
SITES = PROJECT_ROOT / "data" / "processed" / "chernyakhiv_sites.gpkg"
OUT_LOCAL = PROJECT_ROOT / "data" / "processed" / "chernyakhiv_model"
RESULTS = HERE / "results"

SEED = 42
N_BACKGROUND = 5000
WINDOW = 17  # cells, +-240 m at 30 m
GRID_STEP = 5  # map every 5th cell -> 150 m
BLOCK_M = 25_000
VILLAGE_RADIUS_M = 2000
N_FOLDS = 5
N_REPEATS = 3
DIST_BINS = [0, 1, 100, 200, 300, 400, 600, 800, 1000, 1500, 2000, 1e9]  # m to built-up
SOILS = {1: "soil_chernozem", 2: "soil_podzolized", 3: "soil_grey_forest"}
LABELS = {
    "facc_max": "Найбільший водотік поруч, log₁₀ км²",
    "dist_ch5": "Відстань до водотоку ≥ 5 км² (мін. у вікні), м",
    "hand_ch5": "Висота над водотоком ≥ 5 км² (сер.), м",
    "dist_ch50": "Відстань до річки ≥ 50 км² (мін.), м",
    "tpi300": "TPI 300 м (макс. у вікні)",
    "tpi1000": "TPI 1000 м",
    "rel_valley": "Висота над дном долини, м",
    "slope_mean": "Схил (середній), °",
    "northness": "Північність схилу",
    "eastness": "Східність схилу",
    "soil_chernozem": "Ґрунт: чорноземи",
    "soil_podzolized": "Ґрунт: опідзолені",
    "soil_grey_forest": "Ґрунт: сірі лісові",
}
FEATURES = list(LABELS)
CONTINUOUS = FEATURES[:10]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"water_only": "#8a8984", "logistic": "#eb6834", "gbm": "#2a78d6"}
MODEL_LABELS = {
    "water_only": "лише водозбір (логіст.)",
    "logistic": "логістична, усі ознаки",
    "gbm": "градієнтний бустинг",
}


def read(name: str):
    with rasterio.open(ENV / f"{name}.tif") as src:
        a = src.read(1).astype("float32")
        a[a == src.nodata] = np.nan
        return a, src.profile


def window_mean(a: np.ndarray) -> np.ndarray:
    valid = ~np.isnan(a)
    num = ndimage.uniform_filter(np.where(valid, a, 0), WINDOW)
    den = ndimage.uniform_filter(valid.astype("float32"), WINDOW)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (num / den).astype("float32")


def window_min(a: np.ndarray) -> np.ndarray:
    return ndimage.minimum_filter(a, WINDOW)


def window_max(a: np.ndarray) -> np.ndarray:
    nan = np.isnan(a)
    out = ndimage.maximum_filter(np.where(nan, -np.inf, a), WINDOW)
    return np.where(nan | np.isinf(out), np.nan, out).astype("float32")


def channel_distance(acc, elevation, threshold, cap, cell):
    channels = np.nan_to_num(acc) >= threshold
    dist, (iy, ix) = ndimage.distance_transform_edt(~channels, sampling=cell, return_indices=True)
    hand = elevation - elevation[iy, ix]
    return np.minimum(dist, cap).astype("float32"), hand.astype("float32")


def feature_rasters():
    """Yield (name, full-resolution array); the caller samples and frees it."""
    elevation, profile = read("elevation")
    cell = profile["transform"].a
    inside = ~np.isnan(elevation)
    acc, _ = read("flow_acc_km2")
    facc = ndimage.maximum_filter(np.nan_to_num(acc, nan=0.0), WINDOW)
    yield "facc_max", np.where(inside, np.log10(np.maximum(facc, 1e-3)), np.nan)
    del facc
    dist, hand = channel_distance(acc, elevation, 5.0, 3000, cell)
    yield "dist_ch5", np.where(inside, window_min(dist), np.nan)
    yield "hand_ch5", window_mean(np.where(inside, hand, np.nan))
    dist, _ = channel_distance(acc, elevation, 50.0, 10_000, cell)
    yield "dist_ch50", np.where(inside, window_min(dist), np.nan)
    del acc, dist, hand
    yield "tpi300", window_max(read("tpi300")[0])
    yield "tpi1000", window_mean(read("tpi1000")[0])
    yield "rel_valley", window_mean(read("rel_valley")[0])
    slope, _ = read("slope")
    yield "slope_mean", window_mean(slope)
    aspect, _ = read("aspect")
    steep = np.sin(np.radians(slope))
    yield "northness", window_mean(np.cos(np.radians(aspect)) * steep)
    yield "eastness", window_mean(np.sin(np.radians(aspect)) * steep)
    del slope, aspect, steep
    soil, _ = read("soil")
    for code, name in SOILS.items():
        yield name, np.where(np.isnan(soil), np.nan, (soil == code).astype("float32"))


def background_points(sites_rc, inside, dist_builtup, rng):
    """Random cells with the settlements' distance-to-built-up distribution."""
    rows, cols = np.nonzero(inside)
    pick = rng.choice(len(rows), 40 * N_BACKGROUND, replace=False)
    rows, cols = rows[pick], cols[pick]
    d_bg = dist_builtup[rows, cols]
    d_site = dist_builtup[sites_rc[:, 0], sites_rc[:, 1]]
    bins = np.digitize(d_bg, DIST_BINS) - 1
    target = np.bincount(np.digitize(d_site, DIST_BINS) - 1, minlength=len(DIST_BINS))
    have = np.bincount(bins, minlength=len(DIST_BINS))
    weight = np.where(have > 0, target / np.maximum(have, 1), 0)[bins]
    keep = rng.choice(len(rows), N_BACKGROUND, replace=False, p=weight / weight.sum())
    return np.c_[rows[keep], cols[keep]]


def squares(x):
    return np.c_[x, x[:, : len(CONTINUOUS)] ** 2]


def make_models():
    return {
        "water_only": (
            ["facc_max", "dist_ch5"],
            make_pipeline(
                StandardScaler(),
                LogisticRegression(class_weight="balanced", max_iter=2000),
            ),
        ),
        "logistic": (
            FEATURES,
            make_pipeline(
                StandardScaler(),
                FunctionTransformer(squares),
                StandardScaler(),
                LogisticRegression(class_weight="balanced", C=0.3, max_iter=5000),
            ),
        ),
        "gbm": (
            FEATURES,
            HistGradientBoostingClassifier(
                max_depth=3,
                learning_rate=0.05,
                max_iter=300,
                min_samples_leaf=40,
                l2_regularization=1.0,
                class_weight="balanced",
                random_state=SEED,
            ),
        ),
    }


def boyce(pred_presence, pred_background, n_bins=10) -> float:
    """Boyce index: Spearman of predicted/expected ratio over background deciles."""
    edges = np.quantile(pred_background, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(pred_presence, edges)[0] / len(pred_presence)
    return float(stats.spearmanr(np.arange(n_bins), p * n_bins).statistic)


def top_share(pred_presence, pred_background, share) -> float:
    """Fraction of settlements in the top `share` of available area."""
    return float((pred_presence >= np.quantile(pred_background, 1 - share)).mean())


def spatial_cv(data, stage2, xy, rng):
    blocks = pd.Series(
        list(zip(*(np.floor(xy.T / BLOCK_M).astype(int)), strict=True)), dtype=object
    )
    block_ids = pd.factorize(blocks)[0]
    y = data["y"].to_numpy()
    out = {name: [] for name in make_models()}
    oof = {name: np.zeros((N_REPEATS, len(y))) for name in make_models()}
    # The same settlements at their stage-2 positions (not pulled to water).
    oof2 = {name: np.full((N_REPEATS, len(stage2)), np.nan) for name in make_models()}
    n_sites = len(stage2)
    importances = []
    for rep in range(N_REPEATS):
        fold_of_block = rng.integers(N_FOLDS, size=block_ids.max() + 1)
        folds = fold_of_block[block_ids]
        for k in range(N_FOLDS):
            test = folds == k
            if y[test].sum() < 10:
                continue
            for name, (cols, model) in make_models().items():
                model.fit(data.loc[~test, cols], y[~test])
                pred = model.predict_proba(data.loc[test, cols])[:, 1]
                oof[name][rep, test] = pred
                out[name].append(roc_auc_score(y[test], pred))
                t2 = test[:n_sites] & stage2.notna().all(axis=1).to_numpy()
                oof2[name][rep, t2] = model.predict_proba(stage2.loc[t2, cols])[:, 1]
                if name == "gbm":
                    imp = permutation_importance(
                        model,
                        data.loc[test, cols],
                        y[test],
                        scoring="roc_auc",
                        n_repeats=5,
                        random_state=SEED,
                    )
                    importances.append(imp.importances_mean)
    summary = {}
    for name in out:
        aucs = np.array(out[name])
        boy, top10, top20, pooled, unsnapped, stage2_auc = [], [], [], [], [], []
        free = (data["snap"] == "none").to_numpy() | (y == 0)
        for rep in range(N_REPEATS):
            pr, bg = oof[name][rep, y == 1], oof[name][rep, y == 0]
            unsnapped.append(roc_auc_score(y[free], oof[name][rep, free]))
            p2 = oof2[name][rep][~np.isnan(oof2[name][rep])]
            stage2_auc.append(
                roc_auc_score(np.r_[np.ones(len(p2)), np.zeros(len(bg))], np.r_[p2, bg])
            )
            boy.append(boyce(pr, bg))
            top10.append(top_share(pr, bg, 0.1))
            top20.append(top_share(pr, bg, 0.2))
            pooled.append(roc_auc_score(y, oof[name][rep]))
        summary[name] = {
            "auc_fold_mean": round(float(aucs.mean()), 3),
            "auc_fold_sd": round(float(aucs.std()), 3),
            "auc_pooled": round(float(np.mean(pooled)), 3),
            # settlements left at stage-2 positions: free of the 5.1 offset
            "auc_unsnapped_sites": round(float(np.mean(unsnapped)), 3),
            # model trained on refined positions, tested on stage-2 positions
            "auc_stage2_positions": round(float(np.mean(stage2_auc)), 3),
            "boyce": round(float(np.mean(boy)), 3),
            "sites_in_top10pct": round(float(np.mean(top10)), 3),
            "sites_in_top20pct": round(float(np.mean(top20)), 3),
            "n_folds": len(aucs),
        }
    imp = pd.Series(np.mean(importances, axis=0), index=FEATURES).sort_values(ascending=False)
    return summary, imp, oof


def cv_map(data, grid_df, valid, grid_xy, xy, rng) -> dict[str, np.ndarray]:
    """Out-of-fold suitability (0-100) on the grid: each 25 km block is predicted
    by a model that never saw the settlements in it."""
    keys_pts = np.floor(xy / BLOCK_M).astype(int)
    keys_pts = keys_pts[:, 0] * 100_000 + keys_pts[:, 1]
    keys_grid = np.floor(grid_xy / BLOCK_M).astype(int)
    keys_grid = keys_grid[:, 0] * 100_000 + keys_grid[:, 1]
    uniq = np.unique(np.r_[keys_pts, keys_grid])
    fold_of = dict(zip(uniq, rng.integers(N_FOLDS, size=len(uniq)), strict=True))
    f_pts = np.array([fold_of[k] for k in keys_pts])
    f_grid = np.array([fold_of[k] for k in keys_grid])
    y = data["y"].to_numpy()
    out = {}
    for name, (cols, model) in make_models().items():
        rank = np.full(len(grid_df), np.nan, dtype="float32")
        for k in range(N_FOLDS):
            train = f_pts != k
            model.fit(data.loc[train, cols], y[train])
            bg = np.sort(model.predict_proba(data.loc[train & (y == 0), cols])[:, 1])
            cells = valid & (f_grid == k)
            if cells.any():
                pred = model.predict_proba(grid_df.loc[cells, cols])[:, 1]
                rank[cells] = 100 * np.searchsorted(bg, pred) / len(bg)
        out[name] = rank
    return out


def village_check(maps: dict[str, np.ndarray], grid_shape, grid_mask, grid_transform):
    """AUC: villages with catalogue sites vs other villages, by the p90 of the
    (out-of-fold) suitability within VILLAGE_RADIUS_M of the village node."""
    spec = importlib.util.spec_from_file_location("geo", HERE / "02_geocode.py")
    geo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geo)
    oblast, _ = geo.load_admin()
    places = geo.load_places(oblast)
    places = places[places.within(oblast) & places["place"].isin(["village", "hamlet", "town"])]
    used = set(gpd.read_file(SITES, layer="sites")["place_idx"].astype(int))
    y = places.index.isin(used).astype(int)
    pts = places.to_crs(geo.METRIC_CRS).geometry
    k = int(VILLAGE_RADIUS_M / grid_transform.a)
    yy, xx = np.mgrid[-k : k + 1, -k : k + 1]
    disk = xx * xx + yy * yy <= k * k
    inv = ~grid_transform
    centres = [tuple(int(v) for v in (inv @ (p.x, p.y))[::-1]) for p in pts]
    result = {"n_villages": len(y), "n_with_sites": int(y.sum())}
    for name, values in maps.items():
        arr = np.full(grid_shape, np.nan, dtype="float32")
        arr[grid_mask] = values
        score = []
        for r, c in centres:
            ys, xs = r + yy[disk], c + xx[disk]
            ok = (ys >= 0) & (ys < grid_shape[0]) & (xs >= 0) & (xs < grid_shape[1])
            v = arr[ys[ok], xs[ok]]
            v = v[~np.isnan(v)]
            score.append(np.percentile(v, 90) if v.size else np.nan)
        score = np.array(score)
        ok = ~np.isnan(score)
        result[name] = round(float(roc_auc_score(y[ok], score[ok])), 3)
    return result


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_importance(imp: pd.Series, path: Path) -> None:
    imp = imp.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([LABELS[n] for n in imp.index], imp.to_numpy(), color=COLORS["gbm"])
    ax.axvline(0, color=INK_2, lw=0.8)
    ax.set_title(
        "Важливість ознак (бустинг): падіння AUC при перемішуванні, просторова КВ",
        fontsize=10,
        color=INK,
        loc="left",
    )
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_capture(oof, y, path: Path) -> None:
    """Share of settlements vs share of available area, ranked by prediction."""
    fig, ax = plt.subplots(figsize=(6, 5))
    shares = np.linspace(0, 1, 101)
    for name, pred in oof.items():
        p, bg = pred[0, y == 1], pred[0, y == 0]
        cap = [top_share(p, bg, s) if s > 0 else 0 for s in shares]
        ax.plot(shares, cap, color=COLORS[name], lw=2, label=MODEL_LABELS[name])
    ax.plot([0, 1], [0, 1], color=INK_2, lw=1, ls="--", label="випадково")
    ax.set_xlabel("Частка доступної площі (найімовірніші спершу)", fontsize=9, color=INK_2)
    ax.set_ylabel("Частка поселень", fontsize=9, color=INK_2)
    ax.set_title("Скільки поселень «ловить» модель (поза вибіркою)", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_partial(model, data, names, path: Path) -> None:
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3.6), sharey=True)
    for ax, name in zip(axes, names, strict=True):
        pd_res = partial_dependence(model, data[FEATURES], [name], grid_resolution=40)
        x, v = pd_res["grid_values"][0], pd_res["average"][0]
        ax.plot(x, v, color=COLORS["gbm"], lw=2)
        q = np.nanpercentile(data.loc[data["y"] == 1, name], [5, 95])
        ax.axvspan(*q, color=GRID, alpha=0.6, lw=0)
        ax.set_title(LABELS[name], fontsize=9, color=INK, loc="left")
        style(ax)
    axes[0].set_ylabel("Прогноз (частковий вплив)", fontsize=9, color=INK_2)
    fig.text(0.01, 0.01, "Сіра смуга — 5–95 % значень у поселень.", fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def build_dataset(rng) -> dict:
    """Presence + matched background with features, and the same features on the
    150 m map grid. Shared by 05c (evaluation) and 05d (map)."""
    sites = gpd.read_file(SITES, layer="sites_refined")
    elevation, profile = read("elevation")
    transform = profile["transform"]
    inside = ~np.isnan(elevation)
    del elevation
    inv = ~transform
    sites_rc = np.array([[int(r), int(c)] for c, r in (inv @ (p.x, p.y) for p in sites.geometry)])
    stage2 = gpd.read_file(SITES, layer="sites")
    stage2 = stage2[stage2["type"] == "Поселення"].reset_index(drop=True)
    assert (stage2["no"].to_numpy() == sites["no"].to_numpy()).all()
    s2_rc = np.array([[int(r), int(c)] for c, r in (inv @ (p.x, p.y) for p in stage2.geometry)])
    dist_builtup, _ = read("dist_builtup")
    bg_rc = background_points(sites_rc, inside, dist_builtup, rng)
    del dist_builtup
    points = np.r_[sites_rc, bg_rc]
    y = np.r_[np.ones(len(sites_rc)), np.zeros(len(bg_rc))].astype(int)

    print("feature rasters ...")
    grid = inside[::GRID_STEP, ::GRID_STEP]
    columns, grid_cols, s2_cols = {}, {}, {}
    for name, arr in feature_rasters():
        print(f"  {name}")
        columns[name] = arr[points[:, 0], points[:, 1]]
        s2_cols[name] = arr[s2_rc[:, 0], s2_rc[:, 1]]
        grid_cols[name] = arr[::GRID_STEP, ::GRID_STEP][grid]
    snap = np.r_[sites["snap_how"].to_numpy(), np.full(len(bg_rc), "background")]
    data = pd.DataFrame(columns).assign(y=y, snap=snap)
    ok = data[FEATURES].notna().all(axis=1)
    print(f"dropped (nodata): {(~ok).sum()}")
    data, points = data[ok].reset_index(drop=True), points[ok.to_numpy()]
    grid_df = pd.DataFrame(grid_cols)
    gi, gj = np.nonzero(grid)
    grid_profile = profile.copy()
    grid_profile.update(
        height=grid.shape[0],
        width=grid.shape[1],
        transform=transform * transform.scale(GRID_STEP),
        dtype="float32",
        nodata=-9999.0,
    )
    return {
        "data": data,
        "stage2": pd.DataFrame(s2_cols)[ok.to_numpy()[: len(s2_rc)]].reset_index(drop=True),
        "xy": np.c_[
            transform.c + points[:, 1] * transform.a, transform.f + points[:, 0] * transform.e
        ],
        "grid": grid,
        "grid_df": grid_df,
        "grid_valid": grid_df.notna().all(axis=1).to_numpy(),
        "grid_xy": np.c_[
            transform.c + gj * GRID_STEP * transform.a, transform.f + gi * GRID_STEP * transform.e
        ],
        "grid_profile": grid_profile,
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    RESULTS.mkdir(exist_ok=True)
    ds = build_dataset(rng)
    data, xy, grid, grid_df = ds["data"], ds["xy"], ds["grid"], ds["grid_df"]

    print("spatial CV ...")
    summary, imp, oof = spatial_cv(data, ds["stage2"], xy, rng)
    print(json.dumps(summary, indent=1))

    print("final fit ...")
    models = make_models()
    for _, (cols, model) in models.items():
        model.fit(data[cols], data["y"])
    gbm = models["gbm"][1]
    logistic = models["logistic"][1]

    print("out-of-fold map + village check ...")
    maps = cv_map(data, grid_df, ds["grid_valid"], ds["grid_xy"], xy, rng)
    # No fitting at all: the largest catchment around the village.
    maps["raw_catchment"] = grid_df["facc_max"].to_numpy()
    villages = village_check(maps, grid.shape, grid, ds["grid_profile"]["transform"])
    print(villages)

    coef = logistic[-1].coef_[0][: len(FEATURES)]
    plot_importance(imp, RESULTS / "model_importance.png")
    plot_capture(oof, data["y"].to_numpy(), RESULTS / "model_capture.png")
    plot_partial(gbm, data, list(imp.index[:4]), RESULTS / "model_partial.png")
    stats_out = {
        "n_presence": int(data["y"].sum()),
        "n_background": int((data["y"] == 0).sum()),
        "params": {
            "block_km": BLOCK_M / 1000,
            "folds": N_FOLDS,
            "repeats": N_REPEATS,
            "window_m": WINDOW * 30,
            "grid_m": GRID_STEP * 30,
        },
        "cv": summary,
        "gbm_permutation_importance_auc": imp.round(4).to_dict(),
        "logistic_linear_coef_std": dict(zip(FEATURES, np.round(coef, 3).tolist(), strict=True)),
        "village_auc_p90_within_2km_oof": villages,
    }
    (RESULTS / "model_stats.json").write_text(
        json.dumps(stats_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats_out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
