"""
Stage 5e: are soils worth adding to the settlement model?  SoilGrids (03c) vs HWSD.

Two independent checks:

1. Site level (same data and spatial CV as 05c): does a soil feature set add to
   the catchment-only model?  Soil at the point, soil in the 1.5 km "resource
   zone", soil properties.
2. Village level (no geocoding error at all): villages WITH catalogue sites vs
   the other villages of the oblast, by the soil around the village node.

Outputs: research/chernyakhiv/results/soil_stats.json
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PUBLIC_VILLAGES = HERE.parents[1] / "docs" / "data" / "chernyakhiv_villages.geojson"
N_REPEATS, N_FOLDS = 5, 5
VILLAGE_ZONE_M = 2000


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODEL = load_module("05c_model")
SOIL = load_module("03c_soilgrids")
GEO = load_module("02_geocode")

WATER = ["facc_max", "dist_ch5"]
HWSD = ["soil_chernozem", "soil_podzolized", "soil_grey_forest"]
CLASSES = [f"wrb_{g}" for g in SOIL.WRB_GROUPS]
SETS = {
    "water_only": WATER,
    "water+hwsd": WATER + HWSD,
    "water+soilgrids_point": WATER + CLASSES,
    "water+soilgrids_zone": WATER + [c + "_zone" for c in CLASSES],
    "water+properties_zone": WATER + [p + "_zone" for p in SOIL.PROPERTIES],
    "water+all_soil": WATER + [c + "_zone" for c in CLASSES + SOIL.PROPERTIES],
    "soilgrids_zone_only": [c + "_zone" for c in CLASSES + SOIL.PROPERTIES],
}


def logistic():
    return make_pipeline(
        StandardScaler(), LogisticRegression(class_weight="balanced", C=0.3, max_iter=5000)
    )


def boosting():
    return HistGradientBoostingClassifier(
        max_depth=3,
        learning_rate=0.05,
        max_iter=300,
        min_samples_leaf=40,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=MODEL.SEED,
    )


def spatial_cv(data: pd.DataFrame, y, blocks, sets, make, rng, extra_mask=None) -> dict:
    """Fold AUC (mean, sd) per feature set; optionally the AUC on a subset too."""
    fold_auc = {name: [] for name in sets}
    sub_auc = {name: [] for name in sets}
    for _ in range(N_REPEATS):
        folds = rng.integers(N_FOLDS, size=blocks.max() + 1)[blocks]
        oof = {name: np.zeros(len(y)) for name in sets}
        for k in range(N_FOLDS):
            test = folds == k
            if y[test].sum() < 10 or y[~test].sum() < 10:
                continue
            for name, cols in sets.items():
                model = make().fit(data.loc[~test, cols], y[~test])
                pred = model.predict_proba(data.loc[test, cols])[:, 1]
                oof[name][test] = pred
                fold_auc[name].append(roc_auc_score(y[test], pred))
        if extra_mask is not None:
            for name in sets:
                sub_auc[name].append(roc_auc_score(y[extra_mask], oof[name][extra_mask]))
    out = {}
    for name in sets:
        out[name] = {
            "auc": round(float(np.mean(fold_auc[name])), 3),
            "sd": round(float(np.std(fold_auc[name])), 3),
        }
        if extra_mask is not None:
            out[name]["auc_unsnapped_sites"] = round(float(np.mean(sub_auc[name])), 3)
    return out


def block_ids(xy: np.ndarray) -> np.ndarray:
    keys = pd.Series(list(zip(*np.floor(xy.T / MODEL.BLOCK_M).astype(int), strict=True)))
    return pd.factorize(keys)[0]


def univariate(data: pd.DataFrame, y, cols) -> dict:
    out = {}
    for c in cols:
        v = data[c].to_numpy(dtype=float)
        ok = ~np.isnan(v)
        u = stats.mannwhitneyu(v[ok & (y == 1)], v[ok & (y == 0)])
        out[c] = {
            "auc": round(float(u.statistic / ((ok & (y == 1)).sum() * (ok & (y == 0)).sum())), 3),
            "p": float(f"{u.pvalue:.2g}"),
            "median_sites": round(float(np.nanmedian(v[y == 1])), 1),
            "median_other": round(float(np.nanmedian(v[y == 0])), 1),
        }
    return out


def site_level(rng) -> dict:
    ds = MODEL.build_dataset(rng)
    data, xy = ds["data"], ds["xy"]
    y = data["y"].to_numpy()
    for name, values in SOIL.sample(xy).items():
        data[name] = values
    soil_cols = sorted({c for cols in SETS.values() for c in cols} - set(WATER) - set(HWSD))
    data[soil_cols] = data[soil_cols].fillna(data[soil_cols].median())
    free = (data["snap"] == "none").to_numpy() | (y == 0)
    blocks = block_ids(xy)
    return {
        "n_sites": int(y.sum()),
        "n_background": int((y == 0).sum()),
        "univariate": univariate(data, y, soil_cols),
        "logistic_cv": spatial_cv(data, y, blocks, SETS, logistic, rng, free),
        "boosting_cv": spatial_cv(
            data, y, blocks, {k: SETS[k] for k in ("water_only", "water+all_soil")}, boosting, rng
        ),
    }


def village_level(rng) -> dict:
    with_sites = gpd.read_file(PUBLIC_VILLAGES).to_crs(GEO.METRIC_CRS).geometry
    oblast, _ = GEO.load_admin()
    places = GEO.load_places(oblast)
    places = places[places.within(oblast) & places["place"].isin(["village", "hamlet", "town"])]
    pts = places.to_crs(GEO.METRIC_CRS).geometry
    d, _ = cKDTree(np.c_[with_sites.x, with_sites.y]).query(np.c_[pts.x, pts.y])
    other = pts[d > 200]
    xy = np.r_[np.c_[with_sites.x, with_sites.y], np.c_[other.x, other.y]]
    y = np.r_[np.ones(len(with_sites)), np.zeros(len(other))].astype(int)
    data = pd.DataFrame(SOIL.sample(xy, zone_m=VILLAGE_ZONE_M))
    zone = [c for c in data.columns if c.endswith("_zone")]
    data[zone] = data[zone].fillna(data[zone].median())
    sets = {"soilgrids_zone": zone, "classes_zone": [c + "_zone" for c in CLASSES]}
    return {
        "n_villages_with_sites": int(y.sum()),
        "n_other_villages": int((y == 0).sum()),
        "zone_m": VILLAGE_ZONE_M,
        "univariate": univariate(data, y, zone),
        "logistic_cv": spatial_cv(data, y, block_ids(xy), sets, logistic, rng),
    }


def main() -> None:
    rng = np.random.default_rng(MODEL.SEED)
    SOIL.download()
    out = {"site_level": site_level(rng), "village_level": village_level(rng)}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "soil_stats.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
