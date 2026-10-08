"""
Stage 5d: suitability map for Chernyakhiv settlements and its public 2 km version.

Model: the catchment-only logistic regression of 05c (model.md: the richer
models add nothing out of sample). Fitted on all settlements vs the matched
background; suitability = percentile among the background (available) points.

Public layer (decision 1 in PLAN.md: coarse only): 2 x 2 km cells, scored by the
90th percentile of the 150 m suitability inside (the village-level check of 05c
used the same statistic). Classes by share of the oblast's cells:
    high       top 10 %
    elevated   next 15 %
"candidate": a high cell whose centre is more than CANDIDATE_KM from every
village with known sites (where nobody has reported one yet: either a gap in the
record or simply an area nobody surveyed).

Outputs:
    data/processed/chernyakhiv_model/suitability_150m.tif   (LOCAL ONLY)
    docs/data/chernyakhiv_potential.geojson                 (public, 2 km cells)
    research/chernyakhiv/results/map_stats.json, potential_map.png
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
from scipy.spatial import cKDTree
from shapely.geometry import box

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
OUT_PUBLIC = PROJECT_ROOT / "docs" / "data" / "chernyakhiv_potential.geojson"
PUBLIC_VILLAGES = PROJECT_ROOT / "docs" / "data" / "chernyakhiv_villages.geojson"

CELL_M = 2000
MIN_COVERAGE = 0.5  # share of a cell inside the oblast
HIGH_SHARE, ELEVATED_SHARE = 0.10, 0.25
CANDIDATE_KM = 3.0
CLASS_LABELS = {"high": "висока", "elevated": "підвищена"}
COLORS = {"high": "#b2182b", "elevated": "#f4a582", "candidate": "#2a78d6"}
INK, INK_2 = "#0b0b0b", "#52514e"


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODEL = load_module("05c_model")
GEO = load_module("02_geocode")


def suitability(ds) -> np.ndarray:
    """Catchment-only model on the 150 m grid, as background percentile 0-100."""
    cols, model = MODEL.make_models()["water_only"]
    data = ds["data"]
    model.fit(data[cols], data["y"])
    bg = np.sort(model.predict_proba(data.loc[data["y"] == 0, cols])[:, 1])
    rank = np.full(len(ds["grid_df"]), np.nan, dtype="float32")
    valid = ds["grid_valid"]
    rank[valid] = 100 * np.searchsorted(bg, model.predict_proba(ds["grid_df"][valid][cols])[:, 1])
    rank[valid] /= len(bg)
    return rank


def write_local(rank, ds) -> None:
    MODEL.OUT_LOCAL.mkdir(parents=True, exist_ok=True)
    out = np.full(ds["grid"].shape, -9999.0, dtype="float32")
    out[ds["grid"]] = np.nan_to_num(rank, nan=-9999.0)
    with rasterio.open(MODEL.OUT_LOCAL / "suitability_150m.tif", "w", **ds["grid_profile"]) as f:
        f.write(out, 1)


def cells(rank, ds) -> gpd.GeoDataFrame:
    step = MODEL.GRID_STEP * 30
    xy = ds["grid_xy"]  # each 150 m sample is one 30 m pixel; its corner is close enough
    ix, iy = np.floor(xy[:, 0] / CELL_M).astype(int), np.floor(xy[:, 1] / CELL_M).astype(int)
    df = pd.DataFrame({"ix": ix, "iy": iy, "rank": rank})
    full = (CELL_M / step) ** 2
    agg = df.groupby(["ix", "iy"]).agg(
        score=("rank", lambda v: np.nanpercentile(v, 90) if v.notna().any() else np.nan),
        n=("rank", "size"),
    )
    agg = agg[(agg["n"] / full >= MIN_COVERAGE) & agg["score"].notna()].reset_index()
    high = agg["score"].quantile(1 - HIGH_SHARE)
    elevated = agg["score"].quantile(1 - ELEVATED_SHARE)
    agg["class"] = np.select(
        [agg["score"] >= high, agg["score"] >= elevated], ["high", "elevated"], ""
    )
    geom = [
        box(i * CELL_M, j * CELL_M, (i + 1) * CELL_M, (j + 1) * CELL_M)
        for i, j in zip(agg["ix"], agg["iy"], strict=True)
    ]
    return gpd.GeoDataFrame(agg, geometry=geom, crs=GEO.METRIC_CRS)


def site_villages() -> gpd.GeoSeries:
    return gpd.read_file(PUBLIC_VILLAGES).to_crs(GEO.METRIC_CRS).geometry


def other_villages(sites_v: gpd.GeoSeries) -> gpd.GeoSeries:
    oblast, _ = GEO.load_admin()
    places = GEO.load_places(oblast)
    places = places[places.within(oblast) & places["place"].isin(["village", "hamlet", "town"])]
    pts = places.to_crs(GEO.METRIC_CRS).geometry
    d, _ = cKDTree(np.c_[sites_v.x, sites_v.y]).query(np.c_[pts.x, pts.y])
    return pts[d > 200]  # drop the nodes that are the site villages themselves


def class_at(points: gpd.GeoSeries, grid: gpd.GeoDataFrame) -> pd.Series:
    joined = gpd.sjoin(gpd.GeoDataFrame(geometry=points.values, crs=grid.crs), grid, how="left")
    return joined["class"].fillna("outside").replace("", "none")


def plot(grid: gpd.GeoDataFrame, oblast_m, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    gpd.GeoSeries([oblast_m]).boundary.plot(ax=ax, color=INK_2, lw=0.8)
    grid[grid["class"] == "elevated"].plot(ax=ax, color=COLORS["elevated"], lw=0)
    grid[(grid["class"] == "high") & ~grid["candidate"]].plot(ax=ax, color=COLORS["high"], lw=0)
    grid[grid["candidate"]].plot(ax=ax, color=COLORS["candidate"], lw=0)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS["high"], label="висока (10 % площі)"),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            color=COLORS["candidate"],
            label=f"висока, > {CANDIDATE_KM:g} км від сіл з пам'ятками",
        ),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["elevated"], label="підвищена (наступні 15 %)"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="lower left")
    ax.set_title(
        "Придатність для черняхівських поселень (модель «водозбір»), клітинки 2 км",
        fontsize=10,
        color=INK,
        loc="left",
    )
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    rng = np.random.default_rng(MODEL.SEED)
    ds = MODEL.build_dataset(rng)
    rank = suitability(ds)
    write_local(rank, ds)

    grid = cells(rank, ds)
    sites_v = site_villages()
    centres = grid.geometry.centroid
    d, _ = cKDTree(np.c_[sites_v.x, sites_v.y]).query(np.c_[centres.x, centres.y])
    grid["candidate"] = (grid["class"] == "high") & (d > CANDIDATE_KM * 1000)

    others = other_villages(sites_v)
    shares = {
        name: class_at(pts, grid).value_counts(normalize=True).round(3).to_dict()
        for name, pts in (("site_villages", sites_v), ("other_villages", others))
    }
    stats = {
        "cells": int(len(grid)),
        "cell_km": CELL_M / 1000,
        "high_cells": int((grid["class"] == "high").sum()),
        "elevated_cells": int((grid["class"] == "elevated").sum()),
        "candidate_cells": int(grid["candidate"].sum()),
        "candidate_km": CANDIDATE_KM,
        "village_class_shares": shares,
        "n_site_villages": int(len(sites_v)),
        "n_other_villages": int(len(others)),
    }

    public = grid[grid["class"] != ""].dissolve(by=["class", "candidate"]).reset_index()
    public = public[["class", "candidate", "geometry"]].assign(
        label=lambda g: g["class"].map(CLASS_LABELS),
        source="Модель: Магомедов 2022 (каталог) + Copernicus DEM; клітинки 2 км",
    )
    OUT_PUBLIC.unlink(missing_ok=True)
    public.to_crs(4326).to_file(OUT_PUBLIC, driver="GeoJSON", COORDINATE_PRECISION=4)
    stats["public_kb"] = OUT_PUBLIC.stat().st_size // 1024

    oblast, _ = GEO.load_admin()
    oblast_m = gpd.GeoSeries([oblast], crs=4326).to_crs(GEO.METRIC_CRS).iloc[0]
    plot(grid, oblast_m, MODEL.RESULTS / "potential_map.png")
    (MODEL.RESULTS / "map_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
