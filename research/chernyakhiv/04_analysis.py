"""
Stage 4: where are the known Chernyakhiv settlements? Compare their setting
with two controls.

Groups:
    sites    catalogue settlements placed in stage 2 (burials excluded)
    pseudo   OSM villages WITHOUT catalogue sites, placed by the same rule
             (random direction drawn from the catalogue's direction mix):
             what any village edge looks like -> controls survey/geocoding bias
    uniform  random points over the oblast: the landscape as a whole

Features are taken in a 500 m disk around each point (the geocoding error is
~0.5-1 km, so a single pixel would be noise), plus a few point values.

Outputs (aggregated only, safe to commit):
    research/chernyakhiv/results/features_summary.csv
    research/chernyakhiv/results/*.png
    research/chernyakhiv/results/stats.json
Local only: data/processed/chernyakhiv_features.csv (per-point values).
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import rasterio
from scipy import stats
from scipy.spatial import cKDTree
from shapely.geometry import Point

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
ENV = PROJECT_ROOT / "data" / "processed" / "env"
SITES = PROJECT_ROOT / "data" / "processed" / "chernyakhiv_sites.gpkg"
CATALOG_PDF_TEXT_SOURCE = PROJECT_ROOT / "data" / "raw" / "magomedov_2022.pdf"
OUT_LOCAL = PROJECT_ROOT / "data" / "processed" / "chernyakhiv_features.csv"
RESULTS = HERE / "results"

RADIUS_M = 500
N_UNIFORM = 5000
SEED = 42
GROUPS = ["sites", "pseudo", "uniform"]
GROUP_LABELS = {
    "sites": "Черняхівські поселення",
    "pseudo": "Краї інших сіл (контроль)",
    "uniform": "Випадкові точки області",
}
COLORS = {"sites": "#2a78d6", "pseudo": "#eb6834", "uniform": "#1baf7a"}  # validated, slots 1-3
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SOIL_NAMES = {0: "інші", 1: "чорноземи", 2: "опідзолені чорноземи", 3: "сірі лісові"}

# feature -> (label, unit, kind); kind: how it is taken from the 500 m disk
FEATURES = {
    "dist_dstream": ("Відстань до водотоку (з DEM)", "м", "min"),
    "dist_water": ("Відстань до водотоку (OSM)", "м", "min"),
    "hand_dstream": ("Висота над водотоком", "м", "point"),
    "rel_valley": ("Висота над дном долини (500 м)", "м", "point"),
    "tpi300": ("Положення в рельєфі TPI 300 м (макс.)", "м", "max"),
    "slope": ("Схил (середній)", "°", "mean"),
    "south_share": ("Частка південних схилів", "частка", "south"),
    "flow_acc_km2": ("Водозбір найближчого водотоку", "км²", "logmax"),
    "elevation": ("Висота над рівнем моря", "м", "point"),
    "dist_builtup": ("Відстань до сучасної забудови", "м", "point"),
}


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_rasters():
    names = set(FEATURES) - {"south_share"} | {"aspect", "soil"}
    data = {}
    for name in names:
        with rasterio.open(ENV / f"{name}.tif") as src:
            arr = src.read(1).astype("float32")
            arr[arr == src.nodata] = np.nan
            data[name] = arr
            transform = src.transform
    return data, transform


def disk_offsets(cell: float):
    r = int(round(RADIUS_M / cell))
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    keep = xx * xx + yy * yy <= r * r
    return yy[keep], xx[keep]


def sample(points: gpd.GeoSeries, rasters, transform) -> pd.DataFrame:
    dy, dx = disk_offsets(transform.a)
    inv = ~transform
    shape = rasters["elevation"].shape
    rows = []
    for p in points:
        col, row = (int(v) for v in inv * (p.x, p.y))
        ys, xs = row + dy, col + dx
        ok = (ys >= 0) & (ys < shape[0]) & (xs >= 0) & (xs < shape[1])
        ys, xs = ys[ok], xs[ok]
        inside = 0 <= row < shape[0] and 0 <= col < shape[1]
        rec = {}
        for name, (_, _, kind) in FEATURES.items():
            if kind == "south":
                slope, aspect = rasters["slope"][ys, xs], rasters["aspect"][ys, xs]
                valid = ~np.isnan(slope)
                steep = valid & (slope > 2)
                south = steep & (aspect >= 135) & (aspect <= 225)
                rec[name] = south.sum() / max(steep.sum(), 1)
                continue
            if kind == "point":
                rec[name] = rasters[name][row, col] if inside else np.nan
                continue
            vals = rasters[name][ys, xs]
            vals = vals[~np.isnan(vals)]
            if not vals.size:
                rec[name] = np.nan
            elif kind == "min":
                rec[name] = vals.min()
            elif kind == "max":
                rec[name] = vals.max()
            elif kind == "mean":
                rec[name] = vals.mean()
            elif kind == "logmax":
                rec[name] = np.log10(max(vals.max(), 1e-3))
        rec["soil"] = rasters["soil"][row, col] if inside else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def pseudo_sites(geo, sites, rng) -> gpd.GeoSeries:
    """Villages without catalogue sites, shifted by the same rule as stage 2."""
    oblast, _ = geo.load_admin()
    places = geo.load_places(oblast)
    places = places[places.within(oblast)]
    used = set(sites["place_idx"].astype(int))
    pool = places[~places.index.isin(used) & places["place"].isin(["village", "hamlet", "town"])]
    pool_m = pool.to_crs(geo.METRIC_CRS)
    polygons = geo.built_up(oblast)
    sindex = polygons.sindex
    mix = sites["direction"].fillna("").to_numpy()
    out = []
    for anchor in pool_m.geometry:
        direction = rng.choice(mix)
        out.append(geo.shift(anchor, direction, False, polygons, sindex))
    return gpd.GeoSeries(out, crs=geo.METRIC_CRS)


def uniform_points(geo, n, rng) -> gpd.GeoSeries:
    oblast, _ = geo.load_admin()
    poly = gpd.GeoSeries([oblast], crs=4326).to_crs(geo.METRIC_CRS).iloc[0]
    x0, y0, x1, y1 = poly.bounds
    pts = []
    while len(pts) < n:
        p = Point(rng.uniform(x0, x1), rng.uniform(y0, y1))
        if poly.contains(p):
            pts.append(p)
    return gpd.GeoSeries(pts, crs=geo.METRIC_CRS), poly.area


def auc(a, b) -> float:
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    u = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(u / (len(a) * len(b)))


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, (label, unit, kind) in FEATURES.items():
        rec = {"feature": name, "label": label, "unit": unit}
        for g in GROUPS:
            v = df.loc[df["group"] == g, name].to_numpy(dtype=float)
            if kind == "logmax":
                v = 10**v
            q = np.nanpercentile(v, [25, 50, 75])
            rec.update({f"{g}_q25": q[0], f"{g}_median": q[1], f"{g}_q75": q[2]})
        s = df.loc[df["group"] == "sites", name].to_numpy(dtype=float)
        for g in ("pseudo", "uniform"):
            rec[f"auc_vs_{g}"] = auc(s, df.loc[df["group"] == g, name].to_numpy(dtype=float))
        rows.append(rec)
    return pd.DataFrame(rows)


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_ecdfs(df: pd.DataFrame, path: Path) -> None:
    names = [n for n in FEATURES if n != "dist_builtup"]
    fig, axes = plt.subplots(3, 3, figsize=(13, 11))
    for ax, name in zip(axes.flat, names, strict=True):
        label, unit, kind = FEATURES[name]
        for g in GROUPS:
            v = np.sort(df.loc[df["group"] == g, name].dropna().to_numpy())
            if kind == "logmax":
                v = 10**v
            ax.plot(
                v, np.arange(1, len(v) + 1) / len(v), color=COLORS[g], lw=2, label=GROUP_LABELS[g]
            )
        if kind == "logmax":
            ax.set_xscale("log")
        if name in ("dist_dstream", "dist_water"):
            ax.set_xlim(0, 2500)
        ax.set_title(label, fontsize=10, color=INK, loc="left")
        ax.set_xlabel(unit, fontsize=8, color=INK_2)
        style(ax)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=10)
    fig.text(
        0.01,
        0.005,
        "Кумулятивна частка точок (ECDF). Ознаки взято в радіусі 500 м.",
        color=INK_2,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_soils(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    shares = (
        df.dropna(subset=["soil"])
        .assign(soil=lambda d: d["soil"].astype(int).map(SOIL_NAMES))
        .groupby("group")["soil"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .reindex(index=GROUPS, columns=list(SOIL_NAMES.values()), fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.26
    x = np.arange(len(shares.columns))
    for i, g in enumerate(GROUPS):
        bars = ax.bar(
            x + (i - 1) * (width + 0.02),
            shares.loc[g],
            width,
            color=COLORS[g],
            label=GROUP_LABELS[g],
        )
        ax.bar_label(
            bars, labels=[f"{v:.0%}" for v in shares.loc[g]], fontsize=7, color=INK_2, padding=2
        )
    ax.set_xticks(x, shares.columns)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Ґрунти в точці (HWSD, ~1 км)", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return shares


def nearest_neighbour(points: gpd.GeoSeries, area: float) -> dict:
    xy = np.c_[points.x, points.y]
    d, _ = cKDTree(xy).query(xy, k=2)
    nn = d[:, 1]
    expected = 0.5 / np.sqrt(len(xy) / area)
    return {
        "n": len(xy),
        "nn_median_m": float(np.median(nn)),
        "nn_q25_m": float(np.percentile(nn, 25)),
        "nn_q75_m": float(np.percentile(nn, 75)),
        "clark_evans_R": float(nn.mean() / expected),
    }


def coin_villages() -> list[str]:
    """Village names from the catalogue's section 3 (Roman coin finds)."""
    from pypdf import PdfReader

    text = "\n".join(p.extract_text() or "" for p in PdfReader(CATALOG_PDF_TEXT_SOURCE).pages)
    text = text.replace(" ", " ")
    start = text.index("3. Знахідки монет римської доби у Вінницькій області")
    section = text[start:]
    end = re.search(r"\n4\. |\nЛІТЕРАТУРА|\nСПИСОК", section)
    section = section[: end.start()] if end else section
    names = re.findall(r"(?m)^(?:с\.|смт|м\.) ([А-ЯІЇЄҐ][\w’'-]+(?: [А-ЯІЇЄҐ][\w’'-]+)?)", section)
    return sorted(set(names))


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    geo = load_module("02_geocode")

    sites = gpd.read_file(SITES)
    settlements = sites[sites["type"] == "Поселення"].copy()
    print(f"sites: {len(settlements)} settlements")
    print("building controls ...")
    pseudo = pseudo_sites(geo, settlements, rng)
    uniform, area = uniform_points(geo, N_UNIFORM, rng)

    print("sampling rasters ...")
    rasters, transform = load_rasters()
    parts = []
    for name, pts in (("sites", settlements.geometry), ("pseudo", pseudo), ("uniform", uniform)):
        df = sample(pts, rasters, transform)
        df["group"] = name
        df["accuracy"] = settlements["accuracy"].to_numpy() if name == "sites" else ""
        parts.append(df)
    features = pd.concat(parts, ignore_index=True)
    features.to_csv(OUT_LOCAL, index=False)

    summary = summarise(features)
    summary.round(3).to_csv(RESULTS / "features_summary.csv", index=False)
    summary_b = summarise(features[(features["group"] != "sites") | (features["accuracy"] == "B")])

    plot_ecdfs(features, RESULTS / "ecdf_features.png")
    soils = plot_soils(features, RESULTS / "soils.png")

    coins = coin_villages()
    cat_villages = {geo.norm(v) for v in sites["village"]}
    coins_with_sites = [c for c in coins if geo.norm(c) in cat_villages]

    out = {
        "n": {g: int((features["group"] == g).sum()) for g in GROUPS},
        "auc_vs_pseudo": dict(
            zip(summary["feature"], summary["auc_vs_pseudo"].round(3), strict=True)
        ),
        "auc_vs_pseudo_B_only": dict(
            zip(summary_b["feature"], summary_b["auc_vs_pseudo"].round(3), strict=True)
        ),
        "auc_vs_uniform": dict(
            zip(summary["feature"], summary["auc_vs_uniform"].round(3), strict=True)
        ),
        "soil_shares": soils.round(3).to_dict(orient="index"),
        "nearest_neighbour": {
            "sites": nearest_neighbour(settlements.geometry, area),
            "sites_village_level": nearest_neighbour(
                settlements.dissolve(by="place_idx").geometry.centroid, area
            ),
            "pseudo": nearest_neighbour(pseudo, area),
        },
        "coins": {
            "villages_with_coin_finds": len(coins),
            "of_them_with_chernyakhiv_sites": len(coins_with_sites),
        },
    }
    (RESULTS / "stats.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
