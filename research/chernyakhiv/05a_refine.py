"""
Stage 5a: refine site positions by snapping them to the bank of the watercourse
named in the catalogue description, then check whether relief features sharpen.

Stage 2 puts a site just past the village edge in the stated direction (error
~0.5-1.5 km). Most descriptions also say WHICH water and WHICH bank ("on the left
bank of the r. Murashka"). Here:
    1. target channel = DEM drainage cells whose catchment fits the water type
       (river >= 10 km2, stream/pond >= 1, ravine >= 0.3); if the water is named
       and found in OSM, only cells within NAMED_TOL_M of that OSM line;
    2. search window: RADIUS_M around the stage-2 point, and (if a direction is
       given) the +-SECTOR_DEG sector from the anchor village in that direction;
    3. nearest such cell to the stage-2 point; flow direction from the catchment
       gradient along the channel; the site goes OFFSET_M to the stated bank
       (left/right facing downstream), or to the side of the stage-2 point.
The offset is fixed on purpose: choosing it by relief would make the relief
features confirm themselves.

Control: the stage-4 "edges of other villages", refined by the same procedure
with water type and bank drawn from the catalogue's mix (no names).

Outputs:
    data/processed/chernyakhiv_sites.gpkg, layer "sites_refined"   (LOCAL ONLY)
    research/chernyakhiv/results/refine_stats.json, refine_ecdf.png (aggregated)
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import rasterio
from scipy import stats
from shapely.geometry import Point
from shapely.ops import unary_union

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
ENV = PROJECT_ROOT / "data" / "processed" / "env"
SITES = PROJECT_ROOT / "data" / "processed" / "chernyakhiv_sites.gpkg"
RESULTS = HERE / "results"

RADIUS_M = 1500  # search window around the stage-2 point
SECTOR_DEG = 67.5  # half-width of the direction sector seen from the village
NAMED_TOL_M = 250  # DEM channel cells this close to the named OSM line
OFFSET_M = 150  # from the channel to the stated bank (fixed, see docstring)
DIR_RADIUS_CELLS = 4  # neighbourhood for the flow direction along the channel
CLASS_KM2 = {"river": 10.0, "stream": 1.0, "ravine": 0.3}
SAMPLE_RADIUS_M = 250
SEED = 42

# feature -> (label, unit, kind)
FEATURES = {
    "hand_dstream": ("Висота над водотоком", "м", "point"),
    "rel_valley": ("Висота над дном долини (500 м)", "м", "point"),
    "tpi300": ("TPI 300 м", "м", "point"),
    "tpi1000": ("TPI 1000 м", "м", "point"),
    "slope": ("Схил (середній, 250 м)", "°", "mean"),
    "elevation": ("Висота н.р.м.", "м", "point"),
    "flow_acc_km2": ("Водозбір найбільшого водотоку (250 м)", "км²", "logmax"),
}
GROUP_LABELS = {
    "sites": "Черняхівські поселення",
    "pseudo": "Краї інших сіл (контроль)",
}
COLORS = {"sites": "#2a78d6", "pseudo": "#eb6834"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GEO = load_module("02_geocode")


class Raster:
    def __init__(self, name: str):
        with rasterio.open(ENV / f"{name}.tif") as src:
            self.a = src.read(1).astype("float32")
            self.a[self.a == src.nodata] = np.nan
            self.t = src.transform
        self.cell = self.t.a

    def rc(self, x, y):
        col, row = ~self.t @ (x, y)
        return int(math.floor(row)), int(math.floor(col))

    def xy(self, rows, cols):
        return self.t.c + (cols + 0.5) * self.t.a, self.t.f + (rows + 0.5) * self.t.e

    def window(self, x, y, radius):
        r0, c0 = self.rc(x, y)
        n = int(radius / self.cell) + 1
        rows, cols = np.mgrid[
            max(r0 - n, 0) : min(r0 + n + 1, self.a.shape[0]),
            max(c0 - n, 0) : min(c0 + n + 1, self.a.shape[1]),
        ]
        return rows.ravel(), cols.ravel()


def water_class(row) -> str | None:
    if row.get("w_river"):
        return "river"
    if row.get("w_stream") or row.get("w_pond"):
        return "stream"
    if row.get("w_ravine_water"):
        return "ravine"
    return "stream" if isinstance(row.get("bank"), str) else None


def flow_vector(acc: Raster, r: int, c: int) -> tuple[float, float] | None:
    """Downstream unit vector at a channel cell: from low to high catchment."""
    a0 = acc.a[r, c]
    k = DIR_RADIUS_CELLS
    sub = acc.a[max(r - k, 0) : r + k + 1, max(c - k, 0) : c + k + 1]
    rows, cols = np.nonzero((sub >= 0.5 * a0) & (sub <= 2 * a0))
    if len(rows) < 2:
        return None
    vals = sub[rows, cols]
    lo, hi = np.argmin(vals), np.argmax(vals)
    dx, dy = (cols[hi] - cols[lo]), -(rows[hi] - rows[lo])  # rows grow southward
    norm = math.hypot(dx, dy)
    return (dx / norm, dy / norm) if norm else None


def snap(point, anchor, direction, wclass, bank, named_line, acc: Raster):
    """Return (new point, info dict); new point is None when nothing fits."""
    info = {"water_class": wclass, "bank_used": bank, "snap_how": "none"}
    if wclass is None:
        return None, info
    rows, cols = acc.window(point.x, point.y, RADIUS_M)
    xs, ys = acc.xy(rows, cols)
    d = np.hypot(xs - point.x, ys - point.y)
    keep = d <= RADIUS_M
    if direction in GEO.BEARINGS:
        bearing = math.radians(GEO.BEARINGS[direction])
        ang = np.arctan2(xs - anchor.x, ys - anchor.y)  # clockwise from north
        diff = np.abs((ang - bearing + math.pi) % (2 * math.pi) - math.pi)
        keep &= np.degrees(diff) <= SECTOR_DEG
    # "р." also covers small rivers: fall back to the stream threshold.
    for tier in [wclass] + (["stream"] if wclass == "river" else []):
        found = keep & (acc.a[rows, cols] >= CLASS_KM2[tier])
        if found.any():
            break
    else:
        return None, info
    rows, cols, xs, ys, d = rows[found], cols[found], xs[found], ys[found], d[found]
    info["water_class"] = tier if tier == wclass else f"{wclass}->{tier}"
    info["snap_how"] = "dem"
    if named_line is not None:
        near = np.array(
            [named_line.distance(Point(x, y)) <= NAMED_TOL_M for x, y in zip(xs, ys, strict=True)]
        )
        if near.any():
            rows, cols, xs, ys, d = rows[near], cols[near], xs[near], ys[near], d[near]
            info["snap_how"] = "named"
    i = int(np.argmin(d))
    r, c, cx, cy = rows[i], cols[i], xs[i], ys[i]
    info["channel_km2"] = float(acc.a[r, c])
    vec = flow_vector(acc, r, c)
    if vec is None:
        side = (point.x - cx, point.y - cy)
        bank = None
    else:
        left = (-vec[1], vec[0])
        if bank not in ("left", "right"):
            cross = vec[0] * (point.y - cy) - vec[1] * (point.x - cx)
            bank = "left" if cross > 0 else "right"
            info["bank_used"] = f"{bank}(side)"
        side = left if bank == "left" else (-left[0], -left[1])
    norm = math.hypot(*side) or 1.0
    new = Point(cx + side[0] / norm * OFFSET_M, cy + side[1] / norm * OFFSET_M)
    info["shift_m"] = float(new.distance(point))
    return new, info


def named_lines(oblast):
    ways = GEO.load_waterways(oblast)
    return {k: unary_union(g.geometry.values) for k, g in ways.groupby("key")}


def pseudo_with_anchors(sites, rng) -> pd.DataFrame:
    """Stage-4 control (same rng sequence), keeping anchor and direction."""
    oblast, _ = GEO.load_admin()
    places = GEO.load_places(oblast)
    places = places[places.within(oblast)]
    used = set(sites["place_idx"].astype(int))
    pool = places[~places.index.isin(used) & places["place"].isin(["village", "hamlet", "town"])]
    polygons = GEO.built_up(oblast)
    sindex = polygons.sindex
    mix = sites["direction"].fillna("").to_numpy()
    out = []
    for anchor in pool.to_crs(GEO.METRIC_CRS).geometry:
        direction = rng.choice(mix)
        out.append(
            {
                "anchor": anchor,
                "direction": direction,
                "point": GEO.shift(anchor, direction, False, polygons, sindex),
            }
        )
    return pd.DataFrame(out)


def sample(points, rasters: dict[str, Raster]) -> pd.DataFrame:
    ref = rasters["elevation"]
    r = int(round(SAMPLE_RADIUS_M / ref.cell))
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    disk = xx * xx + yy * yy <= r * r
    dy, dx = yy[disk], xx[disk]
    shape = ref.a.shape
    recs = []
    for p in points:
        row, col = ref.rc(p.x, p.y)
        ys, xs = row + dy, col + dx
        ok = (ys >= 0) & (ys < shape[0]) & (xs >= 0) & (xs < shape[1])
        ys, xs = ys[ok], xs[ok]
        inside = 0 <= row < shape[0] and 0 <= col < shape[1]
        rec = {}
        for name, (_, _, kind) in FEATURES.items():
            a = rasters[name].a
            if kind == "point":
                rec[name] = a[row, col] if inside else np.nan
                continue
            vals = a[ys, xs]
            vals = vals[~np.isnan(vals)]
            if not vals.size:
                rec[name] = np.nan
            elif kind == "mean":
                rec[name] = vals.mean()
            elif kind == "logmax":
                rec[name] = np.log10(max(vals.max(), 1e-3))
        recs.append(rec)
    return pd.DataFrame(recs)


def auc(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if not len(a) or not len(b):
        return float("nan")
    return float(stats.mannwhitneyu(a, b).statistic / (len(a) * len(b)))


def plot(feat: pd.DataFrame, path: Path) -> None:
    names = ["hand_dstream", "rel_valley", "tpi300", "slope"]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharey=True)
    for i, stage in enumerate(("before", "after")):
        for ax, name in zip(axes[i], names, strict=True):
            label, unit, _ = FEATURES[name]
            for g in GROUP_LABELS:
                v = np.sort(feat.loc[feat["group"] == g, f"{name}_{stage}"].dropna().to_numpy())
                ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=COLORS[g], lw=2,
                        label=GROUP_LABELS[g])  # fmt: skip
            title = "етап 2" if stage == "before" else "уточнено"
            ax.set_title(f"{label} — {title}", fontsize=9, color=INK, loc="left")
            ax.set_xlabel(unit, fontsize=8, color=INK_2)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(colors=INK_2, labelsize=8)
            ax.grid(True, color=GRID, linewidth=0.6)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    rng = np.random.default_rng(SEED)

    sites = gpd.read_file(SITES, layer="sites")
    sites = sites[sites["type"] == "Поселення"].reset_index(drop=True)
    oblast, _ = GEO.load_admin()
    lines = named_lines(oblast)
    acc = Raster("flow_acc_km2")

    print(f"refining {len(sites)} settlements ...")
    new_pts, infos = [], []
    for _, row in sites.iterrows():
        key = GEO.norm(row["water_name"]) if isinstance(row["water_name"], str) else ""
        anchor = Point(row["anchor_x"], row["anchor_y"])
        line = lines.get(key)
        if line is not None and line.distance(anchor) > 5000:
            line = None  # same name, other river
        direction = row["direction"] if isinstance(row["direction"], str) else ""
        bank = row["bank"] if isinstance(row["bank"], str) else None
        p, info = snap(row.geometry, anchor, direction, water_class(row), bank, line, acc)
        new_pts.append(p if p is not None else row.geometry)
        infos.append(info)
    info = pd.DataFrame(infos)
    refined = sites.assign(**info, geometry=new_pts)
    refined["shift_m"] = refined["shift_m"].fillna(0.0)
    refined.to_file(SITES, layer="sites_refined", driver="GPKG")

    print("refining the control ...")
    pseudo = pseudo_with_anchors(sites, rng)
    classes = [water_class(r) for _, r in sites.iterrows()]
    banks = sites["bank"].where(sites["bank"].notna(), None).tolist()
    p_new, p_info = [], []
    for _, row in pseudo.iterrows():
        i = rng.integers(len(sites))
        p, inf = snap(row["point"], row["anchor"], row["direction"], classes[i], banks[i], None,
                      acc)  # fmt: skip
        p_new.append(p if p is not None else row["point"])
        p_info.append(inf)
    p_info = pd.DataFrame(p_info)
    del acc

    print("sampling ...")
    rasters = {n: Raster(n) for n in FEATURES}
    groups = {
        "sites": (sites.geometry, refined.geometry, info["snap_how"]),
        "pseudo": (pseudo["point"], pd.Series(p_new), p_info["snap_how"]),
    }
    parts = []
    for g, (before, after, how) in groups.items():
        b = sample(before, rasters).add_suffix("_before")
        a = sample(after, rasters).add_suffix("_after")
        parts.append(pd.concat([b, a], axis=1).assign(group=g, snap_how=how.to_numpy()))
    feat = pd.concat(parts, ignore_index=True)
    sites_feat = feat[feat["group"] == "sites"].reset_index(drop=True)

    def aucs(mask_s, mask_p):
        out = {}
        for name in FEATURES:
            out[name] = {
                stage: round(
                    auc(
                        feat.loc[(feat["group"] == "sites") & mask_s, f"{name}_{stage}"],
                        feat.loc[(feat["group"] == "pseudo") & mask_p, f"{name}_{stage}"],
                    ),
                    3,
                )
                for stage in ("before", "after")
            }
        return out

    every = pd.Series(True, index=feat.index)
    snapped = feat["snap_how"] != "none"

    # Internal check: does text landform agree with DEM relief at the position?
    lf = sites[["lf_plateau", "lf_slope", "lf_floodplain", "lf_terrace", "lf_cape"]].astype(bool)
    high = (lf["lf_plateau"] | lf["lf_cape"]) & ~lf["lf_floodplain"] & ~lf["lf_terrace"]
    low = (lf["lf_floodplain"] | lf["lf_terrace"]) & ~lf["lf_plateau"]
    sn = (info["snap_how"] != "none").to_numpy()
    landform = {"n_high": int((high & sn).sum()), "n_low": int((low & sn).sum())}
    for name in ("hand_dstream", "rel_valley", "tpi300", "tpi1000", "slope"):
        landform[name] = {
            stage: round(
                auc(
                    sites_feat.loc[high & sn, f"{name}_{stage}"],
                    sites_feat.loc[low & sn, f"{name}_{stage}"],
                ),
                3,
            )
            for stage in ("before", "after")
        }

    shift = refined.loc[sn, "shift_m"]
    out = {
        "params": {
            "radius_m": RADIUS_M,
            "sector_deg": SECTOR_DEG,
            "named_tol_m": NAMED_TOL_M,
            "offset_m": OFFSET_M,
            "class_km2": CLASS_KM2,
            "sample_radius_m": SAMPLE_RADIUS_M,
        },
        "n_settlements": len(sites),
        "snap_counts": info["snap_how"].value_counts().to_dict(),
        "snap_by_class": pd.crosstab(info["water_class"].fillna("none"), info["snap_how"]).to_dict(
            orient="index"
        ),
        "bank_used": info["bank_used"].fillna("none").value_counts().to_dict(),
        "shift_m": {
            "median": round(float(shift.median())),
            "q25": round(float(shift.quantile(0.25))),
            "q75": round(float(shift.quantile(0.75))),
            "q90": round(float(shift.quantile(0.9))),
        },
        "pseudo_snap_counts": p_info["snap_how"].value_counts().to_dict(),
        "medians": {
            g: {
                f"{n}_{s}": round(float(np.nanmedian(feat.loc[feat["group"] == g, f"{n}_{s}"])), 2)
                for n in FEATURES
                for s in ("before", "after")
            }
            for g in GROUP_LABELS
        },
        "auc_sites_vs_pseudo_all": aucs(every, every),
        "auc_sites_vs_pseudo_snapped": aucs(snapped, snapped),
        "landform_check_high_vs_low": landform,
    }
    (RESULTS / "refine_stats.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot(feat[snapped], RESULTS / "refine_ecdf.png")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
