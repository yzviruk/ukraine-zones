"""
Stage 5: export soil layers for Ukraine from HWSD v2.0 (FAO & IIASA, 2023).

HWSD v2.0 for Ukraine is a 30" (~1 km) rasterisation of the European Soil
Database 1:1M polygons -- surveyed soil units, not ML predictions.

Each raster cell holds a soil mapping unit (SMU) id; the attribute database
lists the SMU's soil components with their area share. Every SMU is classified
by its DOMINANT component (largest SHARE), mapped to one of three layers:

    chernozems             WRB2 == "CH"               (типові, звичайні, південні)
    podzolized_chernozems  WRB2 == "PH", not greyzemic (опідзолені чорноземи)
    grey_forest            WRB4 == "PHgz"             (сірі та темно-сірі лісові)

Output: docs/data/<layer>.geojson in EPSG:4326, clipped to the Ukraine border.

Licence: HWSD v2.0 is CC BY-NC-SA 3.0 IGO; cite FAO & IIASA (2023), doi:10.4060/cc3823en.
"""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from access_parser import AccessParser
from rasterio.features import shapes
from rasterio.windows import from_bounds
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.ops import unary_union

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
HWSD_DIR = RAW_DIR / "hwsd2"
BORDER = RAW_DIR / "ua-border.geojson"
OUT_DIR = PROJECT_ROOT / "docs" / "data"

HWSD_URL = "https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/HWSD/%s"
HWSD_RASTER = HWSD_DIR / "HWSD2.bil"
HWSD_DB = HWSD_DIR / "HWSD2.mdb"

UA_BBOX = (22.0, 44.0, 41.0, 53.0)

LAYERS = ["chernozems", "podzolized_chernozems", "grey_forest"]

BORDER_SIMPLIFY_DEG = 0.002  # ~200 m; keeps output small after clipping
SIMPLIFY_DEG = 0.004  # ~half a pixel; removes pixel staircase
MIN_POLY_AREA_DEG = 1e-4  # ~1 km^2 (one HWSD cell)
COORD_PRECISION = 4  # ~10 m


def download_hwsd() -> None:
    HWSD_DIR.mkdir(parents=True, exist_ok=True)
    for archive, member in (("HWSD2_RASTER.zip", HWSD_RASTER), ("HWSD2_DB.zip", HWSD_DB)):
        if member.exists():
            print(f"  {member.name} exists, skip")
            continue
        zip_path = HWSD_DIR / archive
        print(f"Downloading {archive} ...")
        urllib.request.urlretrieve(HWSD_URL % archive, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                # Flatten archive folders; never trust member paths.
                (HWSD_DIR / Path(info.filename).name).write_bytes(zf.read(info))
        zip_path.unlink()


def classify(wrb2: str | None, wrb4: str | None) -> str | None:
    if wrb2 == "CH":
        return "chernozems"
    if wrb4 == "PHgz":
        return "grey_forest"
    if wrb2 == "PH":
        return "podzolized_chernozems"
    return None


def smu_classes() -> dict[int, tuple[str, str]]:
    """SMU id -> (layer, WRB4 code) of its dominant component."""
    smu = AccessParser(str(HWSD_DB)).parse_table("HWSD2_SMU")
    dominant: dict[int, tuple[float, str | None, str | None]] = {}
    for sid, share, wrb2, wrb4 in zip(
        smu["HWSD2_SMU_ID"], smu["SHARE"], smu["WRB2"], smu["WRB4"], strict=True
    ):
        if sid not in dominant or share > dominant[sid][0]:
            dominant[sid] = (share, wrb2, wrb4)
    out = {}
    for sid, (_, wrb2, wrb4) in dominant.items():
        layer = classify(wrb2, wrb4)
        if layer is not None:
            out[int(sid)] = (layer, wrb4)
    return out


def load_border():
    border = gpd.read_file(BORDER)
    border = border[border.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    if border.empty:
        raise SystemExit(f"No polygon geometry in {BORDER.name}")
    return unary_union(border.geometry.values).simplify(BORDER_SIMPLIFY_DEG)


def vectorise(mask: np.ndarray, transform) -> list[Polygon]:
    polys = []
    for geom_dict, val in shapes(mask.astype("uint8"), mask=mask, transform=transform):
        if val == 1:
            poly = shape(geom_dict)
            if poly.area >= MIN_POLY_AREA_DEG:
                polys.append(poly)
    return polys


def as_multipolygon(geom) -> MultiPolygon:
    parts = getattr(geom, "geoms", [geom])
    return MultiPolygon(
        [p for p in parts if p.geom_type == "Polygon" and p.area >= MIN_POLY_AREA_DEG]
    )


def main() -> None:
    for f in (BORDER,):
        if not f.exists():
            raise SystemExit(f"Missing {f}.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    download_hwsd()

    print("Loading HWSD raster window ...")
    with rasterio.open(HWSD_RASTER) as src:
        window = from_bounds(*UA_BBOX, transform=src.transform).round_offsets().round_lengths()
        smu_ids = src.read(1, window=window)
        transform = src.window_transform(window)
    print(f"  window shape {smu_ids.shape}")

    print("Reading SMU attributes ...")
    classes = smu_classes()

    print("Loading Ukraine border ...")
    border = load_border()

    for layer in LAYERS:
        ids = [sid for sid, (lyr, _) in classes.items() if lyr == layer]
        mask = np.isin(smu_ids, ids)
        codes = sorted({classes[int(s)][1] for s in np.unique(smu_ids[mask])})
        print(f"Vectorising {layer} ({', '.join(codes)}) ...")
        polys = vectorise(mask, transform)
        if not polys:
            print("  -> 0 polygons, skip")
            continue
        # shapes() returns disjoint polygons, so MultiPolygon needs no union.
        geom = MultiPolygon(polys).simplify(SIMPLIFY_DEG, preserve_topology=True)
        geom = as_multipolygon(geom.intersection(border))

        gdf = gpd.GeoDataFrame(
            {
                "class": [layer],
                "wrb": [", ".join(codes)],
                "source": ["HWSD v2.0, FAO & IIASA 2023, CC BY-NC-SA 3.0 IGO"],
            },
            geometry=[geom],
            crs="EPSG:4326",
        )
        out_path = OUT_DIR / f"{layer}.geojson"
        out_path.unlink(missing_ok=True)
        gdf.to_file(out_path, driver="GeoJSON", COORDINATE_PRECISION=COORD_PRECISION)
        size_kb = out_path.stat().st_size // 1024
        print(f"  -> {out_path.name}: {len(geom.geoms)} polygons, {size_kb} KB")


if __name__ == "__main__":
    main()
