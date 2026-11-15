"""
Invariants for the SoilGrids layers exported by preprocessing/06_export_soilgrids.py:
the smoothing/vectorising helpers on synthetic data, then the committed GeoJSON
files in docs/data/ (no raw data needed).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from rasterio.transform import from_origin
from shapely.geometry import box

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "docs" / "data"
LAYERS = ["soilgrids_chernozems", "soilgrids_phaeozems"]
UA_BOUNDS = (22.0, 44.0, 40.5, 52.5)
VINNYTSIA = box(27.3, 47.9, 30.0, 49.9)
MAX_BYTES = 6 * 1024 * 1024  # each layer must stay light enough for the browser


def _load():
    path = PROJECT_ROOT / "preprocessing" / "06_export_soilgrids.py"
    spec = importlib.util.spec_from_file_location("export_soilgrids", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage6 = _load()


@pytest.fixture(scope="module")
def layers() -> dict[str, gpd.GeoDataFrame]:
    out = {}
    for name in LAYERS:
        path = DATA_DIR / f"{name}.geojson"
        if not path.exists():
            pytest.skip(f"{path.name} missing; run preprocessing/06_export_soilgrids.py")
        out[name] = gpd.read_file(path)
    return out


def test_smoothing_removes_specks_and_fills_pinholes():
    mask = np.zeros((20, 20), dtype=bool)
    mask[2, 2] = True  # a lone pixel
    mask[8:16, 8:16] = True  # a block ...
    mask[11, 11] = False  # ... with a one-pixel hole
    out = stage6.smooth(mask, 3)
    assert not out[2, 2]
    assert out[11, 11] and out[9:15, 9:15].all()


def test_vectorise_drops_small_patches_and_small_holes():
    mask = np.zeros((40, 40), dtype=bool)
    mask[1:3, 1:3] = True  # 4 cells
    mask[10:30, 10:30] = True  # 400 cells
    mask[19:21, 19:21] = False  # a 4-cell hole
    transform = from_origin(0, 40, 1, 1)
    polys = stage6.vectorise(mask, transform, min_area=10)
    assert len(polys) == 1
    assert polys[0].area == pytest.approx(400)  # the hole is closed


def test_wrb_codes_follow_the_alphabetical_order():
    groups = ["Acrisols", "Albeluvisols", "Alisols", "Andosols", "Arenosols", "Calcisols"]
    groups += ["Cambisols", "Chernozems", "Cryosols", "Durisols", "Ferralsols", "Fluvisols"]
    groups += ["Gleysols", "Gypsisols", "Histosols", "Kastanozems", "Leptosols", "Lixisols"]
    groups += ["Luvisols", "Nitisols", "Phaeozems"]
    assert groups.index("Chernozems") == stage6.WRB_CODES["chernozems"]
    assert groups.index("Phaeozems") == stage6.WRB_CODES["phaeozems"]


def test_layers_are_wgs84_valid_and_light(layers):
    for name, gdf in layers.items():
        assert gdf.crs.to_epsg() == 4326
        assert len(gdf) == 1 and gdf.geometry.is_valid.all()
        assert gdf["class"][0] == name.removeprefix("soilgrids_")
        assert "model prediction" in gdf["source"][0] and "CC BY 4.0" in gdf["source"][0]
        assert (DATA_DIR / f"{name}.geojson").stat().st_size < MAX_BYTES


def test_layers_inside_ukraine(layers):
    for gdf in layers.values():
        w, s, e, n = gdf.total_bounds
        assert UA_BOUNDS[0] <= w and UA_BOUNDS[1] <= s
        assert e <= UA_BOUNDS[2] and n <= UA_BOUNDS[3]


def test_layers_barely_overlap(layers):
    # Each class is smoothed and simplified on its own, so slivers may overlap.
    a, b = (layers[name].to_crs(6381).geometry[0] for name in LAYERS)
    assert a.intersection(b).area < 0.01 * min(a.area, b.area)


def test_soilgrids_calls_most_of_vinnytsia_chernozem(layers):
    # The known disagreement with HWSD (~20 %): keep it visible if the source changes.
    clip = gpd.GeoSeries([VINNYTSIA], crs=4326).to_crs(6381)[0]
    cher = layers["soilgrids_chernozems"].to_crs(6381).geometry[0]
    assert cher.intersection(clip).area / clip.area > 0.4
