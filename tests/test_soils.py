"""
Invariants for the soil layers exported by preprocessing/05_export_soils.py.

Checks the classification rule on its own, then the generated GeoJSON files
in docs/data/ (they are committed, so no raw data is needed here).
"""

from __future__ import annotations

import importlib.util
from itertools import combinations
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import box
from shapely.ops import unary_union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "docs" / "data"
LAYERS = ["chernozems", "podzolized_chernozems", "grey_forest"]

UA_BOUNDS = (22.0, 44.0, 40.5, 52.5)  # lon/lat, with margin around the border
VINNYTSIA = box(27.3, 47.9, 30.0, 49.9)


def _load_stage5():
    path = PROJECT_ROOT / "preprocessing" / "05_export_soils.py"
    spec = importlib.util.spec_from_file_location("export_soils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def layers() -> dict[str, gpd.GeoDataFrame]:
    out = {}
    for name in LAYERS:
        path = DATA_DIR / f"{name}.geojson"
        if not path.exists():
            pytest.skip(f"{path.name} missing; run preprocessing/05_export_soils.py")
        out[name] = gpd.read_file(path)
    return out


# Classification rule ---------------------------------------------------------


@pytest.mark.parametrize(
    ("wrb2", "wrb4", "expected"),
    [
        ("CH", "CHha", "chernozems"),
        ("CH", "CHlv", "chernozems"),
        ("PH", "PHgz", "grey_forest"),
        ("PH", "PHlv", "podzolized_chernozems"),
        ("PH", "PHha", "podzolized_chernozems"),
        ("LV", "LVha", None),
        (None, None, None),
    ],
)
def test_classify(wrb2, wrb4, expected):
    assert _load_stage5().classify(wrb2, wrb4) == expected


# Exported GeoJSON --------------------------------------------------------------


def test_layers_are_wgs84_valid_and_non_empty(layers):
    for name, gdf in layers.items():
        assert gdf.crs.to_epsg() == 4326, name
        assert not gdf.empty, name
        assert gdf.geometry.is_valid.all(), name
        assert unary_union(gdf.geometry.values).area > 0, name


def test_layers_inside_ukraine(layers):
    bounds = box(*UA_BOUNDS)
    for name, gdf in layers.items():
        assert bounds.contains(box(*gdf.total_bounds)), name


def test_layers_do_not_overlap(layers):
    geoms = {name: unary_union(gdf.geometry.values) for name, gdf in layers.items()}
    for a, b in combinations(LAYERS, 2):
        overlap = geoms[a].intersection(geoms[b]).area
        # Independent simplification may leave slivers along shared edges.
        assert overlap < 0.01 * min(geoms[a].area, geoms[b].area), (a, b)


def test_vinnytsia_dominated_by_phaeozems(layers):
    area = {
        name: unary_union(gdf.geometry.values).intersection(VINNYTSIA).area
        for name, gdf in layers.items()
    }
    total = sum(area.values())
    ch_share = area["chernozems"] / total
    assert 0.15 < ch_share < 0.40
    assert area["podzolized_chernozems"] + area["grey_forest"] > area["chernozems"]
