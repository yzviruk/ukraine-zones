"""
Invariants for the public Chernyakhiv layer (research/chernyakhiv/02_geocode.py).

The public layer must stay at village precision: one point per catalogue
village, no site-level descriptions or coordinates (anti-looting decision).
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import box

PUBLIC = Path(__file__).resolve().parent.parent / "docs" / "data" / "chernyakhiv_villages.geojson"
VINNYTSIA_BOUNDS = box(27.3, 47.9, 30.0, 49.95)
ALLOWED_PROPERTIES = {"district", "village", "settlements", "burials", "catalogue_no", "source"}


@pytest.fixture(scope="module")
def villages() -> gpd.GeoDataFrame:
    if not PUBLIC.exists():
        pytest.skip("run research/chernyakhiv/02_geocode.py")
    return gpd.read_file(PUBLIC)


def test_points_in_wgs84_inside_vinnytsia(villages):
    assert villages.crs.to_epsg() == 4326
    assert (villages.geometry.geom_type == "Point").all()
    assert villages.within(VINNYTSIA_BOUNDS).all()


def test_only_village_level_properties(villages):
    assert set(villages.columns) - {"geometry"} == ALLOWED_PROPERTIES


def test_every_catalogue_entry_counted_once(villages):
    numbers = [int(n) for s in villages["catalogue_no"] for n in s.split(", ")]
    assert sorted(numbers) == list(range(1, 514))
    assert (villages["settlements"] + villages["burials"]).sum() == 513


def test_coordinates_are_coarse(villages):
    text = PUBLIC.read_text(encoding="utf-8")
    assert "desc" not in text
    for geom in villages.geometry:
        for value in (geom.x, geom.y):
            assert round(value, 4) == pytest.approx(value, abs=1e-9)


POTENTIAL = PUBLIC.parent / "chernyakhiv_potential.geojson"
POTENTIAL_PROPERTIES = {"class", "candidate", "label", "source"}


@pytest.fixture(scope="module")
def potential() -> gpd.GeoDataFrame:
    if not POTENTIAL.exists():
        pytest.skip("run research/chernyakhiv/05d_map.py")
    return gpd.read_file(POTENTIAL)


def test_potential_layer_is_coarse_polygons_in_vinnytsia(potential):
    assert potential.crs.to_epsg() == 4326
    assert potential.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all()
    assert potential.within(VINNYTSIA_BOUNDS.buffer(0.05)).all()
    assert set(potential.columns) - {"geometry"} == POTENTIAL_PROPERTIES
    assert set(potential["class"]) <= {"high", "elevated"}


def test_potential_cells_are_at_least_2_km(potential):
    # Dissolved 2 km cells: every part is at least one whole cell (4 km2).
    parts = potential.to_crs(6381).explode(index_parts=False)
    assert parts.area.min() >= 4e6 * 0.99
