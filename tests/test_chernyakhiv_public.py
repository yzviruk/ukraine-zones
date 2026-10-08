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
