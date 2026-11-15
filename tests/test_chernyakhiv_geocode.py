"""
Unit tests for the site placement rules (research/chernyakhiv/02_geocode.py) on
synthetic villages, so no local data is needed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point, box

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load():
    path = PROJECT_ROOT / "research" / "chernyakhiv" / "02_geocode.py"
    spec = importlib.util.spec_from_file_location("geocode", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


geo = _load()
NODE = Point(0, 0)
VILLAGE = box(-400, -1000, 400, 1000)  # 0.8 x 2 km, node in the middle


def test_part_of_the_village_is_inside_the_built_up_area():
    p = geo.place(NODE, VILLAGE, "S", "inside")
    assert VILLAGE.contains(p)
    assert -1000 < p.y < -300 and abs(p.x) < 400


def test_outskirts_on_the_edge_and_beyond_past_it():
    edge = geo.place(NODE, VILLAGE, "N", "outskirts")
    assert (edge.x, edge.y) == pytest.approx((0, 1000))
    past = geo.place(NODE, VILLAGE, "N", "beyond")
    assert past.y == pytest.approx(1000 + geo.BEYOND_EDGE_M)


def test_stated_distance_counts_from_the_edge():
    p = geo.place(NODE, VILLAGE, "W", "beyond", distance_m=4000)
    assert (p.x, p.y) == pytest.approx((-400 - 4000, 0))


def test_no_direction_stays_on_the_node():
    assert geo.place(NODE, VILLAGE, "", "near").equals(NODE)
    assert geo.place(NODE, VILLAGE, "C", "inside").equals(NODE)


def test_village_without_polygon_gets_typical_distances():
    inside = geo.place(NODE, None, "E", "inside").x
    edge = geo.place(NODE, None, "E", "outskirts").x
    past = geo.place(NODE, None, "E", "beyond").x
    assert 0 < inside < edge < past == pytest.approx(geo.NO_POLYGON_M)


def test_own_extent_stops_at_the_neighbouring_village():
    # Two villages 3 km apart whose built-up areas are merged into one polygon.
    places = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(3000, 0)], crs=geo.METRIC_CRS)
    merged = gpd.GeoSeries([box(-500, -300, 3500, 300)], crs=geo.METRIC_CRS)
    extents = geo.Extents(places, merged)
    own = extents.own(0)
    assert own.bounds[2] == pytest.approx(1500)  # half-way to the neighbour
    east = geo.place(places.geometry[0], own, "E", "outskirts")
    assert east.x == pytest.approx(1500)  # not 3500, the far end of the merged polygon
    assert extents.own(1).bounds[0] == pytest.approx(1500)


def test_node_far_from_any_polygon_has_no_extent():
    places = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(9000, 0)], crs=geo.METRIC_CRS)
    polygons = gpd.GeoSeries([box(1500, -200, 2000, 200)], crs=geo.METRIC_CRS)
    assert geo.Extents(places, polygons).own(0) is None


def test_two_villages_cross():
    # south-west of A and north-west of B (B lies 4 km south of A)
    p = geo.two_rays(Point(0, 0), "SW", Point(0, -4000), "NW")
    assert (p.x, p.y) == pytest.approx((-2000, -2000))
    # directions that never meet, or meet far away from both villages
    assert geo.two_rays(Point(0, 0), "N", Point(0, -4000), "S") is None
    assert geo.two_rays(Point(0, 0), "E", Point(0, -4000), "E") is None


def test_village_named_in_the_text_is_the_nearest_namesake():
    places = gpd.GeoDataFrame(
        {"name": ["Степове", "Степове", "Вітрівка"]},
        geometry=[Point(3000, 0), Point(70_000, 0), Point(0, 0)],
        crs=geo.METRIC_CRS,
    )
    index = {"степове": [0, 1], "вітрівка": [2]}
    assert geo.resolve_near("Степове", Point(0, 0), places, index) == 0
    assert geo.resolve_near("Степове", Point(40_000, 0), places, index) is None  # no neighbour
    assert geo.resolve_near("Немає", Point(0, 0), places, index) is None


def test_control_points_follow_the_same_rule():
    places = gpd.GeoDataFrame(
        {"place": ["village", "village"]},
        geometry=[Point(0, 0), Point(5000, 0)],
        crs=geo.METRIC_CRS,
    )
    polygons = gpd.GeoSeries(
        [box(-300, -300, 300, 300), box(4700, -300, 5300, 300)], crs=geo.METRIC_CRS
    )
    extents = geo.Extents(places, polygons)
    rng = np.random.default_rng(0)
    control = geo.control_points(places, extents, {0}, [("N", "outskirts")], rng)
    assert len(control) == 1  # village 0 has catalogue sites
    assert (control["point"][0].x, control["point"][0].y) == pytest.approx((5000, 300))
