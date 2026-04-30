"""
Sanity tests for the buffer/difference logic.

Verifies the algebra
    far_zone = ukraine - buffer(settlement_areas, X)
on synthetic inputs, and explicitly proves that the buffer is measured
from the EDGE of a settlement polygon, not from its centroid.
"""

from __future__ import annotations

import math

import pytest
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union


def far_zone(country: Polygon, settlement_areas, distance_m: float) -> Polygon:
    return country.difference(settlement_areas.buffer(distance_m))


@pytest.fixture
def country_100km() -> Polygon:
    return box(0, 0, 100_000, 100_000)


# Algebra basics -------------------------------------------------------------


def test_buffer_zero_returns_country_minus_settlement(country_100km):
    settlement = box(40_000, 40_000, 60_000, 60_000)
    result = far_zone(country_100km, settlement, 0.0)
    assert math.isclose(result.area, country_100km.area - settlement.area, rel_tol=1e-9)


def test_buffer_covers_country(country_100km):
    settlement = Point(50_000, 50_000)
    result = far_zone(country_100km, settlement, 1_000_000.0)
    assert result.is_empty or result.area < 1.0


def test_two_overlapping_buffers_unioned(country_100km):
    a = Point(50_000, 50_000)
    b = Point(55_000, 50_000)
    distance = 10_000
    union_geom = unary_union([a.buffer(distance), b.buffer(distance)])
    expected = country_100km.area - union_geom.area
    actual = far_zone(country_100km, unary_union([a, b]), distance).area
    assert math.isclose(actual, expected, rel_tol=1e-9)


def test_far_zone_is_subset_of_country(country_100km):
    settlement = box(10_000, 10_000, 12_000, 12_000)
    result = far_zone(country_100km, settlement, 5_000)
    assert country_100km.contains(result) or country_100km.equals(result.union(country_100km))


# KEY: buffer is measured from the EDGE of the settlement, not the centre. -----


def test_buffer_is_measured_from_polygon_edge():
    square = box(0, 0, 10_000, 10_000)
    buffered = square.buffer(1_000)

    # 0.5 km outside the east edge -> INSIDE 1 km buffer.
    assert buffered.contains(Point(10_500, 5_000))
    # 1.5 km outside the east edge -> OUTSIDE 1 km buffer.
    assert not buffered.contains(Point(11_500, 5_000))
    # If buffer were from the centroid (5km, 5km), a 1 km buffer would NOT
    # contain (5500, 5000) (which is 500m from centroid + 4500m extra...).
    # Wait, (5500, 5000) is only 500m from centroid -> inside even the
    # centroid-only 1 km buffer. Use a stronger point:
    # (9500, 5000) is 4500m from centroid -> would be OUTSIDE a 1 km
    # centroid-only buffer, but INSIDE the 1 km edge buffer.
    assert buffered.contains(Point(9_500, 5_000))


def test_polygon_buffer_grows_in_all_directions():
    """A 10x10 km square buffered by 1 km has area
    (10+2*1)^2 - 4*(1 - pi/4) ~= 143.14 km^2."""
    square = box(0, 0, 10_000, 10_000)
    buffered_area_km2 = square.buffer(1_000).area / 1e6
    expected_km2 = 144 - 4 * (1 - math.pi / 4)
    assert math.isclose(buffered_area_km2, expected_km2, rel_tol=1e-2)


# Fallback semantics: buffer only points NOT covered by landuse. ----------------


def test_fallback_only_for_uncovered_points():
    residential = box(0, 0, 10_000, 10_000)
    point_inside = Point(5_000, 5_000)
    point_outside = Point(20_000, 20_000)

    assert residential.contains(point_inside)
    assert not residential.contains(point_outside)

    fallback_radius_m = 300.0
    fallback = point_outside.buffer(fallback_radius_m)
    settlement_areas = unary_union([residential, fallback])

    # Covered point added nothing; only uncovered one did.
    assert math.isclose(
        settlement_areas.area,
        residential.area + fallback.area,
        rel_tol=1e-9,
    )

    country = box(-50_000, -50_000, 50_000, 50_000)
    one_km = settlement_areas.buffer(1_000)
    fz = country.difference(one_km)

    # 2 km east of the residential east edge -> in far zone.
    assert fz.contains(Point(12_000, 5_000))
    # 0.5 km east of the residential east edge -> NOT in far zone.
    assert not fz.contains(Point(10_500, 5_000))
