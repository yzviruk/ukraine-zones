"""
Unit tests for the bank snapping (research/chernyakhiv/05a_refine.py) on a
synthetic catchment raster, so no local DEM data is needed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from rasterio.transform import from_origin
from shapely.geometry import Point

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load():
    path = PROJECT_ROOT / "research" / "chernyakhiv" / "05a_refine.py"
    spec = importlib.util.spec_from_file_location("refine", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


refine = _load()
CELL = 30.0


@pytest.fixture
def acc():
    """A river flowing east along row 50; catchment grows downstream."""
    raster = object.__new__(refine.Raster)
    a = np.full((101, 101), 0.01, dtype="float32")
    a[50, :] = np.linspace(20, 40, 101)
    raster.a, raster.t, raster.cell = a, from_origin(0, 101 * CELL, CELL, CELL), CELL
    return raster


def channel_y(acc):
    return acc.xy(np.array([50]), np.array([0]))[1][0]


def test_flow_vector_points_downstream(acc):
    dx, dy = refine.flow_vector(acc, 50, 50)
    assert dx > 0.99 and abs(dy) < 0.01


@pytest.mark.parametrize(("bank", "north"), [("left", True), ("right", False)])
def test_snap_to_stated_bank(acc, bank, north):
    # Flowing east, the left bank is the northern one.
    start = Point(50 * CELL, channel_y(acc) - 600)  # south of the river
    new, info = refine.snap(start, start, "", "river", bank, None, acc)
    assert info["snap_how"] == "dem"
    assert abs(abs(new.y - channel_y(acc)) - refine.OFFSET_M) < 1
    assert (new.y > channel_y(acc)) == north


def test_side_of_point_without_bank(acc):
    start = Point(50 * CELL, channel_y(acc) - 600)
    new, info = refine.snap(start, start, "", "stream", None, None, acc)
    assert new.y < channel_y(acc) and info["bank_used"] == "right(side)"


def test_sector_excludes_water_behind_village(acc):
    # Site "south of the village"; the only river is north of the anchor.
    anchor = Point(50 * CELL, channel_y(acc) - 300)
    start = Point(anchor.x, anchor.y - 400)
    new, info = refine.snap(start, anchor, "S", "river", "left", None, acc)
    assert new is None and info["snap_how"] == "none"


def test_no_water_type_no_snap(acc):
    p = Point(50 * CELL, 50 * CELL)
    assert refine.snap(p, p, "N", None, None, None, acc)[0] is None
