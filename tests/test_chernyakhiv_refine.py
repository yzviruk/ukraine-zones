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


@pytest.fixture
def forked():
    """The east-flowing river of `acc` joined at column 60 by a stream from the north."""
    raster = object.__new__(refine.Raster)
    a = np.full((101, 101), 0.01, dtype="float32")
    a[50, :] = np.linspace(20, 40, 101)
    a[10:50, 60] = np.linspace(1, 5, 40)
    raster.a, raster.t, raster.cell = a, from_origin(0, 101 * CELL, CELL, CELL), CELL
    return raster


def test_junction_is_found_only_at_the_confluence(forked):
    rows = np.array([50, 50, 50])
    cols = np.array([10, 60, 95])
    assert refine.near_junction(forked, rows, cols).tolist() == [False, True, False]


def test_confluence_text_prefers_the_channel_near_the_junction(forked):
    y = channel_y(forked)
    start = Point(30 * CELL, y + 200)  # nearest channel cell is 30 columns upstream
    plain, _ = refine.snap(start, start, "", "river", "left", None, forked)
    at_fork, info = refine.snap(start, start, "", "river", "left", None, forked, confluence=True)
    assert abs(plain.x - 30 * CELL) <= CELL
    assert info["confluence"] and abs(at_fork.x - 60.5 * CELL) <= refine.JUNCTION_M + CELL


def test_snap_reports_channel_and_bank_side(acc):
    start = Point(50 * CELL, channel_y(acc) + 400)
    new, info = refine.snap(start, start, "", "river", "left", None, acc)
    assert (info["side_x"], info["side_y"]) == pytest.approx((0, 1))  # left bank = north
    assert new.y - info["channel_y"] == pytest.approx(refine.OFFSET_M)


@pytest.fixture
def valley():
    """Valley side rising north of a channel at y = 0: flat 90 m, slope to 300 m, plateau."""
    raster = object.__new__(refine.Raster)
    y = 1500 - (np.arange(100) + 0.5) * CELL  # row centres, north at the top
    height = np.interp(y, [0, 90, 300, 1500], [0, 1, 22, 24])
    raster.a = np.repeat(height[:, None], 20, axis=1).astype("float32")
    raster.t, raster.cell = from_origin(0, 1500, CELL, CELL), CELL
    return raster


def test_best_estimate_follows_the_landform(valley):
    def offset(landform):
        return refine.best_point(valley, 300, 15, (0, 1), landform).y - 15

    assert offset("floodplain") == pytest.approx(60)
    assert offset("") == pytest.approx(refine.OFFSET_M)
    assert offset("floodplain") < offset("terrace") < offset("slope") < offset("plateau")
    assert 250 <= offset("plateau") <= refine.WALK_M  # the top of the slope


def test_landform_priority():
    assert refine.landform_of({"lf_plateau": 1, "lf_slope": 1}) == "slope"  # "схил плато"
    assert refine.landform_of({"lf_cape": 1}) == "plateau"
    assert refine.landform_of({"lf_floodplain": 1, "lf_terrace": 1}) == "floodplain"
    assert refine.landform_of({}) == ""
