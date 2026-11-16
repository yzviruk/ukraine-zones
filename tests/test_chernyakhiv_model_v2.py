"""
Unit tests for the feature builders of research/chernyakhiv/05g_model_v2.py
(synthetic arrays, no local data needed).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load():
    path = PROJECT_ROOT / "research" / "chernyakhiv" / "05g_model_v2.py"
    spec = importlib.util.spec_from_file_location("model_v2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v2 = _load()


def _valley(axis: str):
    """A 41 x 41 grid with one straight channel through the middle."""
    acc = np.full((41, 41), 0.01, dtype="float32")
    if axis == "east-west":
        acc[20, :] = 10.0
    else:
        acc[:, 20] = 10.0
    return acc


def test_sunny_side_is_north_bank_of_an_east_west_stream():
    dist, iy, _ = v2.nearest_channel(_valley("east-west"), 1.0, 30.0)
    sun = v2.sunny_side(dist, iy, 30.0)
    assert sun[5, 10] == pytest.approx(1.0)  # north of the channel: faces south
    assert sun[35, 10] == pytest.approx(-1.0)  # south of it: faces north
    assert sun[20, 10] == 0.0  # the channel itself
    assert dist[5, 10] == pytest.approx(15 * 30.0)


def test_sunny_side_is_neutral_on_a_north_south_stream():
    dist, iy, _ = v2.nearest_channel(_valley("north-south"), 1.0, 30.0)
    sun = v2.sunny_side(dist, iy, 30.0)
    assert np.abs(sun).max() == 0.0  # both sides face east or west


def test_nearest_channel_ignores_nodata_and_small_catchments():
    acc = _valley("east-west")
    acc[0, :] = np.nan
    acc[10, 10] = 0.9  # below the threshold
    dist, iy, ix = v2.nearest_channel(acc, 1.0, 30.0)
    assert iy[10, 10] == 20 and ix[10, 10] == 10
    assert iy[0, 3] == 20


def test_terrace_and_floodplain_masks():
    slope = np.array([0.5, 3.0, 3.0, 3.0, 9.0], dtype="float32")
    dist = np.array([100.0, 100.0, 20.0, 100.0, 100.0], dtype="float32")
    hand = np.array([1.0, 8.0, 8.0, 40.0, 8.0], dtype="float32")
    assert v2.terrace_mask(slope, dist, hand).tolist() == [False, True, False, False, False]
    assert v2.floodplain_mask(slope, hand).tolist() == [True, False, False, False, False]


def test_focal_mean_skips_gaps_and_reports_coverage():
    a = np.full((40, 40), np.nan, dtype="float32")
    a[:, :20] = 2.0
    mean, share = v2.focal(a, 5)
    assert mean[20, 10] == pytest.approx(2.0) and share[20, 10] == pytest.approx(1.0)
    assert mean[20, 20] == pytest.approx(2.0) and share[20, 20] == pytest.approx(0.4)
    assert np.isnan(mean[20, 30]) and share[20, 30] == 0.0


def test_darkness_gaps_are_filled_and_flagged():
    inside = np.ones((300, 300), dtype=bool)
    inside[:, :5] = False
    dark = np.full((300, 300), np.nan, dtype="float32")
    dark[:, 150:] = 1.2  # fields in the east, forest in the west
    dark[100:110, 200:210] = 0.8  # a dark patch
    f = v2.darkness_features(dark, inside)
    assert all(np.isnan(v[0, 0]) for v in f.values())  # outside stays nodata
    assert all(np.isfinite(v[inside]).all() for v in f.values())  # no gaps inside
    # deep in the forest: no measurement, the oblast median, and coverage says so
    assert f["cover_point"][150, 30] == 0.0 and f["cover_zone"][150, 30] == 0.0
    assert f["dark_rel"][150, 30] == 0.0
    # forest edge: no fields in the window, the zone's value is used
    assert f["cover_point"][150, 120] == 0.0 and f["dark_point"][150, 120] == pytest.approx(1.2)
    # the dark patch is darker than its surroundings
    assert f["dark_rel"][105, 205] < -0.05
    assert f["cover_point"][250, 250] == pytest.approx(1.0)


def test_feature_sets_are_consistent():
    sets = v2.model_sets()
    assert set(v2.MAPPED) <= set(sets) and v2.MAP_MODEL in v2.MAPPED
    assert not set(v2.GROUPS["dark"]) & set(v2.V2_NO_DARK)
    assert set(sets["v2"][0]) == set(v2.V2_NO_DARK) | set(v2.GROUPS["dark"])
    assert all(c in v2.LABELS for c in v2.V2)
    for cols, _ in sets.values():
        assert len(cols) == len(set(cols))
