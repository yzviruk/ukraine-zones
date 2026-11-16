"""
Stage S (bare-soil composite from Sentinel-2): the masking and compositing
helpers of research/chernyakhiv/03d_bare_soil.py on synthetic arrays.
No network and no rasters needed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load():
    path = PROJECT_ROOT / "research" / "chernyakhiv" / "03d_bare_soil.py"
    spec = importlib.util.spec_from_file_location("bare_soil", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bs():
    return _load()


def test_reflectance_removes_offset_and_marks_nodata(bs):
    dn = np.array([[0, 1000, 2000, 11000]], np.uint16)
    r = bs.reflectance(dn)
    assert np.isnan(r[0, 0])
    np.testing.assert_allclose(r[0, 1:], [0.0, 0.1, 1.0], atol=1e-6)


def _pixels():
    # reflectance: red, nir, swir1, swir2
    return {
        "bare_dark": (0.08, 0.12, 0.20, 0.18),  # NDVI 0.20, NBR2 0.05
        "bare_light": (0.20, 0.28, 0.35, 0.31),  # NDVI 0.17, NBR2 0.06
        "crop": (0.04, 0.40, 0.20, 0.10),  # NDVI 0.82
        "stubble": (0.15, 0.22, 0.30, 0.20),  # NDVI 0.19, NBR2 0.20
        "water": (0.03, 0.01, 0.005, 0.004),  # NDVI -0.5
    }


def test_bare_mask_separates_soil_from_crop_stubble_and_water(bs):
    px = _pixels()
    arr = np.array(list(px.values()), np.float32).T  # (4, n)
    mask = bs.bare_mask(*arr, **bs.STRICT)
    assert dict(zip(px, mask.tolist(), strict=True)) == {
        "bare_dark": True,
        "bare_light": True,
        "crop": False,
        "stubble": False,
        "water": False,
    }


def test_loose_mask_contains_strict(bs):
    rng = np.random.default_rng(0)
    arr = rng.uniform(0.01, 0.5, size=(4, 5000)).astype(np.float32)
    strict = bs.bare_mask(*arr, **bs.STRICT)
    loose = bs.bare_mask(*arr, **bs.LOOSE)
    assert strict.any() and (loose | ~strict).all() and loose.sum() > strict.sum()


def test_bare_mask_is_false_for_nan(bs):
    nan = np.array([np.nan], np.float32)
    one = np.array([0.1], np.float32)
    assert not bs.bare_mask(nan, one, one, one, **bs.STRICT)[0]
    assert not bs.bare_mask(one, one, nan, one, **bs.STRICT)[0]


def test_clear_mask_drops_clouds_with_a_margin(bs):
    scl = np.full((40, 40), 5, np.uint8)  # not vegetated
    scl[20, 20] = 9  # one cloud pixel
    scl[0, 0] = 6  # water
    scl[0, 1] = 11  # snow
    scl[5, 5] = 2  # dark area: kept (dark bare chernozem lands here)
    clear = bs.clear_mask(scl, dilate_px=3)
    assert not clear[20, 20] and not clear[20, 23] and not clear[17, 20]
    assert clear[20, 24] and clear[5, 5]
    assert not clear[0, 0] and not clear[0, 1]


def test_masked_median_ignores_invalid_dates(bs):
    stack = np.array([10, 20, 30, 9999, 40], np.float32).reshape(5, 1, 1) * np.ones((1, 2, 2))
    valid = np.ones((5, 2, 2), bool)
    valid[3] = False  # the outlier is a cloudy date
    valid[:, 1, 1] = False  # a pixel that was never bare
    valid[1:, 0, 1] = False  # a pixel bare once
    med, n = bs.masked_median(stack, valid)
    assert med[0, 0] == pytest.approx(25.0) and n[0, 0] == 4  # even count: mean of 20, 30
    assert med[0, 1] == pytest.approx(10.0) and n[0, 1] == 1
    assert np.isnan(med[1, 1]) and n[1, 1] == 0


def test_masked_median_matches_numpy_on_random_data(bs):
    rng = np.random.default_rng(1)
    stack = rng.integers(1000, 4000, size=(15, 6, 7)).astype(np.uint16)
    valid = rng.random((15, 6, 7)) > 0.4
    med, n = bs.masked_median(stack, valid)
    ref = np.nanmedian(np.where(valid, stack, np.nan), axis=0)
    np.testing.assert_allclose(med, ref, equal_nan=True)
    assert (n == valid.sum(axis=0)).all()


def test_darker_soil_gives_lower_composite(bs):
    """Two fields, the same dates bare; the humus-rich one stays darker."""
    rng = np.random.default_rng(2)
    dates = 8
    dark = 0.07 + rng.normal(0, 0.01, (dates, 1, 1))
    light = 0.16 + rng.normal(0, 0.01, (dates, 1, 1))
    stack = np.concatenate([dark, light], axis=2)
    valid = np.ones_like(stack, bool)
    valid[2] = False
    med, _ = bs.masked_median(stack, valid)
    assert med[0, 0] < med[0, 1] - 0.05


# --- analysis helpers (05f_bare_soil.py): pure array functions only ---------------
def _source(name: str) -> str:
    return (PROJECT_ROOT / "research" / "chernyakhiv" / name).read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def helpers():
    """focal_mean / anomaly / field_residual, taken out of 05f without importing the
    whole model stack (which needs the local rasters and the catalogue)."""
    import ast

    import pandas as pd
    from scipy import ndimage

    tree = ast.parse(_source("05f_bare_soil.py"))
    wanted = {"focal_mean", "anomaly", "field_residual"}
    consts = {"SMALL_PX", "LARGE_PX", "FIELD_EDGE", "MIN_FIELD_PX"}
    body = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in wanted)
        or (
            isinstance(node, ast.Assign)
            and any(
                n.id in consts for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name)
            )
        )
    ]
    scope = {"np": np, "ndimage": ndimage, "pd": pd}
    exec(compile(ast.Module(body, []), "05f_helpers", "exec"), scope)  # noqa: S102
    assert wanted <= scope.keys()
    return scope


def test_focal_mean_skips_nodata(helpers):
    a = np.full((9, 9), 2.0, np.float32)
    a[4, 4] = np.nan
    a[0, :] = np.nan
    mean, share = helpers["focal_mean"](a, 3, 0.2)
    assert mean[4, 4] == pytest.approx(2.0) and share[4, 4] == pytest.approx(8 / 9)
    only_nan = np.full((9, 9), np.nan, np.float32)
    assert np.isnan(helpers["focal_mean"](only_nan, 3)[0]).all()


def test_anomaly_is_negative_on_a_dark_spot_and_zero_on_flat_ground(helpers):
    a = np.ones((201, 201), np.float32)
    a[95:106, 95:106] = 0.7  # a 220 m dark spot
    out = helpers["anomaly"](a)
    assert out[100, 100] < -0.2
    assert abs(out[30, 30]) < 1e-5


def test_field_residual_removes_the_contrast_between_fields(helpers):
    """Two fields of different brightness and bare-date count; a dark spot in one."""
    dark = np.ones((60, 80), np.float32)
    n = np.full((60, 80), 20.0, np.float32)
    dark[:, 40:] = 1.4  # the brighter field
    n[:, 40:] = 40.0
    dark[25:30, 10:15] = 0.8  # the spot
    resid = helpers["field_residual"](dark, n)
    assert abs(np.nanmedian(resid[:, :36])) < 1e-6 and abs(np.nanmedian(resid[:, 44:])) < 1e-6
    assert resid[27, 12] == pytest.approx(-0.2, abs=1e-5)
    assert np.isnan(resid[:, 39:41]).all()  # the border itself is dropped


def test_field_residual_drops_tiny_patches(helpers):
    dark = np.full((40, 40), np.nan, np.float32)
    dark[5:9, 5:9] = 1.0  # 16 px, far below MIN_FIELD_PX
    n = np.full((40, 40), 10.0, np.float32)
    assert np.isnan(helpers["field_residual"](dark, n)).all()


# --- public layer (preprocessing/07_export_bare_soil.py) ---------------------------
DATA_DIR = PROJECT_ROOT / "docs" / "data"
VINNYTSIA_BOUNDS = (27.3, 47.9, 30.1, 49.95)  # west, south, east, north


@pytest.fixture(scope="module")
def export():
    path = PROJECT_ROOT / "preprocessing" / "07_export_bare_soil.py"
    spec = importlib.util.spec_from_file_location("export_bare_soil", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_block_mean_needs_enough_data(export):
    a = np.full((10, 10), np.nan, np.float32)
    a[:5, :5] = 2.0  # full block
    a[0, 5] = 4.0  # one cell of 25 in the next block
    out = export.block_mean(a, 5, 0.3)
    assert out.shape == (2, 2)
    assert out[0, 0] == pytest.approx(2.0)
    assert np.isnan(out[0, 1]) and np.isnan(out[1, 1])


def test_classes_run_from_dark_to_light_and_nodata_is_transparent(export):
    breaks = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    values = np.array([[0.5, 0.85, 1.05, 1.5, np.nan]], np.float32)
    classes = export.classify(values, breaks)
    assert classes.tolist() == [[1, 2, 4, 7, 0]]
    rgba = export.colorize(classes)
    assert rgba.shape == (4, 1, 5)
    assert rgba[3, 0, 4] == 0 and (rgba[3, 0, :4] == export.ALPHA).all()
    brightness = rgba[:3, 0, :4].astype(int).sum(axis=0)
    assert (np.diff(brightness) > 0).all()  # darker soil -> darker colour
    assert len(export.COLORS) == len(export.BREAKS_Q) + 1


def test_public_layer_files():
    import json

    import rasterio

    meta = json.loads((DATA_DIR / "bare_soil_darkness.json").read_text(encoding="utf-8"))
    (south, west), (north, east) = meta["bounds"]
    w, s, e, n = VINNYTSIA_BOUNDS
    assert w < west < east < e and s < south < north < n
    assert meta["breaks"] == sorted(meta["breaks"]) and len(meta["breaks"]) == 6
    assert "Copernicus Sentinel" in meta["attribution"] and "WorldCover" in meta["attribution"]
    png = DATA_DIR / "bare_soil_darkness.png"
    assert png.stat().st_size < 6 * 1024 * 1024  # light enough for the browser
    with rasterio.open(png) as src:
        assert src.count == 4 and [src.width, src.height] == meta["size"]
        alpha = src.read(4)
    share = (alpha > 0).mean()
    assert 0.2 < share < 0.8  # arable land of the oblast inside its bounding box
    assert alpha[0, 0] == 0  # outside the oblast is transparent


def test_site_mentions_the_layer_and_its_sources():
    html = (PROJECT_ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    js = (PROJECT_ROOT / "docs" / "app.js").read_text(encoding="utf-8")
    assert 'id="toggle-bare-soil"' in html and 'id="bare-soil-legend"' in html
    assert "Copernicus Sentinel data" in html and "ESA WorldCover" in html
    assert "bare_soil_darkness.png" in js and "bare_soil_darkness.json" in js
