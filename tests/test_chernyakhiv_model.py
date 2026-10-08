"""
Unit tests for the evaluation helpers of research/chernyakhiv/05c_model.py
(synthetic numbers, no local data needed).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load():
    path = PROJECT_ROOT / "research" / "chernyakhiv" / "05c_model.py"
    spec = importlib.util.spec_from_file_location("model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


model = _load()
rng = np.random.default_rng(0)
BACKGROUND = rng.uniform(0, 1, 10_000)


def test_boyce_monotone_prediction():
    presence = rng.beta(4, 1, 2000)  # more presences where prediction is high
    assert model.boyce(presence, BACKGROUND) > 0.9


def test_boyce_random_prediction_is_near_zero():
    presence = rng.uniform(0, 1, 2000)
    assert abs(model.boyce(presence, BACKGROUND)) < 0.7


@pytest.mark.parametrize("share", [0.1, 0.2, 0.5])
def test_top_share_random_equals_area_share(share):
    presence = rng.uniform(0, 1, 20_000)
    assert model.top_share(presence, BACKGROUND, share) == pytest.approx(share, abs=0.02)


def test_window_max_ignores_nan():
    a = np.zeros((30, 30), dtype="float32")
    a[20, 20] = 5.0
    a[:3, :3] = np.nan  # outside the oblast stays nodata
    out = model.window_max(a)
    assert out[20, 25] == 5.0 and out[5, 5] == 0.0 and np.isnan(out[0, 0])
