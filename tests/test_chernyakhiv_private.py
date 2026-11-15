"""
Guard for the encrypted viewer on GitHub Pages (research/chernyakhiv/05b_viewer.py
--private): docs/private must never hold the site data in the clear.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import pytest

PRIVATE = Path(
    os.environ.get("CHERNYAKHIV_PRIVATE_DIR")
    or Path(__file__).resolve().parent.parent / "docs" / "private"
)
ALLOWED = {
    "index.html",
    "sites.enc.json",
    "channels.png",
    "hillshade.webp",
    "suitability.png",
    "waterways.geojson",
}


@pytest.fixture(scope="module")
def files() -> set[str]:
    if not PRIVATE.exists():
        pytest.skip("run research/chernyakhiv/05b_viewer.py --private")
    return {p.name for p in PRIVATE.iterdir()}


def test_only_known_files(files):
    assert files <= ALLOWED, files - ALLOWED
    assert "sites.enc.json" in files


def test_site_data_is_encrypted(files):
    raw = (PRIVATE / "sites.enc.json").read_text(encoding="ascii")
    enc = json.loads(raw)
    assert enc["cipher"] == "AES-256-GCM" and enc["kdf"] == "PBKDF2-SHA256"
    assert enc["iterations"] >= 300_000
    assert len(base64.b64decode(enc["salt"])) >= 16 and len(base64.b64decode(enc["iv"])) == 12
    assert set(enc) == {"kdf", "iterations", "cipher", "compression", "salt", "iv", "ct"}


def test_page_ships_no_plain_site_data(files):
    html = (PRIVATE / "index.html").read_text(encoding="utf-8")
    assert '"encrypted"' in html and "noindex" in html
    assert "Поселення «" not in html
    ways = (PRIVATE / "waterways.geojson").read_text(encoding="utf-8")
    assert '"snap_how"' not in ways and '"desc"' not in ways


# --- generator logic (synthetic data, no local files needed) -----------------


def _viewer():
    import importlib.util

    path = Path(__file__).resolve().parent.parent / "research" / "chernyakhiv" / "05b_viewer.py"
    spec = importlib.util.spec_from_file_location("viewer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_candidates_keep_clear_of_known_sites():
    import numpy as np
    from rasterio.transform import from_origin

    viewer = _viewer()
    rank = np.full((40, 40), 50.0, dtype="float32")
    rank[:, 10:14] = 95.0  # a suitable strip, 150 m cells
    rank[0, 0] = -9999.0  # nodata is never a candidate
    transform = from_origin(0, 6000, 150, 150)
    site = np.array([[1800.0, 3000.0]])  # inside the strip
    mask = viewer.candidate_mask(rank, transform, site, gap_m=1000, min_rank=90)
    assert mask.any() and not mask[rank < 90].any()
    rows, cols = np.nonzero(mask)
    x, y = (cols + 0.5) * 150, 6000 - (rows + 0.5) * 150
    assert (np.hypot(x - site[0, 0], y - site[0, 1]) > 1000).all()
    # ... and everything suitable beyond the gap is kept
    far = (rank >= 90) & (np.abs(6000 - (np.arange(40)[:, None] + 0.5) * 150 - 3000) > 1300)
    assert mask[far].all()


def test_accuracy_classes():
    viewer = _viewer()
    assert viewer.accuracy("named", "N") == "A"
    assert viewer.accuracy("dem", "") == "B"
    assert viewer.accuracy("none", "SW") == "C"
    assert viewer.accuracy("none", "C") == "D"  # "in the centre": the village alone
    assert viewer.accuracy("none", None) == "D"
    assert set(viewer.ACCURACY_M) == {"A", "B", "C", "D"}


def test_candidates_never_go_into_a_plain_file(files):
    # Their holes mark the known sites, so they must stay inside sites.enc.json.
    # (index.html only names the role in its script; the data files must not.)
    for name in files - {"sites.enc.json", "index.html"}:
        if name.endswith((".png", ".webp")):
            continue
        assert '"candidate"' not in (PRIVATE / name).read_text(encoding="utf-8")
