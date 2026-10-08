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
