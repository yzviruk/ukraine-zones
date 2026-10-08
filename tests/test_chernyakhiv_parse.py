"""
Unit tests for the catalogue parser (research/chernyakhiv/01_parse_catalog.py).

Uses short synthetic snippets in the catalogue's style, so the copyrighted
PDF is not needed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_parser():
    path = PROJECT_ROOT / "research" / "chernyakhiv" / "01_parse_catalog.py"
    spec = importlib.util.spec_from_file_location("parse_catalog", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


parser = _load_parser()


@pytest.mark.parametrize(
    ("desc", "expected"),
    [
        ("Поселення на південний схід від села на правому березі", "SE"),
        ("Поселення східніше села на південному схилі плато", "E"),
        ("Поселення на північно-західній околиці села", "NW"),
        ("Поселення в центрі села на лівому березі", "C"),
        ("Поселення на північно-східному схилі плато", ""),  # aspect only
        ("Поселення поблизу села.", ""),
    ],
)
def test_direction(desc, expected):
    assert parser.direction(desc) == expected


def test_join_undoes_hyphenation():
    lines = ["на правому бе-", "резі р. Рось. Взято на облік Обл -", "держадміністрації"]
    assert parser.join(lines) == "на правому березі р. Рось. Взято на облік Облдержадміністрації"


def test_village_header():
    m = parser.VILLAGE_HEADER.match(
        "с. Бабин, кол. Іллінецький район (Синарна — Соб — Південний Буг)"
    )
    assert m["name"] == "Бабин"
    assert m["old"] == "Іллінецький"
    assert m["rivers"].startswith("Синарна")


def test_parse_entry_fields():
    text = (
        "7. Поселення «Бабин 3» в центрі села на схилах плато вздовж лівого берега "
        "струмка і над ставом. Розміри 400 × 100 м, культурний шар 0,1 м. Ореться. "
        "Знахідки: ліпна кераміка. Виявив П. І. Хавлюк у 1982 р. "
        "Взято на облік рішенням Облради, охоронний № 59."
    )
    row = parser.parse_entry(7, text)
    assert row["type"] == "Поселення"
    assert row["site_name"] == "Бабин 3"
    assert row["direction"] == "C"
    assert row["bank"] == "left"
    assert (row["length_m"], row["width_m"]) == ("400", "100")
    assert row["year_found"] == "1982"
    assert row["protection_no"] == "59"
    assert row["text"] == text
    assert row["ploughed"] == 1
    assert row["lf_plateau"] == row["lf_slope"] == row["w_stream"] == row["w_pond"] == 1


@pytest.mark.parametrize(
    ("text", "water", "is_river"),
    [
        ("12. Поселення поблизу села. Виявив М. В. Потупчик у 2003 р. Потупчик 2012.", "", 0),
        ("13. Поселення на лівому березі р. Мурашка. Виявив у 1996 р. Левада.", "Мурашка", 1),
    ],
)
def test_year_is_not_a_river(text, water, is_river):
    row = parser.parse_entry(12, text)
    assert row["water_name"] == water
    assert row["w_river"] == is_river
