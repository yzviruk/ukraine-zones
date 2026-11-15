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


@pytest.mark.parametrize(
    ("desc", "expected"),
    [
        ("Поселення в південній частині села на плато", "inside"),
        ("Поселення в центрі села на схилах плато", "inside"),
        ("Поселення на території села, частково забудоване", "inside"),
        ("Поселення на східній околиці села", "outskirts"),
        ("Поселення на східному краю села на схилі плато", "outskirts"),
        ("Поселення на схід від села на правому березі", "beyond"),
        ("Поселення за східною околицею смт", "beyond"),
        ("Поселення навпроти (через річку) північної околиці села", "beyond"),
        ("Поселення поблизу села.", "near"),
    ],
)
def test_position(desc, expected):
    assert parser.position(desc, parser.direction(desc)) == expected


def test_edge_of_the_village_gives_a_direction():
    assert parser.direction("Поселення на східному краю села") == "E"


@pytest.mark.parametrize(
    ("desc", "km"),
    [
        ("Поселення у 4 км західніше села на мисі", 4.0),
        ("Поселення в 0,5 км на схід від с. Пісочин на схилах", 0.5),
        ("Поселення за 1,5 км західніше південно-західної околиці села", 1.5),
        ("Поселення у 800 м на захід від села на терасі", 0.8),
        ("Поселення у 1,5 км від с. Попова Гребля на мису", 1.5),
        # distances to other landmarks are not positions relative to the village
        ("Поселення на околиці села у 5 км від впадіння р. Рудка", ""),
        ("Поселення в північній частині села в 100 м нижче греблі", ""),
        ("Поселення на території смт, у 300 м східніше поселення «Бродецьке 1»", ""),
        ("Поселення за околицею смт в 30 м від лівого берега р. Гнилоп’ять", ""),
    ],
)
def test_distance_only_from_the_village(desc, km):
    assert parser.parse_entry(1, "1. " + desc + ". Розміри 300 × 100 м.")["distance_km"] == km


def test_second_village_and_between():
    row = parser.parse_entry(
        1, "1. Поселення на південний захід від с. Красносілка і на північний захід від с. Залісся."
    )
    assert (row["direction"], row["ref_village"]) == ("SW", "Красносілка")
    assert (row["ref2_village"], row["ref2_direction"]) == ("Залісся", "NW")
    row = parser.parse_entry(2, "2. Поселення між с. Вітрівка та с. Степове на схилі плато.")
    assert (row["ref_village"], row["between_village"]) == ("Вітрівка", "Степове")
    row = parser.parse_entry(
        3, "3. Поселення на схід від села с. Свитинці і на захід від с. Ширмівка."
    )
    assert row["ref_village"] == "" and row["ref2_village"] == "Ширмівка"


def test_figure_caption_inside_a_word_is_removed():
    text = (
        "276. Поселення на південній околиці села вздовж лівого береРис. 2.27. Поселення "
        "Мар’янівка І (с. Мар’янівка): 1, 3 — миски (за: Милян 2014, рис. 4; 5) га р. Рів. "
        "Розміри 400 × 100 м."
    )
    row = parser.parse_entry(276, text)
    assert "Рис." not in row["desc"]
    assert (row["bank"], row["water_name"]) == ("left", "Рів")
