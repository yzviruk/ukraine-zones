"""
Stage 1: parse the catalogue of Chernyakhiv-culture sites of Vinnytsia oblast
(Магомедов Б. В., ІА НАН України, 2022) into a flat table.

Input:  data/raw/magomedov_2022.pdf  (http://www.vgosau.kiev.ua/load_books/Magomedov_2022.pdf)
Output: data/processed/chernyakhiv_catalog.csv  (gitignored: copyrighted source)

One row per catalogue entry. Location fields are extracted from the free-text
description with regexes; empty means "not mentioned", not "absent".
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PDF = PROJECT_ROOT / "data" / "raw" / "magomedov_2022.pdf"
OUT_CSV = PROJECT_ROOT / "data" / "processed" / "chernyakhiv_catalog.csv"

DISTRICTS = [
    "вінницький",
    "гайсинський",
    "жмеринський",
    "могилів-подільський",
    "тульчинський",
    "хмільницький",
]
CATALOGUE_END = "ЗНАХІДКИ МОНЕТ"

UPPER = "А-ЯІЇЄҐ"
PAGE_NO = re.compile(r"^\[ \d+ \]$")
ENTRY_START = re.compile(r"^(\d{1,3})\. (\w.*)$")
RUNNING_TITLE = "2. Каталог пам’яток черняхівської культури"
VILLAGE_START = re.compile(rf"^(с\.|смт|м\.) [{UPPER}]")
VILLAGE_HEADER = re.compile(
    rf"^(?P<kind>с|смт|м)\.? (?P<name>[{UPPER}][^,(]*?)"
    r"(?:, (?:кол\. )?(?P<old>[^()]+?) район)?\s*\((?P<rivers>[^)]+)\)$"
)
CROSS_REF = re.compile(rf"^(с\.|смт|м\.) [{UPPER}][^,]*, див\. ")

ASPECTS = (
    "північно-східн|північно-західн|південно-східн|південно-західн|північн|південн|східн|західн"
)
_N, _S, _E, _W = r"північн", r"південн", r"сх[іо]д", r"зах[іо]д"
_NOT_ASPECT = r"\w*+(?! схил)"  # "південному схилі" is slope aspect, not direction
COMPASS = [
    ("NE", _N + r"\w*[ -]" + _E + _NOT_ASPECT),
    ("NW", _N + r"\w*[ -]" + _W + _NOT_ASPECT),
    ("SE", _S + r"\w*[ -]" + _E + _NOT_ASPECT),
    ("SW", _S + r"\w*[ -]" + _W + _NOT_ASPECT),
    ("N", r"на північ\b|північніше|північн\w* (?:околиц|частин|кра[йюї])"),
    ("S", r"на південь\b|південніше|південн\w* (?:околиц|частин|кра[йюї])"),
    ("E", r"на сх[іо]д\b|східніше|східн\w* (?:околиц|частин|кра[йюї])"),
    ("W", r"на зах[іо]д\b|західніше|західн\w* (?:околиц|частин|кра[йюї])"),
    ("C", r"(?:в|у) центр|на території|в межах (?:села|міста|селища)"),
]
# Where the site lies relative to the village (stage 2 turns it into a distance):
#   inside     "у південній частині села", "в центрі", "на території", "на городах"
#   outskirts  "на східній околиці", "на краю села"
#   beyond     "на схід від села", "східніше", "за околицею", "навпроти (через річку)"
#   near       "поблизу села" and no compass direction
_SETTLEMENT = r"(?:села|селища|міста|смт)"
POSITION = [
    ("beyond", r"за \S+ околицею|від \S+ околиці|навпроти|через річку"),
    ("outskirts", r"околиц|\bкра[юї] " + _SETTLEMENT),
    (
        "inside",
        r"частин\w+ " + _SETTLEMENT + r"|(?:в|у) центр|на території|в межах|\bв селі\b"
        r"|на городах (?:села|садиб)",
    ),
]
_DIR_WORD = r"(?:північн|південн|сх[іо]д|зах[іо]д)[\w-]*(?: (?:сх[іо]д|зах[іо]д)\w*)?"
_NAME = rf"[{UPPER}][\w’'-]+(?: [{UPPER}][\w’'-]+)?"
# Distance FROM THE VILLAGE only: "у 4 км західніше села", "в 0,5 км на схід від
# с. Пісочин", "за 1,5 км західніше південно-західної околиці села". Distances to
# other landmarks ("в 100 м нижче греблі", "у 300 м східніше поселення «…»") are
# not positions relative to the village and are ignored.
VILLAGE_DISTANCE = re.compile(
    r"\b(?:за|у|в)\s+(\d+(?:,\d+)?)\s*(км|м)\s+(?:на\s+)?"
    + _DIR_WORD
    + r"\s+(?:від\s+)?(?:[\w’'-]+\s+){0,2}?(?:села|селища|міста|смт|с\.|околиці)"
    r"|\b(?:за|у|в)\s+(\d+(?:,\d+)?)\s*(км|м)\s+від\s+(?:села|селища|міста|смт|с\.)"
)
# A second village fixing the position: "... і на захід від с. Ширмівка".
SECOND_REF = re.compile(
    r" (?:і|та|й) (?:за [\d,]+ к?м )?((?:на )?" + _DIR_WORD + r") від (?:[\w-]+ околиці )?"
    rf"(?:кол\. )?(?:с\.|смт|м\.) ({_NAME})"
)
# "між с. Вітрівка та с. Степове", "між кол. с. Паланка (...) і с. Вигнанка"
BETWEEN = re.compile(
    rf"між (?:[\w-]+ околицею )?(?:кол\. )?(?:с\.|смт|м\.) ({_NAME})[^.]*? (?:та|і|й) "
    rf"(?:[\w-]+ околицею )?(?:кол\. )?(?:с\.|смт|м\.) ({_NAME})"
)
# A figure caption that the PDF text layer drops into the middle of a word:
# "береРис. 2.27. Поселення … (за: Милян, Чорний 2014, рис. 4; 5) га р. Рів".
CAPTION_IN_WORD = re.compile(r"(?<=[а-яіїєґ])Рис\. \d+\.\d+\. .*?\)\s?(?=[а-яіїєґ])")
LANDFORMS = {
    "plateau": r"плато",
    "slope": r"схил",
    "terrace": r"терас",
    "floodplain": r"заплав",
    "cape": r"мис",
    "ravine": r"балк|\bяр(?:у|ок|ка|ками|и|ом|ами|ів)?\b",
    "rise": r"підвищен|пагорб",
}
NOT_YEAR = r"(?<!\d)(?<!\d )"  # "у 2003 р. Потупчик" is a year, not a river
WATERS = {
    "river": NOT_YEAR + r"\bр\. [" + UPPER + r"]",
    "stream": r"струм",
    "pond": r"став",
    "spring": r"джерел",
    "ravine_water": r"балк",
}


def read_lines() -> list[str]:
    text = "\n".join(page.extract_text() or "" for page in PdfReader(PDF).pages)
    text = text.replace(" ", " ").replace(" ", " ")
    return [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]


def catalogue_lines(lines: list[str]) -> list[tuple[str, str]]:
    """(district, line) pairs for the catalogue body, page furniture removed."""
    start = next(i for i, ln in enumerate(lines) if ln.lower() == "вінницький район")
    out, district = [], DISTRICTS[0]
    for ln in lines[start:]:
        if ln.startswith(CATALOGUE_END):
            break
        if not ln:
            continue
        if ln.startswith(RUNNING_TITLE):
            continue
        if PAGE_NO.match(ln):
            continue
        # District headers: section starts and running headers on odd pages.
        low = ln.lower()
        if low.endswith(" район") and low[: -len(" район")] in DISTRICTS:
            district = low[: -len(" район")]
            continue
        out.append((district, ln))
    return out


def join(lines: list[str]) -> str:
    """Join wrapped lines, undoing hyphenation ("бе-\\nрезі", "Обл -\\nдерж")."""
    text = ""
    for ln in lines:
        if text.endswith(("-", "—")):
            text = text.rstrip(" -") if text.endswith("-") else text
            text += ln
        else:
            text += (" " if text else "") + ln
    text = re.sub(r"(\w) -(\w)", r"\1\2", text)  # leftover " -" splits
    return re.sub(r"\s+", " ", text).strip()


def blocks(pairs: list[tuple[str, str]]):
    """Yield ("village", district, text) and ("entry", district, text) blocks."""
    kind, district, buf = None, None, []
    for i, (dist, ln) in enumerate(pairs):
        is_entry = ENTRY_START.match(ln)
        is_village = False
        if VILLAGE_START.match(ln):
            # A village header is only the name/old district/river system:
            # look ahead until the next entry and check the whole chunk.
            chunk = [ln]
            for _, nxt in pairs[i + 1 : i + 5]:
                if ENTRY_START.match(nxt) or VILLAGE_START.match(nxt):
                    break
                chunk.append(nxt)
            joined = join(chunk)
            is_village = bool(VILLAGE_HEADER.match(joined) or CROSS_REF.match(joined))
        if is_entry or is_village:
            if kind:
                yield kind, district, join(buf)
            kind, district, buf = ("entry" if is_entry else "village"), dist, [ln]
        elif kind:
            buf.append(ln)
    if kind:
        yield kind, district, join(buf)


def first(pattern: str, text: str, group: int = 1) -> str:
    m = re.search(pattern, text)
    return m.group(group) if m else ""


def direction(desc: str) -> str:
    """Compass position relative to the village: the earliest one mentioned."""
    low = desc.lower()
    hits = []
    for code, pat in COMPASS:
        m = re.search(pat, low)
        if m:
            hits.append((m.start(), -len(code), code))
    return min(hits)[2] if hits else ""


def position(desc: str, compass: str) -> str:
    low = desc.lower()
    for kind, pat in POSITION:
        if re.search(pat, low):
            return kind
    if compass == "C":
        return "inside"
    return "beyond" if compass else "near"


def compass_of(phrase: str) -> str:
    """Compass code of a bare direction phrase ("на південний захід", "західніше")."""
    low = phrase.lower()
    north, south = "північ" in low, "півден" in low
    east, west = bool(re.search(_E, low)), bool(re.search(_W, low))
    return ("N" if north else "S" if south else "") + ("E" if east else "W" if west else "")


def parse_entry(no: int, text: str) -> dict:
    text = CAPTION_IN_WORD.sub("", text)
    body = text.split(" Взято на облік")[0]
    desc = re.split(r" Знахідки| Рис\. \d", body)[0]
    low = desc.lower()
    size = re.search(r"(\d+)\s*×\s*(\d+)\s*м", body)
    row = {
        "no": no,
        "type": "Могильник" if re.match(r"^\d+\. Могильник", text) else "Поселення",
        "site_name": first(r"«([^»]+)»", desc),
        "bank": first(r"(лів|прав)\w* бер[еі][гз]", low)
        .replace("лів", "left")
        .replace("прав", "right"),
        "water_name": first(rf"{NOT_YEAR}\bр\. ([{UPPER}][\w’'-]+(?: [{UPPER}][\w’'-]+)?)", desc),
        # Position is sometimes given relative to ANOTHER village.
        "ref_village": first(
            rf"(?:від|біля|поблизу|між) (?:с\.|смт|м\.) ([{UPPER}][\w’'-]+(?: [{UPPER}][\w’'-]+)?)",
            desc,
        ),
        "aspect": first(r"(" + ASPECTS + r")\w* схил", low),
        "direction": direction(desc),
        "position": "",
        "distance_km": "",
        # A second village that pins the site down, see stage 2:
        "ref2_village": "",  # "... і на захід від с. X" (with ref2_direction)
        "ref2_direction": "",
        "between_village": "",  # "між с. A та с. X"
        "length_m": size.group(1) if size else "",
        "width_m": size.group(2) if size else "",
        "layer_m": first(r"(?:культурний шар|товщина культурного шару) ([\d,—–-]+) м", body),
        "ploughed": int("ореться" in body.lower()),
        "year_found": first(r"(?:Вияв|Відкри|Обстеж)\w*\b.{0,80}?\b(1[89]\d\d|20[0-2]\d) р", body),
        "protection_no": first(r"охоронний № (\d+)", text),
        "desc": desc,
        "text": text,  # the full catalogue entry, for the private viewer
    }
    for key, pat in LANDFORMS.items():
        row["lf_" + key] = int(bool(re.search(pat, low)))
    for key, pat in WATERS.items():
        row["w_" + key] = int(bool(re.search(pat, desc if key == "river" else low)))
    row["position"] = position(desc, row["direction"])
    m = VILLAGE_DISTANCE.search(desc)
    if m:
        value, unit = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        val = float(value.replace(",", "."))
        row["distance_km"] = val if unit == "км" else round(val / 1000, 3)
    m = SECOND_REF.search(desc)
    if m and compass_of(m.group(1)):
        row["ref2_village"], row["ref2_direction"] = m.group(2), compass_of(m.group(1))
    m = BETWEEN.search(desc)
    if m:
        row["ref_village"], row["between_village"] = m.group(1), m.group(2)
    if row["ref_village"] and row["ref_village"] == row["ref2_village"]:
        row["ref_village"] = ""  # the first village is the header one
    return row


def main() -> None:
    if not PDF.exists():
        raise SystemExit(f"Missing {PDF}; download it from the URL in the docstring.")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    village = {"village": "", "village_kind": "", "old_district": "", "river_system": ""}
    rows, problems = [], []
    for kind, district, text in blocks(catalogue_lines(read_lines())):
        if kind == "village":
            m = VILLAGE_HEADER.match(text)
            if m:
                village = {
                    "village": m["name"].strip(),
                    "village_kind": m["kind"],
                    "old_district": (m["old"] or "").strip(),
                    "river_system": re.sub(r"\s*—\s*", " — ", m["rivers"]).strip(),
                }
            continue
        no = int(ENTRY_START.match(text).group(1))
        row = {"district": district, **village, **parse_entry(no, text)}
        rows.append(row)
        if not row["village"] or not row["desc"]:
            problems.append(no)

    nos = [r["no"] for r in rows]
    missing = sorted(set(range(1, max(nos) + 1)) - set(nos))
    dupes = sorted({n for n in nos if nos.count(n) > 1})
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} entries -> {OUT_CSV.relative_to(PROJECT_ROOT)}")
    print(f"missing numbers: {missing}")
    print(f"duplicate numbers: {dupes}")
    print(f"entries without village/desc: {problems}")
    if missing or dupes:
        sys.exit(1)


if __name__ == "__main__":
    main()
