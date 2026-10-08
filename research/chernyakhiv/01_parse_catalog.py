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
    ("N", r"на північ\b|північніше|північн\w* (?:околиц|частин|край)"),
    ("S", r"на південь\b|південніше|південн\w* (?:околиц|частин|край)"),
    ("E", r"на сх[іо]д\b|східніше|східн\w* (?:околиц|частин|край)"),
    ("W", r"на зах[іо]д\b|західніше|західн\w* (?:околиц|частин|край)"),
    ("C", r"(?:в|у) центр|на території|в межах (?:села|міста|селища)"),
]
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


def parse_entry(no: int, text: str) -> dict:
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
        "distance_km": "",
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
    # Distance from the village/landmark: "у 5 км від", "за 300 м на схід".
    m = re.search(r"\b(?:за|у|в|на відстані)\s+(\d+(?:,\d+)?)\s*(км|м)\b", desc)
    if m:
        val = float(m.group(1).replace(",", "."))
        row["distance_km"] = val if m.group(2) == "км" else round(val / 1000, 3)
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
