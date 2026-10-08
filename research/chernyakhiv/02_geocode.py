"""
Stage 2: geocode the Chernyakhiv catalogue (stage 1) onto OSM villages.

Input:
    data/processed/chernyakhiv_catalog.csv      (stage 1)
    data/raw/ua-places.geojson                  (OSM place nodes, with koatuu)
    data/raw/ua-border.geojson                  (oblast and raion polygons)
    data/processed/stage1.gpkg                  (built-up areas, project stage 1)
    docs/data/rivers.geojson, streams.geojson   (named waterways)
Output:
    data/processed/chernyakhiv_sites.gpkg       (approximate site points, LOCAL ONLY)
    docs/data/chernyakhiv_villages.geojson      (public: one point per village)

Village matching: name (+ old/alt names) inside the oblast, then disambiguated by
(1) the new raion polygon, (2) the old raion via KOATUU district code learned from
unambiguous matches, (3) proximity to the river named in the description.

Site position: the anchor village (or the other village the text refers to),
shifted in the stated compass direction to just outside the built-up edge.
Accuracy classes:
    A  village only (no direction, or "in the centre")
    B  shifted by direction
    C  village not found -> not placed
The public layer deliberately stays at village precision (anti-looting decision,
see research/chernyakhiv/PLAN.md).
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW = PROJECT_ROOT / "data" / "raw"
PROCESSED = PROJECT_ROOT / "data" / "processed"
CATALOG = PROCESSED / "chernyakhiv_catalog.csv"
PLACES = RAW / "ua-places.geojson"
BORDER = RAW / "ua-border.geojson"
STAGE1 = PROCESSED / "stage1.gpkg"
WATERWAYS = [PROJECT_ROOT / "docs" / "data" / f for f in ("rivers.geojson", "streams.geojson")]
OUT_SITES = PROCESSED / "chernyakhiv_sites.gpkg"
OUT_PUBLIC = PROJECT_ROOT / "docs" / "data" / "chernyakhiv_villages.geojson"

OBLAST = "Вінницька область"
METRIC_CRS = 6381
BEYOND_EDGE_M = 300  # "east of the village" -> this far past the built-up edge
EDGE_M = 0  # "on the eastern outskirts" -> on the edge
NO_POLYGON_M = 750  # shift when the village has no built-up polygon
RIVER_SEARCH_M = 3000

BEARINGS = {"N": 0, "NE": 45, "E": 90, "SE": 135, "S": 180, "SW": 225, "W": 270, "NW": 315}
NAME_COLUMNS = ["name", "name:uk", "old_name", "alt_name"]
# Catalogue spelling -> OSM spelling, for names the generic match misses.
ALIASES = {"перепеличе": "перепеличчя"}


def norm(name) -> str:
    if not isinstance(name, str):
        return ""
    name = name.lower().replace("i", "і")  # Latin i appears in the PDF text
    name = re.sub(r"[ʼ’`]", "'", name)
    return re.sub(r"\s+", " ", name).strip()


def raion_title(district: str) -> str:
    return "-".join(p.capitalize() for p in district.split("-")) + " район"


def load_admin():
    border = gpd.read_file(BORDER)
    border = border[border.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    oblast = border[border["name"] == OBLAST].geometry.iloc[0]
    raions = border[border["admin_level"] == "6"]
    raions = raions[raions.geometry.representative_point().within(oblast)]
    return oblast, dict(zip(raions["name"], raions.geometry, strict=True))


def load_places(oblast) -> gpd.GeoDataFrame:
    places = gpd.read_file(PLACES, columns=NAME_COLUMNS + ["place", "koatuu"])
    places = places[places.geometry.geom_type == "Point"]
    places = places[places.within(oblast.buffer(0.02))].reset_index(drop=True)
    places["district_code"] = places["koatuu"].str[2:5]
    return places


def name_index(places) -> dict[str, list[int]]:
    index = defaultdict(set)
    for i, row in places.iterrows():
        for col in NAME_COLUMNS:
            if isinstance(row[col], str):
                for name in row[col].split(";"):
                    index[norm(name)].add(i)
    return {k: sorted(v) for k, v in index.items()}


def load_waterways(oblast) -> gpd.GeoDataFrame:
    parts = [gpd.read_file(f, bbox=oblast.bounds) for f in WATERWAYS]
    ways = pd.concat(parts, ignore_index=True)
    ways = ways[ways["name"].notna()].copy()
    ways["key"] = ways["name"].map(norm)
    return ways.to_crs(METRIC_CRS)


def river_candidates(row) -> list[str]:
    names = [row["water_name"]] if isinstance(row["water_name"], str) else []
    if isinstance(row["river_system"], str):
        names += [n.strip() for n in re.split(r"—|,", row["river_system"])[:1]]
    return [norm(n) for n in names if n]


def resolve(name, row, places, index, raions, code_of, ways) -> tuple[int | None, str]:
    """Return (place index, how) for a village name of a catalogue row."""
    key = norm(name)
    cands = index.get(ALIASES.get(key, key), [])
    if len(cands) <= 1:
        return (cands[0], "unique") if cands else (None, "not_found")

    raion = raions.get(raion_title(row["district"]))
    if raion is not None:
        inside = [i for i in cands if places.geometry[i].within(raion)]
        cands = inside or cands
    if len(cands) == 1:
        return cands[0], "raion"

    code = code_of.get(old_district_key(row))
    if code:
        same = [i for i in cands if places["district_code"][i] == code]
        cands = same or cands
    if len(cands) == 1:
        return cands[0], "koatuu"

    is_town = row["village_kind"] in ("смт", "м")
    kinds = ("town", "city") if is_town else ("village",)  # catalogue "с." is rarely a hamlet
    same_kind = [i for i in cands if places["place"][i] in kinds]
    cands = same_kind or cands
    if len(cands) == 1:
        return cands[0], "kind"

    for river in river_candidates(row):
        lines = ways[ways["key"] == river]
        if lines.empty:
            continue
        geom = unary_union(lines.geometry.values)
        pts = gpd.GeoSeries([places.geometry[i] for i in cands], crs=4326).to_crs(METRIC_CRS)
        dists = [p.distance(geom) for p in pts]
        if min(dists) < RIVER_SEARCH_M:
            return cands[dists.index(min(dists))], "river"
    # Usually duplicate OSM nodes of one village: prefer the coded one.
    coded = [i for i in cands if isinstance(places["koatuu"][i], str)]
    if len(coded) == 1:
        return coded[0], "koatuu_node"
    return cands[0], "ambiguous"


def old_district_key(row) -> str:
    old = row["old_district"] if isinstance(row["old_district"], str) else ""
    return norm(old or row["district"])


def learn_district_codes(catalog, places, index) -> dict[str, str]:
    """Old raion name -> KOATUU district code, voted by unambiguous matches."""
    votes = defaultdict(Counter)
    for _, row in catalog.iterrows():
        cands = index.get(norm(row["village"]), [])
        code = places["district_code"][cands[0]] if len(cands) == 1 else None
        if isinstance(code, str):
            votes[old_district_key(row)][code] += 1
    return {k: c.most_common(1)[0][0] for k, c in votes.items()}


def built_up(oblast):
    areas = gpd.read_file(STAGE1, layer="settlement_areas", bbox=None)
    oblast_m = gpd.GeoSeries([oblast], crs=4326).to_crs(METRIC_CRS).iloc[0]
    parts = areas.explode(index_parts=False).geometry
    return gpd.GeoSeries(parts[parts.intersects(oblast_m)].values, crs=METRIC_CRS)


def shift(anchor: Point, direction: str, outskirts: bool, polygons, sindex) -> Point:
    """Move from the village node to its built-up edge in the given direction."""
    if direction not in BEARINGS:
        return anchor
    angle = math.radians(BEARINGS[direction])
    dx, dy = math.sin(angle), math.cos(angle)
    near = [polygons.iloc[i] for i in sindex.query(anchor.buffer(500))]
    poly = next((p for p in near if p.distance(anchor) < 500), None)
    if poly is None:
        return Point(anchor.x + dx * NO_POLYGON_M, anchor.y + dy * NO_POLYGON_M)
    ray = LineString([anchor, (anchor.x + dx * 20_000, anchor.y + dy * 20_000)])
    hit = ray.intersection(poly.boundary)
    pts = list(getattr(hit, "geoms", [hit])) if not hit.is_empty else []
    edge = max((anchor.distance(p) for p in pts), default=0.0)
    extra = EDGE_M if outskirts else BEYOND_EDGE_M
    return Point(anchor.x + dx * (edge + extra), anchor.y + dy * (edge + extra))


def main() -> None:
    catalog = pd.read_csv(CATALOG)
    oblast, raions = load_admin()
    places = load_places(oblast)
    index = name_index(places)
    ways = load_waterways(oblast)
    code_of = learn_district_codes(catalog, places, index)
    polygons = built_up(oblast)
    sindex = polygons.sindex
    places_m = places.to_crs(METRIC_CRS)

    records = []
    for _, row in catalog.iterrows():
        village_idx, how = resolve(row["village"], row, places, index, raions, code_of, ways)
        anchor_idx = village_idx
        ref = row["ref_village"] if isinstance(row["ref_village"], str) else ""
        if ref and norm(ref) != norm(row["village"]):
            ref_idx, _ = resolve(ref, row, places, index, raions, code_of, ways)
            anchor_idx = ref_idx if ref_idx is not None else village_idx
        if anchor_idx is None:
            records.append({**row, "match": how, "accuracy": "C", "geometry": None})
            continue
        village_idx = anchor_idx if village_idx is None else village_idx
        direction = row["direction"] if isinstance(row["direction"], str) else ""
        outskirts = "околиц" in str(row["desc"]).lower()
        anchor = places_m.geometry[anchor_idx]
        point = shift(anchor, direction, outskirts, polygons, sindex)
        records.append(
            {
                **row,
                "match": how,
                "place_idx": village_idx,
                "anchor_x": anchor.x,
                "anchor_y": anchor.y,
                "osm_village": places["name"][village_idx],
                "accuracy": "B" if direction in BEARINGS else "A",
                "geometry": point,
            }
        )

    sites = gpd.GeoDataFrame(records, geometry="geometry", crs=METRIC_CRS)
    OUT_SITES.unlink(missing_ok=True)
    sites[sites.geometry.notna()].to_file(OUT_SITES, layer="sites", driver="GPKG")

    # Public layer: one point per catalogue village, at the OSM village node.
    placed = sites[sites.geometry.notna()]
    villages = (
        placed.groupby("place_idx", sort=False)
        .agg(
            district=("district", "first"),
            village=("village", "first"),
            settlements=("type", lambda t: int((t == "Поселення").sum())),
            burials=("type", lambda t: int((t == "Могильник").sum())),
            catalogue_no=("no", lambda n: ", ".join(str(x) for x in sorted(n))),
        )
        .reset_index()
    )
    public = gpd.GeoDataFrame(
        villages.drop(columns="place_idx"),
        geometry=[places.geometry[int(i)] for i in villages["place_idx"]],
        crs=4326,
    )
    public["source"] = "Магомедов Б. В. Пам'ятки черняхівської культури Вінницької обл., 2022"
    OUT_PUBLIC.unlink(missing_ok=True)
    public.to_file(OUT_PUBLIC, driver="GeoJSON", COORDINATE_PRECISION=4)

    print(f"sites: {len(sites)}; accuracy {Counter(sites['accuracy'])}")
    print(f"match: {Counter(sites['match'])}")
    print(f"not placed: {sites[sites['accuracy'] == 'C'][['no', 'village']].values.tolist()}")
    print(f"public villages: {len(public)} -> {OUT_PUBLIC.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
