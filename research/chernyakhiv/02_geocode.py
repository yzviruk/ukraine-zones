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

Site position: from the anchor village (or the other village the text refers to)
in the stated compass direction, as far as the text says (see place()):
"in the ... part of the village" - inside the built-up area; "on the outskirts" -
on its edge; "east of the village" - just past the edge; a stated distance -
that far from the edge. The edge is that of the village's OWN built-up area
(Extents). Two villages in the text ("east of A and west of B", "between A and
B") give the crossing of the two directions or the midpoint.
Accuracy classes:
    A  village only (no direction, or "in the centre")
    B  placed by direction
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
from shapely import voronoi_polygons
from shapely.geometry import LineString, MultiPoint, Point, Polygon
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
OWN_RADIUS_M = 2500  # built-up parts of a village are looked for this close to its node
NEIGHBOUR_M = 12_000  # a village named inside a description lies this close
NODE_TOL_M = 500  # ... and at least one of them must be this close
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


def resolve_near(name, near: Point | None, places_m, index) -> int | None:
    """A village mentioned INSIDE a description ("... від с. X"): the namesake
    closest to the entry's own village, and only if it is a neighbour."""
    key = norm(name)
    cands = index.get(ALIASES.get(key, key), [])
    if not cands or near is None:
        return None
    best = min(cands, key=lambda i: places_m.geometry[i].distance(near))
    return best if places_m.geometry[best].distance(near) <= NEIGHBOUR_M else None


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


class Extents:
    """Built-up area of each village on its own.

    The built-up polygons of neighbouring villages are merged in stage1.gpkg
    (one polygon can span several villages, up to 28 km2), so "the edge of the
    village" is taken inside the village's Voronoi cell among all place nodes:
    the built-up land that is closer to this village's node than to any other.
    """

    def __init__(self, places_m: gpd.GeoDataFrame, polygons: gpd.GeoSeries):
        self.polygons = polygons
        self.sindex = polygons.sindex
        self.anchors = places_m.geometry
        cells = voronoi_polygons(MultiPoint(list(self.anchors.values)))
        cells = gpd.GeoSeries(list(cells.geoms), crs=polygons.crs)
        hits = cells.sindex.query(self.anchors.values, predicate="within")
        self.cell = {self.anchors.index[i]: cells.iloc[j] for i, j in zip(*hits, strict=True)}
        self._own = {}

    def own(self, idx):
        """Built-up geometry of place `idx`, or None when it has no polygon."""
        if idx not in self._own:
            anchor, cell = self.anchors[idx], self.cell.get(idx)
            parts = []
            for i in self.sindex.query(anchor.buffer(OWN_RADIUS_M)):
                part = self.polygons.iloc[i]
                part = part.intersection(cell) if cell is not None else part
                if not part.is_empty and part.distance(anchor) < OWN_RADIUS_M:
                    parts.append(part)
            near = any(p.distance(anchor) < NODE_TOL_M for p in parts)
            self._own[idx] = unary_union(parts) if near else None
        return self._own[idx]


def unit(direction: str) -> tuple[float, float]:
    angle = math.radians(BEARINGS[direction])
    return math.sin(angle), math.cos(angle)


def edge_distance(anchor: Point, direction: str, own) -> float:
    """Distance from the node to the far built-up edge of its own village."""
    dx, dy = unit(direction)
    ray = LineString([anchor, (anchor.x + dx * 20_000, anchor.y + dy * 20_000)])
    hit = ray.intersection(own.boundary)
    pts = list(getattr(hit, "geoms", [hit])) if not hit.is_empty else []
    return max((anchor.distance(p) for p in pts if p.geom_type == "Point"), default=0.0)


def place(anchor: Point, own, direction: str, position: str, distance_m=None) -> Point:
    """Site position from the village node, by what the text says:

    inside     "in the southern part of the village": the middle of that part
    outskirts  on the built-up edge in that direction
    beyond     BEYOND_EDGE_M past the edge
    distance   stated distance from the village, counted from its edge
    Without a compass direction the site stays on the node.
    """
    if direction not in BEARINGS:
        return anchor
    dx, dy = unit(direction)
    if own is None:  # no built-up polygon: typical sizes of a small village
        dist = {"inside": NO_POLYGON_M * 0.3, "outskirts": NO_POLYGON_M * 0.6}.get(
            position, NO_POLYGON_M
        )
        if distance_m:
            dist = NO_POLYGON_M * 0.6 + distance_m
        return Point(anchor.x + dx * dist, anchor.y + dy * dist)
    edge = edge_distance(anchor, direction, own)
    if distance_m:
        dist = edge + distance_m
    elif position == "inside":
        # the part of the village lying in that direction: a 90-degree wedge
        left, right = BEARINGS[direction] - 45, BEARINGS[direction] + 45
        far = [
            (
                anchor.x + math.sin(math.radians(b)) * 20_000,
                anchor.y + math.cos(math.radians(b)) * 20_000,
            )
            for b in (left, right)
        ]
        part = own.intersection(Polygon([(anchor.x, anchor.y), *far]))
        if not part.is_empty and part.area > 0:
            return (
                part.representative_point() if part.centroid.distance(part) > 0 else part.centroid
            )
        dist = edge * 0.6
    elif position == "outskirts":
        dist = edge + EDGE_M
    else:
        dist = edge + BEYOND_EDGE_M
    return Point(anchor.x + dx * dist, anchor.y + dy * dist)


def two_rays(a: Point, dir_a: str, b: Point, dir_b: str) -> Point | None:
    """Where "<dir_a> of village A" meets "<dir_b> of village B", if they do."""
    (ax, ay), (bx, by) = unit(dir_a), unit(dir_b)
    det = ax * (-by) - ay * (-bx)
    if abs(det) < 1e-9:
        return None
    px, py = b.x - a.x, b.y - a.y
    t = (px * (-by) - py * (-bx)) / det
    u = (ax * py - ay * px) / det
    reach = 1.2 * a.distance(b) + 500  # the crossing lies between the two villages
    if t <= 0 or u <= 0 or t > reach or u > reach:
        return None
    return Point(a.x + ax * t, a.y + ay * t)


def control_points(places, extents, used: set, mix, rng) -> pd.DataFrame:
    """Control for the analyses: villages WITHOUT catalogue sites, with a point
    placed by the same rule and the sites' own mix of (direction, position)."""
    pool = places[~places.index.isin(used) & places["place"].isin(["village", "hamlet", "town"])]
    out = []
    for idx in pool.index:
        anchor = extents.anchors[idx]
        direction, position = mix[rng.integers(len(mix))]
        out.append(
            {
                "anchor": anchor,
                "direction": direction,
                "point": place(anchor, extents.own(idx), direction, position),
            }
        )
    return pd.DataFrame(out)


def site_mix(sites) -> list[tuple[str, str]]:
    return list(zip(sites["direction"].fillna(""), sites["position"].fillna("beyond"), strict=True))


def main() -> None:
    catalog = pd.read_csv(CATALOG)
    oblast, raions = load_admin()
    places = load_places(oblast)
    index = name_index(places)
    ways = load_waterways(oblast)
    code_of = learn_district_codes(catalog, places, index)
    places_m = places.to_crs(METRIC_CRS)
    extents = Extents(places_m, built_up(oblast))

    records = []
    for _, row in catalog.iterrows():
        village_idx, how = resolve(row["village"], row, places, index, raions, code_of, ways)
        anchor_idx = village_idx
        ref = row["ref_village"] if isinstance(row["ref_village"], str) else ""
        if ref and norm(ref) != norm(row["village"]):
            home = places_m.geometry[village_idx] if village_idx is not None else None
            ref_idx = resolve_near(ref, home, places_m, index)
            if ref_idx is None and home is None:
                ref_idx, _ = resolve(ref, row, places, index, raions, code_of, ways)
            anchor_idx = ref_idx if ref_idx is not None else village_idx
        if anchor_idx is None:
            records.append({**row, "match": how, "accuracy": "C", "geometry": None})
            continue
        village_idx = anchor_idx if village_idx is None else village_idx
        direction = row["direction"] if isinstance(row["direction"], str) else ""
        position = row["position"] if isinstance(row["position"], str) else "near"
        distance_m = row["distance_km"] * 1000 if pd.notna(row["distance_km"]) else None
        anchor = places_m.geometry[anchor_idx]
        point = place(anchor, extents.own(anchor_idx), direction, position, distance_m)
        placed = "distance" if distance_m and direction in BEARINGS else position
        if direction not in BEARINGS:
            placed = "village"
        # A second village in the text pins the site down better than one direction.
        other, other_dir = None, ""
        for col in ("between_village", "ref2_village"):
            if isinstance(row[col], str) and other is None:
                idx = resolve_near(row[col], anchor, places_m, index)
                if idx is not None and idx != anchor_idx:
                    other = places_m.geometry[idx]
                    other_dir = row["ref2_direction"] if col == "ref2_village" else ""
        if other is not None and other_dir in BEARINGS and direction in BEARINGS:
            crossing = two_rays(anchor, direction, other, other_dir)
            opposite = abs(BEARINGS[direction] - BEARINGS[other_dir]) == 180
            if crossing is not None:
                point, placed = crossing, "two_villages"
            elif opposite:  # "east of A and west of B": on the line between them
                point = Point((anchor.x + other.x) / 2, (anchor.y + other.y) / 2)
                placed = "between"
        elif other is not None and not other_dir:
            point = Point((anchor.x + other.x) / 2, (anchor.y + other.y) / 2)
            placed = "between"
        records.append(
            {
                **row,
                "match": how,
                "place_idx": village_idx,
                "anchor_x": anchor.x,
                "anchor_y": anchor.y,
                "osm_village": places["name"][village_idx],
                "placed": placed,
                "accuracy": "A" if placed == "village" else "B",
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
    print(f"placed: {Counter(sites['placed'].dropna())}")
    print(f"not placed: {sites[sites['accuracy'] == 'C'][['no', 'village']].values.tolist()}")
    print(f"public villages: {len(public)} -> {OUT_PUBLIC.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
