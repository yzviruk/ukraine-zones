"""
Stage 1: Build a single "settlement areas" geometry that approximates the
ACTUAL BUILT-UP BOUNDARIES of all populated places in Ukraine.

Strategy (landuse=residential + fallback):
  1. Take all landuse=residential polygons from OSM.
  2. For place=* NODES NOT covered by any residential polygon, add a
     type-dependent fallback buffer:
       city    -> 2000 m
       town    -> 1000 m
       village ->  300 m
       hamlet  ->  100 m
  3. Union everything into a single (Multi)Polygon.

Stage 2 will buffer THIS geometry, so distances are measured from the
EDGES of populated areas, not from a single centre point.

Output: data/processed/stage1.gpkg with two layers:
  - settlement_areas (one MultiPolygon, EPSG:6381)
  - ukraine          (Ukraine boundary,   EPSG:6381)
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from pyrosm import OSM
from shapely.geometry import MultiPolygon
from shapely.ops import unary_union

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
RAW_PBF = PROJECT_ROOT / "data" / "raw" / "ukraine-latest.osm.pbf"
OUT_GPKG = PROJECT_ROOT / "data" / "processed" / "stage1.gpkg"

TARGET_CRS = 6381  # Ukraine 2000 / Transverse Mercator (metric)
PLACE_TYPES = ["city", "town", "village", "hamlet"]
FALLBACK_BUFFER_M = {
    "city": 2000.0,
    "town": 1000.0,
    "village": 300.0,
    "hamlet": 100.0,
}


def extract_residential_landuse(osm):
    print("Extracting landuse=residential polygons ...")
    landuse = osm.get_landuse(custom_filter={"landuse": ["residential"]})
    if landuse is None or landuse.empty:
        raise SystemExit("No landuse=residential found - PBF may be incomplete.")
    landuse = landuse[landuse.geometry.notna()].copy()
    landuse = landuse[landuse.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    landuse = landuse.to_crs(epsg=TARGET_CRS)
    print("  -> %d residential polygons" % len(landuse))
    return landuse


def extract_place_points(osm):
    print("Extracting place nodes (city/town/village/hamlet) ...")
    places = osm.get_pois(custom_filter={"place": PLACE_TYPES})
    if places is None or places.empty:
        raise SystemExit("No place=* features found.")
    places = places[places.geometry.notna()].copy()
    places = places[places.geometry.geom_type == "Point"]
    if "place" not in places.columns:
        raise SystemExit("Column 'place' missing in pyrosm output.")
    places = places.to_crs(epsg=TARGET_CRS)
    print("  -> %d place=* point nodes" % len(places))
    return places[["geometry", "place"]].reset_index(drop=True)


def fallback_buffers_for_uncovered_points(points, landuse):
    """Buffer place=* points NOT covered by any residential polygon."""
    print("Spatial join: which place points are inside residential polygons?")
    joined = gpd.sjoin(
        points,
        landuse[["geometry"]],
        how="left",
        predicate="within",
    )
    covered_idx = joined.dropna(subset=["index_right"]).index.unique()
    uncovered = points.loc[~points.index.isin(covered_idx)].copy()
    print("  -> %d points uncovered by residential landuse" % len(uncovered))
    if uncovered.empty:
        return gpd.GeoSeries([], crs=points.crs)

    radii = uncovered["place"].map(FALLBACK_BUFFER_M)
    if radii.isna().any():
        unknown = uncovered.loc[radii.isna(), "place"].unique()
        raise SystemExit("Unknown place types: %s" % list(unknown))

    buffered = uncovered.geometry.buffer(radii)
    by_type = uncovered.groupby("place").size().to_dict()
    print("  -> fallback buffers by type: %s" % by_type)
    return buffered


def extract_ukraine_boundary(osm):
    print("Extracting Ukraine boundary (admin_level=2) ...")
    boundaries = osm.get_boundaries(boundary_type="administrative")
    if boundaries is None or boundaries.empty:
        raise SystemExit("No administrative boundaries returned.")
    ukraine = boundaries[boundaries["admin_level"] == "2"].copy()
    if ukraine.empty:
        raise SystemExit("Ukraine admin_level=2 boundary not found.")
    ukraine = ukraine.to_crs(epsg=TARGET_CRS)
    print("  -> %d Ukraine boundary feature(s)" % len(ukraine))
    cols = ["geometry"] + (["name"] if "name" in ukraine.columns else [])
    return ukraine[cols]


def main():
    if not RAW_PBF.exists():
        raise SystemExit(
            "OSM PBF not found at %s. Download from "
            "https://download.geofabrik.de/europe/ukraine-latest.osm.pbf" % RAW_PBF
        )
    OUT_GPKG.parent.mkdir(parents=True, exist_ok=True)

    print("Reading %s ..." % RAW_PBF.name)
    osm = OSM(str(RAW_PBF))

    landuse = extract_residential_landuse(osm)
    place_points = extract_place_points(osm)
    fallback = fallback_buffers_for_uncovered_points(place_points, landuse)
    ukraine = extract_ukraine_boundary(osm)

    print("Unioning landuse + fallback buffers into settlement_areas ...")
    parts = list(landuse.geometry) + list(fallback)
    settlement_geom = unary_union(parts)
    if settlement_geom.geom_type == "Polygon":
        settlement_geom = MultiPolygon([settlement_geom])

    settlement_gdf = gpd.GeoDataFrame(
        {"source": ["landuse_residential + place_node_fallback"]},
        geometry=[settlement_geom],
        crs="EPSG:%d" % TARGET_CRS,
    )

    print("Writing %s ..." % OUT_GPKG)
    settlement_gdf.to_file(OUT_GPKG, layer="settlement_areas", driver="GPKG")
    ukraine.to_file(OUT_GPKG, layer="ukraine", driver="GPKG")

    area_km2 = settlement_geom.area / 1e6
    print("  settlement_areas covers ~%.0f km^2" % area_km2)
    print("Stage 1 done.")


if __name__ == "__main__":
    main()
