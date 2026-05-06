"""
Stage 1 (osmium-based): build settlement_areas geometry from
osmium-extracted GeoJSON layers.

Inputs:
  data/raw/ua-residential.geojson  - landuse=residential polygons
  data/raw/ua-places.geojson       - place=city/town/village/hamlet
  data/raw/ua-border.geojson       - Ukraine boundary (relation 60199)

Output: data/processed/stage1.gpkg
  - settlement_areas (one MultiPolygon, EPSG:6381)
  - ukraine          (Ukraine boundary, EPSG:6381)
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from shapely.geometry import MultiPolygon
from shapely.ops import unary_union

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
RAW = PROJECT_ROOT / "data" / "raw"
RES = RAW / "ua-residential.geojson"
PLACES = RAW / "ua-places.geojson"
BORDER = RAW / "ua-border.geojson"
OUT_GPKG = PROJECT_ROOT / "data" / "processed" / "stage1.gpkg"

TARGET_CRS = 6381
PLACE_TYPES = ["city", "town", "village", "hamlet"]
FALLBACK_BUFFER_M = {"city": 2000.0, "town": 1000.0, "village": 300.0, "hamlet": 100.0}


def main() -> None:
    for f in (RES, PLACES, BORDER):
        if not f.exists():
            raise SystemExit("Missing %s. Run osmium tags-filter / export first." % f)
    OUT_GPKG.parent.mkdir(parents=True, exist_ok=True)

    print("Loading residential landuse ...")
    landuse = gpd.read_file(RES)
    landuse = landuse[landuse.geometry.notna()]
    landuse = landuse[landuse.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    landuse = landuse.to_crs(epsg=TARGET_CRS)
    print("  -> %d residential polygons" % len(landuse))

    print("Loading place=* features ...")
    places = gpd.read_file(PLACES)
    places = places[places.geometry.notna()]
    places = places[places.geometry.geom_type == "Point"]
    if "place" not in places.columns:
        raise SystemExit("Column 'place' missing in ua-places.geojson")
    places = places[places["place"].isin(PLACE_TYPES)]
    places = places.to_crs(epsg=TARGET_CRS).reset_index(drop=True)
    print("  -> %d place=* point nodes" % len(places))

    print("Loading Ukraine border ...")
    border = gpd.read_file(BORDER)
    border = border[border.geometry.notna()]
    border = border[border.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    border = border.to_crs(epsg=TARGET_CRS)
    if border.empty:
        raise SystemExit("No polygon geometry in ua-border.geojson")
    print("  -> %d border polygon feature(s)" % len(border))

    print("Spatial join: which place points are inside residential polygons?")
    joined = gpd.sjoin(places, landuse[["geometry"]], how="left", predicate="within")
    covered_idx = joined.dropna(subset=["index_right"]).index.unique()
    uncovered = places.loc[~places.index.isin(covered_idx)].copy()
    print("  -> %d uncovered points -> need fallback buffer" % len(uncovered))

    if uncovered.empty:
        fallback_geoms = []
    else:
        radii = uncovered["place"].map(FALLBACK_BUFFER_M)
        if radii.isna().any():
            unknown = uncovered.loc[radii.isna(), "place"].unique()
            raise SystemExit("Unknown place types: %s" % list(unknown))
        fallback_geoms = list(uncovered.geometry.buffer(radii))
        by_type = uncovered.groupby("place").size().to_dict()
        print("  -> fallback by type: %s" % by_type)

    print("Unioning landuse + fallback into settlement_areas ...")
    parts = list(landuse.geometry) + fallback_geoms
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
    border.to_file(OUT_GPKG, layer="ukraine", driver="GPKG")

    area_km2 = settlement_geom.area / 1e6
    print("  settlement_areas covers ~%.0f km^2" % area_km2)
    print("Stage 1 done.")


if __name__ == "__main__":
    main()