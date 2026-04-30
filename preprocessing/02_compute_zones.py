"""
Stage 2: For each X in 1..10 km compute the "far zone":
    far_zone[X] = ukraine - buffer(settlement_areas, X km)

settlement_areas is a (Multi)Polygon approximating the BUILT-UP boundaries
of populated places (produced in stage 1 from landuse=residential plus
fallback buffers around uncovered place=* nodes). Therefore buffer(X km)
measures distance from the EDGES of populated areas, not from a point.

Inputs:  data/processed/stage1.gpkg  (layers: settlement_areas, ukraine)
Outputs: data/processed/stage2.gpkg  (layers: zone_1km .. zone_10km)
All in EPSG:6381 (metric).
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from shapely.ops import unary_union

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
SRC = PROJECT_ROOT / "data" / "processed" / "stage1.gpkg"
OUT = PROJECT_ROOT / "data" / "processed" / "stage2.gpkg"

DISTANCES_KM = list(range(1, 11))
TARGET_CRS = 6381


def main():
    if not SRC.exists():
        raise SystemExit("Missing %s. Run 01_extract_settlements.py first." % SRC)

    OUT.parent.mkdir(parents=True, exist_ok=True)

    print("Loading stage1 ...")
    settlement_areas = gpd.read_file(SRC, layer="settlement_areas")
    ukraine = gpd.read_file(SRC, layer="ukraine")

    if not (settlement_areas.crs and settlement_areas.crs.to_epsg() == TARGET_CRS):
        raise SystemExit("settlement_areas CRS must be EPSG:%d" % TARGET_CRS)
    if not (ukraine.crs and ukraine.crs.to_epsg() == TARGET_CRS):
        raise SystemExit("ukraine CRS must be EPSG:%d" % TARGET_CRS)

    print("Preparing %d settlement_areas feature(s) ..." % len(settlement_areas))
    settlement_geom = unary_union(settlement_areas.geometry)

    print("Preparing %d ukraine polygon(s) ..." % len(ukraine))
    ukraine_geom = unary_union(ukraine.geometry)

    for km in DISTANCES_KM:
        print("  computing zone for %d km (buffer from built-up edge) ..." % km)
        buffered = settlement_geom.buffer(km * 1000)
        far_zone = ukraine_geom.difference(buffered)
        gdf = gpd.GeoDataFrame(
            {"distance_km": [km]},
            geometry=[far_zone],
            crs="EPSG:%d" % TARGET_CRS,
        )
        gdf.to_file(OUT, layer="zone_%dkm" % km, driver="GPKG")

    print("Stage 2 done.")


if __name__ == "__main__":
    main()
