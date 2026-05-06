"""
Stage 4: simplify and export waterways for the web map.

Reads ua-rivers.geojson (output of `osmium export`), simplifies geometry
(50 m tolerance is invisible at country zoom but cuts file size 10x+),
keeps only LineString/MultiLineString, splits into rivers vs streams
for differentiated styling, and exports to web/data/.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
SRC = PROJECT_ROOT / "data" / "raw" / "ua-rivers.geojson"
OUT_DIR = PROJECT_ROOT / "web" / "data"

WORK_CRS = 6381  # metric for simplify
WEB_CRS = 4326
TOLERANCE_M = 100  # 100 m simplification


def main() -> None:
    if not SRC.exists():
        raise SystemExit("Missing %s. Run osmium tags-filter+export first." % SRC)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading %s ..." % SRC.name)
    gdf = gpd.read_file(SRC)
    gdf = gdf[gdf.geometry.notna()]
    gdf = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])]
    if "waterway" not in gdf.columns:
        raise SystemExit("No 'waterway' column in source")

    gdf = gdf.to_crs(epsg=WORK_CRS)
    gdf["geometry"] = gdf.geometry.simplify(TOLERANCE_M, preserve_topology=True)

    keep_cols = ["geometry", "waterway"]
    if "name" in gdf.columns:
        keep_cols.append("name")

    gdf = gdf[keep_cols].to_crs(epsg=WEB_CRS)

    # Split: rivers (thick) vs streams/canals (thin) - reduces file size by
    # letting the user choose to skip streams on slow connections.
    rivers = gdf[gdf["waterway"] == "river"]
    streams = gdf[gdf["waterway"].isin(["stream", "canal"])]

    out_rivers = OUT_DIR / "rivers.geojson"
    out_streams = OUT_DIR / "streams.geojson"
    if out_rivers.exists():
        out_rivers.unlink()
    if out_streams.exists():
        out_streams.unlink()
    rivers.to_file(out_rivers, driver="GeoJSON")
    streams.to_file(out_streams, driver="GeoJSON")

    print("  rivers.geojson : %s features, %d KB" %
          (len(rivers), out_rivers.stat().st_size // 1024))
    print("  streams.geojson: %s features, %d KB" %
          (len(streams), out_streams.stat().st_size // 1024))


if __name__ == "__main__":
    main()