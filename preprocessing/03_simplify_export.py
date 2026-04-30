"""
Stage 3: Simplify geometry, reproject to WGS84 (EPSG:4326),
export 10 GeoJSON files to web/data/.

Why simplify here:
- Source geometry has node-level precision; for a country-scale web map
  this is overkill and bloats file size by 10-20x.
- 100 m Douglas-Peucker tolerance is invisible at zoom <= 11.

If files are still too large, run mapshaper as a follow-up:
  npx mapshaper web/data/zones_*.geojson \\
    -simplify dp 5% keep-shapes -o force web/data/
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
SRC = PROJECT_ROOT / "data" / "processed" / "stage2.gpkg"
OUT_DIR = PROJECT_ROOT / "web" / "data"

DISTANCES_KM = list(range(1, 11))
TOLERANCE_METERS = 100  # in source CRS (EPSG:6381)
WEB_CRS = 4326


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"Missing {SRC}. Run 02_compute_zones.py first.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for km in DISTANCES_KM:
        layer = f"zone_{km}km"
        print(f"Processing {layer} ...")
        gdf = gpd.read_file(SRC, layer=layer)

        # simplify in metric CRS, then reproject to WGS84
        gdf["geometry"] = gdf.geometry.simplify(TOLERANCE_METERS, preserve_topology=True)
        gdf = gdf.to_crs(epsg=WEB_CRS)

        out_path = OUT_DIR / f"zones_{km}km.geojson"
        if out_path.exists():
            out_path.unlink()
        gdf.to_file(out_path, driver="GeoJSON")
        size_kb = out_path.stat().st_size // 1024
        print(f"  -> {out_path.name} ({size_kb:,} KB)")

    print("Stage 3 done. Open web/index.html via a local server.")


if __name__ == "__main__":
    main()
