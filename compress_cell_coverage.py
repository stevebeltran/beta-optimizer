"""
Recompress cell_coverage/*.parquet files:
  1. geometry_wkb hex string  ->  binary bytes (halves geometry size)
  2. Shapely simplify(tolerance=0.001)  (reduces vertex count, invisible at map zoom)
  3. Parquet compression SNAPPY -> ZSTD level 9
"""
import os
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from shapely.wkb import loads as wkb_loads, dumps as wkb_dumps


TOLERANCE = 0.001   # degrees (~100m); invisible on any web map zoom level
COVERAGE_DIR = Path("cell_coverage")


def decode(value):
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray, memoryview)):
        return wkb_loads(bytes(value))
    if isinstance(value, str):
        return wkb_loads(bytes.fromhex(value))
    return None


def process_file(path: Path) -> tuple[int, int]:
    df = pd.read_parquet(path)
    if "geometry_wkb" not in df.columns:
        return 0, 0

    def transform(value):
        geom = decode(value)
        if geom is None or geom.is_empty:
            return None
        simplified = geom.simplify(TOLERANCE, preserve_topology=True)
        return wkb_dumps(simplified)   # returns bytes

    df["geometry_wkb"] = df["geometry_wkb"].apply(transform)

    table = pa.Table.from_pandas(df)
    pq.write_table(table, path, compression="zstd", compression_level=9)

    return 1, 0


def main():
    files = sorted(COVERAGE_DIR.glob("*.parquet"))
    if not files:
        print("No parquet files found in cell_coverage/")
        sys.exit(1)

    total_before = sum(f.stat().st_size for f in files)
    print(f"Found {len(files)} files — total before: {total_before / 1e6:.1f} MB\n")

    for f in files:
        before = f.stat().st_size
        process_file(f)
        after = f.stat().st_size
        pct = (1 - after / before) * 100
        print(f"  {f.name:30s}  {before/1e6:6.1f} MB -> {after/1e6:6.1f} MB  ({pct:.0f}% reduction)")

    total_after = sum(f.stat().st_size for f in files)
    pct_total = (1 - total_after / total_before) * 100
    print(f"\nTotal: {total_before/1e6:.1f} MB -> {total_after/1e6:.1f} MB  ({pct_total:.0f}% reduction)")


if __name__ == "__main__":
    main()
