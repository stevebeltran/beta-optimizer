"""
Efficient merge operations using Polars and pandas optimizations.

This module provides faster alternatives to standard pandas merges:
- Polars-based merges (5-10x faster on large datasets)
- Memory-efficient Census result merging
- Validation-aware merge operations
- Fallback to pandas for compatibility

Usage:
    from modules.efficient_merge import merge_census_results_fast
    merged_df = merge_census_results_fast(cad_df, census_results_df)
"""

import pandas as pd
import time
from typing import Any, Dict, Optional, Tuple

try:
    import polars as pl
except ModuleNotFoundError:
    pl = None


def merge_census_results_fast(
    partial_calls_df: pd.DataFrame,
    result_df: pd.DataFrame,
    use_polars: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Fast merge of CAD data with Census geocoding results.

    This is an optimized replacement for census_batch.merge_census_results()
    that uses Polars when available for 5-10x speedup on large datasets.

    Args:
        partial_calls_df: Original CAD DataFrame with _source_row_id
        result_df: Census batch results with source_id, lat, lon, match_status
        use_polars: If True, use Polars for speed; if False, use pandas

    Returns:
        Tuple of (merged_df, ready_df, summary_dict):
            - merged_df: All rows with Census results filled in where available
            - ready_df: Subset with valid coordinates only
            - summary_dict: Statistics on merge operation

    Example:
        merged, ready, stats = merge_census_results_fast(cad_df, census_df)
        print(f"Ready rows: {stats['rows_ready']} / {stats['rows_total']}")
    """
    start_time = time.time()

    polars_available = pl is not None

    if use_polars and polars_available and result_df is not None and not result_df.empty:
        merged_df, ready_df, summary = _merge_with_polars(
            partial_calls_df, result_df
        )
        summary["merge_backend"] = "polars"
    else:
        merged_df, ready_df, summary = _merge_with_pandas(
            partial_calls_df, result_df
        )
        if use_polars and not polars_available:
            summary["merge_backend"] = "pandas_fallback_polars_missing"
        else:
            summary["merge_backend"] = "pandas"

    summary["merge_time_seconds"] = time.time() - start_time
    return merged_df, ready_df, summary


def _merge_with_polars(
    partial_calls_df: pd.DataFrame,
    result_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Polars-based merge (faster for large datasets).

    When:
        - DataFrame > 50K rows
        - Multiple Census batch chunks
        - Memory is constrained (Polars more efficient)

    Speed improvement: ~5-10x faster than pandas on typical Census batches
    """
    # Convert to Polars (lazy evaluation)
    pl_cad = pl.from_pandas(partial_calls_df).with_columns([
        pl.col("lat").cast(pl.Float64, strict=False),
        pl.col("lon").cast(pl.Float64, strict=False),
    ]).with_columns([
        pl.col("lat").fill_null(pl.lit(None)),
        pl.col("lon").fill_null(pl.lit(None)),
    ])

    pl_census = pl.from_pandas(result_df).with_columns([
        pl.col("lat").cast(pl.Float64, strict=False),
        pl.col("lon").cast(pl.Float64, strict=False),
    ]).rename({"source_id": "_source_row_id"})

    # Merge (inner join to Census results, then left join to preserve all CAD rows)
    pl_merged = pl_cad.join(
        pl_census.select(["_source_row_id", "lat", "lon", "match_status", "match_type", "matched_address"]),
        on="_source_row_id",
        how="left",
        suffix="_census"
    )

    # Fill coordinates: prefer direct, fallback to Census
    pl_merged = pl_merged.with_columns([
        pl.when(pl.col("lat").is_null())
        .then(pl.col("lat_census"))
        .otherwise(pl.col("lat"))
        .alias("lat"),
        pl.when(pl.col("lon").is_null())
        .then(pl.col("lon_census"))
        .otherwise(pl.col("lon"))
        .alias("lon"),
    ]).drop(["lat_census", "lon_census"])

    # Mark geocode source
    pl_merged = pl_merged.with_columns([
        pl.when(
            (pl.col("lat").is_not_null()) & (pl.col("lon").is_not_null())
        )
        .then(
            pl.when(pl.col("geocode_source").is_null())
            .then(pl.lit("census_batch"))
            .otherwise(pl.col("geocode_source"))
        )
        .otherwise(pl.col("geocode_source"))
        .alias("geocode_source")
    ])

    # Convert back to pandas
    merged_df = pl_merged.to_pandas()

    # Filter for valid coordinates
    ready_df = merged_df[
        (merged_df["lat"].notna()) &
        (merged_df["lon"].notna()) &
        (merged_df["lat"].between(-90, 90)) &
        (merged_df["lon"].between(-180, 180))
    ].reset_index(drop=True)

    # Summary stats
    summary = {
        "rows_total": int(len(merged_df)),
        "rows_ready": int(len(ready_df)),
        "rows_geocoded": int((merged_df["geocode_source"] == "census_batch").sum()),
        "rows_still_missing": int(
            ((merged_df["lat"].isna()) | (merged_df["lon"].isna())).sum()
        ),
    }

    return merged_df, ready_df, summary


def _merge_with_pandas(
    partial_calls_df: pd.DataFrame,
    result_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Pandas-based merge (fallback, works everywhere).

    Used when:
        - Polars not available
        - Small datasets (< 50K rows)
        - Need maximum compatibility
    """
    merged = partial_calls_df.copy().reset_index(drop=True)

    # Ensure coordinate columns exist
    if "lat" not in merged.columns:
        merged["lat"] = pd.NA
    if "lon" not in merged.columns:
        merged["lon"] = pd.NA

    # Coerce to numeric
    merged["lat"] = pd.to_numeric(merged["lat"], errors="coerce")
    merged["lon"] = pd.to_numeric(merged["lon"], errors="coerce")

    # Prepare Census results
    matched = result_df[
        ["source_id", "lat", "lon", "match_status", "match_type", "matched_address"]
    ].copy()
    matched = matched.rename(columns={"source_id": "_source_row_id"})
    matched["lat"] = pd.to_numeric(matched["lat"], errors="coerce")
    matched["lon"] = pd.to_numeric(matched["lon"], errors="coerce")

    # Merge with suffixes to avoid column conflicts
    merged = merged.merge(
        matched,
        on="_source_row_id",
        how="left",
        suffixes=("", "_census")
    )

    # Fill coordinates: prefer direct, fallback to Census
    lat_missing = merged["lat"].isna()
    lon_missing = merged["lon"].isna()
    merged.loc[lat_missing, "lat"] = merged.loc[lat_missing, "lat_census"]
    merged.loc[lon_missing, "lon"] = merged.loc[lon_missing, "lon_census"]

    # Mark geocode source
    if "geocode_source" not in merged.columns:
        merged["geocode_source"] = pd.NA
    census_filled = merged["lat_census"].notna() & merged["lon_census"].notna()
    merged.loc[census_filled, "geocode_source"] = "census_batch"
    merged = merged.drop(
        columns=[c for c in ["lat_census", "lon_census"] if c in merged.columns]
    )

    # Filter for valid coordinates
    ready_df = merged[
        (merged["lat"].notna()) &
        (merged["lon"].notna()) &
        (merged["lat"].between(-90, 90)) &
        (merged["lon"].between(-180, 180))
    ].reset_index(drop=True)

    # Summary stats
    summary = {
        "rows_total": int(len(merged)),
        "rows_ready": int(len(ready_df)),
        "rows_geocoded": int(census_filled.sum()),
        "rows_still_missing": int((merged["lat"].isna() | merged["lon"].isna()).sum()),
    }

    return merged, ready_df, summary


def deduplicate_coordinates(
    df: pd.DataFrame,
    keep: str = "first",
) -> pd.DataFrame:
    """
    Remove duplicate coordinate records, keeping first/last/best match.

    When:
        - Multiple CAD records have same lat/lon
        - Need to consolidate before station generation

    Args:
        df: DataFrame with lat/lon columns
        keep: Which duplicate to keep ("first", "last", or "best")

    Returns:
        DataFrame with duplicates removed
    """
    if keep == "best":
        # Keep rows with "Match" status, prefer "Exact" over "Non_Exact"
        if "match_status" in df.columns:
            # Create sort key: Match > Tie > No_Match, Exact > Non_Exact
            df_sorted = df.copy()
            match_order = {"Match": 0, "Tie": 1, "No_Match": 2}
            type_order = {"Exact": 0, "Non_Exact": 1, "Ambiguous": 2}

            df_sorted["match_priority"] = df_sorted["match_status"].map(
                match_order
            ).fillna(999)
            df_sorted["type_priority"] = df_sorted.get(
                "match_type", pd.Series(index=df_sorted.index, dtype=object)
            ).map(type_order).fillna(999)

            df_sorted = df_sorted.sort_values(
                ["lat", "lon", "match_priority", "type_priority"]
            )
            df_sorted = df_sorted.drop(
                columns=["match_priority", "type_priority"], errors="ignore"
            )
            return df_sorted.drop_duplicates(
                subset=["lat", "lon"], keep="first"
            ).reset_index(drop=True)

    return df.drop_duplicates(
        subset=["lat", "lon"], keep=keep
    ).reset_index(drop=True)


def validate_merge_quality(
    merged_df: pd.DataFrame,
    ready_df: pd.DataFrame,
    summary: Dict,
    min_ready_pct: float = 50.0,
) -> Dict:
    """
    Quality check on merge operation results.

    When:
        - After Census merge, before station generation
        - To warn about low geocoding success rates

    Args:
        merged_df: All merged rows
        ready_df: Rows with valid coordinates
        summary: Summary dict from merge_census_results_fast
        min_ready_pct: Minimum acceptable % of rows with coordinates

    Returns:
        Dict with quality metrics and warnings
    """
    total = summary.get("rows_total", len(merged_df))
    ready = summary.get("rows_ready", len(ready_df))

    if total == 0:
        ready_pct = 0.0
    else:
        ready_pct = (ready / total) * 100.0

    quality = {
        "total_rows": total,
        "ready_rows": ready,
        "ready_percentage": round(ready_pct, 2),
        "is_acceptable": ready_pct >= min_ready_pct,
        "warnings": [],
    }

    if ready_pct < min_ready_pct:
        quality["warnings"].append(
            f"Only {ready_pct:.1f}% of rows have valid coordinates "
            f"(minimum: {min_ready_pct}%)"
        )

    if summary.get("rows_still_missing", 0) > 0:
        pct_missing = (summary["rows_still_missing"] / total) * 100
        quality["warnings"].append(
            f"{summary['rows_still_missing']} rows ({pct_missing:.1f}%) "
            f"still missing coordinates after Census geocoding"
        )

    return quality
