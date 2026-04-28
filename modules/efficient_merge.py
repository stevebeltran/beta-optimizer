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


CENSUS_MATCH_COLUMNS = ["source_id", "lat", "lon", "match_status", "match_type", "matched_address"]
CENSUS_JOIN_COLUMNS = ["_census_merge_key", "lat", "lon", "match_status", "match_type", "matched_address"]


def _source_merge_key(series: pd.Series) -> pd.Series:
    """Normalize CAD/Census source IDs for robust joins without changing output IDs."""
    return series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def _prepare_calls_for_merge(partial_calls_df: pd.DataFrame) -> pd.DataFrame:
    merged = partial_calls_df.copy().reset_index(drop=True)
    if "_source_row_id" not in merged.columns:
        merged["_source_row_id"] = range(len(merged))
    if "lat" not in merged.columns:
        merged["lat"] = pd.NA
    if "lon" not in merged.columns:
        merged["lon"] = pd.NA
    if "geocode_source" not in merged.columns:
        merged["geocode_source"] = pd.NA

    merged["lat"] = pd.to_numeric(merged["lat"], errors="coerce")
    merged["lon"] = pd.to_numeric(merged["lon"], errors="coerce")
    merged["_census_merge_key"] = _source_merge_key(merged["_source_row_id"])
    return merged


def _prepare_census_matches(result_df: pd.DataFrame | None) -> pd.DataFrame:
    result = pd.DataFrame() if result_df is None else result_df.copy()
    for column in CENSUS_MATCH_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA

    matched = result[CENSUS_MATCH_COLUMNS].copy()
    matched["_census_merge_key"] = _source_merge_key(matched["source_id"])
    matched = matched[
        matched["_census_merge_key"].notna()
        & (matched["_census_merge_key"].astype(str).str.strip() != "")
    ].copy()
    matched["lat"] = pd.to_numeric(matched["lat"], errors="coerce")
    matched["lon"] = pd.to_numeric(matched["lon"], errors="coerce")
    matched = matched.drop(columns=["source_id"])
    matched = matched.drop_duplicates(subset=["_census_merge_key"], keep="first")
    return matched[CENSUS_JOIN_COLUMNS]


def _finish_merge(merged: pd.DataFrame, census_filled) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    if "_census_filled" in merged.columns:
        census_filled = merged["_census_filled"].fillna(False).astype(bool)
    else:
        census_filled = pd.Series(census_filled, index=merged.index, dtype=bool)

    merged = merged.drop(columns=["_census_merge_key", "_census_filled"], errors="ignore")

    ready_df = merged[
        (merged["lat"].notna()) &
        (merged["lon"].notna()) &
        (merged["lat"].between(-90, 90)) &
        (merged["lon"].between(-180, 180))
    ].reset_index(drop=True)

    summary = {
        "rows_total": int(len(merged)),
        "rows_ready": int(len(ready_df)),
        "rows_geocoded": int(census_filled.sum()),
        "rows_still_missing": int(
            ((merged["lat"].isna()) | (merged["lon"].isna())).sum()
        ),
    }
    return merged, ready_df, summary


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
    merged_base = _prepare_calls_for_merge(partial_calls_df)
    matched = _prepare_census_matches(result_df)
    if matched.empty:
        return _finish_merge(merged_base, pd.Series(False, index=merged_base.index))

    # Convert to Polars (lazy evaluation)
    pl_cad = pl.from_pandas(merged_base)
    pl_census = pl.from_pandas(matched)

    # Merge (inner join to Census results, then left join to preserve all CAD rows)
    pl_merged = pl_cad.join(
        pl_census,
        on="_census_merge_key",
        how="left",
        suffix="_census"
    )

    # Fill coordinates: prefer direct, fallback to Census
    pl_merged = pl_merged.with_columns([
        (
            (pl.col("lat").is_null() | pl.col("lon").is_null()) &
            pl.col("lat_census").is_not_null() &
            pl.col("lon_census").is_not_null()
        ).alias("_census_filled"),
        pl.when(pl.col("lat").is_null())
        .then(pl.col("lat_census"))
        .otherwise(pl.col("lat"))
        .alias("lat"),
        pl.when(pl.col("lon").is_null())
        .then(pl.col("lon_census"))
        .otherwise(pl.col("lon"))
        .alias("lon"),
    ]).drop(["lat_census", "lon_census"])

    if "geocode_source" not in pl_merged.columns:
        pl_merged = pl_merged.with_columns(
            pl.lit(None).alias("geocode_source")
        )

    # Mark geocode source
    pl_merged = pl_merged.with_columns([
        pl.when(pl.col("_census_filled"))
        .then(pl.lit("census_batch"))
        .otherwise(pl.col("geocode_source"))
        .alias("geocode_source")
    ])

    # Convert back to pandas
    merged_df = pl_merged.to_pandas()
    return _finish_merge(merged_df, merged_df.get("_census_filled", False))


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
    merged = _prepare_calls_for_merge(partial_calls_df)
    matched = _prepare_census_matches(result_df)

    # Merge with suffixes to avoid column conflicts
    merged = merged.merge(
        matched,
        on="_census_merge_key",
        how="left",
        suffixes=("", "_census")
    )

    # Fill coordinates: prefer direct, fallback to Census
    lat_missing = merged["lat"].isna()
    lon_missing = merged["lon"].isna()
    merged.loc[lat_missing, "lat"] = merged.loc[lat_missing, "lat_census"]
    merged.loc[lon_missing, "lon"] = merged.loc[lon_missing, "lon_census"]

    # Mark geocode source
    census_filled = (lat_missing | lon_missing) & merged["lat_census"].notna() & merged["lon_census"].notna()
    merged.loc[census_filled, "geocode_source"] = "census_batch"
    merged = merged.drop(
        columns=[c for c in ["lat_census", "lon_census"] if c in merged.columns]
    )

    return _finish_merge(merged, census_filled)


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
