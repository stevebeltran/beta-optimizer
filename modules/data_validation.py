"""
Pandera schemas for DataFrame validation.

These schemas enforce data integrity at each stage:
- CAD file parsing validation
- Census geocoding results validation
- Merged data validation
- Export data validation
"""

import logging
import pandera.pandas as pa
from pandera.pandas import Column, Index, Check
import pandas as pd

PA_OBJECT = getattr(pa, "object_", getattr(pa, "Object", object))


# ── CAD Parsing Validation ────────────────────────────────────────────────────

cad_raw_schema = pa.DataFrameSchema(
    columns={
        "_source_row_id": Column(
            pa.Int64,
            checks=[Check.greater_than_or_equal_to(0)],
            nullable=False,
            description="Unique row identifier from CAD source"
        ),
        "date": Column(
            PA_OBJECT,
            nullable=True,
            description="Incident date (will be coerced to datetime)"
        ),
        "address": Column(
            PA_OBJECT,
            nullable=True,
            description="Street address"
        ),
        "city": Column(
            PA_OBJECT,
            nullable=True,
            description="City name"
        ),
        "state": Column(
            PA_OBJECT,
            nullable=True,
            description="State code (2 letters)"
        ),
        "zip": Column(
            PA_OBJECT,
            nullable=True,
            description="ZIP code"
        ),
        "priority": Column(
            pa.Int64,
            checks=[Check.between(1, 5)],
            nullable=True,
            description="Dispatch priority 1-5"
        ),
        "agency": Column(
            PA_OBJECT,
            nullable=True,
            description="Responding agency"
        ),
    },
    strict="filter",  # Allow extra columns, but validate these
    description="Raw CAD data after initial parsing"
)


cad_with_coords_schema = pa.DataFrameSchema(
    columns={
        "_source_row_id": Column(
            pa.Int64,
            checks=[Check.greater_than_or_equal_to(0)],
            nullable=False,
        ),
        "lat": Column(
            pa.Float64,
            checks=[
                Check.between(-90, 90, raise_warning=False),
            ],
            nullable=True,
            description="Latitude (-90 to 90)"
        ),
        "lon": Column(
            pa.Float64,
            checks=[
                Check.between(-180, 180, raise_warning=False),
            ],
            nullable=True,
            description="Longitude (-180 to 180)"
        ),
        "geocode_source": Column(
            PA_OBJECT,
            checks=[Check.isin(["direct", "census_batch", "manual"])],
            nullable=True,
            description="Source of coordinate data"
        ),
    },
    strict="filter",
    description="CAD data with coordinate columns"
)


# ── Census Geocoding Results Validation ────────────────────────────────────────

census_result_schema = pa.DataFrameSchema(
    columns={
        "source_id": Column(
            pa.Int64,
            checks=[Check.greater_than_or_equal_to(0)],
            nullable=False,
            description="Row ID from source CAD"
        ),
        "lat": Column(
            pa.Float64,
            checks=[Check.between(-90, 90, raise_warning=False)],
            nullable=True,
            description="Matched latitude"
        ),
        "lon": Column(
            pa.Float64,
            checks=[Check.between(-180, 180, raise_warning=False)],
            nullable=True,
            description="Matched longitude"
        ),
        "match_status": Column(
            PA_OBJECT,
            checks=[Check.isin(["Match", "Tie", "No_Match"])],
            nullable=False,
            description="Census API match status"
        ),
        "match_type": Column(
            PA_OBJECT,
            checks=[Check.isin(["Exact", "Non_Exact", "Ambiguous", None])],
            nullable=True,
            description="Type of geocoding match"
        ),
        "matched_address": Column(
            PA_OBJECT,
            nullable=True,
            description="Full matched address"
        ),
    },
    strict="filter",
    description="Census batch geocoding results"
)


# ── Merged Results Validation ─────────────────────────────────────────────────

merged_cad_census_schema = pa.DataFrameSchema(
    columns={
        "_source_row_id": Column(
            pa.Int64,
            nullable=False,
        ),
        "lat": Column(
            pa.Float64,
            checks=[Check.between(-90, 90, raise_warning=False)],
            nullable=True,
        ),
        "lon": Column(
            pa.Float64,
            checks=[Check.between(-180, 180, raise_warning=False)],
            nullable=True,
        ),
        "geocode_source": Column(
            PA_OBJECT,
            checks=[Check.isin(["direct", "census_batch", "manual"])],
            nullable=True,
        ),
        "match_status": Column(
            PA_OBJECT,
            checks=[Check.isin(["Match", "Tie", "No_Match", None])],
            nullable=True,
        ),
    },
    strict="filter",
    description="Merged CAD and Census geocoding results"
)


# ── Export Data Validation ────────────────────────────────────────────────────

export_ready_schema = pa.DataFrameSchema(
    columns={
        "_source_row_id": Column(
            pa.Int64,
            nullable=False,
        ),
        "lat": Column(
            pa.Float64,
            checks=[Check.between(-90, 90)],
            nullable=False,
            description="Latitude must be valid for export"
        ),
        "lon": Column(
            pa.Float64,
            checks=[Check.between(-180, 180)],
            nullable=False,
            description="Longitude must be valid for export"
        ),
    },
    strict="filter",
    description="Data ready for geographic export (all coords valid)"
)


# ── Helper Functions ──────────────────────────────────────────────────────────

def validate_cad_raw(df: pd.DataFrame, raise_exceptions: bool = False) -> bool:
    """
    Validate raw CAD DataFrame after parsing.

    Args:
        df: DataFrame to validate
        raise_exceptions: If True, raise validation errors; if False, log warnings

    Returns:
        True if valid, False otherwise
    """
    try:
        cad_raw_schema.validate(df)
        return True
    except pa.errors.SchemaError as e:
        if raise_exceptions:
            raise
        logging.warning("CAD raw validation warning: %s", e)
        return False


def validate_cad_with_coords(df: pd.DataFrame, raise_exceptions: bool = False) -> bool:
    """
    Validate CAD DataFrame with coordinate columns.

    Args:
        df: DataFrame to validate
        raise_exceptions: If True, raise validation errors

    Returns:
        True if valid, False otherwise
    """
    try:
        cad_with_coords_schema.validate(df)
        return True
    except pa.errors.SchemaError as e:
        if raise_exceptions:
            raise
        logging.warning("CAD with coordinates validation warning: %s", e)
        return False


def validate_census_results(df: pd.DataFrame, raise_exceptions: bool = False) -> bool:
    """
    Validate Census batch geocoding results.

    Args:
        df: DataFrame to validate
        raise_exceptions: If True, raise validation errors

    Returns:
        True if valid, False otherwise
    """
    try:
        census_result_schema.validate(df)
        return True
    except pa.errors.SchemaError as e:
        if raise_exceptions:
            raise
        logging.warning("Census results validation warning: %s", e)
        return False


def validate_merged_data(df: pd.DataFrame, raise_exceptions: bool = False) -> bool:
    """
    Validate merged CAD + Census data.

    Args:
        df: DataFrame to validate
        raise_exceptions: If True, raise validation errors

    Returns:
        True if valid, False otherwise
    """
    try:
        merged_cad_census_schema.validate(df)
        return True
    except pa.errors.SchemaError as e:
        if raise_exceptions:
            raise
        logging.warning("Merged data validation warning: %s", e)
        return False


def validate_export_ready(df: pd.DataFrame, raise_exceptions: bool = False) -> bool:
    """
    Validate data ready for export (all coordinates must be valid).

    Args:
        df: DataFrame to validate
        raise_exceptions: If True, raise validation errors

    Returns:
        True if valid, False otherwise
    """
    try:
        export_ready_schema.validate(df)
        return True
    except pa.errors.SchemaError as e:
        if raise_exceptions:
            raise
        logging.warning("Export ready validation warning: %s", e)
        return False
