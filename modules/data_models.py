"""
Pydantic data models for strict type validation and data contracts.

These models enforce data integrity throughout the pipeline:
- CAD record validation
- Geocoding result validation
- Station candidate validation
"""

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    ConfigDict,
)
from typing import Optional, Literal
from datetime import datetime


class CoordinateBounds(BaseModel):
    """Validate latitude and longitude bounds."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    lat: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="Latitude must be between -90 and 90"
    )
    lon: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="Longitude must be between -180 and 180"
    )


class GeocodingResult(BaseModel):
    """
    Validated geocoding result from Census batch API.

    Attributes:
        source_id: Original row ID from CAD data
        lat: Latitude coordinate
        lon: Longitude coordinate
        match_status: Status from Census (Match, Tie, No_Match)
        match_type: Type of match (Exact, Non_Exact, Ambiguous)
        matched_address: Full matched address string
        geocode_source: Source of geocoding (direct, census_batch, manual)
    """
    source_id: int = Field(..., description="Row ID linking back to CAD source")
    lat: Optional[float] = Field(None, ge=-90.0, le=90.0)
    lon: Optional[float] = Field(None, ge=-180.0, le=180.0)
    match_status: Literal["Match", "Tie", "No_Match"] = Field(
        "No_Match",
        description="Census batch API match status"
    )
    match_type: Optional[Literal["Exact", "Non_Exact", "Ambiguous"]] = Field(
        None,
        description="Type of geocoding match"
    )
    matched_address: Optional[str] = Field(
        None,
        description="Full address as matched by Census"
    )
    geocode_source: Literal["direct", "census_batch", "manual"] = Field(
        "direct",
        description="Source of coordinate data"
    )

    @field_validator("lat", "lon", mode="before")
    @classmethod
    def coerce_numeric(cls, v):
        """Convert numeric strings to float, handle NaN."""
        if v is None or (isinstance(v, float) and v != v):  # NaN check
            return None
        try:
            return float(v)
        except (ValueError, TypeError):
            return None


class CADRecord(BaseModel):
    """
    Validated CAD record with optional coordinates.

    Attributes:
        _source_row_id: Unique row identifier
        lat: Latitude (optional if will be geocoded)
        lon: Longitude (optional if will be geocoded)
        address: Street address
        city: City name
        state: State code (2 letters)
        zip: ZIP code
        priority: Dispatch priority (1-5)
        agency: Responding agency
        date: Date of incident
    """
    source_row_id: int = Field(..., alias="_source_row_id")
    lat: Optional[float] = Field(None, ge=-90.0, le=90.0)
    lon: Optional[float] = Field(None, ge=-180.0, le=180.0)
    address: Optional[str] = Field(None, max_length=255)
    city: Optional[str] = Field(None, max_length=100)
    state: Optional[str] = Field(None, min_length=2, max_length=2)
    zip: Optional[str] = Field(None, max_length=10)
    priority: int = Field(3, ge=1, le=5)
    agency: str = Field("police", max_length=50)
    date: Optional[datetime] = Field(None)

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )

    @field_validator("lat", "lon", mode="before")
    @classmethod
    def coerce_numeric(cls, v):
        """Convert numeric strings to float."""
        if v is None or (isinstance(v, float) and v != v):
            return None
        try:
            if isinstance(v, str):
                return float(v) if v.strip() else None
            return float(v)
        except (ValueError, TypeError):
            return None

    @field_validator("state", mode="before")
    @classmethod
    def uppercase_state(cls, v):
        """Normalize state to uppercase."""
        if isinstance(v, str):
            return v.upper().strip()
        return v


class StationCandidate(BaseModel):
    """
    Station location candidate with spatial metadata.

    Attributes:
        id: Unique station identifier
        lat: Latitude of station
        lon: Longitude of station
        name: Station name/identifier
        coverage_radius_m: Coverage radius in meters
        jurisdiction: Administrative jurisdiction
        priority_score: Deployment priority (0-100)
    """
    id: str = Field(..., description="Unique station ID")
    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)
    name: str = Field(..., max_length=255)
    coverage_radius_m: float = Field(1000.0, gt=0)
    jurisdiction: Optional[str] = Field(None)
    priority_score: float = Field(50.0, ge=0.0, le=100.0)


class MergeReport(BaseModel):
    """
    Report from a merge operation showing statistics.

    Attributes:
        rows_total: Total rows processed
        rows_ready: Rows with valid coordinates
        rows_geocoded: Rows that received Census geocoding
        rows_still_missing: Rows without coordinates
        merge_time_seconds: Time taken for merge operation
    """
    rows_total: int = Field(..., ge=0)
    rows_ready: int = Field(..., ge=0)
    rows_geocoded: int = Field(..., ge=0)
    rows_still_missing: int = Field(..., ge=0)
    merge_time_seconds: float = Field(0.0, ge=0.0)

    @field_validator("rows_ready", mode="after")
    @classmethod
    def validate_ready_le_total(cls, v, info):
        """Ensure ready rows don't exceed total."""
        if "rows_total" in info.data:
            if v > info.data["rows_total"]:
                raise ValueError("rows_ready cannot exceed rows_total")
        return v
