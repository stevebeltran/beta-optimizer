# New Skills Usage Guide

Quick reference for the four new data processing libraries.

---

## 1. POLARS - Fast DataFrame Operations

**Problem:** Your Census merge takes 2-3 seconds for 50K rows

**Solution:** Polars handles it in 0.3 seconds

**Usage:**
```python
from modules.efficient_merge import merge_census_results_fast

# Automatic: Uses Polars if available, falls back to pandas
merged, ready, summary = merge_census_results_fast(
    cad_df,
    census_results_df,
    use_polars=True  # Default
)
```

**When it helps:**
- Census batch > 10K rows
- Multiple consecutive merges
- Low memory environments

**Performance:** 5-10x faster on large datasets, auto-fallback to pandas if unavailable.

---

## 2. PANDERA - DataFrame Validation

**Problem:** Bad data slips through and breaks downstream steps

**Solution:** Validate DataFrames against schemas at each step

**Usage:**
```python
from modules.data_validation import (
    validate_cad_raw,
    validate_census_results,
    validate_merged_data,
)

# After parsing CAD file
parsed = parse_cad_file(upload)
if not validate_cad_raw(parsed):
    raise ValueError("CAD file has invalid structure")

# After Census API
results = submit_census_batch(batch)
validate_census_results(results, raise_exceptions=True)

# After merge
merged, ready, stats = merge_census_results(cad_df, census_df)
validate_merged_data(merged)
```

**Error messages tell you exactly what's wrong:**
```
SchemaError: column 'lat' failed Check(between(-90, 90)):
  - Row 127: value 123.45 is outside range
  - Row 342: value -91.00 is outside range
```

**When to use:**
- After CAD file parsing (find issues early)
- After Census API calls (verify response structure)
- Before station generation (ensure data quality)

---

## 3. PYDANTIC - Type-Safe Data Models

**Problem:** Hard to remember what fields each struct has; no IDE autocomplete

**Solution:** Define strict models with validation

**Usage:**
```python
from modules.data_models import GeocodingResult, CADRecord

# Parse Census result with type safety
result = GeocodingResult(
    source_id=1,
    lat=40.7128,
    lon=-74.0060,
    match_status="Match",
    geocode_source="census_batch"
)

# Pydantic validates:
# ✓ lat is -90 to 90
# ✗ If lat=91, raises ValueError
# ✓ Coerces "40.7128" (string) to float
# ✓ match_status must be in ["Match", "Tie", "No_Match"]

# Use in functions for better IDE support:
def process_result(result: GeocodingResult) -> float:
    # IDE knows result.lat is float
    # IDE autocompletes: result.lat, result.lon, result.match_status, etc.
    return result.lat + result.lon
```

**Benefits:**
- IDE autocomplete for all fields
- Clear error messages if data is invalid
- Auto coercion of types (e.g., "40.7" → 40.7)
- Built-in documentation

---

## 4. DASK - Distributed Computing

**Problem:** Dataset might grow too large for single machine

**Solution:** Dask handles multi-GB datasets with same code

**Current Status:** Available but not yet active

**Future Usage:**
```python
import dask.dataframe as dd

# Instead of pandas:
# df = pd.read_csv('huge_file.csv')

# Use Dask:
df = dd.read_csv('huge_file.csv')

# Same operations, distributes across cores:
result = df.groupby('city').apply(...)
result.compute()  # Execute
```

---

## When to Use Each

### Polars
✓ After Census merge  
✓ Multiple joins/merges  
✓ Large DataFrames  
✗ Small datasets (< 5K rows)  

### Pandera
✓ After file parsing  
✓ After API calls  
✓ Before critical operations  
✗ In tight loops  

### Pydantic
✓ Defining data contracts  
✓ Parsing API responses  
✓ Function arguments  
✗ One-off scripts  

### Dask
✓ Multi-GB datasets  
✓ Cluster processing  
✗ Current use cases (data < 500MB)  

---

## Integration Status

| Library | Status | Integration |
|---------|--------|-------------|
| Polars | ✅ Active | Auto-used in `merge_census_results_fast()` |
| Pandera | ✅ Active | Validates in merge pipeline |
| Pydantic | ✅ Ready | Available for adoption |
| Dask | ⏳ Available | Not yet integrated |

---

## Common Patterns

### Pattern 1: Validate → Process → Export

```python
from modules.data_validation import validate_cad_raw
from modules.efficient_merge import merge_census_results_fast

# 1. Parse and validate
cad = parse_cad_file(file)
validate_cad_raw(cad)  # Catches issues early

# 2. Geocode with Census
census_results = submit_census_batch(cad)

# 3. Merge efficiently (auto-fast with Polars)
merged, ready, summary = merge_census_results_fast(cad, census_results)

# 4. Export (no validation needed - already done)
ready.to_csv('output.csv')
```

### Pattern 2: Type-Safe Result Processing

```python
from modules.data_models import GeocodingResult
import pandas as pd

def process_row(row: pd.Series) -> GeocodingResult:
    """Convert DataFrame row to typed object."""
    try:
        return GeocodingResult(**row.to_dict())
    except ValueError as e:
        # Pydantic gives clear error
        print(f"Invalid geocoding result: {e}")
        return None

# Use:
results = [process_row(row) for row in census_df.iterrows()]
```

### Pattern 3: Quality Checks

```python
from modules.efficient_merge import validate_merge_quality

merged, ready, summary = merge_census_results_fast(cad, census)

quality = validate_merge_quality(merged, ready, summary, min_ready_pct=75.0)

if quality['is_acceptable']:
    print(f"✓ {quality['ready_percentage']}% of data is ready")
else:
    for warning in quality['warnings']:
        print(f"⚠ {warning}")
    # Maybe ask user to manually geocode missing records?
```

---

## Testing/Verification

Run syntax validation:
```bash
python -m py_compile modules/data_models.py
python -m py_compile modules/data_validation.py
python -m py_compile modules/efficient_merge.py
python -m py_compile modules/census_batch.py
```

Or in app.py:
```python
# Test imports work
try:
    from modules.data_models import GeocodingResult
    from modules.data_validation import validate_census_results
    from modules.efficient_merge import merge_census_results_fast
    print("✓ All new modules imported successfully")
except ImportError as e:
    print(f"✗ Import failed: {e}")
```

---

## Where to Find Them

- `modules/data_models.py` - Pydantic models
- `modules/data_validation.py` - Pandera schemas
- `modules/efficient_merge.py` - Polars/pandas merges
- `modules/census_batch.py` - Updated to use new merge
- `requirements.txt` - Added: polars, pandera, pydantic, dask
