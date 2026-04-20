# Project Skills

## Core App
- Main entrypoint: `app.py`
- Main mode of operation: Streamlit app with CAD upload, jurisdiction detection, station generation, optimization, and export flows
- Key support modules:
  - `modules/cad_parser.py`
  - `modules/census_batch.py`
  - `modules/onboarding.py`
  - `modules/dashboard_helpers.py`
  - `modules/session_state.py`

## CAD Upload Skill
- Path 02 is the real CAD upload path in `app.py`
- Normal flow:
  - parse uploaded CAD
  - detect usable `lat/lon`
  - load or generate stations
  - detect jurisdiction
  - resolve boundaries
  - continue to stations workflow

## Coordinate Recovery Skill
- The app should try to recover coordinates from:
  - explicit `lat/lon` headers
  - bad headers with coordinate-like numeric cell values
  - combined coordinate strings
  - loose numeric coordinate pairs
- Only if coordinates are truly unrecoverable should the app fall back to Census geocoding

## Census Batch Geocoding Skill
- Implemented in `modules/census_batch.py`
- Current automated flow:
  - load raw CAD workbook/CSV
  - normalize headerless Excel exports when needed
  - infer `street`, `city`, `state`, `zip`
  - create Census batch chunks
  - submit chunks directly to the Census batch endpoint
  - parse mixed-width Census responses (`Match`, `Tie`, `No_Match`)
  - merge returned coordinates back into source rows
- Current chunk size: `5000`
- If a chunk fails and is larger than `1000` rows, the app splits it into smaller chunks and retries

## Known Data Characteristics
- Some CAD Excel files are effectively headerless exports
- Some staged files contain duplicate column names
- ZIP values may appear as `51401.0`
- Census results are not fixed-width CSV rows:
  - `No_Match` / `Tie` rows may have 3 columns
  - `Match` rows may have 8 columns

## Current UX Skill
- Upload overlay in `app.py` shows:
  - progress bar
  - active step
  - rolling log
  - error state for failed chunk or parse step
- Sidebar should expose corrected-file download after successful Census merge

## Important Merge Behavior
- If structured CAD parsing fails, the app falls back to staged source rows for Census merge-back
- Fallback merge injects safe defaults:
  - `priority = 3`
  - `agency = "police"`

## Validation Commands
```powershell
python -m py_compile app.py modules\cad_parser.py modules\census_batch.py modules\session_state.py
```

## Files To Inspect First For Upload/Geocoding Work
- `app.py`
- `modules/cad_parser.py`
- `modules/census_batch.py`
- `modules/session_state.py`
