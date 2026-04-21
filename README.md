# beta-optimizer

`beta-optimizer` is a Streamlit application for planning BRINC Drone as First Responder deployments. It ingests incident/CAD data, derives a jurisdiction boundary, generates candidate stations, solves fleet placement with mixed-integer optimization, and produces map, budget, RF, and export outputs for proposal workflows.

## What It Does

- Uploads and normalizes CAD / call-for-service data
- Identifies relevant city or county boundaries from local cached shapefiles
- Generates and scores candidate drone stations
- Optimizes Responder and Guardian placement with PuLP
- Renders interactive coverage maps with FAA and infrastructure overlays
- Estimates budget, time savings, operational savings, and grant potential
- Exports deployment plans, executive-summary HTML, and Google Earth KML

## Stack

- Python
- Streamlit
- Pandas / GeoPandas / Shapely / PyProj
- Plotly
- PuLP
- Google Sheets + Gmail integrations via `gspread`, `google-auth`, and SMTP

## Repository Layout

- `app.py`: primary application entry point and most business logic
- `requirements.txt`: Python dependencies
- `download_regulatory_layers.py`: downloads and caches FAA / infrastructure overlays
- `jurisdiction_data/`: local boundary shapefile cache
- `regulatory_layers/`: cached parquet overlays, generated locally
- `cell_coverage/`: local coverage data
- `modules/`: extracted helpers for parsing, onboarding, optimization, reporting, and config
- `QUICKSTART.md`: operational quick start for regulatory layers

## Local Setup

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

The advanced RF section imports `scipy`, but `scipy` is not currently listed in `requirements.txt`. Install it manually if you use that section:

```powershell
pip install scipy
```

### 3. Configure Streamlit secrets

The app reads configuration from `.streamlit/secrets.toml`.

Keys referenced in the code include:

- `[auth]` for Google OAuth login
- `GMAIL_ADDRESS`
- `GMAIL_APP_PASSWORD`
- `NOTIFY_EMAIL`
- `GOOGLE_SHEET_ID`
- `gcp_service_account`

These integrations are optional for basic local exploration, but features such as login, notification emails, and Sheets logging depend on them.

### 4. Cache regulatory layers

Recommended for normal use:

```powershell
python download_regulatory_layers.py
```

This populates `regulatory_layers/*.parquet` and makes FAA / hazards / cell tower / no-fly overlays load quickly.

### 5. Run the app

```powershell
streamlit run app.py
```

## Main User Flows

- Upload CAD or incident data
- Confirm or refine the inferred jurisdiction
- Tune station generation and deployment strategy in the sidebar
- Set Responder and Guardian counts
- Review coverage, response, budget, and RF outputs
- Export a deployment plan, executive summary, or KML

## Data Expectations

The repo already contains several large local datasets and caches. In normal operation the app also expects:

- local jurisdiction boundary files in `jurisdiction_data/`
- cached regulatory parquet files in `regulatory_layers/`
- uploaded CAD / XLSX / CSV / related incident exports from the user

Some generated data is intentionally ignored by git. See `.gitignore`.

## Architecture Notes

Current implementation characteristics:

- The application is mostly a monolith in `app.py`.
- UI, geospatial data access, optimization, export generation, and external integrations are tightly coupled.
- The `pages/` directory exists but is effectively unused, so this is still a single-app layout rather than a split Streamlit multipage app.

That structure works, but it increases change risk. The most natural refactor boundaries are:

- auth and external integrations
- boundary and geocoding utilities
- regulatory / map overlay loaders
- optimization engine
- export generation
- RF coverage modeling

## Cleanup Notes

- `public_reports/`, `streamlit_start.out`, `streamlit_start.err`, and `jurisdiction_data/temp_tiger_states/` are generated runtime artifacts and should not be committed.
- `pages/` currently appears unused and can be removed once you confirm there is no planned multipage UI split.
- `app.py` is still very large, so the next real cleanup step after repo hygiene is code extraction rather than more file deletion.
