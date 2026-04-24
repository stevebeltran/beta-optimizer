# Project Skills

Use the smallest set of skills that matches the task. For this app, default to the five skills below.

## Core App
- Main entrypoint: `app.py`
- App type: Streamlit planning app with CAD ingest, jurisdiction detection, map layers, optimization, and export flows
- Main supporting modules:
  - `modules/cad_parser.py`
  - `modules/census_batch.py`
  - `modules/dashboard_helpers.py`
  - `modules/session_state.py`
  - `modules/optimization.py`
  - `modules/html_reports.py`
  - `modules/geospatial.py`

## Streamlit Monolith Surgery
- `app.py` is large and tightly coupled; make narrow edits
- Trace the exact UI flow before changing code
- Prefer fixing the specific branch, helper call, CSS block, or session-state interaction involved
- Avoid broad refactors unless explicitly requested

## Geospatial Data Safety
- Prefer source-code fixes over regenerating local data caches
- Avoid bulk edits in:
  - `jurisdiction_data/`
  - `regulatory_layers/`
  - `cell_coverage/`
  - `public_reports/`
- Only touch generated or cached map assets when the task explicitly requires it

## Current UX
- Use this skill for main-page layout, badges, hero sections, overlays, logos, and visual polish
- Check inline CSS and `st.markdown()` HTML blocks in `app.py` first
- Keep UI fixes scoped to the affected screen

## Validation
- Start with targeted syntax validation for touched files
- Preferred command for `app.py`-only work:

```powershell
python -m py_compile app.py
```

- Expand to touched modules only when needed

## Files To Inspect First
- `app.py`
- `modules/dashboard_helpers.py`
- `modules/geospatial.py`
- `modules/html_reports.py`
- `AGENTS.md`
- `CLAUDE.md`
