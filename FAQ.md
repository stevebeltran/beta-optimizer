# Frankenstein FAQ

## What is this software?

`Frankenstein` is a Streamlit-based planning tool for BRINC Drone as First Responder deployments. It uses incident or CAD data, jurisdiction boundaries, station candidates, and optimization logic to recommend drone placement, estimate coverage, and generate proposal-ready outputs.

## Who is it for?

It is designed for public safety, operations, sales engineering, and proposal workflows. Typical users include police, fire, emergency management, and internal teams evaluating where BRINC Responder and Guardian drones should be deployed.

## What problem does it solve?

The software helps answer practical deployment questions such as:

- Where should drones be staged?
- How many Responders and Guardians are needed?
- How much of the jurisdiction can be covered?
- How quickly can drones arrive?
- What budget and operational impact does the proposed fleet create?

## What data do I need to use it?

The most common input is a CAD or incident export, usually as CSV or Excel. The app can also work from saved app state and local cached geographic data such as jurisdiction boundaries, FAA layers, and cell coverage data already stored in the workspace.

## How does it determine the jurisdiction?

The app analyzes uploaded incident coordinates and matches them against local jurisdiction boundary data. If multiple jurisdictions are relevant, the sidebar lets you review and refine the selected area before optimization runs.

## What are Responder and Guardian drones in the app?

- `Responder` is the short-range, faster-response fleet used for dense tactical coverage.
- `Guardian` is the longer-range fleet used for broader overwatch and extended reach.

The software models them separately, then combines the results into one deployment plan.

## Does the app choose stations automatically?

Yes. It can generate candidate stations and score them against call density, geography, and deployment strategy. Users can also add or lock custom stations when they want to force specific sites into the plan.

## What outputs can the software generate?

The app can produce:

- Interactive coverage and jurisdiction maps
- Budget and fleet summaries
- Response and coverage analysis
- Executive-summary HTML reports
- Google Earth KML briefing files
- Saved planning files for later reuse

## Are FAA and regulatory overlays included?

Yes, if the local regulatory cache has been downloaded. The app supports overlays such as FAA LAANC airspace, flight hazards, cell towers, and no-fly reference layers. These are optional for basic usage but useful for planning and presentation.

## Is this a real-time dispatch system?

No. It is a planning and proposal tool, not a live dispatch platform. It uses historical incident data and geographic modeling to estimate coverage, travel, and fleet performance.

## Do I need internet access to run it?

Not always. Once dependencies and local data caches are in place, much of the app can run locally. Internet access may still be needed for initial setup, downloading regulatory layers, authentication, or connected services such as Google Sheets and email.

## How do I start the software locally?

Typical local startup:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python download_regulatory_layers.py
streamlit run app.py
```

## What if maps or overlays are missing?

Usually that means the local cache is incomplete or outdated. Re-run:

```powershell
python download_regulatory_layers.py
```

Then restart the app and verify the `regulatory_layers/` folder contains parquet files.

## What if uploaded calls do not produce useful coverage?

The most common causes are bad coordinates, the wrong jurisdiction selection, or incident points falling outside the chosen boundary. Review the uploaded file, confirm the detected jurisdiction in the sidebar, and check whether the incident records include valid latitude and longitude values.

## Where should I look next for more detail?

- `README.md` for the product overview and architecture
- `QUICKSTART.md` for regulatory-layer setup
- `INTEGRATION_GUIDE.md` for connected services
- `app.py` and `modules/` for implementation details
