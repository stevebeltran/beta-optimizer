# BRINC DFR Planning Software FAQ

## Frequently Asked Questions

### 1. What does this software do?

This software helps agencies and planning teams design a BRINC Drone as First Responder deployment for a specific jurisdiction. It uses incident or CAD data, local boundary data, candidate station locations, and optimization logic to recommend drone placement, estimate coverage, and generate planning outputs.

### 2. Who is this software built for?

It is built for public safety agencies, operations teams, proposal teams, and internal stakeholders who need to evaluate how a BRINC DFR program could perform in a real city, county, or service area.

### 3. What kind of inputs does it use?

The most common input is a CAD or incident export in CSV or Excel format. The software also uses local geographic data such as jurisdiction boundaries, regulatory overlays, and coverage layers to support analysis and mapping.

### 4. How does it know which jurisdiction to analyze?

The software reviews uploaded incident coordinates and matches them to local jurisdiction boundary data. If the incident data touches more than one area, the user can review and adjust the selected jurisdiction before running the optimization.

### 5. What is the difference between Responder and Guardian in the software?

The software models two different drone roles:

- `Responder` is used for shorter-range, fast tactical response.
- `Guardian` is used for longer-range coverage and broader overwatch.

Both fleets can be planned together, which allows the software to model a mixed deployment strategy instead of forcing a one-size-fits-all plan.

### 6. Does the software pick station locations automatically?

Yes. It can generate candidate stations, score them against call density and geography, and recommend placements based on the selected deployment strategy. Users can also add or lock custom stations if they want to include specific sites in the final plan.

### 7. What outputs does the software generate?

Depending on the workflow, the software can generate:

- Coverage maps
- Jurisdiction maps
- Fleet and budget summaries
- Response and coverage analysis
- Executive-summary HTML reports
- Google Earth KML briefing files
- Saved planning files for later reuse

### 8. Does it include FAA or regulatory map layers?

Yes. When the local regulatory cache has been set up, the software can display overlays such as FAA LAANC airspace, flight hazards, cell towers, and no-fly reference layers. These layers help with planning, review, and presentations.

### 9. Is this a live dispatch system?

No. This is a planning and analysis platform, not a real-time dispatch product. It uses historical incident patterns and geographic modeling to estimate how a drone deployment could perform.

### 10. Can it be used without internet access?

In many cases, yes. Once the required dependencies and local data caches are installed, much of the software can run locally. Internet access may still be needed for first-time setup, regulatory data downloads, authentication, email notifications, or Google integrations.

### 11. What should I do if the map layers are missing?

If map overlays are missing, the most common cause is that the local regulatory cache has not been downloaded or is out of date. Rebuilding the regulatory layer cache usually resolves the issue.

### 12. What should I do if my incident upload does not produce useful results?

The most common causes are:

- invalid or missing coordinates
- incorrect jurisdiction selection
- uploaded calls falling outside the selected boundary

Review the file, confirm the detected jurisdiction, and verify that latitude and longitude values are valid.

### 13. What is the main benefit of using this software?

The main benefit is that it ties deployment recommendations to real jurisdiction-specific geography and incident demand. Instead of guessing where drones should go, users get a defendable planning model based on actual data.

## Short Description For Website Use

This BRINC DFR planning software helps agencies and proposal teams evaluate drone deployment strategy using real incident data, jurisdiction boundaries, station modeling, and optimization. It produces coverage maps, fleet recommendations, budget summaries, and presentation-ready outputs for jurisdiction-specific planning.

## Contact / Support Line

For questions, planning support, or a guided review of your jurisdiction, contact the BRINC team directly.
