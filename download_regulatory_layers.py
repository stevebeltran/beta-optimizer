#!/usr/bin/env python3
"""
Download and Cache Regulatory/Infrastructure Layers (SIMPLIFIED)
==================================================================

Creates fallback/mock regulatory data layers for the app.
Since live APIs (Overpass, OpenCelliD, FAA) have rate limits,
timeouts, and incomplete data, this script:

  1. Creates mock FAA LAANC airspace (state-level grids)
  2. Creates mock FAA obstacles (sparse points)
  3. Provides instructions for manual OpenCelliD download
  4. Creates basic no-fly zone polygons
  5. Caches predownloaded airfields

The app can then load these cached files instantly instead of
querying slow APIs during every map interaction.

Usage:
  python download_regulatory_layers.py                    # Create all
  python download_regulatory_layers.py --airfields-only   # Just airfields
  python download_regulatory_layers.py --quick            # Quick mock setup

Output:
  regulatory_layers/
    ├── faa_airspace_{STATE}.parquet      (mock grid)
    ├── faa_obstacles.parquet             (mock points)
    ├── no_fly_zones.parquet              (mock polygons)
    └── airfields_us.parquet              (cached from Overpass)

Requires:
  pip install geopandas pyarrow pandas shapely requests
"""

import os
import sys
import json
import time
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from shapely.geometry import Point, Polygon, box, shape
import requests
from tqdm import tqdm

# ─── CONFIG ───────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("regulatory_layers")
OUTPUT_DIR.mkdir(exist_ok=True)

US_STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID",
    "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS",
    "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK",
    "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV",
    "WI", "WY", "DC"
]

# Approximate state bounding boxes (for mock data generation)
STATE_BOUNDS = {
    "AL": (-88.5, 30.2, -84.9, 35.0), "AK": (-172.0, 51.3, -130.0, 71.6),
    "AZ": (-114.8, 31.3, -109.0, 37.0), "AR": (-94.4, 33.0, -89.6, 36.5),
    "CA": (-124.5, 32.5, -114.1, 42.0), "CO": (-109.1, 36.9, -102.0, 41.0),
    "CT": (-73.7, 40.9, -71.8, 42.1), "DE": (-75.8, 38.4, -75.0, 39.8),
    "FL": (-87.6, 24.5, -80.0, 31.0), "GA": (-85.6, 30.4, -80.8, 35.0),
    "HI": (-160.2, 18.9, -154.8, 22.2), "ID": (-117.2, 42.0, -111.0, 49.0),
    "IL": (-91.5, 37.0, -87.0, 42.5), "IN": (-88.1, 37.8, -84.8, 41.8),
    "IA": (-96.6, 40.3, -90.1, 43.5), "KS": (-102.0, 37.0, -94.6, 40.0),
    "KY": (-89.6, 36.5, -81.9, 39.1), "LA": (-94.0, 29.0, -88.8, 33.0),
    "ME": (-71.1, 43.0, -66.9, 47.5), "MD": (-79.5, 37.9, -75.0, 39.7),
    "MA": (-73.5, 41.2, -69.9, 42.9), "MI": (-90.4, 41.7, -83.3, 48.3),
    "MN": (-97.2, 43.5, -89.5, 49.4), "MS": (-91.7, 30.2, -88.1, 35.0),
    "MO": (-95.8, 36.0, -90.1, 40.6), "MT": (-116.0, 45.0, -104.0, 49.0),
    "NE": (-104.1, 40.0, -95.3, 43.0), "NV": (-120.0, 35.0, -114.4, 42.0),
    "NH": (-72.6, 42.7, -70.7, 45.3), "NJ": (-75.6, 38.9, -73.9, 41.4),
    "NM": (-109.0, 31.8, -103.0, 37.0), "NY": (-79.8, 40.5, -71.9, 45.0),
    "NC": (-84.3, 33.8, -75.4, 36.6), "ND": (-104.0, 45.9, -96.6, 49.0),
    "OH": (-84.8, 38.4, -80.5, 42.3), "OK": (-103.0, 33.6, -94.4, 37.0),
    "OR": (-124.6, 42.0, -116.5, 46.3), "PA": (-80.5, 39.7, -74.7, 42.3),
    "RI": (-71.9, 41.1, -71.1, 42.0), "SC": (-83.4, 32.0, -78.5, 35.2),
    "SD": (-104.0, 42.5, -96.4, 45.9), "TN": (-90.3, 35.0, -81.6, 36.7),
    "TX": (-106.6, 25.8, -93.5, 36.5), "UT": (-114.0, 37.0, -109.0, 42.0),
    "VT": (-73.4, 42.7, -71.5, 45.0), "VA": (-83.7, 36.5, -75.2, 39.5),
    "WA": (-124.7, 45.6, -116.9, 49.0), "WV": (-82.6, 37.2, -77.7, 40.6),
    "WI": (-92.9, 42.5, -86.8, 47.3), "WY": (-111.0, 41.0, -104.0, 45.0),
    "DC": (-77.1, 38.8, -76.9, 39.0),
}

# ─── MOCK FAA AIRSPACE GENERATOR ──────────────────────────────────────────────

def generate_mock_faa_airspace_for_state(state):
    """
    Generate mock FAA LAANC airspace grid for a state.

    In reality, FAA data requires their official LAANC API (with auth).
    Since we can't auto-fetch that, we create a simplified mock grid
    that shows the fallback behavior the app uses anyway.

    The app's load_faa_parquet() function will use this mock data
    until real FAA data is available.
    """
    bounds = STATE_BOUNDS.get(state, None)
    if not bounds:
        return gpd.GeoDataFrame()

    minx, miny, maxx, maxy = bounds

    # Create a grid of airspace zones (simplified)
    features = []
    grid_size = 0.5  # degrees (~30 miles)

    lat = miny
    while lat < maxy:
        lon = minx
        while lon < maxx:
            # Vary ceiling based on location (higher near coasts/cities)
            ceiling_ft = 400
            if state in ["CA", "TX", "FL", "NY", "IL"]:
                ceiling_ft = 300  # More restricted in major metros
            elif state in ["AK", "MT", "WY", "ND"]:
                ceiling_ft = 400  # Less restricted in rural areas

            cell = box(lon, lat, lon + grid_size, lat + grid_size)
            features.append({
                'geometry': cell,
                'ceiling_ft': ceiling_ft,
                'airspace_class': 'Mock (Fallback)',
                'name': f"{state} Zone",
                'state': state,
            })
            lon += grid_size
        lat += grid_size

    if features:
        gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
        return gdf

    return gpd.GeoDataFrame()

def generate_mock_faa_obstacles():
    """
    Generate mock FAA Digital Obstacle File (DOF) — obstacles > 200 ft.

    Real DOF requires manual download from FAA. We generate mock data
    so the app can run without it. The app gracefully handles missing
    obstacle data, so this is purely a fallback for testing.

    For real deployment, download DOF_Public.zip from:
    https://www.faa.gov/air_traffic/publications/notices/
    """
    print("  Generating mock FAA obstacles...")

    features = []

    # Scatter some mock obstacles across the country
    # (tall towers, buildings, antennas)
    major_metros = [
        (40.7128, -74.0060, "NYC Tower"),      # New York
        (34.0522, -118.2437, "LA Building"),   # LA
        (41.8781, -87.6298, "Chicago Tower"),  # Chicago
        (29.7604, -95.3698, "Houston Tower"),  # Houston
        (33.7490, -84.3880, "Atlanta Tower"),  # Atlanta
        (39.7392, -104.9903, "Denver Tower"),  # Denver
        (47.6062, -122.3321, "Seattle Tower"), # Seattle
    ]

    for lat, lon, name in major_metros:
        features.append({
            'geometry': Point(lon, lat),
            'id': np.random.randint(1000, 9999),
            'agl_height': 250 + np.random.randint(0, 300),
            'type': 'Mock Tower',
            'verified': False,
        })

    if features:
        gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
        return gdf

    return gpd.GeoDataFrame()

# ─── OPENCELID CELL TOWERS DOWNLOADER ────────────────────────────────────

def load_cell_towers_from_local_gz(state):
    """
    Load REAL cell tower data from local .gz files in cell_coverage/raw/.

    Files are OpenCellID format with columns:
    radio,mcc,mnc,lac,cellid,unknown,lon,lat,...

    Filters by state bounds and returns GeoDataFrame.
    """
    import gzip

    try:
        # Get state bounds
        bounds = STATE_BOUNDS.get(state, None)
        if not bounds:
            return gpd.GeoDataFrame()

        minx, miny, maxx, maxy = bounds
        features = []

        # Try loading from MCC files (310 = USA, 311 = PR/USVI)
        for mcc in ["310", "311", "312", "313", "314"]:
            gz_file = Path("cell_coverage/raw") / f"{mcc}.csv.gz"

            if not gz_file.exists():
                continue

            try:
                with gzip.open(gz_file, 'rt', encoding='utf-8', errors='ignore') as f:
                    for line_num, line in enumerate(f):
                        if line_num > 500000:  # Limit to avoid memory
                            break

                        parts = line.strip().split(',')
                        if len(parts) < 8:
                            continue

                        try:
                            lon = float(parts[6])
                            lat = float(parts[7])

                            # Check if within state bounds
                            if minx <= lon <= maxx and miny <= lat <= maxy:
                                features.append({
                                    'geometry': Point(lon, lat),
                                    'radio': parts[0],
                                    'mcc': parts[1],
                                    'mnc': parts[2],
                                })
                        except (ValueError, IndexError):
                            continue

            except Exception:
                continue

        if features:
            gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
            return gdf

        return gpd.GeoDataFrame()

    except Exception:
        return gpd.GeoDataFrame()

def generate_mock_cell_towers_for_state(state):
    """
    Generate mock cell tower data for fallback testing.

    Real cell tower data is available from OpenCelliD:
    https://opencellid.org/downloads.php

    Instructions for getting real data:
    1. Visit: https://opencellid.org/downloads.php
    2. Download: cells_{state}.csv.zip (free, no key required)
    3. Extract: cells_{state}.csv to regulatory_layers/
    4. Re-run this script

    For now, we generate sparse mock towers in major metros so the app
    can demonstrate the feature without live data.
    """
    bounds = STATE_BOUNDS.get(state, None)
    if not bounds:
        return gpd.GeoDataFrame()

    minx, miny, maxx, maxy = bounds
    features = []

    # Scatter ~5–10 mock cell towers per state
    np.random.seed(hash(state) % 2**32)  # Reproducible per state
    num_towers = np.random.randint(5, 12)

    for _ in range(num_towers):
        lon = np.random.uniform(minx, maxx)
        lat = np.random.uniform(miny, maxy)
        features.append({
            'geometry': Point(lon, lat),
            'cell_id': np.random.randint(1000000, 9999999),
            'radio': np.random.choice(['4G', '4G', '4G', '5G']),  # Mostly 4G
            'mcc': 310,  # USA
            'mnc': np.random.choice([10, 260, 410]),  # AT&T, Verizon, T-Mobile
            'mcc_mnc': f"310-{np.random.choice([10, 260, 410])}",
        })

    if features:
        gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
        return gdf

    return gpd.GeoDataFrame()

# ─── NO-FLY ZONES GENERATOR ───────────────────────────────────────────────

def generate_mock_no_fly_zones():
    """
    Generate mock no-fly zones (parks, protected areas, water).

    Real data is available from OpenStreetMap (Overpass API), but
    it has rate limits and is slow. We generate mock zones for
    demonstration and fallback testing.

    For production, you could extend this to query Overpass with
    better rate limiting (e.g., retry with exponential backoff).
    """
    print("  Generating mock no-fly zones...")

    features = []

    # Mock national parks and protected areas
    mock_parks = [
        (37.8, -119.5, "Yosemite", "National Park"),
        (47.5, -121.0, "Mount Rainier", "National Park"),
        (42.3, -71.1, "Boston Common", "Park"),
        (34.0, -118.2, "Griffith Park", "Park"),
        (41.0, -88.0, "Grant Park", "Park"),
    ]

    for lat, lon, name, ptype in mock_parks:
        # Create ~0.5 degree radius polygon around point
        angle = np.linspace(0, 2*np.pi, 32)
        r = 0.3  # degrees
        lons = lon + r * np.cos(angle)
        lats = lat + r * np.sin(angle)
        poly = Polygon(zip(lons, lats))

        features.append({
            'geometry': poly,
            'zone_type': ptype,
            'name': name,
        })

    if features:
        gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
        print(f"    Created {len(gdf)} mock no-fly zones")
        return gdf

    return gpd.GeoDataFrame()

# ─── AIRFIELDS PREDOWNLOAD ────────────────────────────────────────────────

def download_airfields_us_with_retry():
    """
    Download all US airfields from OpenStreetMap with retry logic.

    Overpass API has rate limits (429) and timeouts (504). This version:
    1. Tries to query Overpass with retries
    2. Falls back to mock airfields if needed
    3. Caches result so it's instant during app runtime
    """
    print("Predownloading all US airfields (with retry)...")

    # Try Overpass API with exponential backoff
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"  Attempt {attempt + 1}/{max_retries}...")

            query = """[out:json];
            (
              node["aeroway"~"aerodrome|heliport"](25,-125,50,-65);
              way["aeroway"~"aerodrome|heliport"](25,-125,50,-65);
            );
            out center;
            """

            headers = {'User-Agent': 'BRINC-DataDownload/1.0'}
            response = requests.post(
                "https://overpass-api.de/api/interpreter",
                data=query,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            features = []
            for el in data.get('elements', []):
                lat = el.get('lat') or el.get('center', {}).get('lat')
                lon = el.get('lon') or el.get('center', {}).get('lon')
                name = el.get('tags', {}).get('name', 'Unknown')
                iata = el.get('tags', {}).get('iata_code', '')
                icao = el.get('tags', {}).get('icao_code', '')

                if lat and lon:
                    features.append({
                        'geometry': Point(lon, lat),
                        'name': name,
                        'iata': iata,
                        'icao': icao,
                        'type': el.get('tags', {}).get('aeroway', 'unknown'),
                    })

            if features:
                gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
                out_path = OUTPUT_DIR / "airfields_us.parquet"
                gdf.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")
                print(f"    [OK] Saved {len(gdf)} airfields to {out_path}")
                return gdf

        except requests.exceptions.Timeout:
            wait = 2 ** attempt
            print(f"    Timeout. Waiting {wait}s before retry...")
            time.sleep(wait)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Rate limited
                wait = 10 * (2 ** attempt)
                print(f"    Rate limited (429). Waiting {wait}s before retry...")
                time.sleep(wait)
            else:
                print(f"    HTTP Error {e.response.status_code}: {e}")
                break
        except Exception as e:
            print(f"    Error: {e}")
            break

    # Fallback: create mock airfields
    print("  Overpass API failed. Creating mock airfields for fallback...")
    return generate_mock_airfields_us()

def generate_mock_airfields_us():
    """Generate mock US airfields for fallback when API is unavailable."""
    features = []
    mock_airports = [
        (47.6, -122.3, "Seattle-Tacoma", "KSEA", "SEA"),
        (37.6, -122.4, "San Francisco", "KSFO", "SFO"),
        (34.1, -118.2, "Los Angeles", "KLAX", "LAX"),
        (33.7, -116.2, "San Diego", "KSAN", "SAN"),
        (39.9, -104.9, "Denver", "KDEN", "DEN"),
        (40.8, -111.9, "Salt Lake", "KSLC", "SLC"),
        (41.9, -87.9, "Chicago", "KORD", "ORD"),
        (42.4, -83.3, "Detroit", "KDTW", "DTW"),
        (40.8, -73.9, "New York", "KJFK", "JFK"),
        (40.8, -74.0, "Newark", "KEWR", "EWR"),
        (38.7, -77.0, "Washington DC", "KDCA", "DCA"),
        (28.4, -81.3, "Orlando", "KMCO", "MCO"),
        (25.8, -80.3, "Miami", "KMIA", "MIA"),
        (29.6, -95.2, "Houston", "KIAH", "IAH"),
        (32.9, -97.0, "Dallas", "KDFW", "DFW"),
        (35.0, -106.6, "Albuquerque", "KABQ", "ABQ"),
        (33.7, -84.4, "Atlanta", "KATL", "ATL"),
    ]

    for lat, lon, name, icao, iata in mock_airports:
        features.append({
            'geometry': Point(lon, lat),
            'name': name,
            'iata': iata,
            'icao': icao,
            'type': 'aerodrome',
        })

    if features:
        gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
        out_path = OUTPUT_DIR / "airfields_us.parquet"
        gdf.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")
        print(f"    [OK] Saved {len(gdf)} mock airfields to {out_path}")
        return gdf

    return gpd.GeoDataFrame()

# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate regulatory/infrastructure layer cache files",
        epilog="Creates parquet files for fast loading during app runtime. "
               "Uses mock data + Overpass API with retry logic."
    )
    parser.add_argument("--quick", action="store_true",
                       help="Quick setup: mock data only (< 10 sec)")
    parser.add_argument("--airfields-only", action="store_true",
                       help="Download airfields only (with retry logic)")
    parser.add_argument("--state", nargs="+",
                       help="Process specific states (default: all)")
    args = parser.parse_args()

    states_to_process = args.state if args.state else US_STATES

    print("=" * 80)
    print(" REGULATORY & INFRASTRUCTURE LAYER CACHE GENERATOR")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    if not args.quick and not args.airfields_only:
        print(f"States to process: {', '.join(states_to_process)}")
    print()

    if args.airfields_only:
        # ── AIRFIELDS ONLY ──
        print("STEP 1: US Airfields (Attempting Overpass API with retries)")
        print("-" * 80)
        download_airfields_us_with_retry()
        print()

    else:
        # ── FAA AIRSPACE ──
        print("STEP 1: FAA Airspace (Mock Grid — Fallback Mode)")
        print("-" * 80)
        print("Note: Real FAA LAANC requires official API (authentication needed).")
        print("Creating mock grid so app can demonstrate features.")
        print()
        for state in tqdm(states_to_process, desc="FAA Airspace"):
            gdf = generate_mock_faa_airspace_for_state(state)
            if gdf is not None and len(gdf) > 0:
                out_path = OUTPUT_DIR / f"faa_airspace_{state}.parquet"
                gdf.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")
        print(f"  [OK] Created FAA airspace mock for {len(states_to_process)} states")
        print()

        # ── FAA OBSTACLES ──
        print("STEP 1b: FAA Digital Obstacle File (Mock)")
        print("-" * 80)
        print("Note: Real DOF requires manual download from FAA.")
        print("Creating mock obstacles for demonstration.")
        gdf_obstacles = generate_mock_faa_obstacles()
        if gdf_obstacles is not None and len(gdf_obstacles) > 0:
            out_path = OUTPUT_DIR / "faa_obstacles.parquet"
            gdf_obstacles.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")
            print(f"  [OK] Created {len(gdf_obstacles)} mock obstacles")
        print()

        # ── CELL TOWERS ──
        print("STEP 2: Cell Towers (Real OpenCellID Data from Local Files)")
        print("-" * 80)
        print("Loading REAL cell tower data from local .gz files...")
        print()
        tower_count = 0
        for state in tqdm(states_to_process, desc="Cell Towers"):
            # Load real cell tower data from local files, fall back to mock if unavailable
            gdf = load_cell_towers_from_local_gz(state)
            if gdf is None or len(gdf) == 0:
                gdf = generate_mock_cell_towers_for_state(state)

            if gdf is not None and len(gdf) > 0:
                out_path = OUTPUT_DIR / f"cell_towers_{state}.parquet"
                gdf.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")
                tower_count += len(gdf)

        print(f"  [OK] Created cell tower data for {len(states_to_process)} states ({tower_count} total)")
        print()

        # ── NO-FLY ZONES ──
        print("STEP 3: No-Fly Zones (Mock)")
        print("-" * 80)
        print("Note: Real data from OpenStreetMap (Overpass API)")
        print("Creating mock no-fly zones for demonstration.")
        gdf_nfz = generate_mock_no_fly_zones()
        if gdf_nfz is not None and len(gdf_nfz) > 0:
            out_path = OUTPUT_DIR / "no_fly_zones.parquet"
            gdf_nfz.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")
            print(f"  [OK] Created {len(gdf_nfz)} mock no-fly zones")
        print()

        # ── AIRFIELDS ──
        print("STEP 4: US Airfields (Attempting Overpass API with retries)")
        print("-" * 80)
        download_airfields_us_with_retry()
        print()

    print("=" * 80)
    print("[OK] CACHE SETUP COMPLETE!")
    print("=" * 80)
    print()
    print("NEXT STEPS:")
    print("  1. All files cached in: regulatory_layers/")
    print("  2. Run: streamlit run app.py")
    print("  3. Toggle new layers in Coverage Map section:")
    print("     [ ] FAA LAANC Airspace (now instant!)")
    print("     [ ] Flight Hazards")
    print("     [ ] Cell Towers")
    print("     [ ] No-Fly Zones")
    print()
    print("NOTES:")
    print("  - Mock data is for demonstration/fallback")
    print("  - Real FAA data requires official LAANC API")
    print("  - Real cell towers: download from opencellid.org")
    print("  - Real obstacles: download DOF_Public.zip from FAA")
    print("  - Airfields use Overpass API (slower but functional)")
    print()
    print("App will work fine with mock data!")
    print()

if __name__ == "__main__":
    main()
