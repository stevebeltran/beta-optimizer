# Copyright (c) Steven Beltran. Created by Steven Beltran in partnership with BRINC Drones.
"""Highway corridor mode for state police / highway patrol agencies.

Each selected highway is simulated as an independent deployment plan.
Call generation uses corridor length + traffic volume rather than Census population.
"""
import datetime
import random

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge, unary_union

# ---------------------------------------------------------------------------
# State Police primary interstates — used when only a state is entered
# ---------------------------------------------------------------------------
STATE_PRIMARY_INTERSTATES: dict = {
    'AL': ['I-20', 'I-65', 'I-85'],
    'AZ': ['I-10', 'I-17', 'I-19', 'I-40'],
    'AR': ['I-30', 'I-40', 'I-55'],
    'CA': ['I-5', 'I-10', 'I-15', 'I-40', 'I-80'],
    'CO': ['I-25', 'I-70', 'I-76'],
    'CT': ['I-84', 'I-91', 'I-95'],
    'DE': ['I-95'],
    'FL': ['I-4', 'I-10', 'I-75', 'I-95'],
    'GA': ['I-16', 'I-20', 'I-75', 'I-85', 'I-95'],
    'ID': ['I-15', 'I-84', 'I-90'],
    'IL': ['I-55', 'I-57', 'I-64', 'I-70', 'I-74', 'I-80', 'I-90', 'I-94'],
    'IN': ['I-65', 'I-69', 'I-70', 'I-74', 'I-80', 'I-90', 'I-94'],
    'IA': ['I-29', 'I-35', 'I-80'],
    'KS': ['I-35', 'I-70', 'I-135'],
    'KY': ['I-24', 'I-64', 'I-65', 'I-75'],
    'LA': ['I-10', 'I-20', 'I-49', 'I-55'],
    'ME': ['I-95'],
    'MD': ['I-68', 'I-70', 'I-81', 'I-83', 'I-95', 'I-97'],
    'MA': ['I-90', 'I-93', 'I-95'],
    'MI': ['I-69', 'I-75', 'I-94', 'I-96'],
    'MN': ['I-35', 'I-90', 'I-94'],
    'MS': ['I-20', 'I-55', 'I-59'],
    'MO': ['I-29', 'I-35', 'I-44', 'I-55', 'I-70'],
    'MT': ['I-15', 'I-90', 'I-94'],
    'NE': ['I-80', 'I-29', 'I-680'],
    'NV': ['I-15', 'I-80'],
    'NH': ['I-89', 'I-93', 'I-95'],
    'NJ': ['I-78', 'I-80', 'I-95', 'I-287'],
    'NM': ['I-10', 'I-25', 'I-40'],
    'NY': ['I-81', 'I-87', 'I-90', 'I-95'],
    'NC': ['I-26', 'I-40', 'I-77', 'I-85', 'I-95'],
    'ND': ['I-29', 'I-94'],
    'OH': ['I-70', 'I-71', 'I-75', 'I-76', 'I-77', 'I-80', 'I-90'],
    'OK': ['I-35', 'I-40', 'I-44'],
    'OR': ['I-5', 'I-84'],
    'PA': ['I-76', 'I-78', 'I-80', 'I-81', 'I-83', 'I-90', 'I-95'],
    'RI': ['I-95'],
    'SC': ['I-20', 'I-26', 'I-77', 'I-85', 'I-95'],
    'SD': ['I-29', 'I-90'],
    'TN': ['I-24', 'I-40', 'I-65', 'I-75', 'I-81'],
    'TX': ['I-10', 'I-20', 'I-27', 'I-30', 'I-35', 'I-37', 'I-40', 'I-45'],
    'UT': ['I-15', 'I-70', 'I-80', 'I-84'],
    'VT': ['I-89', 'I-91'],
    'VA': ['I-64', 'I-66', 'I-77', 'I-81', 'I-85', 'I-95'],
    'WA': ['I-5', 'I-82', 'I-90'],
    'WV': ['I-64', 'I-68', 'I-77', 'I-79'],
    'WI': ['I-39', 'I-43', 'I-90', 'I-94'],
    'WY': ['I-25', 'I-80', 'I-90'],
}

# Approximate bounding boxes per state [south, west, north, east]
_STATE_BBOXES = {
    'AL': (30.1, -88.5, 35.0, -84.9),
    'AK': (54.6, -168.1, 71.5, -129.9),
    'AZ': (31.3, -114.8, 37.0, -109.0),
    'AR': (33.0, -94.6, 36.5, -89.6),
    'CA': (32.5, -124.4, 42.0, -114.1),
    'CO': (36.9, -109.1, 41.0, -102.0),
    'CT': (40.9, -73.7, 42.1, -71.8),
    'DE': (38.4, -75.8, 39.8, -75.0),
    'FL': (24.5, -87.6, 31.0, -80.0),
    'GA': (30.4, -85.6, 35.0, -80.8),
    'HI': (18.9, -160.2, 22.2, -154.8),
    'ID': (41.9, -117.2, 49.0, -111.0),
    'IL': (36.9, -91.5, 42.5, -87.0),
    'IN': (37.8, -88.1, 41.8, -84.8),
    'IA': (40.4, -96.6, 43.5, -90.1),
    'KS': (36.9, -102.1, 40.0, -94.6),
    'KY': (36.5, -89.6, 39.1, -81.9),
    'LA': (28.9, -94.0, 33.0, -88.8),
    'ME': (43.1, -71.1, 47.5, -66.9),
    'MD': (37.9, -79.5, 39.7, -75.0),
    'MA': (41.2, -73.5, 42.9, -69.9),
    'MI': (41.7, -90.4, 48.3, -82.4),
    'MN': (43.5, -97.2, 49.4, -89.5),
    'MS': (30.2, -91.7, 35.0, -88.1),
    'MO': (35.9, -95.8, 40.6, -89.1),
    'MT': (44.4, -116.1, 49.0, -104.0),
    'NE': (40.0, -104.1, 43.0, -95.3),
    'NV': (35.0, -120.0, 42.0, -114.0),
    'NH': (42.7, -72.6, 45.3, -70.6),
    'NJ': (38.9, -75.6, 41.4, -73.9),
    'NM': (31.3, -109.1, 37.0, -103.0),
    'NY': (40.5, -79.8, 45.0, -71.9),
    'NC': (33.8, -84.3, 36.6, -75.5),
    'ND': (45.9, -104.1, 49.0, -96.6),
    'OH': (38.4, -84.8, 41.9, -80.5),
    'OK': (33.6, -103.0, 37.0, -94.4),
    'OR': (41.9, -124.6, 46.3, -116.5),
    'PA': (39.7, -80.5, 42.3, -74.7),
    'RI': (41.1, -71.9, 42.0, -71.1),
    'SC': (32.0, -83.4, 35.2, -78.5),
    'SD': (42.5, -104.1, 45.9, -96.4),
    'TN': (34.9, -90.3, 36.7, -81.6),
    'TX': (25.8, -106.6, 36.5, -93.5),
    'UT': (36.9, -114.1, 42.0, -109.0),
    'VT': (42.7, -73.4, 45.0, -71.5),
    'VA': (36.5, -83.7, 39.5, -75.2),
    'WA': (45.5, -124.7, 49.0, -116.9),
    'WV': (37.2, -82.6, 40.6, -77.7),
    'WI': (42.5, -92.9, 47.1, -86.2),
    'WY': (41.0, -111.1, 45.0, -104.0),
}


def fetch_highway_geometry(highway_ref, state_abbr):
    """Fetch highway geometry from Overpass API, clipped to the given state.

    Args:
        highway_ref: e.g. "I-80", "I-25"
        state_abbr:  two-letter state abbreviation, e.g. "NE"

    Returns:
        GeoDataFrame (EPSG:4326) with a single merged LineString, or None on failure.
    """
    ref_num = highway_ref.strip().upper().replace('I-', '').replace('I ', '').strip()
    south, west, north, east = _STATE_BBOXES.get(state_abbr.upper(), (24.0, -125.0, 49.0, -66.0))

    overpass_query = (
        f'[out:json][timeout:60];\n'
        f'(\n'
        f'  way["highway"~"motorway|trunk"]["ref"~"(^|;\\\\s*)I[ -]{ref_num}(\\\\s*;|$)"]'
        f'({south},{west},{north},{east});\n'
        f');\n'
        f'out geom;\n'
    )

    _headers = {"Accept": "*/*", "User-Agent": "BRINC-Frankenstein/1.0"}
    try:
        resp = requests.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": overpass_query},
            headers=_headers,
            timeout=90,
        )
        resp.raise_for_status()
        elements = resp.json().get("elements", [])
    except Exception:
        return None

    lines = []
    for el in elements:
        if el.get("type") == "way" and "geometry" in el:
            coords = [(node["lon"], node["lat"]) for node in el["geometry"]]
            if len(coords) >= 2:
                lines.append(LineString(coords))

    if not lines:
        return None

    merged = linemerge(unary_union(lines))
    return gpd.GeoDataFrame(geometry=[merged], crs="EPSG:4326")


def build_corridor_polygon(line_gdf, buffer_miles=1.5):
    """Buffer a highway LineString to create a corridor polygon.

    Returns:
        (corridor_poly_4326, corridor_line_4326, length_miles)
    """
    line_proj = line_gdf.to_crs(epsg=3857)
    buffer_m = buffer_miles * 1609.344
    corridor_proj = line_proj.geometry.buffer(buffer_m).union_all()

    corridor_gdf = gpd.GeoDataFrame(
        geometry=[corridor_proj], crs="EPSG:3857"
    ).to_crs("EPSG:4326")
    corridor_poly = corridor_gdf.geometry.iloc[0]

    corridor_line = line_gdf.geometry.iloc[0]
    length_miles = float(line_proj.geometry.length.sum()) / 1609.344

    return corridor_poly, corridor_line, length_miles


def estimate_corridor_calls(corridor_length_miles):
    """Estimate annual calls for service along a highway corridor.

    Formula: corridor_miles x 0.33 calls/mile/day x 365 days
    0.33 is calibrated to match observed state highway patrol activity rates.
    """
    return max(365, int(corridor_length_miles * 0.33 * 365))


def _generate_corridor_calls(corridor_line, corridor_poly, num_calls, generate_random_points_in_polygon):
    """Place call points along the highway corridor.

    75% distributed near the route centerline with small scatter.
    25% random within the corridor buffer (off-ramps, rest areas, etc.).
    """
    lines = (
        list(corridor_line.geoms)
        if corridor_line.geom_type == "MultiLineString"
        else [corridor_line]
    )
    total_length = sum(line.length for line in lines)

    if total_length == 0:
        return generate_random_points_in_polygon(corridor_poly, num_calls)

    sigma = 0.005  # ~500m standard deviation perpendicular to route

    points = []
    n_along = int(num_calls * 0.75)

    for _ in range(n_along * 6):
        if len(points) >= n_along:
            break
        r = random.uniform(0, total_length)
        cumlen = 0.0
        for line in lines:
            seg_len = line.length
            if cumlen + seg_len >= r:
                t = (r - cumlen) / seg_len
                pt = line.interpolate(t, normalized=True)
                px = pt.x + np.random.normal(0, sigma)
                py = pt.y + np.random.normal(0, sigma * 0.5)
                candidate = Point(px, py)
                points.append((py, px) if corridor_poly.contains(candidate) else (pt.y, pt.x))
                break
            cumlen += seg_len

    n_random = num_calls - len(points)
    if n_random > 0:
        points.extend(generate_random_points_in_polygon(corridor_poly, n_random))

    np.random.shuffle(points)
    return points[:num_calls]


def build_corridor_demo(corridor_line, corridor_poly, annual_cfs, generate_random_points_in_polygon):
    """Build a simulated calls DataFrame for a highway corridor.

    Returns:
        (df_demo, annual_cfs, simulated_points_count)
        Mirrors the return signature of onboarding.build_demo_calls.
    """
    simulated_points_count = min(max(annual_cfs, 365), 36500)
    np.random.seed(42)
    random.seed(42)

    call_points = _generate_corridor_calls(
        corridor_line,
        corridor_poly,
        simulated_points_count,
        generate_random_points_in_polygon,
    )

    base_date = datetime.datetime.now() - datetime.timedelta(days=364)
    fake_datetimes = [
        base_date + datetime.timedelta(
            days=random.randint(0, 364),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
        )
        for _ in range(simulated_points_count)
    ]

    demo_df = pd.DataFrame({
        "lat": [p[0] for p in call_points],
        "lon": [p[1] for p in call_points],
        "priority": np.random.choice(
            [1, 2, 3], simulated_points_count, p=[0.15, 0.35, 0.50]
        ),
        "date": [dt.strftime("%Y-%m-%d") for dt in fake_datetimes],
        "time": [dt.strftime("%H:%M:%S") for dt in fake_datetimes],
    })
    return demo_df, annual_cfs, simulated_points_count
