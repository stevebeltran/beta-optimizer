import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import plotly.graph_objects as go
from shapely.geometry import Point, Polygon, MultiPolygon, box
import shapely.wkt
from shapely.ops import unary_union
import os
import itertools
import glob
import math
import simplekml
from concurrent.futures import ThreadPoolExecutor
import pulp
import re
import random
import json
import urllib.request
import zipfile
import io
import streamlit.components.v1 as components

# --- GLOBAL CONFIGURATION ---
CONFIG = {
    "RESPONDER_COST": 80000,
    "GUARDIAN_COST": 160000,
    "RESPONDER_RANGE_MI": 2.0,
    "OFFICER_COST_PER_CALL": 82,
    "DRONE_COST_PER_CALL": 6,
    "DEFAULT_TRAFFIC_SPEED": 35.0, 
    "RESPONDER_SPEED": 42.0,       
    "GUARDIAN_SPEED": 60.0         
}

STATE_FIPS = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08", "CT": "09",
    "DE": "10", "FL": "12", "GA": "13", "HI": "15", "ID": "16", "IL": "17", "IN": "18",
    "IA": "19", "KS": "20", "KY": "21", "LA": "22", "ME": "23", "MD": "24", "MA": "25",
    "MI": "26", "MN": "27", "MS": "28", "MO": "29", "MT": "30", "NE": "31", "NV": "32",
    "NH": "33", "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39",
    "OK": "40", "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46", "TN": "47",
    "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53", "WV": "54", "WI": "55",
    "WY": "56"
}

KNOWN_POPULATIONS = {
    "New York": 8336817, "Los Angeles": 3822238, "Chicago": 2665039, "Houston": 1304379, 
    "Phoenix": 1644409, "Philadelphia": 1567258, "San Antonio": 2302878, "San Diego": 1472530, 
    "Dallas": 1299544, "San Jose": 1381162, "Austin": 974447, "Jacksonville": 971319, 
    "Fort Worth": 956709, "Columbus": 907971, "Indianapolis": 880621, "Charlotte": 897720, 
    "San Francisco": 971233, "Seattle": 749256, "Denver": 713252, "Washington": 678972, 
    "Nashville": 683622, "Oklahoma City": 694800, "El Paso": 694553, "Boston": 650706, 
    "Portland": 635067, "Las Vegas": 656274, "Detroit": 620376, "Memphis": 633104, 
    "Louisville": 628594, "Baltimore": 620961, "Milwaukee": 620251, "Albuquerque": 677122, 
    "Tucson": 564559, "Fresno": 677102, "Sacramento": 808418, "Kansas City": 697738, 
    "Mesa": 504258, "Atlanta": 499127, "Omaha": 508901, "Colorado Springs": 483956, 
    "Raleigh": 476587, "Miami": 449514, "Virginia Beach": 455369, "Oakland": 530763, 
    "Minneapolis": 563332, "Tulsa": 547239, "Arlington": 398654, "New Orleans": 562503, 
    "Wichita": 402263, "Cleveland": 900000, "Tampa": 449514, "Orlando": 316081
}

# --- PAGE CONFIG ---
st.set_page_config(page_title="BRINC COS Drone Optimizer", layout="wide")

# --- THEME TOGGLE ---
st.sidebar.markdown("<h3 style='margin-bottom:0px;'>🎨 Appearance</h3>", unsafe_allow_html=True)
theme_choice = st.sidebar.radio("Theme", ["Dark Mode", "Light Mode"], horizontal=True, label_visibility="collapsed")
is_dark = theme_choice == "Dark Mode"

# --- DYNAMIC THEME VARIABLES ---
if is_dark:
    bg_main = "#000000"
    bg_sidebar = "#111111"
    text_main = "#ffffff"
    text_muted = "#aaaaaa"
    accent_color = "#00D2FF" 
    card_bg = "#111111"
    card_border = "#333333"
    card_text = "#eeeeee"
    card_title = "#ffffff"
    budget_box_bg = "#0a0a0a"
    budget_box_border = "#00D2FF"
    budget_box_shadow = "rgba(0, 210, 255, 0.15)"
    map_style = "carto-darkmatter"
    map_boundary_color = "#ffffff"
    map_incident_color = "#00D2FF"
    legend_bg = "rgba(0, 0, 0, 0.7)"
    legend_text = "#ffffff"
    
    STATION_COLORS = [
        "#00D2FF", "#39FF14", "#FFD700", "#FF007F", "#FF4500", 
        "#00FFCC", "#FF3333", "#7FFF00", "#00FFFF", "#FF9900"
    ]
    
    theme_css = f"""
    .stApp, .main {{ background-color: {bg_main} !important; }}
    html, body, [class*="css"], p, label, li, h1, h2, h3, h4, h5, h6 {{ font-family: 'Manrope', sans-serif !important; color: {text_main} !important; }}
    [data-testid="stSidebar"] {{ background-color: {bg_sidebar} !important; border-right: 1px solid {card_border}; }}
    [data-testid="stFileUploader"] p, [data-testid="stFileUploader"] small {{ color: {text_muted} !important; }}
    div[data-testid="stMetricValue"] {{ font-family: 'IBM Plex Mono', monospace !important; color: {accent_color} !important; }}
    div[data-testid="stMetricLabel"] * {{ color: {text_muted} !important; }}
    div[data-baseweb="select"] > div {{ background-color: #222222 !important; border-color: #444444 !important; color: #ffffff !important; }}
    div[data-baseweb="select"] > div * {{ color: #ffffff !important; }}
    div[data-baseweb="select"] span[data-baseweb="tag"] {{ background-color: #333333 !important; color: #ffffff !important; font-weight: normal; border: 1px solid #555555 !important; }}
    div[data-baseweb="select"] span[data-baseweb="tag"] * {{ color: #ffffff !important; }}
    div[data-baseweb="popover"] ul {{ background-color: #222222 !important; color: #ffffff !important; }}
    div[data-baseweb="popover"] li:hover {{ background-color: #444444 !important; }}
    """
else:
    bg_main = "#ffffff"
    bg_sidebar = "#f8f9fa"
    text_main = "#222222"
    text_muted = "#666666"
    accent_color = "#ff4b4b" 
    card_bg = "#ffffff"
    card_border = "#e0e0e0"
    card_text = "#222222"
    card_title = "#333333"
    budget_box_bg = "#ffffff"
    budget_box_border = "#ff4b4b" 
    budget_box_shadow = "rgba(0, 0, 0, 0.05)"
    map_style = "open-street-map"
    map_boundary_color = "#222222"
    map_incident_color = "#000080"
    legend_bg = "rgba(255, 255, 255, 0.9)"
    legend_text = "#333333"
    
    STATION_COLORS = [
        "#E6194B", "#3CB44B", "#4363D8", "#F58231", "#911EB4", 
        "#800000", "#333333", "#000075", "#808000", "#9A6324"
    ]
    
    theme_css = f"""
    .stApp, .main {{ background-color: {bg_main} !important; }}
    html, body, [class*="css"], p, label, li, h1, h2, h3, h4, h5, h6 {{ font-family: 'Manrope', sans-serif !important; color: {text_main} !important; }}
    [data-testid="stSidebar"] {{ background-color: {bg_sidebar} !important; border-right: 1px solid {card_border}; }}
    [data-testid="stFileUploader"] p, [data-testid="stFileUploader"] small {{ color: {text_muted} !important; }}
    div[data-testid="stMetricValue"] {{ font-family: 'IBM Plex Mono', monospace !important; color: {accent_color} !important; }}
    div[data-testid="stMetricLabel"] * {{ color: {text_muted} !important; }}
    div[data-baseweb="select"] > div {{ background-color: #ffffff !important; border-color: #cccccc !important; color: #333333 !important; }}
    div[data-baseweb="select"] > div * {{ color: #333333 !important; }}
    div[data-baseweb="select"] span[data-baseweb="tag"] {{ background-color: #eeeeee !important; color: #000000 !important; font-weight: normal; border: 1px solid #cccccc !important; }}
    div[data-baseweb="select"] span[data-baseweb="tag"] * {{ color: #000000 !important; }}
    div[data-baseweb="popover"] ul {{ background-color: #ffffff !important; color: #333333 !important; }}
    div[data-baseweb="popover"] li:hover {{ background-color: #f0f0f0 !important; }}
    """

# --- INJECT CSS ---
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;700&family=Manrope:wght@400;600;700&display=swap');
    {theme_css}
    .stRadio label p, .stMultiSelect label p, .stSlider label p, .stToggle label p, .stCheckbox label p {{
        font-weight: 600 !important;
        font-size: 0.85rem !important;
    }}
    div[role="radiogroup"] {{ gap: 0.5rem !important; }}
    @media print {{
        section[data-testid="stSidebar"], header[data-testid="stHeader"], .stSlider, button, div[data-testid="stToolbar"] {{ display: none !important; }}
        * {{ -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }}
        .block-container, .stApp, .main, div {{ max-width: 100% !important; width: 100% !important; padding: 0 !important; margin: 0 !important; overflow: visible !important; height: auto !important; }}
        div[data-testid="stHorizontalBlock"] {{ display: block !important; width: 100% !important; }}
        div[data-testid="stColumn"] {{ width: 100% !important; max-width: 100% !important; flex: 0 0 100% !important; display: block !important; margin-bottom: 20px !important; }}
        .js-plotly-plot, .plot-container {{ width: 100% !important; page-break-inside: avoid !important; margin-bottom: 30px !important; }}
        div[style*="border-top: 4px solid"] {{ page-break-inside: avoid !important; margin-bottom: 15px !important; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --- LOGO ---
try:
    st.sidebar.image("logo.png", width='stretch')
except FileNotFoundError:
    pass

SHAPEFILE_DIR = "jurisdiction_data" 
if not os.path.exists(SHAPEFILE_DIR):
    os.makedirs(SHAPEFILE_DIR)

if 'csvs_ready' not in st.session_state:
    st.session_state['csvs_ready'] = False
    st.session_state['df_calls'] = None
    st.session_state['df_stations'] = None

if not st.session_state['csvs_ready']:
    st.title("🛰️ BRINC COS Drone Optimizer")

# --- CENSUS TIGER SHAPEFILE & API FETCHER ---
@st.cache_data
def fetch_census_population(state_fips, place_name):
    url = f"https://api.census.gov/data/2020/dec/pl?get=P1_001N,NAME&for=place:*&in=state:{state_fips}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            search_name = place_name.lower().strip()
            
            for row in data[1:]:
                place_full = row[1].lower().split(',')[0].strip()
                if place_full == search_name:
                    return int(row[0])
                elif place_full.startswith(search_name + " "):
                    return int(row[0])
    except Exception:
        pass
    return None

@st.cache_data
def fetch_tiger_city_shapefile(state_fips, city_name, output_dir):
    url = f"https://www2.census.gov/geo/tiger/TIGER2023/PLACE/tl_2023_{state_fips}_place.zip"
    try:
        req = urllib.request.urlopen(url)
        zip_file = zipfile.ZipFile(io.BytesIO(req.read()))
        temp_dir = os.path.join(output_dir, f"temp_tiger_{state_fips}")
        os.makedirs(temp_dir, exist_ok=True)
        zip_file.extractall(temp_dir)
        
        shp_path = glob.glob(os.path.join(temp_dir, "*.shp"))[0]
        gdf = gpd.read_file(shp_path)
        
        search_name = city_name.lower().strip()
        exact_mask = gdf['NAME'].str.lower().str.strip() == search_name
        if exact_mask.any():
            city_gdf = gdf[exact_mask]
        else:
            city_gdf = gdf[gdf['NAME'].str.lower().str.contains(search_name, case=False, na=False)]
        
        if not city_gdf.empty:
            city_gdf = city_gdf.dissolve(by='NAME').reset_index()
            save_path = os.path.join(output_dir, f"{city_name.replace(' ', '_')}_{state_fips}.shp")
            city_gdf.to_file(save_path)
            return True, city_gdf
    except Exception as e:
        print(f"Failed to fetch TIGER shapefile: {e}")
        return False, None
    return False, None

def generate_random_points_in_polygon(polygon, num_points):
    points = []
    minx, miny, maxx, maxy = polygon.bounds
    while len(points) < num_points:
        x_coords = np.random.uniform(minx, maxx, 1000)
        y_coords = np.random.uniform(miny, maxy, 1000)
        for x, y in zip(x_coords, y_coords):
            if len(points) >= num_points:
                break
            if polygon.contains(Point(x, y)):
                points.append((y, x))
    return points

def generate_clustered_calls(polygon, num_points):
    points = []
    minx, miny, maxx, maxy = polygon.bounds
    
    num_hotspots = random.randint(5, 15)
    hotspots = []
    while len(hotspots) < num_hotspots:
        hx, hy = random.uniform(minx, maxx), random.uniform(miny, maxy)
        if polygon.contains(Point(hx, hy)):
            hotspots.append((hx, hy))
            
    target_clustered = int(num_points * 0.75)
    while len(points) < target_clustered:
        hx, hy = random.choice(hotspots)
        px, py = np.random.normal(hx, 0.02), np.random.normal(hy, 0.02)
        if polygon.contains(Point(px, py)):
            points.append((py, px)) 
            
    while len(points) < num_points:
        px, py = random.uniform(minx, maxx), random.uniform(miny, maxy)
        if polygon.contains(Point(px, py)):
            points.append((py, px))
            
    np.random.shuffle(points)
    return points

# --- MAIN UPLOAD & VALIDATION SECTION ---
if not st.session_state['csvs_ready']:
    
    st.markdown("### 📁 Upload Your Mission Data")
    st.info("Upload 'calls.csv' and 'stations.csv' to begin. The map will auto-detect matching jurisdictions.")
    
    uploaded_files = st.file_uploader("Upload Mission Data (CSV)", accept_multiple_files=True, label_visibility="collapsed")
    call_file, station_file = None, None
    
    if uploaded_files:
        for f in uploaded_files:
            fname = f.name.lower()
            if fname == "calls.csv": call_file = f
            elif fname == "stations.csv": station_file = f
            
        if call_file and station_file:
            df_c = pd.read_csv(call_file)
            df_c.columns = [str(c).lower().strip() for c in df_c.columns]
            df_c = df_c.rename(columns={'latitude': 'lat', 'longitude': 'lon'}) 
            
            if 'lat' not in df_c.columns or 'lon' not in df_c.columns:
                st.error(f"❌ **Validation Error:** Your calls.csv must contain 'lat' and 'lon' columns. Found: {', '.join(df_c.columns)}")
                st.stop()
                
            orig_len = len(df_c)
            df_c = df_c.dropna(subset=['lat', 'lon']).reset_index(drop=True)
            if len(df_c) < orig_len:
                st.warning(f"⚠️ Dropped {orig_len - len(df_c)} rows from calls data due to missing or invalid GPS coordinates.")
                
            st.session_state['df_calls'] = df_c
            
            df_s = pd.read_csv(station_file)
            df_s.columns = [str(c).lower().strip() for c in df_s.columns]
            df_s = df_s.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
            
            if 'lat' not in df_s.columns or 'lon' not in df_s.columns:
                st.error(f"❌ **Validation Error:** Your stations.csv must contain 'lat' and 'lon' columns. Found: {', '.join(df_s.columns)}")
                st.stop()
                
            st.session_state['df_stations'] = df_s.dropna(subset=['lat', 'lon']).reset_index(drop=True)
            st.session_state['csvs_ready'] = True
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    
    with st.expander("🚀 Don't have data? Run a Synthetic City Simulation", expanded=False):
        st.write("Generate a highly realistic simulated 911 call history and infrastructure map for any US City.")
        
        with st.form("demo_city_form"):
            col1, col2 = st.columns([3, 1])
            input_city = col1.text_input("Enter City Name", value="Orlando")
            input_state = col2.selectbox("State", list(STATE_FIPS.keys()), index=8) 
            submit_demo = st.form_submit_button("🚀 Simulate City")
            
        if submit_demo:
            with st.spinner(f"Fetching exact TIGER boundary data for {input_city}, {input_state} from US Census Bureau..."):
                success, active_city_gdf = fetch_tiger_city_shapefile(STATE_FIPS[input_state], input_city, SHAPEFILE_DIR)
                
            if not success:
                st.error(f"Could not find a Census boundary for '{input_city}' in {input_state}. Try checking spelling or using a major city.")
            else:
                with st.spinner(f"Querying US Census API for {input_city}'s true population..."):
                    city_poly = active_city_gdf.geometry.union_all()
                    
                    pop = fetch_census_population(STATE_FIPS[input_state], input_city)
                    if pop:
                        estimated_pop = pop
                        st.toast(f"✅ Verified US Census Population: {estimated_pop:,}")
                    else:
                        gdf_proj = active_city_gdf.to_crs(epsg=3857)
                        area_sq_mi = gdf_proj.geometry.area.sum() / 2589988.11
                        estimated_pop = KNOWN_POPULATIONS.get(input_city, int(area_sq_mi * 3500))
                        st.toast(f"⚠️ Census API unavailable. Estimated Population: {estimated_pop:,}")
                        
                    annual_cfs = int(estimated_pop * 0.6) 
                    simulated_points_count = min(int(annual_cfs / 12), 25000)
                    
                with st.spinner("Procedurally mapping true-to-life 911 call clusters..."):
                    np.random.seed(42)
                    call_points = generate_clustered_calls(city_poly, simulated_points_count)
                    st.session_state['df_calls'] = pd.DataFrame({
                        'lat': [p[0] for p in call_points], 
                        'lon': [p[1] for p in call_points], 
                        'priority': np.random.choice(['High', 'Medium', 'Low'], simulated_points_count)
                    })
                    
                with st.spinner("Distributing municipal infrastructure grid..."):
                    station_points = generate_random_points_in_polygon(city_poly, 100)
                    types = ['Police', 'Fire', 'EMS'] * 34
                    st.session_state['df_stations'] = pd.DataFrame({
                        'name': [f'Station {i+1}' for i in range(len(station_points))],
                        'lat': [p[0] for p in station_points], 
                        'lon': [p[1] for p in station_points],
                        'type': types[:len(station_points)]
                    })
                    
                    st.session_state['inferred_daily_calls_override'] = int(annual_cfs / 365)
                    st.session_state['csvs_ready'] = True
                    st.rerun()

def get_circle_coords(lat, lon, r_mi=2.0):
    angles = np.linspace(0, 2*np.pi, 100)
    c_lats = lat + (r_mi/69.172) * np.sin(angles)
    c_lons = lon + (r_mi/(69.172 * np.cos(np.radians(lat)))) * np.cos(angles)
    return c_lats, c_lons

def format_3_lines(name_str):
    match = re.search(r'\s(\d{1,5}\s+[A-Za-z])', name_str)
    if match:
        idx = match.start()
        line1 = name_str[:idx].strip()
        rest = name_str[idx:].strip()
        if ',' in rest:
            parts = rest.split(',', 1)
            return f"{line1}<br>{parts[0].strip()},<br>{parts[1].strip()}"
        return f"{line1}<br>{rest}<br> "
    if ',' in name_str:
        parts = name_str.split(',')
        if len(parts) >= 3:
            return f"{parts[0].strip()},<br>{parts[1].strip()},<br>{','.join(parts[2:]).strip()}"
    return f"{name_str}<br> <br> "

def to_kml_color(hex_str):
    h = hex_str.lstrip('#')
    return f"ff{h[4:6]}{h[2:4]}{h[0:2]}" if len(h) == 6 else "ff0000ff"

def generate_kml(active_gdf, active_drones, calls_gdf):
    kml = simplekml.Kml()
    fol_bounds = kml.newfolder(name="Jurisdictions")
    for _, row in active_gdf.iterrows():
        geoms = [row.geometry] if isinstance(row.geometry, Polygon) else row.geometry.geoms
        for geom in geoms:
            pol = fol_bounds.newpolygon(name=row.get('DISPLAY_NAME', 'Boundary'))
            pol.outerboundaryis = list(geom.exterior.coords)
            pol.style.linestyle.color = simplekml.Color.red
            pol.style.linestyle.width = 3
            pol.style.polystyle.color = simplekml.Color.changealphaint(30, simplekml.Color.red)

    fol_stations = kml.newfolder(name="Stations Points")
    fol_rings = kml.newfolder(name="Coverage Rings")

    def add_kml_station(d_obj, radius, color_hex, name_prefix):
        kml_c = to_kml_color(color_hex)
        pnt = fol_stations.newpoint(name=f"{name_prefix} {d_obj['name']}")
        pnt.coords = [(d_obj['lon'], d_obj['lat'])]
        pnt.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/paddle/blu-blank.png'
        lats, lons = get_circle_coords(d_obj['lat'], d_obj['lon'], r_mi=radius)
        ring_coords = list(zip(lons, lats))
        ring_coords.append(ring_coords[0])
        pol = fol_rings.newpolygon(name=f"Range: {d_obj['name']}")
        pol.outerboundaryis = ring_coords
        pol.style.linestyle.color = kml_c
        pol.style.linestyle.width = 2
        pol.style.polystyle.color = simplekml.Color.changealphaint(60, kml_c)

    for d in active_drones:
        add_kml_station(d, d['radius_m']/1609.34, d['color'], f"[{d['type'][:3]}]")

    fol_calls = kml.newfolder(name="Incident Data (Sample)")
    calls_export = calls_gdf.to_crs(epsg=4326)
    if len(calls_export) > 2000:
        calls_export = calls_export.sample(2000, random_state=42)
        
    for _, row in calls_export.iterrows():
        pnt = fol_calls.newpoint()
        pnt.coords = [(row.geometry.x, row.geometry.y)]
        pnt.style.iconstyle.scale = 0.5
        pnt.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png'

    return kml.kml()

def calculate_zoom(min_lon, max_lon, min_lat, max_lat):
    lon_diff = max_lon - min_lon
    lat_diff = max_lat - min_lat
    if lon_diff <= 0 or lat_diff <= 0: return 12
    zoom_lon = np.log2(360 / lon_diff)
    zoom_lat = np.log2(180 / lat_diff)
    return min(max(min(zoom_lon, zoom_lat) + 1.6, 5), 18)

@st.cache_data
def find_relevant_jurisdictions(calls_df, stations_df, shapefile_dir):
    points_list = []
    if calls_df is not None: points_list.append(calls_df[['lat', 'lon']])
    if stations_df is not None: points_list.append(stations_df[['lat', 'lon']])
    
    if not points_list: return None
    full_points = pd.concat(points_list)
    full_points = full_points[(full_points.lat.abs() > 1) & (full_points.lon.abs() > 1)]
    
    scan_points = full_points.sample(50000, random_state=42) if len(full_points) > 50000 else full_points
    points_gdf = gpd.GeoDataFrame(scan_points, geometry=gpd.points_from_xy(scan_points.lon, scan_points.lat), crs="EPSG:4326")
    
    total_bounds = points_gdf.total_bounds
    shp_files = glob.glob(os.path.join(shapefile_dir, "*.shp"))
    relevant_polys = []
    
    for shp_path in shp_files:
        try:
            gdf_chunk = gpd.read_file(shp_path, bbox=tuple(total_bounds))
            if not gdf_chunk.empty:
                if gdf_chunk.crs is None: gdf_chunk.set_crs(epsg=4269, inplace=True)
                gdf_chunk = gdf_chunk.to_crs(epsg=4326)
                hits = gpd.sjoin(gdf_chunk, points_gdf, how="inner", predicate="intersects")
                if not hits.empty:
                    subset = gdf_chunk.loc[hits.index.unique()].copy()
                    subset['data_count'] = hits.index.value_counts()
                    name_col = next((c for c in ['NAME', 'DISTRICT', 'NAMELSAD'] if c in subset.columns), subset.columns[0])
                    subset['DISPLAY_NAME'] = subset[name_col].astype(str)
                    relevant_polys.append(subset)
        except Exception:
            continue
            
    if not relevant_polys: return None
    master_gdf = pd.concat(relevant_polys, ignore_index=True).sort_values(by='data_count', ascending=False)
    
    master_gdf = master_gdf.dissolve(by='DISPLAY_NAME', aggfunc={'data_count': 'sum'}).reset_index()
    master_gdf = master_gdf.sort_values(by='data_count', ascending=False)
    
    if master_gdf['data_count'].sum() > 0:
        master_gdf['pct_share'] = master_gdf['data_count'] / master_gdf['data_count'].sum()
        master_gdf['cum_share'] = master_gdf['pct_share'].cumsum()
        mask = (master_gdf['cum_share'] <= 0.98) | (master_gdf['pct_share'] > 0.01)
        mask.iloc[0] = True
        return master_gdf[mask]
    return master_gdf

@st.cache_resource
def precompute_spatial_data(df_calls, df_stations_all, _city_m, epsg_code, resp_radius_mi, guard_radius_mi, bounds_hash):
    gdf_calls = gpd.GeoDataFrame(df_calls, geometry=gpd.points_from_xy(df_calls.lon, df_calls.lat), crs="EPSG:4326")
    gdf_calls_utm = gdf_calls.to_crs(epsg=epsg_code)
    
    try: calls_in_city = gdf_calls_utm[gdf_calls_utm.within(_city_m)]
    except: calls_in_city = gdf_calls_utm
        
    radius_resp_m = resp_radius_mi * 1609.34
    radius_guard_m = guard_radius_mi * 1609.34 
    
    station_metadata = []
    total_calls = len(calls_in_city)
    n = len(df_stations_all)
    resp_matrix = np.zeros((n, total_calls), dtype=bool)
    guard_matrix = np.zeros((n, total_calls), dtype=bool)
    
    dist_matrix_r = np.zeros((n, total_calls))
    dist_matrix_g = np.zeros((n, total_calls))
    
    display_calls = calls_in_city.sample(min(5000, total_calls), random_state=42).to_crs(epsg=4326) if not calls_in_city.empty else gpd.GeoDataFrame()
    
    if not calls_in_city.empty:
        calls_array = np.array(list(zip(calls_in_city.geometry.x, calls_in_city.geometry.y)))
        for idx_pos, (i, row) in enumerate(df_stations_all.iterrows()):
            s_pt_m = gpd.GeoSeries([Point(row['lon'], row['lat'])], crs="EPSG:4326").to_crs(epsg=epsg_code).iloc[0]
            dists = np.sqrt((calls_array[:,0] - s_pt_m.x)**2 + (calls_array[:,1] - s_pt_m.y)**2)
            dists_mi = dists / 1609.34
            
            mask_r = dists <= radius_resp_m
            mask_g = dists <= radius_guard_m
            resp_matrix[idx_pos, :] = mask_r
            guard_matrix[idx_pos, :] = mask_g
            
            dist_matrix_r[idx_pos, :] = dists_mi
            dist_matrix_g[idx_pos, :] = dists_mi

            full_buf_2m = s_pt_m.buffer(radius_resp_m)
            try: clipped_2m = full_buf_2m.intersection(_city_m)
            except: clipped_2m = full_buf_2m

            full_buf_guard = s_pt_m.buffer(radius_guard_m)
            try: clipped_guard = full_buf_guard.intersection(_city_m)
            except: clipped_guard = full_buf_guard
            
            avg_dist_r = dists_mi[mask_r].mean() if mask_r.any() else (resp_radius_mi * (2/3))
            avg_dist_g = dists_mi[mask_g].mean() if mask_g.any() else (guard_radius_mi * (2/3))
                
            station_metadata.append({
                'name': row['name'], 'lat': row['lat'], 'lon': row['lon'],
                'clipped_2m': clipped_2m, 'clipped_guard': clipped_guard,
                'avg_dist_r': avg_dist_r, 'avg_dist_g': avg_dist_g
            })
            
    return calls_in_city, display_calls, resp_matrix, guard_matrix, dist_matrix_r, dist_matrix_g, station_metadata, total_calls

def solve_mclp(resp_matrix, guard_matrix, dist_r, dist_g, num_resp, num_guard, allow_redundancy, incremental=True):
    n_stations, n_calls = resp_matrix.shape
    if n_calls == 0 or (num_resp == 0 and num_guard == 0):
        return [], [], [], []

    df_profiles = pd.DataFrame(resp_matrix.T).astype(int).astype(str)
    df_profiles['g'] = pd.DataFrame(guard_matrix.T).astype(int).astype(str).agg(''.join, axis=1)
    df_profiles['r'] = df_profiles.drop(columns='g').agg(''.join, axis=1)
    
    grouped = df_profiles.groupby(['r', 'g'], sort=False)
    weights = grouped.size().values
    unique_idx = grouped.head(1).index
    
    u_resp = resp_matrix[:, unique_idx]
    u_guard = guard_matrix[:, unique_idx]
    
    u_dist_r = dist_r[:, unique_idx]
    u_dist_g = dist_g[:, unique_idx]
    
    n_u = len(weights)

    def run_lp(target_r, target_g, locked_r, locked_g):
        model = pulp.LpProblem("DroneCoverage", pulp.LpMaximize)
        x_r = pulp.LpVariable.dicts("r_st", range(n_stations), 0, 1, pulp.LpBinary)
        x_g = pulp.LpVariable.dicts("g_st", range(n_stations), 0, 1, pulp.LpBinary)

        model += pulp.lpSum(x_r[i] for i in range(n_stations)) == target_r
        model += pulp.lpSum(x_g[i] for i in range(n_stations)) == target_g

        for r in locked_r: model += x_r[r] == 1
        for g in locked_g: model += x_g[g] == 1

        if not allow_redundancy:
            for s in range(n_stations): model += x_r[s] + x_g[s] <= 1

        y = pulp.LpVariable.dicts("cl", range(n_u), 0, 1, pulp.LpBinary)
        
        # Primary Objective: Maximize covered calls
        primary_obj = pulp.lpSum(y[i] * weights[i] for i in range(n_u))

        # Secondary Objective: Penalize long flight distances. 
        # For every station chosen, subtract a tiny fraction of its total required flight distance.
        penalty_factor = 0.00001
        distance_penalty = pulp.lpSum(
            x_r[s] * np.sum(u_dist_r[s, :]) * penalty_factor +
            x_g[s] * np.sum(u_dist_g[s, :]) * penalty_factor
            for s in range(n_stations)
        )

        model += primary_obj - distance_penalty

        for i in range(n_u):
            cover = []
            for s in range(n_stations):
                if u_resp[s, i]: cover.append(x_r[s])
                if u_guard[s, i]: cover.append(x_g[s])
            
            if cover: model += y[i] <= pulp.lpSum(cover)
            else: model += y[i] == 0

        model.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=10, gapRel=0.0))

        res_r = [i for i in range(n_stations) if pulp.value(x_r[i]) == 1]
        res_g = [i for i in range(n_stations) if pulp.value(x_g[i]) == 1]
        return res_r, res_g

    if not incremental:
        res_r, res_g = run_lp(num_resp, num_guard, [], [])
        return res_r, res_g, res_r, res_g
    else:
        curr_r, curr_g = [], []
        chrono_r, chrono_g = [], []
        for tg in range(1, num_guard + 1):
            next_r, next_g = run_lp(0, tg, curr_r, curr_g)
            new_g = [x for x in next_g if x not in curr_g]
            chrono_g.extend(new_g)
            curr_r, curr_g = next_r, next_g
        for tr in range(1, num_resp + 1):
            next_r, next_g = run_lp(tr, num_guard, curr_r, curr_g)
            new_r = [x for x in next_r if x not in curr_r]
            chrono_r.extend(new_r)
            curr_r, curr_g = next_r, next_g
        return curr_r, curr_g, chrono_r, chrono_g

@st.cache_resource
def compute_all_elbow_curves(n_calls, _resp_matrix, _guard_matrix, _geos_r, _geos_g, total_area, _bounds_hash):
    n_st = _resp_matrix.shape[0]
    
    def greedy_calls(matrix):
        uncovered = np.ones(n_calls, dtype=bool)
        curve = [0.0]
        cov_count = 0
        min_gain = max(1, n_calls * 0.001) 
        for _ in range(n_st):
            best_s, best_cov = -1, -1
            for s in range(n_st):
                cov = (matrix[s] & uncovered).sum()
                if cov > best_cov: best_cov, best_s = cov, s
            if best_s != -1 and best_cov >= min_gain:
                uncovered = uncovered & ~matrix[best_s]
                cov_count += best_cov
                curve.append((cov_count / max(1, n_calls)) * 100)
                if cov_count == n_calls: break
            else:
                break
        return curve
        
    def greedy_area(geos):
        if total_area <= 0: return [0.0]
        current_union = Polygon()
        curve = [0.0]
        available = list(range(n_st))
        min_gain = total_area * 0.001 
        for _ in range(n_st):
            best_s, best_poly, best_area = -1, None, -1
            for s in available:
                cand = current_union.union(geos[s])
                if cand.area > best_area:
                    best_area, best_poly, best_s = cand.area, cand, s
            if best_s != -1 and (best_area - current_union.area) >= min_gain:
                current_union = best_poly
                available.remove(best_s)
                curve.append((current_union.area / total_area) * 100)
            else:
                break
        return curve

    c_r = greedy_calls(_resp_matrix) if n_calls > 0 else [0.0]
    c_g = greedy_calls(_guard_matrix) if n_calls > 0 else [0.0]
    a_r = greedy_area(_geos_r)
    a_g = greedy_area(_geos_g)
    
    max_len = max(len(c_r), len(c_g), len(a_r), len(a_g))
    
    def pad(c):
        res = list(c)
        while len(res) < max_len:
            res.append(np.nan)
        return res
    
    return pd.DataFrame({
        'Drones': range(max_len),
        'Responder (Calls)': pad(c_r),
        'Responder (Area)': pad(a_r),
        'Guardian (Calls)': pad(c_g),
        'Guardian (Area)': pad(a_g)
    })

# --- MAIN LOGIC ---
if st.session_state['csvs_ready']:
    df_calls = st.session_state['df_calls'].copy()
    df_stations_all = st.session_state['df_stations'].copy()

    with st.spinner("🌍 Identifying dominant jurisdictions..."):
        master_gdf = find_relevant_jurisdictions(df_calls, df_stations_all, SHAPEFILE_DIR)

    if master_gdf is None or master_gdf.empty:
        min_lon, min_lat = df_calls['lon'].min(), df_calls['lat'].min()
        max_lon, max_lat = df_calls['lon'].max(), df_calls['lat'].max()
        lon_pad = (max_lon - min_lon) * 0.1
        lat_pad = (max_lat - min_lat) * 0.1
        poly = box(min_lon - lon_pad, min_lat - lat_pad, max_lon + lon_pad, max_lat + lat_pad)
        master_gdf = gpd.GeoDataFrame({'DISPLAY_NAME': ['Auto-Generated Boundary'], 'data_count': [len(df_calls)]}, geometry=[poly], crs="EPSG:4326")

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"<h3 style='margin-bottom:0px; color:{text_main};'>📍 Jurisdictions</h3>", unsafe_allow_html=True)
    
    total_pts = master_gdf['data_count'].sum()
    master_gdf['LABEL'] = master_gdf['DISPLAY_NAME'] + " (" + (master_gdf['data_count']/total_pts*100).round(1).astype(str) + "%)"
    options_map = dict(zip(master_gdf['LABEL'], master_gdf['DISPLAY_NAME']))
    all_options = master_gdf['LABEL'].tolist()
    
    selected_labels = st.sidebar.multiselect("Active Jurisdictions", options=all_options, default=all_options, label_visibility="collapsed")
    if not selected_labels:
        st.warning("Please select at least one jurisdiction from the sidebar.")
        st.stop()
        
    selected_names = [options_map[l] for l in selected_labels]
    active_gdf = master_gdf[master_gdf['DISPLAY_NAME'].isin(selected_names)]

    minx, miny, maxx, maxy = active_gdf.to_crs(epsg=4326).total_bounds
    center_lon = (minx + maxx) / 2
    center_lat = (miny + maxy) / 2
    dynamic_zoom = calculate_zoom(minx, maxx, miny, maxy)
    
    utm_zone = int((center_lon + 180) / 6) + 1
    epsg_code = f"326{utm_zone}" if center_lat > 0 else f"327{utm_zone}"
    
    city_m = None
    city_boundary_geom = None
    try:
        active_utm = active_gdf.to_crs(epsg=epsg_code)
        if hasattr(active_utm.geometry, 'union_all'):
            full_boundary_utm = active_utm.geometry.buffer(0.1).union_all().buffer(-0.1)
        else:
            full_boundary_utm = active_utm.geometry.buffer(0.1).unary_union.buffer(-0.1)
            
        city_m = full_boundary_utm
        city_boundary_geom = gpd.GeoSeries([full_boundary_utm], crs=epsg_code).to_crs(epsg=4326).iloc[0]
    except Exception as e:
        st.error(f"Geometry Error: {e}")
        st.stop()

    # --- SIDEBAR LAYOUT CONTAINERS ---
    opt_container = st.sidebar.container()
    strat_expander = st.sidebar.expander("⚙️ Deployment Strategy", expanded=False)
    disp_expander = st.sidebar.expander("👁️ Display Options", expanded=False)
    filter_expander = st.sidebar.expander("⚙️ Data Filters", expanded=False)
    sim_expander = st.sidebar.expander("🚗 Ground Traffic Simulator", expanded=False)

    # --- DYNAMIC MISSION DATA FILTERS ---
    with filter_expander:
        if 'type' in df_stations_all.columns:
            all_types = sorted(df_stations_all['type'].dropna().astype(str).unique().tolist())
            if all_types:
                st.markdown(f"<div style='font-size:0.75rem; color:{text_muted}; font-weight:800; margin-top:15px; margin-bottom:5px; text-transform:uppercase;'>Facility Type</div>", unsafe_allow_html=True)
                selected_types = st.multiselect("Facility Type", options=all_types, default=all_types, label_visibility="collapsed", help="Filter the types of stations available for drone deployment.")
                if not selected_types:
                    st.warning("Please select at least one Facility Type.")
                    st.stop()
                df_stations_all = df_stations_all[df_stations_all['type'].astype(str).isin(selected_types)].copy().reset_index(drop=True)
                df_stations_all['name'] = "[" + df_stations_all['type'].astype(str) + "] " + df_stations_all['name'].astype(str)
                
        if 'priority' in df_calls.columns:
            all_priorities = sorted(df_calls['priority'].dropna().unique().tolist())
            if all_priorities:
                st.markdown(f"<div style='font-size:0.75rem; color:{text_muted}; font-weight:800; margin-top:15px; margin-bottom:5px; text-transform:uppercase;'>Incident Priority</div>", unsafe_allow_html=True)
                selected_priorities = st.multiselect("Incident Priority", options=all_priorities, default=all_priorities, label_visibility="collapsed", help="Filter the historical 911 calls by priority level.")
                if not selected_priorities:
                    st.warning("Please select at least one Incident Priority.")
                    st.stop()
                df_calls = df_calls[df_calls['priority'].isin(selected_priorities)].copy().reset_index(drop=True)

    if len(df_stations_all) == 0:
        st.error("No stations match the selected filters.")
        st.stop()
    if len(df_calls) == 0:
        st.error("No calls match the selected filters.")
        st.stop()

    n = len(df_stations_all)

    # --- OPTIMIZER CONTROLS ---
    with opt_container:
        st.markdown("---")
        st.markdown(f"<h3 style='margin-bottom:0px; color:{text_main};'>🎯 Optimizer Controls</h3>", unsafe_allow_html=True)

        st.markdown(f"<div style='font-size:0.75rem; color:{text_muted}; font-weight:800; margin-top:15px; margin-bottom:5px; text-transform:uppercase;'>Optimization Goal</div>", unsafe_allow_html=True)
        opt_strategy_raw = st.radio("Goal", ("Call Coverage", "Land Coverage"), horizontal=True, label_visibility="collapsed", help="Choose whether the algorithm should prioritize covering the most 911 calls or the most physical land area.")
        opt_strategy = "Maximize Call Coverage" if opt_strategy_raw == "Call Coverage" else "Maximize Land Coverage"
        
        st.markdown(f"<div style='font-size:0.75rem; color:{text_muted}; font-weight:800; margin-top:15px; margin-bottom:5px; text-transform:uppercase;'>Fleet Configuration</div>", unsafe_allow_html=True)
        
        counts_placeholder = st.container()
        ranges_placeholder = st.container()

    with ranges_placeholder:
        resp_radius_mi = st.slider("🚁 Responder Range (Miles)", 2.0, 3.0, 2.0, step=0.5, help="The effective operational flight radius for short-range Responder drones.")
        guard_radius_mi = st.slider("🦅 Guardian Range (Miles)", 1, 8, 8, help="The effective operational flight radius for long-range Guardian drones.")

    bounds_hash = f"{minx}_{miny}_{maxx}_{maxy}_{n}_{resp_radius_mi}_{guard_radius_mi}"

    with st.spinner("⚡ Precomputing spatial optimization matrices..."):
        calls_in_city, display_calls, resp_matrix, guard_matrix, dist_matrix_r, dist_matrix_g, station_metadata, total_calls = precompute_spatial_data(
            df_calls, df_stations_all, city_m, epsg_code, resp_radius_mi, guard_radius_mi, bounds_hash
        )
        
        df_curve = compute_all_elbow_curves(
            total_calls, resp_matrix, guard_matrix, 
            [s['clipped_2m'] for s in station_metadata], 
            [s['clipped_guard'] for s in station_metadata], 
            city_m.area if city_m else 1.0,
            bounds_hash
        )

    def get_max_drones(col_name):
        series = df_curve[col_name].dropna()
        if len(series) == 0: return 1
        idx_99 = series[series >= 99.0].first_valid_index()
        if idx_99 is not None:
            return int(df_curve.loc[idx_99, 'Drones'])
        else:
            return int(df_curve.loc[series.last_valid_index(), 'Drones'])

    max_r = min(max(1, get_max_drones('Responder (Calls)')), n)
    max_g = min(max(1, get_max_drones('Guardian (Calls)')), n)

    with counts_placeholder:
        k_responder = st.slider("🚁 Responder Count", 0, max_r, min(1, max_r), help="Number of short-range tactical drones to deploy.")
        k_guardian = st.slider("🦅 Guardian Count", 0, max_g, 0, help="Number of long-range heavy-lift drones to deploy.")
        
    with strat_expander:
        incremental_build = st.toggle("Phased Rollout", value=True, help="Builds the fleet one-by-one. Existing stations are locked in place as new drones are added.")
        allow_redundancy = st.toggle("Multi-Tier (Allow Overlap)", value=True, help="Allows drone coverage rings to overlap if call volume justifies it. If disabled, forces drones apart.")

    with disp_expander:
        col1, col2 = st.columns(2)
        show_boundaries = col1.toggle("Boundaries", value=True, help="Show or hide jurisdiction borders.")
        show_heatmap = col2.toggle("Heatmap", value=False, help="Overlay a thermal density map of historical 911 calls.")
        show_health = col1.toggle("Health Score", value=False, help="Display the department's overall drone coverage health score.")
        show_satellite = col2.toggle("Satellite", value=False, help="Switch the map background to high-resolution satellite imagery.")
        show_cards = st.toggle("Unit Economics Cards", value=True, help="Show or hide the financial breakdown cards for each deployed drone.")

    with sim_expander:
        simulate_traffic = st.toggle("Enable Traffic Sim", value=False, help="Compare drone flight times against ground vehicle drive times.")
        if simulate_traffic:
            traffic_level = st.slider("Traffic Intensity (%)", 0, 100, 40, help="Adjust the simulated ground traffic congestion.")
        else:
            traffic_level = 40

    budget_placeholder = st.sidebar.container()

    best_resp_names, best_guard_names = [], []
    active_resp_names, active_guard_names = [], []
    chrono_r, chrono_g = [], []
    
    if k_responder + k_guardian > n:
        st.error("⚠️ Over-Deployment: Total requested drones exceed available stations.")
    elif k_responder > 0 or k_guardian > 0:
        
        best_combo = None
        
        if opt_strategy == "Maximize Call Coverage":
            with st.spinner("🧠 Running exact MCLP Optimizer (PuLP)..."):
                r_best, g_best, chrono_r, chrono_g = solve_mclp(resp_matrix, guard_matrix, dist_matrix_r, dist_matrix_g, k_responder, k_guardian, allow_redundancy, incremental=incremental_build)
                best_combo = (tuple(r_best), tuple(g_best))
        else:
            station_indices = list(range(n))
            total_resp_combos = math.comb(n, k_responder)
            total_guard_combos = math.comb(n - k_responder, k_guardian) if n >= k_responder else 1
            total_possible = total_resp_combos * total_guard_combos
            
            best_score = (-1.0, -1, -1.0)
            
            def evaluate_combo(rg_combo):
                r_combo, g_combo = rg_combo
                if allow_redundancy:
                    r_geos = [station_metadata[i]['clipped_2m'] for i in r_combo]
                    g_geos = [station_metadata[i]['clipped_guard'] for i in g_combo]
                    score_area = (unary_union(r_geos).area if r_geos else 0.0) + (unary_union(g_geos).area if g_geos else 0.0)
                else:
                    geos = [station_metadata[i]['clipped_2m'] for i in r_combo] + [station_metadata[i]['clipped_guard'] for i in g_combo]
                    score_area = unary_union(geos).area if geos else 0.0
                
                if total_calls > 0:
                    cov_r = resp_matrix[list(r_combo)].any(axis=0) if r_combo else np.zeros(total_calls, bool)
                    cov_g = guard_matrix[list(g_combo)].any(axis=0) if g_combo else np.zeros(total_calls, bool)
                    score_calls = np.logical_or(cov_r, cov_g).sum()
                else:
                    score_calls = 0
                
                score_cent = sum([station_metadata[i]['centrality'] for i in r_combo]) + sum([station_metadata[i]['centrality'] for i in g_combo])
                return (score_area, score_calls, score_cent, rg_combo)

            with st.spinner(f"Optimizing configurations for land area..."):
                if incremental_build:
                    locked_r = ()
                    locked_g = ()

                    for _ in range(k_guardian):
                        loop_best_score = (-1.0, -1, -1.0)
                        best_pick = None
                        for s in range(n):
                            if s in locked_g or (not allow_redundancy and s in locked_r): continue
                            test_g = tuple(sorted(list(locked_g) + [s]))
                            score = evaluate_combo((locked_r, test_g))
                            if score > loop_best_score:
                                loop_best_score = score
                                best_pick = s
                        if best_pick is not None:
                            locked_g = tuple(sorted(list(locked_g) + [best_pick]))
                            chrono_g.append(best_pick)

                    for _ in range(k_responder):
                        loop_best_score = (-1.0, -1, -1.0)
                        best_pick = None
                        for s in range(n):
                            if s in locked_r or (not allow_redundancy and s in locked_g): continue
                            test_r = tuple(sorted(list(locked_r) + [s]))
                            score = evaluate_combo((test_r, locked_g))
                            if score > loop_best_score:
                                loop_best_score = score
                                best_pick = s
                        if best_pick is not None:
                            locked_r = tuple(sorted(list(locked_r) + [best_pick]))
                            chrono_r.append(best_pick)

                    best_combo = (locked_r, locked_g)
                else:
                    if total_possible > 3000:
                        st.toast(f"Optimization Mode: Sampling ({total_possible:,} options)")
                        sampled_combos = []
                        for _ in range(3000):
                            chosen = np.random.choice(range(n), k_responder + k_guardian, replace=False)
                            r_c = tuple(sorted(chosen[:k_responder]))
                            g_c = tuple(sorted(chosen[k_responder:]))
                            sampled_combos.append((r_c, g_c))
                        combos_to_test = list(set(sampled_combos))
                    else:
                        combos_to_test = []
                        for r_c in itertools.combinations(station_indices, k_responder):
                            rem = [x for x in station_indices if x not in r_c]
                            if k_guardian > 0:
                                for g_c in itertools.combinations(rem, k_guardian):
                                    combos_to_test.append((r_c, g_c))
                            else:
                                combos_to_test.append((r_c, ()))

                    with ThreadPoolExecutor() as executor:
                        results = list(executor.map(evaluate_combo, combos_to_test))
                        
                    for score_area, score_calls, score_cent, combo in results:
                        if (score_area, score_calls, score_cent) > best_score:
                            best_score = (score_area, score_calls, score_cent)
                            best_combo = combo
                            
                    chrono_r = list(best_combo[0])
                    chrono_g = list(best_combo[1])
        
        if best_combo is not None:
            r_best, g_best = best_combo
            active_resp_names = [station_metadata[i]['name'] for i in r_best]
            active_guard_names = [station_metadata[i]['name'] for i in g_best]

    st.markdown("---")
    
    # --- METRICS CALCULATION ---
    area_covered_perc, overlap_perc, calls_covered_perc = 0.0, 0.0, 0.0
    
    active_resp_idx = [i for i, s in enumerate(station_metadata) if s['name'] in active_resp_names]
    active_guard_idx = [i for i, s in enumerate(station_metadata) if s['name'] in active_guard_names]
    
    # --- BUILD STRICTLY ISOLATED DRONE DEPLOYMENTS ---
    ordered_deployments_raw = []
    for idx in chrono_g:
        if idx in active_guard_idx: ordered_deployments_raw.append((idx, 'GUARDIAN'))
    for idx in chrono_r:
        if idx in active_resp_idx: ordered_deployments_raw.append((idx, 'RESPONDER'))

    for idx in active_resp_idx:
        if idx not in chrono_r: ordered_deployments_raw.append((idx, 'RESPONDER'))
    for idx in active_guard_idx:
        if idx not in chrono_g: ordered_deployments_raw.append((idx, 'GUARDIAN'))

    active_color_map = {}
    c_idx = 0
    for idx, d_type in ordered_deployments_raw:
        key = f"{station_metadata[idx]['name']}_{d_type}"
        if key not in active_color_map:
            active_color_map[key] = STATION_COLORS[c_idx % len(STATION_COLORS)]
            c_idx += 1
            
    active_resp_data = [station_metadata[i] for i in active_resp_idx]
    active_guard_data = [station_metadata[i] for i in active_guard_idx]
    
    active_geos = [s['clipped_2m'] for s in active_resp_data] + [s['clipped_guard'] for s in active_guard_data]

    if active_geos:
        if not city_m.is_empty:
            area_covered_perc = (unary_union(active_geos).area / city_m.area) * 100
        
        if total_calls > 0:
            cov_r = resp_matrix[active_resp_idx].any(axis=0) if active_resp_idx else np.zeros(total_calls, bool)
            cov_g = guard_matrix[active_guard_idx].any(axis=0) if active_guard_idx else np.zeros(total_calls, bool)
            calls_covered_perc = (np.logical_or(cov_r, cov_g).sum() / total_calls) * 100
        
        inters = []
        for i in range(len(active_geos)):
            for j in range(i+1, len(active_geos)):
                over = active_geos[i].intersection(active_geos[j])
                if not over.is_empty: inters.append(over)
        if not city_m.is_empty:
            overlap_perc = (unary_union(inters).area / city_m.area * 100) if inters else 0.0

    # ==========================================
    # --- BUDGET IMPACT MODULE & ALLOCATION ---
    # ==========================================
    active_drones = []
    fleet_capex = 0
    
    with budget_placeholder:
        st.markdown("---")
        st.markdown(f"<h3 style='color:{text_main};'>💰 Budget Impact</h3>", unsafe_allow_html=True)
        
        inferred_daily_calls = st.session_state.get('inferred_daily_calls_override', max(1, int(total_calls / 365)))
        max_slider_val = max(100, inferred_daily_calls * 3) 
        
        calls_per_day = st.slider("TOTAL DAILY CALLS (CITYWIDE)", min_value=1, max_value=max_slider_val, value=inferred_daily_calls)
        
        col_r1, col_r2 = st.columns(2)
        dfr_dispatch_rate = col_r1.slider("DFR DISPATCH RATE (%)", min_value=1, max_value=100, value=25) / 100.0
        deflection_rate = col_r2.slider("DRONE-ONLY RESOLUTION (%)", min_value=0, max_value=100, value=30) / 100.0
        
        actual_k_responder = len(active_resp_names)
        actual_k_guardian = len(active_guard_names)
        
        capex_responder_total = actual_k_responder * CONFIG["RESPONDER_COST"]
        capex_guardian_total = actual_k_guardian * CONFIG["GUARDIAN_COST"]
        fleet_capex = capex_responder_total + capex_guardian_total
        
        if fleet_capex > 0:
            effective_coverage_rate = calls_covered_perc / 100.0
            covered_daily_calls = calls_per_day * effective_coverage_rate
            daily_dfr_responses = covered_daily_calls * dfr_dispatch_rate
            daily_drone_only_calls = daily_dfr_responses * deflection_rate
            
            if daily_drone_only_calls > 0:
                monthly_savings = (CONFIG["OFFICER_COST_PER_CALL"] - CONFIG["DRONE_COST_PER_CALL"]) * daily_drone_only_calls * 30.4
                annual_savings = monthly_savings * 12
                fleet_break_even_months = fleet_capex / monthly_savings
                break_even_text = f"{fleet_break_even_months:.1f} MONTHS"
            else:
                annual_savings = 0
                break_even_text = "N/A"
            
            st.markdown(f"""
            <div style="background-color: {budget_box_bg}; border: 1px solid {budget_box_border}; padding: 12px; border-radius: 4px; text-align: center; margin-bottom: 12px; box-shadow: 0 2px 5px {budget_box_shadow};">
                <h6 style="color: {text_muted}; margin: 0; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase;">Annual Capacity Value</h6>
                <h2 style="color: {budget_box_border}; margin: 4px 0; font-family: 'Consolas', monospace; font-size: 1.8rem;">${annual_savings:,.0f}</h2>
                <div style="border-top: 1px solid {card_border}; margin: 8px 0;"></div>
                <div style="display: flex; justify-content: space-between; font-size: 0.75rem; margin-bottom: 3px;">
                    <span style="color: {text_muted};">CALLS IN RANGE:</span>
                    <span style="color: {text_main}; font-weight: 700;">{covered_daily_calls:.1f} / DAY</span>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.75rem; margin-bottom: 3px;">
                    <span style="color: {text_muted};">DFR FLIGHTS ({int(dfr_dispatch_rate*100)}%):</span>
                    <span style="color: {text_main}; font-weight: 700;">{daily_dfr_responses:.1f} / DAY</span>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.75rem; margin-bottom: 6px;">
                    <span style="color: {text_muted};">DEFLECTED (CAPACITY):</span>
                    <span style="color: {text_main}; font-weight: 700;">{daily_drone_only_calls:.1f} / DAY</span>
                </div>
                <div style="border-top: 1px dashed {card_border}; margin: 6px 0;"></div>
                <div style="display: flex; justify-content: space-between; font-size: 0.75rem; margin-bottom: 3px;">
                    <span style="color: {text_muted};">FLEET CAPEX:</span>
                    <span style="color: {text_main}; font-weight: 700;">${fleet_capex:,.0f}</span>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.75rem;">
                    <span style="color: {text_muted};">BREAK-EVEN:</span>
                    <span style="color: {budget_box_border}; font-weight: 700;">{break_even_text}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if actual_k_responder > 0:
                st.markdown(f"""
                <div style="background-color: {card_bg}; border: 1px solid {card_border}; padding: 10px; border-radius: 4px; margin-bottom: 8px;">
                    <h5 style="color: {text_main}; margin: 0 0 4px 0; font-size: 0.85rem;">RESPONDER <span style="color:{text_muted}; font-weight:normal;">(x{actual_k_responder})</span></h5>
                    <div style="color: {text_muted}; font-size: 0.75rem;">COVERAGE: <span style="color:{text_main}; font-weight:600;">{resp_radius_mi} MI RADIUS</span></div>
                    <div style="color: {text_muted}; font-size: 0.75rem;">UNIT CAPEX: <span style="color:{text_main}; font-weight:600;">${CONFIG["RESPONDER_COST"]:,.0f}</span></div>
                    <div style="color: {text_muted}; font-size: 0.75rem; margin-top: 4px; border-top: 1px solid {card_border}; padding-top: 4px;">SUBTOTAL: <span style="color:{text_main}; font-weight:600;">${capex_responder_total:,.0f}</span></div>
                </div>
                """, unsafe_allow_html=True)
                
            if actual_k_guardian > 0:
                st.markdown(f"""
                <div style="background-color: {card_bg}; border: 1px solid {card_border}; padding: 10px; border-radius: 4px; margin-bottom: 8px;">
                    <h5 style="color: {text_main}; margin: 0 0 4px 0; font-size: 0.85rem;">GUARDIAN <span style="color:{text_muted}; font-weight:normal;">(x{actual_k_guardian})</span></h5>
                    <div style="color: {text_muted}; font-size: 0.75rem;">COVERAGE: <span style="color:{text_main}; font-weight:600;">{guard_radius_mi} MI RADIUS</span></div>
                    <div style="color: {text_muted}; font-size: 0.75rem;">UNIT CAPEX: <span style="color:{text_main}; font-weight:600;">${CONFIG["GUARDIAN_COST"]:,.0f}</span></div>
                    <div style="color: {text_muted}; font-size: 0.75rem; margin-top: 4px; border-top: 1px solid {card_border}; padding-top: 4px;">SUBTOTAL: <span style="color:{text_main}; font-weight:600;">${capex_guardian_total:,.0f}</span></div>
                </div>
                """, unsafe_allow_html=True)

            cumulative_mask = np.zeros(total_calls, dtype=bool) if total_calls > 0 else None
            
            step = 1
            for idx, d_type in ordered_deployments_raw:
                if d_type == 'RESPONDER':
                    cov_array = resp_matrix[idx]
                    cost = CONFIG["RESPONDER_COST"]
                    speed_mph = CONFIG["RESPONDER_SPEED"]
                    avg_dist = station_metadata[idx]['avg_dist_r']
                    radius_m = resp_radius_mi * 1609.34
                else:
                    cov_array = guard_matrix[idx]
                    cost = CONFIG["GUARDIAN_COST"]
                    speed_mph = CONFIG["GUARDIAN_SPEED"]
                    avg_dist = station_metadata[idx]['avg_dist_g']
                    radius_m = guard_radius_mi * 1609.34
                    
                map_color = active_color_map[f"{station_metadata[idx]['name']}_{d_type}"]
                avg_time_min = (avg_dist / speed_mph) * 60

                d = {
                    'idx': idx,
                    'name': station_metadata[idx]['name'],
                    'lat': station_metadata[idx]['lat'],
                    'lon': station_metadata[idx]['lon'],
                    'type': d_type,
                    'cost': cost,
                    'cov_array': cov_array,
                    'color': map_color,
                    'deploy_step': step if (idx in chrono_r or idx in chrono_g) else "MANUAL",
                    'avg_time_min': avg_time_min,
                    'speed_mph': speed_mph,
                    'radius_m': radius_m
                }
                
                if total_calls > 0:
                    marginal_mask = cov_array & ~cumulative_mask
                    marginal_historic = np.sum(marginal_mask)
                    d['assigned_indices'] = np.where(marginal_mask)[0] 
                    
                    cumulative_mask = cumulative_mask | cov_array
                    
                    d['marginal_perc'] = marginal_historic / total_calls
                    marginal_daily = calls_per_day * d['marginal_perc']
                    d['marginal_flights'] = marginal_daily * dfr_dispatch_rate
                    d['marginal_deflected'] = d['marginal_flights'] * deflection_rate
                    
                    temp_all_cov = np.vstack([x['cov_array'] for x in [{'cov_array': resp_matrix[i]} for i in active_resp_idx] + [{'cov_array': guard_matrix[i]} for i in active_guard_idx]])
                    overlap_counts = temp_all_cov.sum(axis=0)
                    shared_mask = d['cov_array'] & (overlap_counts > 1)
                    shared_daily_calls = (np.sum(shared_mask) / total_calls) * calls_per_day
                    d['shared_flights'] = shared_daily_calls * dfr_dispatch_rate

                    d['monthly_savings'] = (CONFIG["OFFICER_COST_PER_CALL"] - CONFIG["DRONE_COST_PER_CALL"]) * d['marginal_deflected'] * 30.4
                    d['annual_savings'] = d['monthly_savings'] * 12
                    
                    if d['monthly_savings'] > 0:
                        d['be_text'] = f"{d['cost'] / d['monthly_savings']:.1f} MO"
                    else:
                        d['annual_savings'] = 0
                        d['be_text'] = "N/A"
                else:
                    d['assigned_indices'] = []
                    d['annual_savings'] = 0
                    d['marginal_flights'] = 0
                    d['marginal_deflected'] = 0
                    d['shared_flights'] = 0
                    d['be_text'] = "N/A"
                    
                active_drones.append(d)
                step += 1
                
        else:
            st.info("🚁 Select at least one drone above to calculate budget impact.")

    # ==========================================

    if show_health:
        norm_redundancy = min(overlap_perc / 35.0, 1.0) * 100
        health_score = (calls_covered_perc * 0.50) + (area_covered_perc * 0.35) + (norm_redundancy * 0.15)
        if health_score >= 80: h_color, h_label = accent_color, "OPTIMAL" 
        elif health_score >= 70: h_color, h_label = "#94c11f", "GOOD"
        elif health_score >= 55: h_color, h_label = "#ffc107", "MARGINAL"
        else: h_color, h_label = "#dc3545", "ESSENTIAL"
        
        st.markdown(f"""
            <div style="background-color: {card_bg}; border-left: 5px solid {h_color}; border-top: 1px solid {card_border}; border-right: 1px solid {card_border}; border-bottom: 1px solid {card_border}; padding: 10px; border-radius: 4px; color: {text_main}; margin-bottom: 10px; display: flex; align-items: center; justify-content: space-between;">
                <span style="font-size: 1.5em; font-weight: bold; color: {h_color};">Department Health Score: {health_score:.1f}%</span>
                <span style="font-size: 1.3em; background: rgba(128,128,128,0.15); padding: 2px 10px; border-radius: 4px;">{h_label}</span>
            </div>""", unsafe_allow_html=True)

    if simulate_traffic:
        m1, m2, m3, m4, m5 = st.columns(5)
        
        if len(active_guard_names) > 0:
            eval_dist = guard_radius_mi
            eval_speed = CONFIG["GUARDIAN_SPEED"]
            gain_label = f"Efficiency Gain ({guard_radius_mi}-mi)"
        else:
            eval_dist = resp_radius_mi
            eval_speed = CONFIG["RESPONDER_SPEED"]
            gain_label = f"Efficiency Gain ({resp_radius_mi}-mi)"

        avg_ground_speed = CONFIG["DEFAULT_TRAFFIC_SPEED"] * (1 - (traffic_level / 100))
        
        if len(active_resp_names) == 0 and len(active_guard_names) == 0:
            gain_val = "N/A"
        elif avg_ground_speed > 0:
            drone_t = (eval_dist / eval_speed) * 60 
            ground_t = ((eval_dist * 1.4) / avg_ground_speed) * 60 
            time_saved = ground_t - drone_t
            gain_val = f"{time_saved:.1f} min"
        else:
            gain_val = "Stalled"
            
        m1.metric("Total Incident Points", f"{total_calls:,}")
        m2.metric("Response Capacity %", f"{calls_covered_perc:.1f}%")
        m3.metric("Land Covered", f"{area_covered_perc:.1f}%")
        m4.metric("Redundancy (Overlap)", f"{overlap_perc:.1f}%")
        m5.metric(gain_label, gain_val)
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Incident Points", f"{total_calls:,}")
        m2.metric("Response Capacity %", f"{calls_covered_perc:.1f}%")
        m3.metric("Land Covered", f"{area_covered_perc:.1f}%")
        m4.metric("Redundancy (Overlap)", f"{overlap_perc:.1f}%")

    # --- KML EXPORT ---
    kml_data = generate_kml(
        active_gdf, 
        active_drones, 
        calls_in_city
    )
    
    st.sidebar.markdown("---")
    st.sidebar.download_button(
        label="🌏 Download for Google Earth",
        data=kml_data,
        file_name="drone_deployment.kml",
        mime="application/vnd.google-earth.kml+xml"
    )

    # ==========================================
    # --- MAIN UI SPLIT: MAP (LEFT) & STATS (RIGHT) ---
    # ==========================================
    map_col, stats_col = st.columns([4.2, 1.8])
    
    with map_col:
        fig = go.Figure()
        
        if show_boundaries:
            if city_boundary_geom is not None and not city_boundary_geom.is_empty:
                if isinstance(city_boundary_geom, Polygon):
                    bx, by = city_boundary_geom.exterior.coords.xy
                    fig.add_trace(go.Scattermapbox(mode="lines", lon=list(bx), lat=list(by), line=dict(color=map_boundary_color, width=2), name="Jurisdiction Boundary", hoverinfo='skip'))
                elif isinstance(city_boundary_geom, MultiPolygon):
                    for poly in city_boundary_geom.geoms:
                        bx, by = poly.exterior.coords.xy
                        fig.add_trace(go.Scattermapbox(mode="lines", lon=list(bx), lat=list(by), line=dict(color=map_boundary_color, width=2), name="Jurisdiction Boundary", hoverinfo='skip', showlegend=False))

        if show_heatmap and not display_calls.empty:
            fig.add_trace(go.Densitymapbox(
                lat=display_calls.geometry.y,
                lon=display_calls.geometry.x,
                z=np.ones(len(display_calls)),
                radius=12,
                colorscale='Inferno',
                opacity=0.6,
                showscale=False,
                name="Heatmap",
                hoverinfo='skip'
            ))

        if not display_calls.empty:
            fig.add_trace(go.Scattermapbox(
                lat=display_calls.geometry.y, 
                lon=display_calls.geometry.x, 
                mode='markers', 
                marker=dict(size=4, color=map_incident_color, opacity=0.4), 
                name="Incident Data", 
                hoverinfo='skip'
            ))

        for d in active_drones:
            clats, clons = get_circle_coords(d['lat'], d['lon'], r_mi=d['radius_m'] / 1609.34)
            short_name = d['name'].split(',')[0]
            lbl = f"{short_name} ({'Resp' if d['type'] == 'RESPONDER' else 'Guard'})"

            fig.add_trace(go.Scattermapbox(
                lat=list(clats) + [None, d['lat']], 
                lon=list(clons) + [None, d['lon']], 
                mode='lines+markers', 
                marker=dict(size=[0]*len(clats) + [0, 20], color=d['color']), 
                line=dict(color=d['color'], width=4.5), 
                fill='toself', 
                fillcolor='rgba(0,0,0,0)', 
                name=lbl, 
                hoverinfo='name'
            ))

            if simulate_traffic:
                if traffic_level < 35:
                    t_color = "#28a745"
                    t_fill = "rgba(40, 167, 69, 0.15)"
                    t_label = "Light Traffic"
                elif traffic_level < 75:
                    t_color = "#ffc107"
                    t_fill = "rgba(255, 193, 7, 0.15)"
                    t_label = "Moderate Traffic"
                else:
                    t_color = "#dc3545"
                    t_fill = "rgba(220, 53, 69, 0.15)"
                    t_label = "Heavy Traffic"

                ground_speed_mph = CONFIG["DEFAULT_TRAFFIC_SPEED"] * (1 - (traffic_level / 100))
                drive_time_min = (d['radius_m'] / 1609.34 / d['speed_mph']) * 60 
                
                if ground_speed_mph > 0:
                    ground_range_mi = (ground_speed_mph / 60) * drive_time_min
                    g_angles = np.linspace(0, 2*np.pi, 9) 
                    g_lats = d['lat'] + (ground_range_mi/69.172) * np.sin(g_angles)
                    g_lons = d['lon'] + (ground_range_mi/(69.172 * np.cos(np.radians(d['lat'])))) * np.cos(g_angles)

                    fig.add_trace(go.Scattermapbox(
                        lat=list(g_lats),
                        lon=list(g_lons),
                        mode='lines',
                        line=dict(color=t_color, width=2.5), 
                        fill='toself',
                        fillcolor=t_fill,
                        name=f"Ground Reach ({t_label})",
                        hoverinfo='skip'
                    ))

        mapbox_config = dict(
            center=dict(lat=center_lat, lon=center_lon),
            zoom=dynamic_zoom,
            style=map_style
        )
        
        if show_satellite:
            mapbox_config["style"] = "carto-positron"
            mapbox_config["layers"] = [
                {
                    "below": 'traces',
                    "sourcetype": "raster",
                    "sourceattribution": "Esri, Maxar, Earthstar Geographics",
                    "source": [
                        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                    ]
                }
            ]

        fig.update_layout(
            uirevision="LOCKED_MAP",
            mapbox=mapbox_config,
            margin=dict(l=0, r=0, t=0, b=0), 
            height=800,
            font=dict(size=18),
            showlegend=True,
            legend=dict(
                title=dict(text=""),
                yanchor="top",
                y=0.98,
                xanchor="left",
                x=0.02,
                bgcolor=legend_bg,
                bordercolor=accent_color, 
                borderwidth=1,
                font=dict(size=12, color=legend_text),
                itemclick="toggle"
            )
        )

        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})
        
    with stats_col:
        
        # --- Coverage Elbow Curve ---
        st.markdown(f"<h4 style='margin-top:0px; border-bottom: 1px solid {card_border}; padding-bottom: 8px; color: {text_main};'>Coverage Optimization</h4>", unsafe_allow_html=True)
        
        df_curve = compute_all_elbow_curves(
            total_calls, resp_matrix, guard_matrix, 
            [s['clipped_2m'] for s in station_metadata], 
            [s['clipped_guard'] for s in station_metadata], 
            city_m.area if city_m else 1.0,
            bounds_hash
        )
        
        if not df_curve.empty:
            fig_curve = go.Figure()
            
            line_configs = [
                ('Responder (Calls)', accent_color, 'solid'),
                ('Guardian (Calls)', '#FFD700', 'solid'),
                ('Responder (Area)', accent_color, 'dash'),
                ('Guardian (Area)', '#FFD700', 'dash'),
            ]
            
            for col, color, dash in line_configs:
                y_data = df_curve[col].dropna()
                x_data = df_curve.loc[y_data.index, 'Drones']
                if not y_data.empty:
                    fig_curve.add_trace(go.Scatter(
                        x=x_data, y=y_data, 
                        mode='lines+markers', name=col,
                        line=dict(color=color, width=2, dash=dash), marker=dict(size=4)
                    ))
                    
                    # Highlight the exact dot where it crosses 90% (No floating text)
                    if 'Calls' in col:
                        idx_90 = y_data[y_data >= 90.0].first_valid_index()
                        if idx_90 is not None:
                            drones_90 = int(x_data.loc[idx_90])
                            val_90 = y_data.loc[idx_90]
                            fig_curve.add_trace(go.Scatter(
                                x=[drones_90], y=[val_90],
                                mode='markers',
                                marker=dict(color=color, size=12, symbol='star', line=dict(color='white', width=1)),
                                showlegend=False,
                                hoverinfo='skip'
                            ))
                
            fig_curve.update_layout(
                xaxis_title="Drones", 
                yaxis_title="Coverage %",
                xaxis=dict(showgrid=True, gridcolor=card_border, tickfont=dict(color=text_muted)),
                yaxis=dict(
                    showgrid=True, 
                    gridcolor=card_border, 
                    tickfont=dict(color=text_muted),
                    tickvals=[0, 20, 40, 60, 80, 90, 100],
                    ticktext=['0', '20', '40', '60', '80', f'<b style="color:{accent_color}; font-size:14px;">90</b>', '100'],
                    range=[0, 105]
                ),
                legend=dict(
                    title=dict(text=""),
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=10, color=text_muted)
                ),
                margin=dict(l=10, r=10, t=20, b=10),
                height=260,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_curve, use_container_width=True, config={'displayModeBar': False})
            st.markdown("<br>", unsafe_allow_html=True)

        # --- Unit-Level Economics Cards ---
        if show_cards:
            st.markdown(f"<h4 style='margin-top:0px; border-bottom: 1px solid {card_border}; padding-bottom: 8px; color: {text_main};'>Unit-Level Economics</h4>", unsafe_allow_html=True)
            if fleet_capex > 0:
                with st.container():
                    c1, c2 = st.columns(2)
                    for i, d in enumerate(active_drones):
                        target_col = c1 if i % 2 == 0 else c2
                        formatted_name = format_3_lines(d['name'])
                        
                        html_card = (
                            f'<div style="background-color: {card_bg}; color: {card_text}; border-top: 4px solid {d["color"]}; '
                            f'border-left: 1px solid {card_border}; border-right: 1px solid {card_border}; border-bottom: 1px solid {card_border}; '
                            f'padding: 8px; border-radius: 4px; margin-bottom: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.2); line-height: 1.2;">'
                            f'<div style="font-weight: 700; font-size: 0.7rem; margin-bottom: 6px; min-height: 3.6em; color: {card_title};">{formatted_name}</div>'
                            f'<div style="font-size: 0.6rem; color: #888; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;">{d["type"]} • PH: #{d["deploy_step"]}</div>'
                            f'<div style="font-size: 0.7rem; color: {text_muted}; margin-bottom: 2px;">Capacity Value: <span style="color: {accent_color}; font-weight: 700; float: right;">${d["annual_savings"]:,.0f}</span></div>'
                            f'<div style="border-top: 1px solid {card_border}; margin: 4px 0;"></div>'
                            f'<div style="font-size: 0.65rem; color: {text_muted}; margin-bottom: 2px;">Net New Flights: <span style="font-weight: 600; color: {accent_color}; float: right;">{d["marginal_flights"]:.1f}/d</span></div>'
                            f'<div style="font-size: 0.65rem; color: {text_muted}; margin-bottom: 2px;">Shared Flights: <span style="font-weight: 600; float: right; color:{card_title};">{d["shared_flights"]:.1f}/d</span></div>'
                            f'<div style="font-size: 0.65rem; color: {text_muted}; margin-bottom: 6px;">Deflected: <span style="font-weight: 600; float: right; color:{card_title};">{d["marginal_deflected"]:.1f}/d</span></div>'
                            f'<div style="font-size: 0.65rem; color: {text_muted}; margin-bottom: 6px;">Avg Resp Time: <span style="font-weight: 600; float: right; color:{card_title};">{d["avg_time_min"]:.1f} min</span></div>'
                            f'<div style="border-top: 1px dashed {card_border}; padding-top: 4px; font-size: 0.65rem; color: {text_muted};">'
                            f'CapEx: <strong style="float:right; color:{card_title};">${d["cost"]:,.0f}</strong><br>'
                            f'ROI: <strong style="color: {accent_color}; float:right;">{d["be_text"]}</strong></div>'
                            f'</div>'
                        )
                        
                        with target_col:
                            st.markdown(html_card, unsafe_allow_html=True)
            else:
                st.info("🚁 Deploy drones on the map to see individual unit economics.")

    # ==========================================
    # --- 3D DECK.GL SWARM SIMULATION ---
    # ==========================================
    if fleet_capex > 0:
        st.markdown("---")
        st.markdown(f"<h3 style='margin-bottom:0px; color:{text_main};'>🚁 3D Swarm Simulation</h3>", unsafe_allow_html=True)
        
        show_sim = st.toggle("🎬 Enable 3D Swarm Simulation", value=False)
        
        if show_sim:
            def generate_deckgl_html(active_drones, calls_in_city, dfr_dispatch_rate, lat, lon, zoom, daily_calls):
                stations_json = []
                flights_json = []
                
                calls_coords = np.column_stack((calls_in_city['lon'], calls_in_city['lat']))
                
                # --- RESTRICTED NEAREST NEIGHBOR DISPATCH SIMULATION ---
                sim_assignments = {i: [] for i in range(len(active_drones))}
                for c_idx, call_coord in enumerate(calls_coords):
                    best_d_idx = -1
                    min_dist = float('inf')
                    for d_idx, d in enumerate(active_drones):
                        if d['cov_array'][c_idx]:
                            dist = (call_coord[0] - d['lon'])**2 + (call_coord[1] - d['lat'])**2
                            if dist < min_dist:
                                min_dist = dist
                                best_d_idx = d_idx
                    if best_d_idx != -1:
                        sim_assignments[best_d_idx].append(c_idx)
                
                legend_html = ""
                total_sim_flights = 0
                
                for d_idx, d in enumerate(active_drones):
                    short_name = f"{d['name'].split(',')[0]} ({d['type'][:3]})"
                    hex_c = d['color'].lstrip('#')
                    rgb = [int(hex_c[j:j+2], 16) for j in (0, 2, 4)]
                    
                    stations_json.append({
                        "name": short_name,
                        "lon": d['lon'],
                        "lat": d['lat'],
                        "color": rgb,
                        "radius": d['radius_m']
                    })
                    
                    legend_html += f'<div style="margin-bottom:3px;"><span style="display:inline-block;width:10px;height:10px;background-color:{d["color"]};margin-right:8px;border-radius:50%;"></span>{short_name}</div>'
                    
                    assigned_calls = sim_assignments[d_idx]
                    num_to_simulate = int(len(assigned_calls) * dfr_dispatch_rate)
                    if num_to_simulate > 0:
                        assigned_calls = random.sample(list(assigned_calls), min(num_to_simulate, len(assigned_calls)))
                    else:
                        assigned_calls = []

                    for call_idx in assigned_calls:
                        lon1, lat1 = calls_coords[call_idx]
                        lon0, lat0 = d['lon'], d['lat']
                        
                        dist_mi = ((lon1 - lon0)**2 + (lat1 - lat0)**2)**0.5 * 69.172
                        
                        flight_time_sec = (dist_mi / d['speed_mph']) * 3600
                        vis_time = max(flight_time_sec * 3, 120) 
                        
                        launch = random.randint(0, 86400)
                        
                        mid_lon = (lon0 + lon1) / 2
                        mid_lat = (lat0 + lat1) / 2
                        arc_height = min(max(dist_mi * 40, 50), 200) 
                        
                        flights_json.append({
                            "path": [[lon0, lat0, 0], [mid_lon, mid_lat, arc_height], [lon1, lat1, 0]],
                            "timestamps": [launch, launch + vis_time/2, launch + vis_time],
                            "color": rgb
                        })
                
                warn_html = ""
                if len(flights_json) > 2000:
                    flights_json = random.sample(flights_json, 2000)
                    warn_html = f'<div style="background: #440000; border: 1px solid #ff4b4b; color: #ffbbbb; padding: 5px; font-size: 10px; border-radius: 4px; margin-bottom: 10px;">⚠️ Visuals capped at 2,000 flights for performance (Total Actual: {total_sim_flights:,}).</div>'
                
                drone_svg = "data:image/svg+xml;charset=utf-8,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='white'%3E%3Cpath d='M18 6a2 2 0 100-4 2 2 0 000 4zm-12 0a2 2 0 100-4 2 2 0 000 4zm12 12a2 2 0 100-4 2 2 0 000 4zm-12 0a2 2 0 100-4 2 2 0 000 4z'/%3E%3Cpath stroke='white' stroke-width='2' stroke-linecap='round' d='M8.5 8.5l7 7m0-7l-7 7'/%3E%3Ccircle cx='12' cy='12' r='2' fill='white'/%3E%3C/svg%3E"
                
                html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <script src="https://unpkg.com/deck.gl@8.9.35/dist.min.js"></script>
                    <script src="https://unpkg.com/maplibre-gl@3.0.0/dist/maplibre-gl.js"></script>
                    <link href="https://unpkg.com/maplibre-gl@3.0.0/dist/maplibre-gl.css" rel="stylesheet" />
                    <style>
                        body {{ margin: 0; padding: 0; overflow: hidden; background: #000; font-family: 'Manrope', sans-serif; }}
                        #map {{ width: 100vw; height: 100vh; position: absolute; }}
                        #ui {{ position: absolute; top: 20px; left: 20px; background: rgba(17,17,17,0.9); padding: 20px; border-radius: 8px; color: white; border: 1px solid #333; z-index: 10; box-shadow: 0 4px 10px rgba(0,0,0,0.5); width: 280px;}}
                        button {{ background: #00D2FF; color: black; border: none; padding: 12px; cursor: pointer; font-weight: bold; border-radius: 4px; width: 100%; font-size: 14px; text-transform: uppercase; margin-bottom: 10px;}}
                        button:disabled {{ background: #444; color: #888; cursor: not-allowed; }}
                    </style>
                </head>
                <body>
                    <div id="ui">
                        <h3 style="margin: 0 0 10px 0; color: #00D2FF;">DFR Swarm Sim</h3>
                        {warn_html}
                        <div style="font-size: 13px; color: #aaa; margin-bottom: 15px;">Simulating {total_sim_flights:,} DFR flights ({int(dfr_dispatch_rate*100)}% dispatch rate of {daily_calls:,} daily calls) over a 24-hour cycle.</div>
                        
                        <div style="margin-bottom: 15px;">
                            <label style="font-size: 12px; color: #ccc;">Time Speed Multiplier: <span id="speedLabel">1</span>x</label>
                            <input type="range" id="speedSlider" min="1" max="100" value="1" style="width: 100%;">
                        </div>

                        <button id="runBtn">LAUNCH SWARM</button>
                        <div id="timeDisplay" style="font-family: monospace; font-size: 18px; color: #00ffcc; font-weight: bold; text-align: center;">00:00:00</div>
                        
                        <div style="margin-top: 15px; border-top: 1px solid #333; padding-top: 10px;">
                            <h4 style="margin: 0 0 5px 0; color: #aaa; font-size: 11px; text-transform: uppercase;">Active Stations</h4>
                            <div style="font-size: 11px; color: #ddd; max-height: 120px; overflow-y: auto;">
                                {legend_html}
                            </div>
                        </div>
                    </div>
                    <div id="map"></div>
                    <script>
                        const stations = {json.dumps(stations_json)};
                        const flights = {json.dumps(flights_json)};
                        
                        const speedSlider = document.getElementById('speedSlider');
                        const speedLabel = document.getElementById('speedLabel');
                        speedSlider.oninput = () => {{ speedLabel.innerText = speedSlider.value; }};
                        
                        const map = new deck.DeckGL({{
                            container: 'map',
                            mapStyle: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
                            initialViewState: {{
                                longitude: {lon},
                                latitude: {lat},
                                zoom: {zoom},
                                pitch: 50,
                                bearing: 0
                            }},
                            controller: true
                        }});

                        let time = 0;
                        let timer = null;
                        let lastTime = 0;
                        
                        function render() {{
                            const layers = [
                                new deck.ScatterplotLayer({{
                                    id: 'station-rings',
                                    data: stations,
                                    getPosition: d => [d.lon, d.lat],
                                    getFillColor: d => [d.color[0], d.color[1], d.color[2], 30],
                                    getLineColor: d => [d.color[0], d.color[1], d.color[2], 255],
                                    lineWidthMinPixels: 2,
                                    stroked: true,
                                    filled: true,
                                    getRadius: d => d.radius,
                                    pickable: false
                                }}),
                                new deck.ScatterplotLayer({{
                                    id: 'stations-pad',
                                    data: stations,
                                    getPosition: d => [d.lon, d.lat],
                                    getFillColor: d => [d.color[0], d.color[1], d.color[2], 100],
                                    getRadius: 200,
                                    pickable: false
                                }}),
                                new deck.IconLayer({{
                                    id: 'station-icons',
                                    data: stations,
                                    pickable: false,
                                    getIcon: d => ({{
                                        url: "{drone_svg}",
                                        width: 24,
                                        height: 24,
                                        anchorY: 12
                                    }}),
                                    getPosition: d => [d.lon, d.lat],
                                    getSize: d => 40,
                                    sizeScale: 1
                                }}),
                                new deck.TripsLayer({{
                                    id: 'flights',
                                    data: flights,
                                    getPath: d => d.path,
                                    getTimestamps: d => d.timestamps,
                                    getColor: d => d.color,
                                    opacity: 0.8,
                                    widthMinPixels: 4,
                                    trailLength: 120, 
                                    currentTime: time,
                                    rounded: true
                                }}),
                                new deck.ScatterplotLayer({{
                                    id: 'landed-calls',
                                    data: flights,
                                    getPosition: d => d.path[2],
                                    getFillColor: d => time >= d.timestamps[2] ? [d.color[0], d.color[1], d.color[2], 255] : [0, 0, 0, 0],
                                    getRadius: 25,
                                    radiusMinPixels: 3,
                                    updateTriggers: {{
                                        getFillColor: time
                                    }}
                                }})
                            ];
                            map.setProps({{layers}});
                            
                            let hrs = Math.floor(time / 3600).toString().padStart(2, '0');
                            let mins = Math.floor((time % 3600) / 60).toString().padStart(2, '0');
                            document.getElementById('timeDisplay').innerText = `Sim Time: ${{hrs}}:${{mins}}`;
                        }}

                        const animate = () => {{
                            let now = performance.now();
                            let dt = now - lastTime;
                            lastTime = now;
                            
                            if (dt > 100) dt = 16.6; 
                            
                            let timeIncrement = (dt / 1000) * 2880 * parseFloat(speedSlider.value);
                            time += timeIncrement; 
                            
                            render();
                            if (time < 86400) {{
                                timer = requestAnimationFrame(animate);
                            }} else {{
                                document.getElementById('runBtn').disabled = false;
                                document.getElementById('runBtn').innerText = "RESTART SWARM";
                                time = 0;
                            }}
                        }};

                        document.getElementById('runBtn').onclick = () => {{
                            document.getElementById('runBtn').disabled = true;
                            document.getElementById('runBtn').innerText = "SIMULATING...";
                            time = 0;
                            lastTime = performance.now();
                            if(timer) cancelAnimationFrame(timer);
                            animate();
                        }};
                        
                        render();
                    </script>
                </body>
                </html>
                """
                return html

            sim_html = generate_deckgl_html(active_drones, calls_in_city, dfr_dispatch_rate, center_lat, center_lon, dynamic_zoom, calls_per_day)
            components.html(sim_html, height=700)
