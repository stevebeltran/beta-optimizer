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
import streamlit.components.v1 as components

# --- PAGE CONFIG ---
st.set_page_config(page_title="BRINC COS Drone Optimizer", layout="wide")

# --- THEME TOGGLE ---
st.sidebar.markdown("<h3 style='margin-bottom:0px;'>🎨 Appearance</h3>", unsafe_allow_html=True)
theme_choice = st.sidebar.radio("Theme", ["Dark Mode", "Light Mode"], horizontal=True, label_visibility="collapsed")
is_dark = theme_choice == "Dark Mode"

# --- DYNAMIC THEME VARIABLES ---
if is_dark:
    # Dark Mode Palette
    bg_main = "#000000"
    bg_sidebar = "#111111"
    text_main = "#ffffff"
    text_muted = "#aaaaaa"
    accent_color = "#00D2FF" # Brinc Blue for custom cards/metrics
    
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
    
    # Custom CSS blocks for Dark Mode layout
    theme_css = f"""
    /* Overwrite Streamlit's absolute base theme variables */
    :root, .stApp, .main {{
        --primary-color: {accent_color} !important;
        background-color: {bg_main} !important; 
    }}
    
    html, body, [class*="css"], p, label, li, h1, h2, h3, h4, h5, h6 {{ 
        font-family: 'Manrope', sans-serif !important; 
        color: {text_main} !important; 
    }}
    
    [data-testid="stSidebar"] {{ 
        background-color: {bg_sidebar} !important; 
        border-right: 1px solid {card_border}; 
    }}
    
    [data-testid="stFileUploader"] p, [data-testid="stFileUploader"] small {{ 
        color: {text_muted} !important; 
    }}
    
    /* Metrics */
    div[data-testid="stMetricValue"] {{ 
        font-family: 'IBM Plex Mono', monospace !important; 
        color: {accent_color} !important; 
    }}
    div[data-testid="stMetricLabel"] * {{ 
        color: {text_muted} !important; 
    }}
    
    /* Multiselect Box Darkening */
    div[data-baseweb="select"] > div {{ background-color: #222222 !important; border-color: #444444 !important; color: #ffffff !important; }}
    div[data-baseweb="select"] > div * {{ color: #ffffff !important; }}
    div[data-baseweb="select"] span[data-baseweb="tag"] {{ background-color: #333333 !important; color: #ffffff !important; font-weight: normal; border: 1px solid #555555 !important; }}
    div[data-baseweb="select"] span[data-baseweb="tag"] * {{ color: #ffffff !important; }}
    div[data-baseweb="popover"] ul {{ background-color: #222222 !important; color: #ffffff !important; }}
    div[data-baseweb="popover"] li:hover {{ background-color: #444444 !important; }}

    /* ========================================= */
    /* SURGICAL BRINC BLUE OVERRIDES (DARK MODE) */
    /* ========================================= */

    /* 1. Sliders */
    div[data-testid="stSlider"] div[data-baseweb="slider"] div[role="slider"] {{
        background-color: {accent_color} !important;
        border-color: #ffffff !important;
    }}
    div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div > div:first-of-type {{
        background-color: {accent_color} !important;
    }}
    div[data-testid="stSlider"] div[data-baseweb="slider"] div[data-testid="stTickBar"] > div {{
        background-color: {accent_color} !important;
    }}

    /* 2. Toggles */
    div[data-testid="stToggle"] input[type="checkbox"]:checked + div {{
        background-color: {accent_color} !important;
    }}

    /* 3. Checkboxes */
    div[data-testid="stCheckbox"] input[type="checkbox"]:checked + div {{
        background-color: {accent_color} !important;
        border-color: {accent_color} !important;
    }}

    /* 4. Radio Buttons */
    div[role="radiogroup"] [role="radio"][aria-checked="true"] > div:first-of-type {{
        background-color: {accent_color} !important;
        border-color: {accent_color} !important;
    }}
    div[role="radiogroup"] [role="radio"][aria-checked="true"] > div:first-of-type > div {{
        background-color: {accent_color} !important;
    }}
    """
else:
    # Light Mode Palette (Default Streamlit Red)
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
    
    theme_css = f"""
    .stApp, .main {{ background-color: {bg_main} !important; }}
    html, body, [class*="css"], p, label, li, h1, h2, h3, h4, h5, h6 {{ font-family: 'Manrope', sans-serif !important; color: {text_main} !important; }}
    [data-testid="stSidebar"] {{ background-color: {bg_sidebar} !important; border-right: 1px solid {card_border}; }}
    [data-testid="stFileUploader"] p, [data-testid="stFileUploader"] small {{ color: {text_muted} !important; }}
    
    div[data-testid="stMetricValue"] {{ font-family: 'IBM Plex Mono', monospace !important; color: {accent_color} !important; }}
    div[data-testid="stMetricLabel"] * {{ color: {text_muted} !important; }}
    
    /* Force Multiselect to White */
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
    
    /* General Form Fonts */
    .stRadio label p, .stMultiSelect label p, .stSlider label p, .stToggle label p, .stCheckbox label p {{
        font-weight: 600 !important;
        font-size: 0.85rem !important;
    }}
    div[role="radiogroup"] {{ gap: 0.5rem !important; }}

    /* Print settings */
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

st.title("🛰️ BRINC COS Drone Optimizer")

# --- CONFIGURATION ---
SHAPEFILE_DIR = "jurisdiction_data" 
if not os.path.exists(SHAPEFILE_DIR):
    os.makedirs(SHAPEFILE_DIR)

if 'csvs_ready' not in st.session_state:
    st.session_state['csvs_ready'] = False
    st.session_state['df_calls'] = None
    st.session_state['df_stations'] = None

# --- SIDEBAR: MAP LIBRARY MANAGER (HIDDEN WHEN READY) ---
if not st.session_state['csvs_ready']:
    with st.sidebar.expander("🗺️ Map Library Manager"):
        st.write("Upload shapefiles here to populate the 'jurisdiction_data' folder.")
        map_files = st.file_uploader("Drop .shp, .shx, .dbf, .prj files", accept_multiple_files=True)
        if map_files:
            count = 0
            for f in map_files:
                with open(os.path.join(SHAPEFILE_DIR, f.name), "wb") as buffer:
                    buffer.write(f.getbuffer())
                count += 1
            st.success(f"Saved {count} map files to library!")

# --- MAIN UPLOAD SECTION (CSVs ONLY) ---
if not st.session_state['csvs_ready']:
    st.info("📁 Please upload 'calls.csv' and 'stations.csv' to begin. The map will auto-detect matching jurisdictions.")
    uploaded_files = st.file_uploader("Upload Mission Data", accept_multiple_files=True)
    
    call_file, station_file = None, None
    if uploaded_files:
        for f in uploaded_files:
            fname = f.name.lower()
            if fname == "calls.csv": call_file = f
            elif fname == "stations.csv": station_file = f
            
        if call_file and station_file:
            df_c = pd.read_csv(call_file)
            df_c.columns = [str(c).lower().strip() for c in df_c.columns]
            st.session_state['df_calls'] = df_c.dropna(subset=['lat', 'lon'])
            
            df_s = pd.read_csv(station_file)
            df_s.columns = [str(c).lower().strip() for c in df_s.columns]
            st.session_state['df_stations'] = df_s.dropna(subset=['lat', 'lon'])
            
            st.session_state['csvs_ready'] = True
            st.rerun()

# High-Visibility Neon Palette designed to pop on Dark/Light mode maps
STATION_COLORS = [
    "#00D2FF", # Brinc Blue
    "#39FF14", # Neon Green
    "#FF007F", # Bright Pink
    "#FFD700", # Cyber Yellow
    "#B026FF", # Neon Purple
    "#FF4500", # Orange Red
    "#00FFCC", # Bright Cyan
    "#FF3333", # Bright Red
    "#9D00FF", # Bright Violet
    "#7FFF00"  # Chartreuse
]

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
            line2 = parts[0].strip() + ","
            line3 = parts[1].strip()
            return f"{line1}<br>{line2}<br>{line3}"
        else:
            return f"{line1}<br>{rest}<br> "
    else:
        if ',' in name_str:
            parts = name_str.split(',')
            if len(parts) >= 3:
                return f"{parts[0].strip()},<br>{parts[1].strip()},<br>{','.join(parts[2:]).strip()}"
            return f"{name_str}<br> <br> "
        return f"{name_str}<br> <br> "

def generate_kml(active_gdf, df_stations_all, active_resp_names, active_guard_names, calls_gdf, guard_radius_mi):
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

    def add_kml_station(row, radius, color, name_prefix):
        pnt = fol_stations.newpoint(name=f"{name_prefix} {row['name']}")
        pnt.coords = [(row['lon'], row['lat'])]
        pnt.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/paddle/blu-blank.png'
        lats, lons = get_circle_coords(row['lat'], row['lon'], r_mi=radius)
        ring_coords = list(zip(lons, lats))
        ring_coords.append(ring_coords[0])
        pol = fol_rings.newpolygon(name=f"Range: {row['name']}")
        pol.outerboundaryis = ring_coords
        pol.style.linestyle.color = color
        pol.style.linestyle.width = 2
        pol.style.polystyle.color = simplekml.Color.changealphaint(60, color)

    for _, row in df_stations_all[df_stations_all['name'].isin(active_resp_names)].iterrows():
        add_kml_station(row, 2.0, simplekml.Color.blue, "[Responder]")
        
    for _, row in df_stations_all[df_stations_all['name'].isin(active_guard_names)].iterrows():
        add_kml_station(row, guard_radius_mi, simplekml.Color.orange, "[Guardian]")

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

@st.cache_data
def find_relevant_jurisdictions(calls_df, stations_df, shapefile_dir):
    points_list = []
    if calls_df is not None:
        points_list.append(calls_df[['lat', 'lon']])
    if stations_df is not None:
        points_list.append(stations_df[['lat', 'lon']])
    
    if not points_list: return None
    
    full_points = pd.concat(points_list)
    full_points = full_points[(full_points.lat.abs() > 1) & (full_points.lon.abs() > 1)]
    
    if len(full_points) > 50000:
        scan_points = full_points.sample(50000, random_state=42)
    else:
        scan_points = full_points

    points_gdf = gpd.GeoDataFrame(
        scan_points, 
        geometry=gpd.points_from_xy(scan_points.lon, scan_points.lat), 
        crs="EPSG:4326"
    )
    
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
                    valid_indices = hits.index.unique()
                    subset = gdf_chunk.loc[valid_indices].copy()
                    subset['data_count'] = hits.index.value_counts()
                    
                    name_col = next((c for c in ['NAME', 'DISTRICT', 'NAMELSAD'] if c in subset.columns), subset.columns[0])
                    subset['DISPLAY_NAME'] = subset[name_col].astype(str)
                    
                    relevant_polys.append(subset)
        except Exception:
            continue
            
    if not relevant_polys: return None
        
    master_gdf = pd.concat(relevant_polys, ignore_index=True)
    master_gdf = master_gdf.sort_values(by='data_count', ascending=False)
    
    total_scanned_points = master_gdf['data_count'].sum()
    
    if total_scanned_points > 0:
        master_gdf['pct_share'] = master_gdf['data_count'] / total_scanned_points
        master_gdf['cum_share'] = master_gdf['pct_share'].cumsum()
        mask = (master_gdf['cum_share'] <= 0.98) | (master_gdf['pct_share'] > 0.01)
        mask.iloc[0] = True
        return master_gdf[mask]
    
    return master_gdf

@st.cache_resource
def precompute_spatial_data(df_calls, df_stations_all, city_m_wkt, epsg_code, guard_radius_mi):
    city_m = shapely.wkt.loads(city_m_wkt)
    
    gdf_calls = gpd.GeoDataFrame(df_calls, geometry=gpd.points_from_xy(df_calls.lon, df_calls.lat), crs="EPSG:4326")
    gdf_calls_utm = gdf_calls.to_crs(epsg=epsg_code)
    
    try:
        calls_in_city = gdf_calls_utm[gdf_calls_utm.within(city_m)]
    except:
        calls_in_city = gdf_calls_utm
        
    radius_resp_m = 3218.69   
    radius_guard_m = guard_radius_mi * 1609.34 
    
    station_metadata = []
    total_calls = len(calls_in_city)
    n = len(df_stations_all)
    
    resp_matrix = np.zeros((n, total_calls), dtype=bool)
    guard_matrix = np.zeros((n, total_calls), dtype=bool)
    
    if not calls_in_city.empty:
        display_calls = calls_in_city.sample(min(5000, total_calls), random_state=42).to_crs(epsg=4326)
    else:
        display_calls = gpd.GeoDataFrame()
    
    if not calls_in_city.empty:
        calls_array = np.array(list(zip(calls_in_city.geometry.x, calls_in_city.geometry.y)))
        
        for i, row in df_stations_all.iterrows():
            s_pt_m = gpd.GeoSeries([Point(row['lon'], row['lat'])], crs="EPSG:4326").to_crs(epsg=epsg_code).iloc[0]
            
            dists = np.sqrt((calls_array[:,0] - s_pt_m.x)**2 + (calls_array[:,1] - s_pt_m.y)**2)
            dists_mi = dists / 1609.34
            
            mask_r = dists <= radius_resp_m
            mask_g = dists <= radius_guard_m
            
            resp_matrix[i, :] = mask_r
            guard_matrix[i, :] = mask_g

            full_buf_2m = s_pt_m.buffer(radius_resp_m)
            try: clipped_2m = full_buf_2m.intersection(city_m)
            except: clipped_2m = full_buf_2m

            full_buf_guard = s_pt_m.buffer(radius_guard_m)
            try: clipped_guard = full_buf_guard.intersection(city_m)
            except: clipped_guard = full_buf_guard
            
            avg_dist_r = dists_mi[mask_r].mean() if mask_r.any() else (2.0 * (2/3))
            avg_dist_g = dists_mi[mask_g].mean() if mask_g.any() else (guard_radius_mi * (2/3))
                
            station_metadata.append({
                'name': row['name'], 'lat': row['lat'], 'lon': row['lon'],
                'clipped_2m': clipped_2m, 'clipped_guard': clipped_guard,
                'avg_dist_r': avg_dist_r, 'avg_dist_g': avg_dist_g
            })
            
    return calls_in_city, display_calls, resp_matrix, guard_matrix, station_metadata, total_calls

def solve_mclp(resp_matrix, guard_matrix, num_resp, num_guard, allow_redundancy, tb_area_r, tb_area_g, tb_cent, incremental=True):
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
            for s in range(n_stations):
                model += x_r[s] + x_g[s] <= 1

        if allow_redundancy:
            y_r = pulp.LpVariable.dicts("cl_r", range(n_u), 0, 1, pulp.LpBinary)
            y_g = pulp.LpVariable.dicts("cl_g", range(n_u), 0, 1, pulp.LpBinary)
            
            primary_obj = pulp.lpSum(y_r[i] * weights[i] + y_g[i] * weights[i] for i in range(n_u))
            
            for i in range(n_u):
                cover_r = [x_r[s] for s in range(n_stations) if u_resp[s, i]]
                cover_g = [x_g[s] for s in range(n_stations) if u_guard[s, i]]
                
                if cover_r: model += y_r[i] <= pulp.lpSum(cover_r)
                else: model += y_r[i] == 0
                
                if cover_g: model += y_g[i] <= pulp.lpSum(cover_g)
                else: model += y_g[i] == 0
                
        else:
            y = pulp.LpVariable.dicts("cl", range(n_u), 0, 1, pulp.LpBinary)
            primary_obj = pulp.lpSum(y[i] * weights[i] for i in range(n_u))

            for i in range(n_u):
                cover = []
                for s in range(n_stations):
                    if u_resp[s, i]:
                        cover.append(x_r[s])
                    if u_guard[s, i]:
                        cover.append(x_g[s])
                
                if cover:
                    model += y[i] <= pulp.lpSum(cover)
                else:
                    model += y[i] == 0

        n_drones_step = target_r + target_g
        if n_drones_step == 0: n_drones_step = 1
        area_weight = 0.4 / n_drones_step
        cent_weight = 0.05 / n_drones_step
        
        tie_breaker_obj = pulp.lpSum(
            x_r[s] * (tb_area_r[s] * area_weight + tb_cent[s] * cent_weight) +
            x_g[s] * (tb_area_g[s] * area_weight + tb_cent[s] * cent_weight)
            for s in range(n_stations)
        )

        model += primary_obj + tie_breaker_obj

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

# --- MAIN LOGIC ---
if st.session_state['csvs_ready']:
    df_calls = st.session_state['df_calls'].copy()
    df_stations_all = st.session_state['df_stations'].copy()

    with st.spinner("🌍 Identifying dominant jurisdictions..."):
        master_gdf = find_relevant_jurisdictions(df_calls, df_stations_all, SHAPEFILE_DIR)

    if master_gdf is None or master_gdf.empty:
        st.error("❌ No matching jurisdictions found.")
        st.stop()

    st.sidebar.success(f"**Found {len(master_gdf)} Significant Zones**")
    
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

    # --- DYNAMIC MISSION DATA FILTERS ---
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"<h3 style='margin-bottom:0px; color:{text_main};'>⚙️ Data Filters</h3>", unsafe_allow_html=True)
    
    if 'type' in df_stations_all.columns:
        all_types = sorted(df_stations_all['type'].dropna().astype(str).unique().tolist())
        if all_types:
            st.sidebar.markdown(f"<div style='font-size:0.75rem; color:{text_muted}; font-weight:800; margin-top:15px; margin-bottom:5px; text-transform:uppercase;'>Facility Type</div>", unsafe_allow_html=True)
            selected_types = st.sidebar.multiselect("Facility Type", options=all_types, default=all_types, label_visibility="collapsed")
            if not selected_types:
                st.warning("Please select at least one Facility Type from the sidebar.")
                st.stop()
            df_stations_all = df_stations_all[df_stations_all['type'].astype(str).isin(selected_types)].copy()
            df_stations_all['name'] = "[" + df_stations_all['type'].astype(str) + "] " + df_stations_all['name'].astype(str)
            
    if 'priority' in df_calls.columns:
        all_priorities = sorted(df_calls['priority'].dropna().unique().tolist())
        if all_priorities:
            st.sidebar.markdown(f"<div style='font-size:0.75rem; color:{text_muted}; font-weight:800; margin-top:15px; margin-bottom:5px; text-transform:uppercase;'>Incident Priority</div>", unsafe_allow_html=True)
            selected_priorities = st.sidebar.multiselect("Incident Priority", options=all_priorities, default=all_priorities, label_visibility="collapsed")
            if not selected_priorities:
                st.warning("Please select at least one Incident Priority from the sidebar.")
                st.stop()
            df_calls = df_calls[df_calls['priority'].isin(selected_priorities)].copy()

    if len(df_stations_all) == 0:
        st.error("No stations match the selected filters.")
        st.stop()
        
    if len(df_calls) == 0:
        st.error("No calls match the selected filters.")
        st.stop()

    n = len(df_stations_all)

    # --- OPTIMIZER CONTROLS ---
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"<h3 style='margin-bottom:0px; color:{text_main};'>🎯 Optimizer Controls</h3>", unsafe_allow_html=True)

    st.sidebar.markdown(f"<div style='font-size:0.75rem; color:{text_muted}; font-weight:800; margin-top:15px; margin-bottom:5px; text-transform:uppercase;'>Optimization Goal</div>", unsafe_allow_html=True)
    opt_strategy_raw = st.sidebar.radio(
        "Goal", 
        ("Call Coverage", "Land Coverage"), 
        horizontal=True,
        label_visibility="collapsed"
    )
    opt_strategy = "Maximize Call Coverage" if opt_strategy_raw == "Call Coverage" else "Maximize Land Coverage"
    
    st.sidebar.markdown(f"<div style='font-size:0.75rem; color:{text_muted}; font-weight:800; margin-top:15px; margin-bottom:5px; text-transform:uppercase;'>Fleet Configuration</div>", unsafe_allow_html=True)
    k_responder = st.sidebar.slider("🚁 Responder Count", 0, n, min(1, n))
    k_guardian = st.sidebar.slider("🦅 Guardian Count", 0, n, 0)
    
    st.sidebar.markdown(f"<div style='font-size:0.75rem; color:{text_muted}; font-weight:800; margin-top:15px; margin-bottom:5px; text-transform:uppercase;'>Guardian Range</div>", unsafe_allow_html=True)
    guard_radius_mi = st.sidebar.slider("🦅 Guardian Range (Miles)", 1, 8, 8, label_visibility="collapsed")

    with st.spinner("⚡ Precomputing spatial optimization matrices..."):
        city_m_wkt = city_m.wkt  
        calls_in_city, display_calls, resp_matrix, guard_matrix, station_metadata, total_calls = precompute_spatial_data(
            df_calls, df_stations_all, city_m_wkt, epsg_code, guard_radius_mi
        )

    max_dist = max([((s['lon'] - center_lon)**2 + (s['lat'] - center_lat)**2)**0.5 for s in station_metadata])
    if max_dist == 0: max_dist = 1.0
    
    for s in station_metadata:
        dist = ((s['lon'] - center_lon)**2 + (s['lat'] - center_lat)**2)**0.5
        s['centrality'] = 1.0 - (dist / max_dist)

    max_area = city_m.area if (city_m and city_m.area > 0) else 1.0
    tb_area_r = [s['clipped_2m'].area / max_area for s in station_metadata]
    tb_area_g = [s['clipped_guard'].area / max_area for s in station_metadata]
    tb_cent = [s['centrality'] for s in station_metadata]
    
    st.sidebar.markdown(f"<div style='font-size:0.75rem; color:{text_muted}; font-weight:800; margin-top:15px; margin-bottom:5px; text-transform:uppercase;'>Deployment Strategy</div>", unsafe_allow_html=True)
    incremental_build = st.sidebar.toggle(
        "Phased Rollout", 
        value=True, 
        help="When ON, builds the fleet one-by-one so existing stations never change."
    )
    allow_redundancy = st.sidebar.toggle(
        "Multi-Tier (Allow Overlap)", 
        value=True, 
        help="When ON, drones won't move away just because their coverage rings overlap."
    )
    
    st.sidebar.markdown(f"<div style='font-size:0.75rem; color:{text_muted}; font-weight:800; margin-top:15px; margin-bottom:5px; text-transform:uppercase;'>Map Layers</div>", unsafe_allow_html=True)
    col1, col2 = st.sidebar.columns(2)
    show_boundaries = col1.toggle("Boundaries", value=True)
    show_heatmap = col2.toggle("Heatmap", value=False)
    show_health = col1.toggle("Health Score", value=False)
    show_satellite = col2.toggle("Satellite", value=False)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"<h3 style='margin-bottom:0px; color:{text_main};'>🚗 Ground Traffic Simulator</h3>", unsafe_allow_html=True)
    simulate_traffic = st.sidebar.toggle("Enable Traffic Sim", value=False)
    if simulate_traffic:
        traffic_level = st.sidebar.slider("Traffic Intensity (%)", 0, 100, 40)
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
                r_best, g_best, chrono_r, chrono_g = solve_mclp(resp_matrix, guard_matrix, k_responder, k_guardian, allow_redundancy, tb_area_r, tb_area_g, tb_cent, incremental=incremental_build)
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
    dfr_dispatch_rate = 0.25 # Default 25% if sliders don't render yet
    
    with budget_placeholder:
        st.markdown("---")
        st.markdown(f"<h3 style='color:{text_main};'>💰 Budget Impact</h3>", unsafe_allow_html=True)
        
        inferred_daily_calls = max(1, int(total_calls / 365)) if total_calls > 0 else 20
        max_slider_val = max(100, inferred_daily_calls * 3) 
        
        calls_per_day = st.slider("TOTAL DAILY CALLS (CITYWIDE)", min_value=1, max_value=max_slider_val, value=inferred_daily_calls)
        
        col_r1, col_r2 = st.columns(2)
        dfr_dispatch_rate = col_r1.slider("DFR DISPATCH RATE (%)", min_value=1, max_value=100, value=25) / 100.0
        deflection_rate = col_r2.slider("DRONE-ONLY RESOLUTION (%)", min_value=0, max_value=100, value=30) / 100.0
        
        cost_officer = 82
        cost_drone = 6
        savings_per_call = cost_officer - cost_drone
        
        actual_k_responder = len(active_resp_names)
        actual_k_guardian = len(active_guard_names)
        
        capex_responder_total = actual_k_responder * 80000
        capex_guardian_total = actual_k_guardian * 160000
        fleet_capex = capex_responder_total + capex_guardian_total
        
        if fleet_capex > 0:
            effective_coverage_rate = calls_covered_perc / 100.0
            covered_daily_calls = calls_per_day * effective_coverage_rate
            daily_dfr_responses = covered_daily_calls * dfr_dispatch_rate
            daily_drone_only_calls = daily_dfr_responses * deflection_rate
            
            if daily_drone_only_calls > 0:
                monthly_savings = savings_per_call * daily_drone_only_calls * 30.4
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
                    <div style="color: {text_muted}; font-size: 0.75rem;">COVERAGE: <span style="color:{text_main}; font-weight:600;">2 MI RADIUS</span></div>
                    <div style="color: {text_muted}; font-size: 0.75rem;">UNIT CAPEX: <span style="color:{text_main}; font-weight:600;">$80,000</span></div>
                    <div style="color: {text_muted}; font-size: 0.75rem; margin-top: 4px; border-top: 1px solid {card_border}; padding-top: 4px;">SUBTOTAL: <span style="color:{text_main}; font-weight:600;">${capex_responder_total:,.0f}</span></div>
                </div>
                """, unsafe_allow_html=True)
                
            if actual_k_guardian > 0:
                st.markdown(f"""
                <div style="background-color: {card_bg}; border: 1px solid {card_border}; padding: 10px; border-radius: 4px; margin-bottom: 8px;">
                    <h5 style="color: {text_main}; margin: 0 0 4px 0; font-size: 0.85rem;">GUARDIAN <span style="color:{text_muted}; font-weight:normal;">(x{actual_k_guardian})</span></h5>
                    <div style="color: {text_muted}; font-size: 0.75rem;">COVERAGE: <span style="color:{text_main}; font-weight:600;">{guard_radius_mi} MI RADIUS</span></div>
                    <div style="color: {text_muted}; font-size: 0.75rem;">UNIT CAPEX: <span style="color:{text_main}; font-weight:600;">$160,000</span></div>
                    <div style="color: {text_muted}; font-size: 0.75rem; margin-top: 4px; border-top: 1px solid {card_border}; padding-top: 4px;">SUBTOTAL: <span style="color:{text_main}; font-weight:600;">${capex_guardian_total:,.0f}</span></div>
                </div>
                """, unsafe_allow_html=True)

            ordered_deployments = []
            for idx in chrono_g:
                if idx in active_guard_idx: ordered_deployments.append((idx, 'GUARDIAN'))
            for idx in chrono_r:
                if idx in active_resp_idx: ordered_deployments.append((idx, 'RESPONDER'))

            for idx in active_resp_idx:
                if idx not in chrono_r: ordered_deployments.append((idx, 'RESPONDER'))
            for idx in active_guard_idx:
                if idx not in chrono_g: ordered_deployments.append((idx, 'GUARDIAN'))

            cumulative_mask = np.zeros(total_calls, dtype=bool) if total_calls > 0 else None
            
            step = 1
            for idx, d_type in ordered_deployments:
                if d_type == 'RESPONDER':
                    cov_array = resp_matrix[idx]
                    cost = 80000
                    speed_mph = 42.0
                    avg_dist = station_metadata[idx]['avg_dist_r']
                else:
                    cov_array = guard_matrix[idx]
                    cost = 160000
                    speed_mph = 60.0
                    avg_dist = station_metadata[idx]['avg_dist_g']
                    
                map_color = STATION_COLORS[idx % len(STATION_COLORS)]
                
                avg_time_min = (avg_dist / speed_mph) * 60

                d = {
                    'name': station_metadata[idx]['name'],
                    'lat': station_metadata[idx]['lat'],
                    'lon': station_metadata[idx]['lon'],
                    'type': d_type,
                    'cost': cost,
                    'cov_array': cov_array,
                    'color': map_color,
                    'deploy_step': step if (idx in chrono_r or idx in chrono_g) else "MANUAL",
                    'avg_time_min': avg_time_min,
                    'speed_mph': speed_mph
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

                    d['monthly_savings'] = savings_per_call * d['marginal_deflected'] * 30.4
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
            eval_speed = 60.0
            gain_label = f"Efficiency Gain ({guard_radius_mi}-mi)"
        else:
            eval_dist = 2.0
            eval_speed = 42.0
            gain_label = "Efficiency Gain (2-mi)"

        avg_ground_speed = 35 * (1 - (traffic_level / 100))
        
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
        df_stations_all, 
        active_resp_names,
        active_guard_names,
        calls_in_city,
        guard_radius_mi
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
        
        def calculate_zoom(min_lon, max_lon, min_lat, max_lat):
            lon_diff = max_lon - min_lon
            lat_diff = max_lat - min_lat
            if lon_diff <= 0 or lat_diff <= 0: return 12
            zoom_lon = np.log2(360 / lon_diff)
            zoom_lat = np.log2(180 / lat_diff)
            best_zoom = min(zoom_lon, zoom_lat) + 1.6
            return min(max(best_zoom, 5), 18)

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

        for i, row in df_stations_all.iterrows():
            s_name = row['name']
            color = STATION_COLORS[i % len(STATION_COLORS)]
            
            short_name = s_name.split(',')[0]

            if s_name in active_resp_names:
                clats, clons = get_circle_coords(row['lat'], row['lon'], r_mi=2.0)
                lbl = f"{short_name} (Resp)"
                drive_time_min = (2.0 / 42.0) * 60 
            elif s_name in active_guard_names:
                clats, clons = get_circle_coords(row['lat'], row['lon'], r_mi=guard_radius_mi)
                lbl = f"{short_name} (Guard)"
                drive_time_min = (guard_radius_mi / 60.0) * 60 
            else:
                continue

            fig.add_trace(go.Scattermapbox(
                lat=list(clats) + [None, row['lat']], 
                lon=list(clons) + [None, row['lon']], 
                mode='lines+markers', 
                marker=dict(size=[0]*len(clats) + [0, 20], color=color), 
                line=dict(color=color, width=4.5), 
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

                ground_speed_mph = 35 * (1 - (traffic_level / 100))
                
                if ground_speed_mph > 0:
                    ground_range_mi = (ground_speed_mph / 60) * drive_time_min
                    g_angles = np.linspace(0, 2*np.pi, 9) 
                    g_lats = row['lat'] + (ground_range_mi/69.172) * np.sin(g_angles)
                    g_lons = row['lon'] + (ground_range_mi/(69.172 * np.cos(np.radians(row['lat'])))) * np.cos(g_angles)

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

        dynamic_zoom = calculate_zoom(minx, maxx, miny, maxy)

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
        st.info(f"Watch the deployed drones respond to the assigned 911 calls. The historical volume has been procedurally distributed over a simulated 24-hour timeline. Drones fly true 3D arcs out to the incident.")
        
        show_sim = st.toggle("🎬 Enable 3D Swarm Simulation", value=False)
        
        if show_sim:
            def generate_deckgl_html(active_drones, calls_in_city, dfr_dispatch_rate):
                stations_json = []
                flights_json = []
                
                calls_coords = np.column_stack((calls_in_city['lon'], calls_in_city['lat']))
                
                legend_html = ""
                
                for d in active_drones:
                    short_name = d['name'].split(',')[0]
                    stations_json.append({
                        "name": short_name,
                        "lon": d['lon'],
                        "lat": d['lat'],
                        "color": d['color']
                    })
                    
                    legend_html += f'<div style="margin-bottom:3px;"><span style="display:inline-block;width:10px;height:10px;background-color:{d["color"]};margin-right:8px;border-radius:50%;"></span>{short_name}</div>'
                    
                    hex_c = d['color'].lstrip('#')
                    rgb = [int(hex_c[j:j+2], 16) for j in (0, 2, 4)]
                    
                    assigned_calls = d.get('assigned_indices', [])
                    num_to_simulate = int(len(assigned_calls) * dfr_dispatch_rate)
                    if num_to_simulate > 0:
                        assigned_calls = random.sample(list(assigned_calls), num_to_simulate)
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
                total_flights = len(flights_json)
                if total_flights > 2000:
                    flights_json = random.sample(flights_json, 2000)
                    warn_html = f'<div style="background: #440000; border: 1px solid #ff4b4b; color: #ffbbbb; padding: 5px; font-size: 10px; border-radius: 4px; margin-bottom: 10px;">⚠️ Visuals capped at 2,000 flights for performance (Total: {total_flights}).</div>'
                        
                html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <script src="https://unpkg.com/deck.gl@latest/dist.min.js"></script>
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
                        <div style="font-size: 13px; color: #aaa; margin-bottom: 15px;">Simulating {len(flights_json)} net-new flights (approx {int(dfr_dispatch_rate*100)}% dispatch rate) over a 24-hour cycle.</div>
                        
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
                                longitude: stations.length > 0 ? stations[0].lon : -90,
                                latitude: stations.length > 0 ? stations[0].lat : 38,
                                zoom: 10.5,
                                pitch: 50,
                                bearing: 0
                            }},
                            controller: true
                        }});

                        let time = 0;
                        let timer = null;
                        
                        function render() {{
                            const layers = [
                                new deck.ScatterplotLayer({{
                                    id: 'stations',
                                    data: stations,
                                    getPosition: d => [d.lon, d.lat],
                                    getFillColor: [255, 255, 255, 255],
                                    getRadius: 150,
                                    pickable: true
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

                        document.getElementById('runBtn').onclick = () => {{
                            document.getElementById('runBtn').disabled = true;
                            document.getElementById('runBtn').innerText = "SIMULATING...";
                            time = 0;
                            if(timer) cancelAnimationFrame(timer);
                            
                            const animate = () => {{
                                time += parseInt(speedSlider.value); 
                                render();
                                if (time < 86400) {{
                                    timer = requestAnimationFrame(animate);
                                }} else {{
                                    document.getElementById('runBtn').disabled = false;
                                    document.getElementById('runBtn').innerText = "RESTART SWARM";
                                    time = 0;
                                }}
                            }};
                            animate();
                        }};
                        
                        render();
                    </script>
                </body>
                </html>
                """
                return html

            sim_html = generate_deckgl_html(active_drones, calls_in_city, dfr_dispatch_rate)
            components.html(sim_html, height=700)
