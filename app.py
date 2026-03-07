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

# --- PAGE CONFIG ---
st.set_page_config(page_title="brinc COS Drone Optimizer", layout="wide")

# --- CUSTOM CSS ---
st.markdown(
    """
    <style>
    html, body, [class*="css"]  { font-size: 18px !important; }
    div[role="radiogroup"] label div { font-size: 20px !important; }
    .stRadio label p, .stMultiSelect label p { font-size: 22px !important; font-weight: bold !important; }
    div[data-baseweb="select"] span { font-size: 18px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- CONFIGURATION ---
SHAPEFILE_DIR = "jurisdiction_data" 
if not os.path.exists(SHAPEFILE_DIR):
    os.makedirs(SHAPEFILE_DIR)

# --- HELPER FUNCTIONS ---
def get_circle_coords(lat, lon, r_mi=2.0):
    angles = np.linspace(0, 2*np.pi, 100)
    c_lats = lat + (r_mi/69.172) * np.sin(angles)
    c_lons = lon + (r_mi/(69.172 * np.cos(np.radians(lat)))) * np.cos(angles)
    return c_lats, c_lons

def get_diamond_coords(lat, lon, r_mi=2.0):
    """Generates diamond coordinates to simulate grid-based ground travel."""
    angles = np.array([0, np.pi/2, np.pi, 3*np.pi/2, 0])
    g_lats = lat + (r_mi/69.172) * np.sin(angles)
    g_lons = lon + (r_mi/(69.172 * np.cos(np.radians(lat)))) * np.cos(angles)
    return g_lats, g_lons

# --- KML EXPORT ---
def generate_kml(active_gdf, df_stations_all, active_resp_names, active_guard_names):
    kml = simplekml.Kml()
    fol_stations = kml.newfolder(name="Stations Points")
    fol_rings = kml.newfolder(name="Coverage Rings")

    def add_kml_station(row, radius, color, name_prefix):
        pnt = fol_stations.newpoint(name=f"{name_prefix} {row['name']}")
        pnt.coords = [(row['lon'], row['lat'])]
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
        add_kml_station(row, 8.0, simplekml.Color.orange, "[Guardian]")

    return kml.kml()

# --- INTELLIGENT SCANNER ---
@st.cache_data
def find_relevant_jurisdictions(calls_df, stations_df, shapefile_dir):
    points_list = []
    if calls_df is not None: points_list.append(calls_df[['lat', 'lon']])
    if stations_df is not None: points_list.append(stations_df[['lat', 'lon']])
    if not points_list: return None
    
    full_points = pd.concat(points_list)
    full_points = full_points[(full_points.lat.abs() > 1) & (full_points.lon.abs() > 1)]
    scan_points = full_points.sample(min(50000, len(full_points)), random_state=42)

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
                    valid_indices = hits.index.unique()
                    subset = gdf_chunk.loc[valid_indices].copy()
                    subset['data_count'] = hits.index.value_counts()
                    name_col = next((c for c in ['NAME', 'DISTRICT', 'NAMELSAD'] if c in subset.columns), subset.columns[0])
                    subset['DISPLAY_NAME'] = subset[name_col].astype(str)
                    relevant_polys.append(subset)
        except: continue
            
    if not relevant_polys: return None
    return pd.concat(relevant_polys, ignore_index=True).sort_values(by='data_count', ascending=False)

# --- CACHED SPATIAL MATH ---
@st.cache_resource
def precompute_spatial_data(df_calls, df_stations_all, city_m_wkt, epsg_code):
    city_m = shapely.wkt.loads(city_m_wkt)
    gdf_calls = gpd.GeoDataFrame(df_calls, geometry=gpd.points_from_xy(df_calls.lon, df_calls.lat), crs="EPSG:4326").to_crs(epsg=epsg_code)
    try: calls_in_city = gdf_calls[gdf_calls.within(city_m)]
    except: calls_in_city = gdf_calls
        
    radius_resp_m, radius_guard_m = 3218.69, 12874.75
    n = len(df_stations_all)
    total_calls = len(calls_in_city)
    
    resp_matrix = np.zeros((n, total_calls), dtype=bool)
    guard_matrix = np.zeros((n, total_calls), dtype=bool)
    station_metadata = []

    if not calls_in_city.empty:
        display_calls = calls_in_city.sample(min(5000, total_calls), random_state=42).to_crs(epsg=4326)
        calls_array = np.array(list(zip(calls_in_city.geometry.x, calls_in_city.geometry.y)))
        
        for i, row in df_stations_all.iterrows():
            s_pt_m = gpd.GeoSeries([Point(row['lon'], row['lat'])], crs="EPSG:4326").to_crs(epsg=epsg_code).iloc[0]
            dists = np.sqrt((calls_array[:,0] - s_pt_m.x)**2 + (calls_array[:,1] - s_pt_m.y)**2)
            resp_matrix[i, :] = dists <= radius_resp_m
            guard_matrix[i, :] = dists <= radius_guard_m
            station_metadata.append({
                'name': row['name'], 'lat': row['lat'], 'lon': row['lon'],
                'clipped_2m': s_pt_m.buffer(radius_resp_m).intersection(city_m),
                'clipped_8m': s_pt_m.buffer(radius_guard_m).intersection(city_m)
            })
    return calls_in_city, display_calls, resp_matrix, guard_matrix, station_metadata, total_calls

# --- HIGH-SPEED EXACT OPTIMIZER ---
def solve_mclp(resp_matrix, guard_matrix, num_resp, num_guard):
    n_stations, n_calls = resp_matrix.shape
    if n_calls == 0: return [], []
    df_profiles = pd.DataFrame(resp_matrix.T).astype(int).astype(str)
    df_profiles['g'] = pd.DataFrame(guard_matrix.T).astype(int).astype(str).agg(''.join, axis=1)
    df_profiles['r'] = df_profiles.drop(columns='g').agg(''.join, axis=1)
    weights = df_profiles.groupby(['r', 'g']).size().values
    unique_idx = df_profiles.groupby(['r', 'g']).head(1).index
    u_resp, u_guard = resp_matrix[:, unique_idx], guard_matrix[:, unique_idx]
    
    model = pulp.LpProblem("DroneCoverage", pulp.LpMaximize)
    x_r = pulp.LpVariable.dicts("r_st", range(n_stations), 0, 1, pulp.LpBinary)
    x_g = pulp.LpVariable.dicts("g_st", range(n_stations), 0, 1, pulp.LpBinary)
    y = pulp.LpVariable.dicts("cl", range(len(weights)), 0, 1, pulp.LpBinary)
    model += pulp.lpSum(y[i] * weights[i] for i in range(len(weights)))
    model += pulp.lpSum(x_r[i] for i in range(n_stations)) <= num_resp
    model += pulp.lpSum(x_g[i] for i in range(n_stations)) <= num_guard
    for s in range(n_stations): model += x_r[s] + x_g[s] <= 1
    for i in range(len(weights)):
        cover = [x_r[s] for s in range(n_stations) if u_resp[s, i]] + [x_g[s] for s in range(n_stations) if u_guard[s, i]]
        if cover: model += y[i] <= pulp.lpSum(cover)
    model.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=10))
    return [i for i in range(n_stations) if pulp.value(x_r[i]) == 1], [i for i in range(n_stations) if pulp.value(x_g[i]) == 1]

# --- MAIN APP ---
st.title("🛰️ BRINC COS Drone Optimizer")

uploaded_files = st.file_uploader("Upload 'calls.csv' and 'stations.csv'", accept_multiple_files=True)
call_data = next((f for f in uploaded_files if f.name.lower() == "calls.csv"), None)
station_data = next((f for f in uploaded_files if f.name.lower() == "stations.csv"), None)

if call_data and station_data:
    df_calls = pd.read_csv(call_data).dropna(subset=['lat', 'lon'])
    df_stations_all = pd.read_csv(station_data).dropna(subset=['lat', 'lon'])
    master_gdf = find_relevant_jurisdictions(df_calls, df_stations_all, SHAPEFILE_DIR)

    if master_gdf is not None:
        ctrl_col1, ctrl_col2 = st.columns([1, 2])
        selected_labels = ctrl_col1.multiselect("📍 Active Jurisdictions", options=master_gdf['DISPLAY_NAME'].tolist(), default=master_gdf['DISPLAY_NAME'].tolist())
        
        if selected_labels:
            active_gdf = master_gdf[master_gdf['DISPLAY_NAME'].isin(selected_labels)]
            minx, miny, maxx, maxy = active_gdf.to_crs(epsg=4326).total_bounds
            center_lon, center_lat = (minx + maxx) / 2, (miny + maxy) / 2
            epsg_code = f"326{int((center_lon + 180) / 6) + 1}" if center_lat > 0 else f"327{int((center_lon + 180) / 6) + 1}"
            
            city_m = active_gdf.to_crs(epsg=epsg_code).geometry.buffer(0.1).unary_union.buffer(-0.1)
            calls_in_city, display_calls, resp_matrix, guard_matrix, station_metadata, total_calls = precompute_spatial_data(df_calls, df_stations_all, city_m.wkt, epsg_code)
            
            # --- SIDEBAR: OPTIMIZER & TRAFFIC ---
            st.sidebar.header("🎯 Optimizer Controls")
            k_res = st.sidebar.slider("🚁 Responders", 0, len(df_stations_all), 1)
            k_gua = st.sidebar.slider("🦅 Guardians", 0, len(df_stations_all), 0)
            
            st.sidebar.markdown("---")
            st.sidebar.subheader("🚗 Ground Traffic Simulator")
            show_traffic = st.sidebar.toggle("Show Ground Response Gap", value=False)
            traffic_lvl = st.sidebar.slider("Traffic Intensity (%)", 0, 100, 40)
            
            with st.spinner("🧠 Calculating Optimal Deployment..."):
                r_idx, g_idx = solve_mclp(resp_matrix, guard_matrix, k_res, k_gua)
                best_r = [station_metadata[i]['name'] for i in r_idx]
                best_g = [station_metadata[i]['name'] for i in g_idx]
            
            active_resp_names = ctrl_col2.multiselect("🚁 Active Responders", options=df_stations_all['name'].tolist(), default=best_r)
            active_guard_names = ctrl_col2.multiselect("🦅 Active Guardians", options=df_stations_all['name'].tolist(), default=best_g)

            # --- CALCULATE TIME SAVINGS ---
            drone_speed = 60 # mph
            ground_speed = 35 * (1 - (traffic_lvl / 100))
            t_saved = ((2 * 1.4)/ground_speed * 60) - (2/drone_speed * 60) if ground_speed > 0 else 99
            st.sidebar.info(f"⏱️ **Efficiency Gain:** Drones arrive **{t_saved:.1f} minutes** faster than ground units.")

            # --- MAP ---
            fig = go.Figure()
            if not display_calls.empty:
                fig.add_trace(go.Scattermapbox(lat=display_calls.geometry.y, lon=display_calls.geometry.x, mode='markers', marker=dict(size=4, color='#000080', opacity=0.3), name="Incidents"))
            
            for i, row in df_stations_all.iterrows():
                if row['name'] in active_resp_names: radius, color, label = 2.0, "blue", "Responder"
                elif row['name'] in active_guard_names: radius, color, label = 8.0, "orange", "Guardian"
                else: continue
                
                # Drone Circle
                clats, clons = get_circle_coords(row['lat'], row['lon'], r_mi=radius)
                fig.add_trace(go.Scattermapbox(lat=list(clats), lon=list(clons), mode='lines', line=dict(color=color, width=3), fill='toself', fillcolor=f"rgba{tuple(list(pd.to_numeric(pd.Series(color))))+(0.1,)}", name=f"{row['name']} ({label})"))
                
                # Traffic Diamond (Ground Reach)
                if show_traffic:
                    ground_range = (ground_speed / 60) * (2 if label == "Responder" else 8)
                    glats, glons = get_diamond_coords(row['lat'], row['lon'], r_mi=ground_range)
                    fig.add_trace(go.Scattermapbox(lat=list(glats), lon=list(glons), mode='lines', line=dict(color='red', width=2, dash='dash'), fill='toself', fillcolor='rgba(255, 0, 0, 0.1)', name="Ground Reach (Equal Time)"))

            fig.update_layout(mapbox=dict(style="open-street-map", center=dict(lat=center_lat, lon=center_lon), zoom=11), margin=dict(l=0,r=0,t=0,b=0), height=800)
            st.plotly_chart(fig, use_container_width=True)

            kml_out = generate_kml(active_gdf, df_stations_all, active_resp_names, active_guard_names)
            st.sidebar.download_button("🌏 Download KML", kml_out, "deployment.kml")

else:
    st.info("👋 Upload CSV data to begin.")
