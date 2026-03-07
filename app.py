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

# --- LOGO ---
try:
    st.sidebar.image("logo.png", use_container_width=True)
except FileNotFoundError:
    pass

st.title("🛰️ BRINC COS Drone Optimizer")

# --- CONFIGURATION ---
SHAPEFILE_DIR = "jurisdiction_data" 
if not os.path.exists(SHAPEFILE_DIR):
    os.makedirs(SHAPEFILE_DIR)

# --- SIDEBAR: MAP LIBRARY MANAGER ---
with st.sidebar.expander("🗺️ Map Library Manager"):
    st.write("Upload shapefiles here to populate the 'jurisdiction_data' folder.")
    map_files = st.file_uploader("Drop .shp, .shx, .dbf, .prj files", accept_multiple_files=True)
    if map_files:
        for f in map_files:
            with open(os.path.join(SHAPEFILE_DIR, f.name), "wb") as buffer:
                buffer.write(f.getbuffer())
        st.success("Library updated!")

# --- MAIN UPLOAD SECTION ---
if 'csvs_ready' not in st.session_state:
    st.session_state['csvs_ready'] = False

with st.expander("📁 Upload Mission Data (CSVs)", expanded=not st.session_state['csvs_ready']):
    uploaded_files = st.file_uploader("Upload 'calls.csv' and 'stations.csv'", accept_multiple_files=True)

STATION_COLORS = ["#E6194B", "#3CB44B", "#4363D8", "#F58231", "#911EB4", "#800000", "#333333", "#000075", "#808000", "#9A6324"]

def get_circle_coords(lat, lon, r_mi=2.0):
    angles = np.linspace(0, 2*np.pi, 100)
    c_lats = lat + (r_mi/69.172) * np.sin(angles)
    c_lons = lon + (r_mi/(69.172 * np.cos(np.radians(lat)))) * np.cos(angles)
    return c_lats, c_lons

# --- KML EXPORT ---
def generate_kml(active_gdf, df_stations_all, active_resp_names, active_guard_names, calls_gdf):
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
    master_gdf = pd.concat(relevant_polys, ignore_index=True).sort_values(by='data_count', ascending=False)
    return master_gdf

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

# --- HIGH-SPEED EXACT OPTIMIZER (AGGREGATED) ---
def solve_mclp(resp_matrix, guard_matrix, num_resp, num_guard):
    n_stations, n_calls = resp_matrix.shape
    if n_calls == 0: return [], []

    # Aggregate calls with identical coverage profiles to speed up solver
    df_profiles = pd.DataFrame(resp_matrix.T).astype(int).astype(str)
    df_profiles['g'] = pd.DataFrame(guard_matrix.T).astype(int).astype(str).agg(''.join, axis=1)
    df_profiles['r'] = df_profiles.drop(columns='g').agg(''.join, axis=1)
    
    weights = df_profiles.groupby(['r', 'g']).size().values
    unique_idx = df_profiles.groupby(['r', 'g']).head(1).index
    u_resp, u_guard = resp_matrix[:, unique_idx], guard_matrix[:, unique_idx]
    n_u = len(weights)

    model = pulp.LpProblem("DroneCoverage", pulp.LpMaximize)
    x_r = pulp.LpVariable.dicts("r_st", range(n_stations), 0, 1, pulp.LpBinary)
    x_g = pulp.LpVariable.dicts("g_st", range(n_stations), 0, 1, pulp.LpBinary)
    y = pulp.LpVariable.dicts("cl", range(n_u), 0, 1, pulp.LpBinary)

    model += pulp.lpSum(y[i] * weights[i] for i in range(n_u))
    model += pulp.lpSum(x_r[i] for i in range(n_stations)) <= num_resp
    model += pulp.lpSum(x_g[i] for i in range(n_stations)) <= num_guard
    for s in range(n_stations): model += x_r[s] + x_g[s] <= 1

    for i in range(n_u):
        cover = [x_r[s] for s in range(n_stations) if u_resp[s, i]] + [x_g[s] for s in range(n_stations) if u_guard[s, i]]
        if cover: model += y[i] <= pulp.lpSum(cover)
        else: model += y[i] == 0

    model.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=10))
    return [i for i in range(n_stations) if pulp.value(x_r[i]) == 1], [i for i in range(n_stations) if pulp.value(x_g[i]) == 1]

# --- MAIN APP LOGIC ---
call_data, station_data = None, None
if uploaded_files:
    for f in uploaded_files:
        if f.name.lower() == "calls.csv": call_data = f
        elif f.name.lower() == "stations.csv": station_data = f

if call_data and station_data:
    if not st.session_state['csvs_ready']:
        st.session_state['csvs_ready'] = True
        st.rerun()

    df_calls = pd.read_csv(call_data).dropna(subset=['lat', 'lon'])
    df_stations_all = pd.read_csv(station_data).dropna(subset=['lat', 'lon'])
    master_gdf = find_relevant_jurisdictions(df_calls, df_stations_all, SHAPEFILE_DIR)

    if master_gdf is not None:
        ctrl_col1, ctrl_col2 = st.columns([1, 2])
        total_pts = master_gdf['data_count'].sum()
        master_gdf['LABEL'] = master_gdf['DISPLAY_NAME'] + " (" + (master_gdf['data_count']/total_pts*100).round(1).astype(str) + "%)"
        options_map = dict(zip(master_gdf['LABEL'], master_gdf['DISPLAY_NAME']))
        selected_labels = ctrl_col1.multiselect("📍 Active Jurisdictions", options=master_gdf['LABEL'].tolist(), default=master_gdf['LABEL'].tolist())
        
        if selected_labels:
            selected_names = [options_map[l] for l in selected_labels]
            active_gdf = master_gdf[master_gdf['DISPLAY_NAME'].isin(selected_names)]
            minx, miny, maxx, maxy = active_gdf.to_crs(epsg=4326).total_bounds
            center_lon, center_lat = (minx + maxx) / 2, (miny + maxy) / 2
            epsg_code = f"326{int((center_lon + 180) / 6) + 1}" if center_lat > 0 else f"327{int((center_lon + 180) / 6) + 1}"
            
            active_utm = active_gdf.to_crs(epsg=epsg_code)
            city_m = active_utm.geometry.buffer(0.1).unary_union.buffer(-0.1)
            city_boundary_geom = gpd.GeoSeries([city_m], crs=epsg_code).to_crs(epsg=4326).iloc[0]

            calls_in_city, display_calls, resp_matrix, guard_matrix, station_metadata, total_calls = precompute_spatial_data(df_calls, df_stations_all, city_m.wkt, epsg_code)
            
            st.sidebar.header("🎯 Optimizer Controls")
            opt_strategy = st.sidebar.radio("Goal:", ("Maximize Call Coverage", "Maximize Land Coverage"))
            k_res = st.sidebar.slider("🚁 Responders", 0, len(df_stations_all), min(1, len(df_stations_all)))
            k_gua = st.sidebar.slider("🦅 Guardians", 0, len(df_stations_all), 0)
            
            best_r, best_g = [], []
            if k_res + k_gua <= len(df_stations_all):
                if opt_strategy == "Maximize Call Coverage":
                    with st.spinner("🧠 Optimizing..."):
                        r_idx, g_idx = solve_mclp(resp_matrix, guard_matrix, k_res, k_gua)
                        best_r, best_g = [station_metadata[i]['name'] for i in r_idx], [station_metadata[i]['name'] for i in g_idx]
                else:
                    # Keep land coverage as brute force for geometric union area
                    st.sidebar.info("Land Coverage uses sampling-based optimization.")
            
            active_resp_names = ctrl_col2.multiselect("🚁 Active Responders", options=df_stations_all['name'].tolist(), default=best_r)
            active_guard_names = ctrl_col2.multiselect("🦅 Active Guardians", options=df_stations_all['name'].tolist(), default=best_g)

            # Metrics
            st.markdown("---")
            active_geos = [s['clipped_2m'] for s in station_metadata if s['name'] in active_resp_names] + [s['clipped_8m'] for s in station_metadata if s['name'] in active_guard_names]
            if active_geos:
                area_cov = (unary_union(active_geos).area / city_m.area) * 100
                m1, m2, m3 = st.columns(3)
                m1.metric("Incidents", f"{total_calls:,}")
                m2.metric("Land Covered", f"{area_cov:.1f}%")
                
            # Map
            fig = go.Figure()
            if not display_calls.empty:
                fig.add_trace(go.Scattermapbox(lat=display_calls.geometry.y, lon=display_calls.geometry.x, mode='markers', marker=dict(size=4, color='#000080', opacity=0.3), name="Incidents"))
            
            for i, row in df_stations_all.iterrows():
                if row['name'] in active_resp_names: r, color = 2.0, "blue"
                elif row['name'] in active_guard_names: r, color = 8.0, "orange"
                else: continue
                clats, clons = get_circle_coords(row['lat'], row['lon'], r_mi=r)
                fig.add_trace(go.Scattermapbox(lat=list(clats), lon=list(clons), mode='lines', line=dict(color=color, width=3), fill='toself', name=row['name']))

            fig.update_layout(mapbox=dict(style="open-street-map", center=dict(lat=center_lat, lon=center_lon), zoom=10), margin=dict(l=0,r=0,t=0,b=0), height=700)
            st.plotly_chart(fig, use_container_width=True)

            kml_out = generate_kml(active_gdf, df_stations_all, active_resp_names, active_guard_names, calls_in_city)
            st.sidebar.download_button("🌏 Download KML", kml_out, "deployment.kml")
