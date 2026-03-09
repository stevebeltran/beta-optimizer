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

# --- CUSTOM CSS FOR FONT SIZES ---
st.markdown(
    """
    <style>
    /* 1. Global font size for standard text */
    html, body, [class*="css"]  {
        font-size: 18px !important; 
    }

    /* 2. Change the font size of the Radio Button Options */
    div[role="radiogroup"] label div {
        font-size: 20px !important;
    }

    /* 3. Change the font size of the main Widget Titles */
    .stRadio label p, .stMultiSelect label p {
        font-size: 22px !important;
        font-weight: bold !important;
    }

    /* 4. Change the font size of the Multi-Select box items */
    div[data-baseweb="select"] span {
        font-size: 18px !important;
    }
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

# --- SIDEBAR: MAP LIBRARY MANAGER ---
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
if 'csvs_ready' not in st.session_state:
    st.session_state['csvs_ready'] = False
    st.session_state['df_calls'] = None
    st.session_state['df_stations'] = None

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
            st.session_state['df_calls'] = pd.read_csv(call_file).dropna(subset=['lat', 'lon'])
            st.session_state['df_stations'] = pd.read_csv(station_file).dropna(subset=['lat', 'lon'])
            st.session_state['csvs_ready'] = True
            st.rerun()

# High-Contrast Palette
STATION_COLORS = [
    "#E6194B", "#3CB44B", "#4363D8", "#F58231", "#911EB4", 
    "#800000", "#333333", "#000075", "#808000", "#9A6324"
]

def get_circle_coords(lat, lon, r_mi=2.0):
    angles = np.linspace(0, 2*np.pi, 100)
    c_lats = lat + (r_mi/69.172) * np.sin(angles)
    c_lons = lon + (r_mi/(69.172 * np.cos(np.radians(lat)))) * np.cos(angles)
    return c_lats, c_lons

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
        add_kml_station(row, 8.0, simplekml.Color.orange, "[Guardian]")

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
def precompute_spatial_data(df_calls, df_stations_all, city_m_wkt, epsg_code):
    city_m = shapely.wkt.loads(city_m_wkt)
    
    gdf_calls = gpd.GeoDataFrame(df_calls, geometry=gpd.points_from_xy(df_calls.lon, df_calls.lat), crs="EPSG:4326")
    gdf_calls_utm = gdf_calls.to_crs(epsg=epsg_code)
    
    try:
        calls_in_city = gdf_calls_utm[gdf_calls_utm.within(city_m)]
    except:
        calls_in_city = gdf_calls_utm
        
    radius_resp_m = 3218.69   
    radius_guard_m = 12874.75 
    
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
            
            resp_matrix[i, :] = dists <= radius_resp_m
            guard_matrix[i, :] = dists <= radius_guard_m

            full_buf_2m = s_pt_m.buffer(radius_resp_m)
            try: clipped_2m = full_buf_2m.intersection(city_m)
            except: clipped_2m = full_buf_2m

            full_buf_8m = s_pt_m.buffer(radius_guard_m)
            try: clipped_8m = full_buf_8m.intersection(city_m)
            except: clipped_8m = full_buf_8m
                
            station_metadata.append({
                'name': row['name'], 'lat': row['lat'], 'lon': row['lon'],
                'clipped_2m': clipped_2m, 'clipped_8m': clipped_8m
            })
            
    return calls_in_city, display_calls, resp_matrix, guard_matrix, station_metadata, total_calls

def solve_mclp(resp_matrix, guard_matrix, num_resp, num_guard, allow_redundancy, tb_area_r, tb_area_g, tb_cent):
    n_stations, n_calls = resp_matrix.shape
    if n_calls == 0:
        return [], []

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

    model += pulp.lpSum(x_r[i] for i in range(n_stations)) == num_resp
    model += pulp.lpSum(x_g[i] for i in range(n_stations)) == num_guard

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

    n_drones = num_resp + num_guard
    if n_drones == 0: n_drones = 1
    area_weight = 0.4 / n_drones
    cent_weight = 0.05 / n_drones
    
    tie_breaker_obj = pulp.lpSum(
        x_r[s] * (tb_area_r[s] * area_weight + tb_cent[s] * cent_weight) +
        x_g[s] * (tb_area_g[s] * area_weight + tb_cent[s] * cent_weight)
        for s in range(n_stations)
    )

    model += primary_obj + tie_breaker_obj

    model.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=10))

    responders = [i for i in range(n_stations) if pulp.value(x_r[i]) == 1]
    guardians = [i for i in range(n_stations) if pulp.value(x_g[i]) == 1]

    return responders, guardians

# --- MAIN LOGIC ---
if st.session_state['csvs_ready']:
    df_calls = st.session_state['df_calls']
    df_stations_all = st.session_state['df_stations']

    with st.spinner("🌍 Identifying dominant jurisdictions..."):
        master_gdf = find_relevant_jurisdictions(df_calls, df_stations_all, SHAPEFILE_DIR)

    if master_gdf is None or master_gdf.empty:
        st.error("❌ No matching jurisdictions found.")
        st.stop()

    st.sidebar.success(f"**Found {len(master_gdf)} Significant Zones**")
    
    st.markdown("---")
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)
    
    total_pts = master_gdf['data_count'].sum()
    master_gdf['LABEL'] = master_gdf['DISPLAY_NAME'] + " (" + (master_gdf['data_count']/total_pts*100).round(1).astype(str) + "%)"
    
    options_map = dict(zip(master_gdf['LABEL'], master_gdf['DISPLAY_NAME']))
    all_options = master_gdf['LABEL'].tolist()
    
    selected_labels = ctrl_col1.multiselect("📍 Active Jurisdictions", options=all_options, default=all_options)
    
    if not selected_labels:
        st.warning("Please select at least one jurisdiction.")
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

    with st.spinner("⚡ Precomputing spatial optimization matrices..."):
        city_m_wkt = city_m.wkt  
        calls_in_city, display_calls, resp_matrix, guard_matrix, station_metadata, total_calls = precompute_spatial_data(
            df_calls, df_stations_all, city_m_wkt, epsg_code
        )
    n = len(df_stations_all)

    max_dist = max([((s['lon'] - center_lon)**2 + (s['lat'] - center_lat)**2)**0.5 for s in station_metadata])
    if max_dist == 0: max_dist = 1.0
    
    for s in station_metadata:
        dist = ((s['lon'] - center_lon)**2 + (s['lat'] - center_lat)**2)**0.5
        s['centrality'] = 1.0 - (dist / max_dist)

    max_area = city_m.area if (city_m and city_m.area > 0) else 1.0
    tb_area_r = [s['clipped_2m'].area / max_area for s in station_metadata]
    tb_area_g = [s['clipped_8m'].area / max_area for s in station_metadata]
    tb_cent = [s['centrality'] for s in station_metadata]

    # --- OPTIMIZER CONTROLS ---
    st.sidebar.header("🎯 Optimizer Controls")
    opt_strategy = st.sidebar.radio("Optimization Goal:", ("Maximize Call Coverage", "Maximize Land Coverage"), index=0)
    
    k_responder = st.sidebar.slider("🚁 Responder Drones (2-Mile)", 0, n, min(1, n))
    k_guardian = st.sidebar.slider("🦅 Guardian Drones (8-Mile)", 0, n, 0)
    
    allow_redundancy = st.sidebar.toggle(
        "Multi-Tier (Allow Overlap)", 
        value=True, 
        help="Treat Responders and Guardians as independent layers. When ON, drones won't move away just because their coverage rings overlap."
    )
    
    show_boundaries = st.sidebar.checkbox("Show Jurisdiction Boundaries", value=True)
    show_heatmap = st.sidebar.toggle("🔥 Show Incident Heatmap", value=False)
    show_health = st.sidebar.toggle("Show Health Score Banner", value=True)
    show_satellite = st.sidebar.toggle("🌍 Satellite View", value=False)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🚗 Ground Traffic Simulator")
    simulate_traffic = st.sidebar.toggle("Show Ground Response Gap", value=False)
    traffic_level = st.sidebar.slider("Traffic Intensity (%)", 0, 100, 40)

    budget_placeholder = st.sidebar.container()

    best_resp_names, best_guard_names = [], []
    
    if k_responder + k_guardian > n:
        st.error("⚠️ Over-Deployment: Total requested drones exceed available stations.")
    elif k_responder > 0 or k_guardian > 0:
        
        best_combo = None
        
        if opt_strategy == "Maximize Call Coverage":
            with st.spinner("🧠 Running exact MCLP Optimizer (PuLP)..."):
                r_best, g_best = solve_mclp(resp_matrix, guard_matrix, k_responder, k_guardian, allow_redundancy, tb_area_r, tb_area_g, tb_cent)
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
                    g_geos = [station_metadata[i]['clipped_8m'] for i in g_combo]
                    score_area = (unary_union(r_geos).area if r_geos else 0.0) + (unary_union(g_geos).area if g_geos else 0.0)
                else:
                    geos = [station_metadata[i]['clipped_2m'] for i in r_combo] + [station_metadata[i]['clipped_8m'] for i in g_combo]
                    score_area = unary_union(geos).area if geos else 0.0
                
                if total_calls > 0:
                    cov_r = resp_matrix[list(r_combo)].any(axis=0) if r_combo else np.zeros(total_calls, bool)
                    cov_g = guard_matrix[list(g_combo)].any(axis=0) if g_combo else np.zeros(total_calls, bool)
                    score_calls = np.logical_or(cov_r, cov_g).sum()
                else:
                    score_calls = 0
                
                score_cent = sum([station_metadata[i]['centrality'] for i in r_combo]) + sum([station_metadata[i]['centrality'] for i in g_combo])
                return (score_area, score_calls, score_cent, rg_combo)

            with st.spinner(f"Optimizing {min(total_possible, 3000)} configurations for area..."):
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
        
        if best_combo is not None:
            r_best, g_best = best_combo
            best_resp_names = [station_metadata[i]['name'] for i in r_best]
            best_guard_names = [station_metadata[i]['name'] for i in g_best]

    st.sidebar.markdown("---")
    st.sidebar.subheader("🏆 Recommended Deployment")
    for name in best_resp_names: st.sidebar.write(f"🚁 {name} (Responder)")
    for name in best_guard_names: st.sidebar.write(f"🦅 {name} (Guardian)")

    # --- UI SELECTION ---
    active_resp_names = ctrl_col2.multiselect("🚁 Active Responders (2-Mile)", options=df_stations_all['name'].tolist(), default=best_resp_names)
    active_guard_names = ctrl_col3.multiselect("🦅 Active Guardians (8-Mile)", options=df_stations_all['name'].tolist(), default=best_guard_names)
    
    # --- METRICS CALCULATION ---
    area_covered_perc, overlap_perc, calls_covered_perc = 0.0, 0.0, 0.0
    
    active_resp_idx = [i for i, s in enumerate(station_metadata) if s['name'] in active_resp_names]
    active_guard_idx = [i for i, s in enumerate(station_metadata) if s['name'] in active_guard_names]
    
    active_resp_data = [station_metadata[i] for i in active_resp_idx]
    active_guard_data = [station_metadata[i] for i in active_guard_idx]
    
    active_geos = [s['clipped_2m'] for s in active_resp_data] + [s['clipped_8m'] for s in active_guard_data]

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
    # --- DECOUPLED BUDGET IMPACT MODULE ---
    # ==========================================
    with budget_placeholder:
        st.markdown("---")
        st.subheader("💰 Budget Impact")
        
        inferred_daily_calls = max(1, int(total_calls / 365)) if total_calls > 0 else 20
        max_slider_val = max(100, inferred_daily_calls * 3) 
        
        calls_per_day = st.slider("TOTAL DAILY CALLS (CITYWIDE)", min_value=1, max_value=max_slider_val, value=inferred_daily_calls)
        deflection_rate = st.slider("DRONE-ONLY RESOLUTION (%)", min_value=0, max_value=100, value=30) / 100.0
        
        cost_officer = 82
        cost_drone = 6
        savings_per_call = cost_officer - cost_drone
        
        actual_k_responder = len(active_resp_names)
        actual_k_guardian = len(active_guard_names)
        
        capex_responder_total = actual_k_responder * 80000
        capex_guardian_total = actual_k_guardian * 160000
        fleet_capex = capex_responder_total + capex_guardian_total
        
        if fleet_capex > 0:
            # 1. FLEET LEVEL MATH (Calculates unique total program savings)
            effective_coverage_rate = calls_covered_perc / 100.0
            covered_daily_calls = calls_per_day * effective_coverage_rate
            daily_drone_only_calls = covered_daily_calls * deflection_rate
            
            if daily_drone_only_calls > 0:
                monthly_savings = savings_per_call * daily_drone_only_calls * 30.4
                annual_savings = monthly_savings * 12
                fleet_break_even_months = fleet_capex / monthly_savings
                break_even_text = f"{fleet_break_even_months:.1f} MONTHS"
            else:
                annual_savings = 0
                break_even_text = "N/A"
            
            st.markdown(f"""
            <div style="background: #000000; border: 1px solid #00ff00; padding: 15px; border-radius: 4px; text-align: center; margin-bottom: 15px; box-shadow: 0px 0px 10px rgba(0, 255, 0, 0.15);">
                <h6 style="color: #888; margin: 0; font-size: 0.8rem; letter-spacing: 1px;">ANNUAL TAXPAYER SAVINGS</h6>
                <h2 style="color: #00ff00; margin: 0; font-family: 'Consolas', monospace; text-shadow: 0 0 10px rgba(0,255,0,0.5);">${annual_savings:,.0f}</h2>
                <hr style="border-color: #00ff00; opacity: 0.3; margin: 10px 0;">
                <div style="display: flex; justify-content: space-between; font-size: 0.8rem;">
                    <span style="color: #888;">IN DRONE RANGE:</span>
                    <span style="color: #fff; font-weight: bold;">{covered_daily_calls:.1f} / DAY</span>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.8rem;">
                    <span style="color: #888;">DEFLECTED (SAVINGS):</span>
                    <span style="color: #fff; font-weight: bold;">{daily_drone_only_calls:.1f} / DAY</span>
                </div>
                <hr style="border-color: #333; margin: 5px 0;">
                <div style="display: flex; justify-content: space-between; font-size: 0.8rem;">
                    <span style="color: #888;">FLEET CAPEX:</span>
                    <span style="color: #fff; font-weight: bold;">${fleet_capex:,.0f}</span>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.8rem;">
                    <span style="color: #888;">BREAK-EVEN:</span>
                    <span style="color: #00ff00; font-weight: bold;">{break_even_text}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # 2. RESPONDER TIER MATH
            if actual_k_responder > 0:
                cov_r_only = resp_matrix[active_resp_idx].any(axis=0) if len(active_resp_idx) > 0 else np.zeros(total_calls, bool)
                resp_calls_perc = (cov_r_only.sum() / total_calls) if total_calls > 0 else 0
                avg_resp_perc = resp_calls_perc / actual_k_responder
                
                r_covered_daily = calls_per_day * avg_resp_perc
                r_deflected_daily = r_covered_daily * deflection_rate
                r_monthly_savings = savings_per_call * r_deflected_daily * 30.4
                
                if r_monthly_savings > 0:
                    resp_be_months = 80000 / r_monthly_savings
                    resp_be_text = f"{resp_be_months:.1f} MO"
                else:
                    resp_be_text = "N/A"

                st.markdown(f"""
                <div style="border: 1px solid #444; padding: 10px; border-radius: 4px; margin-bottom: 10px; background: #111;">
                    <h5 style="color: #00ffff; margin: 0; margin-bottom: 4px;">RESPONDER <span style="color:#fff; font-size:0.9rem;">(x{actual_k_responder})</span></h5>
                    <div style="color: #888; font-size: 0.85rem;">COVERAGE: <span style="color:#fff;">2 MI RADIUS</span></div>
                    <div style="color: #888; font-size: 0.85rem;">UNIT CAPEX: <span style="color:#fff;">$80,000</span></div>
                    <div style="color: #888; font-size: 0.85rem;">CALLS IN RANGE: <span style="color:#fff;">{r_covered_daily:.1f} / DAY (AVG)</span></div>
                    <div style="color: #888; font-size: 0.85rem;">UNIT ROI: <span style="color:#00ff00; font-weight:bold;">{resp_be_text}</span></div>
                    <div style="color: #888; font-size: 0.85rem; margin-top: 4px; border-top: 1px dashed #333; padding-top: 4px;">SUBTOTAL: <span style="color:#00ffff; font-weight:bold;">${capex_responder_total:,.0f}</span></div>
                </div>
                """, unsafe_allow_html=True)
                
            # 3. GUARDIAN TIER MATH
            if actual_k_guardian > 0:
                cov_g_only = guard_matrix[active_guard_idx].any(axis=0) if len(active_guard_idx) > 0 else np.zeros(total_calls, bool)
                guard_calls_perc = (cov_g_only.sum() / total_calls) if total_calls > 0 else 0
                avg_guard_perc = guard_calls_perc / actual_k_guardian
                
                g_covered_daily = calls_per_day * avg_guard_perc
                g_deflected_daily = g_covered_daily * deflection_rate
                g_monthly_savings = savings_per_call * g_deflected_daily * 30.4
                
                if g_monthly_savings > 0:
                    guard_be_months = 160000 / g_monthly_savings
                    guard_be_text = f"{guard_be_months:.1f} MO"
                else:
                    guard_be_text = "N/A"

                st.markdown(f"""
                <div style="border: 1px solid #444; padding: 10px; border-radius: 4px; margin-bottom: 10px; background: #111;">
                    <h5 style="color: #00ffff; margin: 0; margin-bottom: 4px;">GUARDIAN <span style="color:#fff; font-size:0.9rem;">(x{actual_k_guardian})</span></h5>
                    <div style="color: #888; font-size: 0.85rem;">COVERAGE: <span style="color:#fff;">8 MI RADIUS</span></div>
                    <div style="color: #888; font-size: 0.85rem;">UNIT CAPEX: <span style="color:#fff;">$160,000</span></div>
                    <div style="color: #888; font-size: 0.85rem;">CALLS IN RANGE: <span style="color:#fff;">{g_covered_daily:.1f} / DAY (AVG)</span></div>
                    <div style="color: #888; font-size: 0.85rem;">UNIT ROI: <span style="color:#00ff00; font-weight:bold;">{guard_be_text}</span></div>
                    <div style="color: #888; font-size: 0.85rem; margin-top: 4px; border-top: 1px dashed #333; padding-top: 4px;">SUBTOTAL: <span style="color:#00ffff; font-weight:bold;">${capex_guardian_total:,.0f}</span></div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("🚁 Select at least one non-redundant drone to calculate budget impact.")
    # ==========================================

    st.markdown("---")
    if show_health:
        norm_redundancy = min(overlap_perc / 35.0, 1.0) * 100
        health_score = (calls_covered_perc * 0.50) + (area_covered_perc * 0.35) + (norm_redundancy * 0.15)
        if health_score >= 80: h_color, h_label = "#28a745", "OPTIMAL"
        elif health_score >= 70: h_color, h_label = "#94c11f", "GOOD"
        elif health_score >= 55: h_color, h_label = "#ffc107", "MARGINAL"
        else: h_color, h_label = "#dc3545", "ESSENTIAL"
        
        st.markdown(f"""
            <div style="background-color: {h_color}; padding: 10px; border-radius: 5px; color: white; margin-bottom: 10px; display: flex; align-items: center; justify-content: space-between;">
                <span style="font-size: 1.5em; font-weight: bold;">Department Health Score: {health_score:.1f}%</span>
                <span style="font-size: 1.3em; background: rgba(0,0,0,0.2); padding: 2px 10px; border-radius: 4px;">{h_label}</span>
            </div>""", unsafe_allow_html=True)

    # --- DYNAMIC METRICS CALCULATION (ADAPTIVE FOR DRONE TYPE) ---
    if simulate_traffic:
        m1, m2, m3, m4, m5 = st.columns(5)
        
        if len(active_guard_names) > 0:
            eval_dist = 8.0
            eval_speed = 60.0
            gain_label = "Efficiency Gain (8-mi)"
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
        calls_in_city
    )
    
    st.sidebar.markdown("---")
    st.sidebar.download_button(
        label="🌏 Download for Google Earth",
        data=kml_data,
        file_name="drone_deployment.kml",
        mime="application/vnd.google-earth.kml+xml"
    )

    # --- MAP RENDERING ---
    fig = go.Figure()
    
    def calculate_zoom(min_lon, max_lon, min_lat, max_lat):
        lon_diff = max_lon - min_lon
        lat_diff = max_lat - min_lat
        
        if lon_diff <= 0 or lat_diff <= 0:
            return 12
            
        zoom_lon = np.log2(360 / lon_diff)
        zoom_lat = np.log2(180 / lat_diff)
        
        best_zoom = min(zoom_lon, zoom_lat) + 1.6
        return min(max(best_zoom, 5), 18)

    if show_boundaries:
        if city_boundary_geom is not None and not city_boundary_geom.is_empty:
            if isinstance(city_boundary_geom, Polygon):
                bx, by = city_boundary_geom.exterior.coords.xy
                fig.add_trace(go.Scattermapbox(mode="lines", lon=list(bx), lat=list(by), line=dict(color="#222", width=3), name="Jurisdiction Boundary", hoverinfo='skip'))
            elif isinstance(city_boundary_geom, MultiPolygon):
                for poly in city_boundary_geom.geoms:
                    bx, by = poly.exterior.coords.xy
                    fig.add_trace(go.Scattermapbox(mode="lines", lon=list(bx), lat=list(by), line=dict(color="#222", width=3), name="Jurisdiction Boundary", hoverinfo='skip', showlegend=False))

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
            marker=dict(size=4, color='#000080', opacity=0.35), 
            name="Incident Data", 
            hoverinfo='skip'
        ))

    for i, row in df_stations_all.iterrows():
        s_name = row['name']
        color = STATION_COLORS[i % len(STATION_COLORS)]

        if s_name in active_resp_names:
            clats, clons = get_circle_coords(row['lat'], row['lon'], r_mi=2.0)
            lbl = f"{s_name} (Responder)"
            drive_time_min = (2.0 / 42.0) * 60 
        elif s_name in active_guard_names:
            clats, clons = get_circle_coords(row['lat'], row['lon'], r_mi=8.0)
            lbl = f"{s_name} (Guardian)"
            drive_time_min = (8.0 / 60.0) * 60 
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
        style="open-street-map"
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
        font=dict(size=18)
    )

    st.plotly_chart(fig, width='stretch', config={"scrollZoom": True})
