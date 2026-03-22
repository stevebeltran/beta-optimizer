import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import plotly.graph_objects as go
from shapely.geometry import Point, Polygon, MultiPolygon, box, shape
from shapely.ops import unary_union
import os, itertools, glob, math, simplekml, heapq, re, random, json, io, datetime, base64, smtplib
from concurrent.futures import ThreadPoolExecutor
import pulp
import urllib.request
import streamlit.components.v1 as components
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import gspread
from google.oauth2.service_account import Credentials

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

US_STATES_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
    "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH",
    "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
    "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
    "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN",
    "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY"
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

STATION_COLORS = ["#00D2FF", "#39FF14", "#FFD700", "#FF007F", "#FF4500", "#00FFCC", "#FF3333", "#7FFF00", "#00FFFF", "#FF9900"]

# ============================================================
# ANALYTICS ENGINE (The "Thinking" Logic)
# ============================================================

def generate_command_center_html(df, export_mode=False):
    """Generates the full Command Center visual suite (Shifts, DOW, Calendar)."""
    if df is None or df.empty or 'date' not in df.columns:
        return "<div style='color:gray; padding:20px;'>Analytics unavailable.</div>"
    
    # 1. Standardize Data
    df['dt_obj'] = pd.to_datetime(df['date'] + ' ' + df.get('time', '00:00:00'), errors='coerce')
    df = df.dropna(subset=['dt_obj'])
    df['hour'] = df['dt_obj'].dt.hour
    df['dow'] = df['dt_obj'].dt.day_name()
    df['date_key'] = df['dt_obj'].dt.date.astype(str)

    # 2. Shift Windows (Rigorous Rolling Sum)
    hourly_counts = df['hour'].value_counts().reindex(range(24), fill_value=0)
    shift_html = ""
    for win in [8, 10, 12]:
        best_v, best_s = 0, 0
        for s in range(24):
            v = sum(hourly_counts[(s + h) % 24] for h in range(win))
            if v > best_v: best_v, best_s = v, s
        pct = (best_v / len(df)) * 100
        shift_html += f"""
        <div class="sc-row" style="display:flex; align-items:center; background:#0c0c12; border:1px solid #252535; padding:8px; margin-bottom:4px; border-radius:4px;">
            <div style="width:50px; font-weight:800; color:#fff;">{win}hr</div>
            <div style="width:130px; font-family:monospace; color:#00D2FF;">{best_s:02d}:00-{(best_s+win)%24:02d}:00</div>
            <div style="flex-grow:1; background:#1a1a26; height:8px; border-radius:4px; margin:0 15px; position:relative;">
                <div style="position:absolute; left:{(best_s/24)*100}%; width:{(win/24)*100}%; background:#00D2FF; height:100%; border-radius:4px; opacity:0.6;"></div>
            </div>
            <div style="width:60px; text-align:right; font-family:monospace; color:#00D2FF;">{pct:.1f}%</div>
        </div>"""

    # 3. DOW Chart
    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_counts = df['dow'].value_counts().reindex(dow_order, fill_value=0)
    max_dow = dow_counts.max()
    dow_html = "".join([f"""
        <div style="flex:1; display:flex; flex-direction:column; align-items:center;">
            <div style="background:#1a1a26; width:20px; height:80px; position:relative; border-radius:2px;">
                <div style="position:absolute; bottom:0; width:100%; height:{(v/max_dow)*100}%; background:#F0B429; border-radius:2px;"></div>
            </div>
            <span style="font-size:10px; color:#7777a0; margin-top:5px;">{d[:3]}</span>
        </div>""" for d, v in dow_counts.items()])

    # 4. Final Container
    return f"""
    <div style="background:#06060a; padding:20px; border-radius:8px; border:1px solid #1a1a26; font-family:sans-serif;">
        <div style="color:#00D2FF; font-weight:800; letter-spacing:2px; font-size:11px; text-transform:uppercase; margin-bottom:15px;">Historical Resource Analysis</div>
        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:15px; margin-bottom:20px;">
            <div style="background:#0c0c12; border-left:3px solid #00D2FF; padding:12px; border-radius:4px;">
                <div style="color:#00D2FF; font-size:24px; font-weight:bold;">{len(df):,}</div>
                <div style="color:#7777a0; font-size:10px; text-transform:uppercase;">Ingested Incidents</div>
            </div>
            <div style="background:#0c0c12; border-left:3px solid #F0B429; padding:12px; border-radius:4px;">
                <div style="color:#F0B429; font-size:24px; font-weight:bold;">{int(df['hour'].mode()[0])}:00</div>
                <div style="color:#7777a0; font-size:10px; text-transform:uppercase;">Peak Activity Hour</div>
            </div>
        </div>
        <div style="margin-bottom:10px; font-size:10px; color:#7777a0; text-transform:uppercase;">Optimized DFR Shift Windows</div>
        {shift_html}
        <div style="margin-top:20px; margin-bottom:10px; font-size:10px; color:#7777a0; text-transform:uppercase;">Volume by Day of Week</div>
        <div style="display:flex; justify-content:space-between; padding:10px; background:#0c0c12; border-radius:4px;">{dow_html}</div>
    </div>
    """

# ============================================================
# AGGRESSIVE DATA PARSER
# ============================================================

def aggressive_parse_calls(uploaded_files):
    all_calls_list = []
    
    # Header synonyms
    CV = {
        'date': ['received date','incident date','call date','call creation date','calldatetime','call datetime','calltime','timestamp','date','datetime','dispatch date','time received','incdate'],
        'time': ['call creation time','call time','dispatch time','received time','time'],
        'priority': ['call priority','priority level','priority','pri','urgency'],
        'lat': ['call original address latitude','address latitude','addresslatitude','latitude','lat','ycoor','y coor','ycoord','y coord','y coordinate','y-coordinate','addressy','ylat','lat y','coord y','coordy','geo lat','geolat','point y','pointy','address y'],
        'lon': ['call original address longitude','address longitude','addresslongitude','longitude','lon','long','xcoor','x coor','xcoord','x coord','x coordinate','x-coordinate','lng','addressx','xlong','lon x','coord x','coordx','geo lon','geolon','point x','pointx','address x']
    }

    def parse_priority(raw):
        s = str(raw).strip().upper()
        if not s or s == 'NAN': return None
        if re.search(r'\bNON[\-\s]?EMERG|\bROUTINE\b|\bINFORMATIONAL\b', s): return 4
        if re.search(r'\bHIGH\b|\bEMERG|\bCRITICAL\b', s): return 1
        if re.search(r'\bMED', s): return 2
        if re.search(r'\bLOW\b', s): return 3
        m = re.search(r'^(\d+)', s)
        if m: return int(m.group(1))
        return 3

    for cfile in uploaded_files:
        try:
            content = cfile.getvalue().decode('utf-8', errors='ignore')
            first_line = content.split('\n')[0]
            delim = ',' if first_line.count(',') > first_line.count('\t') else '\t'
            raw_df = pd.read_csv(io.StringIO(content), sep=delim, dtype=str)
            raw_df.columns = [str(c).lower().strip() for c in raw_df.columns]
            
            res = pd.DataFrame()
            
            # Map Lat/Lon
            for field in ['lat', 'lon']:
                found = [c for c in raw_df.columns if any(s in c for s in CV[field])]
                if found: res[field] = pd.to_numeric(raw_df[found[0]], errors='coerce')
            
            # Map Priority
            p_found = [c for c in raw_df.columns if any(s in c for s in CV['priority'])]
            if p_found: res['priority'] = raw_df[p_found[0]].apply(parse_priority)
            else: res['priority'] = 3
            
            # Map Date/Time
            d_found = [c for c in raw_df.columns if any(s in c for s in CV['date'])]
            t_found = [c for c in raw_df.columns if any(s in c for s in CV['time'])]
            
            if d_found:
                # If time is in a separate column or already merged
                if t_found and d_found[0] != t_found[0]:
                    dt_series = pd.to_datetime(raw_df[d_found[0]] + ' ' + raw_df[t_found[0]], errors='coerce')
                else:
                    dt_series = pd.to_datetime(raw_df[d_found[0]], errors='coerce')
                
                res['date'] = dt_series.dt.strftime('%Y-%m-%d')
                res['time'] = dt_series.dt.strftime('%H:%M:%S')
            
            all_calls_list.append(res)
        except: continue
        
    if not all_calls_list: return pd.DataFrame()
    final_df = pd.concat(all_calls_list, ignore_index=True).dropna(subset=['lat', 'lon'])
    return final_df

# ============================================================
# OPTIMIZATION & SPATIAL ENGINE
# ============================================================

def greedy_area_lazy(geos, total_area, max_st, progress_bar=None):
    """Mathematically precise overlap calculation using Lazy-Greedy algorithm."""
    if total_area <= 0: return [0.0]
    current_union = Polygon()
    curve = [0.0]
    geos_sub = geos[:max_st*2] # Sample pool
    
    pq = [(-g.area, i) for i, g in enumerate(geos_sub) if not g.is_empty]
    heapq.heapify(pq)
    
    for count in range(max_st):
        if not pq: break
        best_s, best_gain = -1, -1
        
        while pq:
            neg_gain, idx = heapq.heappop(pq)
            try:
                actual_gain = current_union.union(geos_sub[idx]).area - current_union.area
            except: actual_gain = 0
            
            # Lazy check: if this gain is still better than the next best on heap
            if not pq or actual_gain >= -pq[0][0]:
                best_s, best_gain = idx, actual_gain
                break
            else:
                heapq.heappush(pq, (-actual_gain, idx))
        
        if best_s != -1 and best_gain > 0:
            current_union = current_union.union(geos_sub[best_s])
            curve.append((current_union.area / total_area) * 100)
        else: break
        if progress_bar: progress_bar.progress(int((count/max_st)*100))
            
    return curve

def solve_mclp_rigorous(resp_matrix, guard_matrix, k_resp, k_guard, allow_overlap):
    """Thinking model: Solves the Maximum Coverage Location Problem using Integer Programming."""
    n_stations, n_calls = resp_matrix.shape
    prob = pulp.LpProblem("DFR_Optimizer", pulp.LpMaximize)
    
    # Decision variables
    x_r = pulp.LpVariable.dicts("st_resp", range(n_stations), 0, 1, pulp.LpBinary)
    x_g = pulp.LpVariable.dicts("st_guard", range(n_stations), 0, 1, pulp.LpBinary)
    y = pulp.LpVariable.dicts("call_covered", range(n_calls), 0, 1, pulp.LpBinary)
    
    # Objective: Maximize covered calls
    prob += pulp.lpSum(y[j] for j in range(n_calls))
    
    # Constraints: Fleet Size
    prob += pulp.lpSum(x_r[i] for i in range(n_stations)) <= k_resp
    prob += pulp.lpSum(x_g[i] for i in range(n_stations)) <= k_guard
    
    if not allow_overlap:
        for i in range(n_stations):
            prob += x_r[i] + x_g[i] <= 1
            
    # Constraint: Call coverage linking
    for j in range(n_calls):
        # A call is covered if ANY station within range of it is selected
        covering_stations = [x_r[i] for i in range(n_stations) if resp_matrix[i, j]] + \
                            [x_g[i] for i in range(n_stations) if guard_matrix[i, j]]
        if covering_stations:
            prob += y[j] <= pulp.lpSum(covering_stations)
        else:
            prob += y[j] == 0
            
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=20))
    
    res_r = [i for i in range(n_stations) if pulp.value(x_r[i]) > 0.5]
    res_g = [i for i in range(n_stations) if pulp.value(x_g[i]) > 0.5]
    return res_r, res_g

# ============================================================
# MAIN APP FLOW
# ============================================================

# Sidebar initialization
st.sidebar.image("https://brincdrones.com/wp-content/uploads/2023/12/brinc_logo_white.png", width=150)
st.sidebar.markdown("---")

if not st.session_state['csvs_ready']:
    st.title("🛰️ BRINC COS Optimizer")
    st.info("Ingest raw CAD exports to optimize drone placement and shift windows.")
    
    uploaded_files = st.file_uploader("Drop one or more CAD CSV files", accept_multiple_files=True)
    
    if uploaded_files:
        with st.spinner("🧠 Analyzing and merging CAD data..."):
            df_calls = aggressive_parse_calls(uploaded_files)
            if not df_calls.empty:
                st.session_state['df_calls'] = df_calls
                
                # Auto-Discover Stations
                lat_c, lon_c = df_calls['lat'].mean(), df_calls['lon'].mean()
                osm_query = f"[out:json];(node['amenity'~'fire_station|police']({lat_c-0.2},{lon_c-0.2},{lat_c+0.2},{lon_c+0.2}););out;"
                try:
                    with urllib.request.urlopen(f"https://overpass-api.de/api/interpreter?data={osm_query}") as r:
                        data = json.loads(r.read().decode())
                        st_list = [{'name': e['tags'].get('name', 'Municipal Station'), 'lat': e['lat'], 'lon': e['lon'], 'type': e['tags'].get('amenity', 'Facility')} for e in data['elements']]
                        st.session_state['df_stations'] = pd.DataFrame(st_list).drop_duplicates(subset=['lat', 'lon'])
                except:
                    st.session_state['df_stations'] = pd.DataFrame([{'name': 'Primary Hub', 'lat': lat_c, 'lon': lon_c, 'type': 'Police'}])
                
                # Reverse Geocode for Title
                st_full, city_name = reverse_geocode_state(lat_c, lon_c)
                st.session_state['active_city'] = city_name
                st.session_state['active_state'] = st_full
                
                st.session_state['csvs_ready'] = True
                st.rerun()

if st.session_state['csvs_ready']:
    df_calls = st.session_state['df_calls']
    df_stations = st.session_state['df_stations']
    
    st.sidebar.header("Fleet Optimization")
    k_resp = st.sidebar.slider("Responder Drones", 0, len(df_stations), 5)
    k_guard = st.sidebar.slider("Guardian Drones", 0, len(df_stations), 2)
    allow_overlap = st.sidebar.checkbox("Allow Overlapping Coverage", value=True)

    # 1. Rigorous Spatial Precompute
    with st.spinner("📡 Computing coverage matrices..."):
        # Convert to UTM for accurate meter-based distance
        utm_zone = int((df_calls['lon'].mean() + 180) / 6) + 1
        epsg = f"326{utm_zone}" if df_calls['lat'].mean() > 0 else f"327{utm_zone}"
        
        gdf_calls = gpd.GeoDataFrame(df_calls, geometry=gpd.points_from_xy(df_calls.lon, df_calls.lat), crs="EPSG:4326").to_crs(epsg)
        gdf_st = gpd.GeoDataFrame(df_stations, geometry=gpd.points_from_xy(df_stations.lon, df_stations.lat), crs="EPSG:4326").to_crs(epsg)
        
        # Build Distance Matrices
        radius_r = CONFIG["RESPONDER_RANGE_MI"] * 1609.34
        radius_g = 6.0 * 1609.34 # Guardian default
        
        st_coords = np.array(list(zip(gdf_st.geometry.x, gdf_st.geometry.y)))
        call_coords = np.array(list(zip(gdf_calls.geometry.x, gdf_calls.geometry.y)))
        
        resp_matrix = np.zeros((len(st_coords), len(call_coords)), dtype=bool)
        guard_matrix = np.zeros((len(st_coords), len(call_coords)), dtype=bool)
        
        for i, s in enumerate(st_coords):
            dists = np.sqrt(np.sum((call_coords - s)**2, axis=1))
            resp_matrix[i] = dists <= radius_r
            guard_matrix[i] = dists <= radius_g

    # 2. Run Optimizer
    idx_r, idx_g = solve_mclp_rigorous(resp_matrix, guard_matrix, k_resp, k_guard, allow_overlap)
    
    # Calculate covered percentages
    final_mask = np.zeros(len(df_calls), dtype=bool)
    if idx_r: final_mask |= resp_matrix[idx_r].any(axis=0)
    if idx_g: final_mask |= guard_matrix[idx_g].any(axis=0)
    call_cov_pct = (final_mask.sum() / len(df_calls)) * 100
    
    # Calculate precise Area Coverage
    city_bounds = gdf_calls.unary_union.convex_hull
    active_geos = [gdf_st.iloc[i].geometry.buffer(radius_r) for i in idx_r] + \
                  [gdf_st.iloc[i].geometry.buffer(radius_g) for i in idx_g]
    area_cov_pct = (unary_union(active_geos).intersection(city_bounds).area / city_bounds.area) * 100

    # 3. Main Dashboard Display
    st.title(f"📍 {st.session_state['active_city']} DFR Deployment Plan")
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Call Coverage", f"{call_cov_pct:.1f}%")
    kpi2.metric("Land Area Covered", f"{area_cov_pct:.1f}%")
    kpi3.metric("Fleet CapEx", f"${(len(idx_r)*CONFIG['RESPONDER_COST'] + len(idx_g)*CONFIG['GUARDIAN_COST']):,}")

    # Map
    fig = go.Figure()
    # Add Calls
    sample_calls = df_calls.sample(min(3000, len(df_calls)))
    fig.add_trace(go.Scattermapbox(lat=sample_calls.lat, lon=sample_calls.lon, mode='markers', marker=dict(size=3, color='#00D2FF', opacity=0.3), name="Incidents"))
    # Add Stations & Rings
    for i in idx_r:
        clat, clon = get_circle_coords(df_stations.iloc[i].lat, df_stations.iloc[i].lon, 2.0)
        fig.add_trace(go.Scattermapbox(lat=clat, lon=clon, mode='lines', fill='toself', fillcolor='rgba(0,210,255,0.1)', line=dict(color='#00D2FF', width=2), name="Responder Range"))
    
    fig.update_layout(mapbox_style="carto-darkmatter", mapbox_zoom=11, mapbox_center={"lat": df_calls.lat.mean(), "lon": df_calls.lon.mean()}, margin={"r":0,"t":0,"l":0,"b":0}, height=600)
    st.plotly_chart(fig, use_container_width=True)

    # 4. Command Center Analytics (BELOW SWARM)
    st.markdown("---")
    st.subheader("📊 CAD Ingestion Analytics")
    analytics_html = generate_command_center_html(df_calls)
    components.html(analytics_html, height=500, scrolling=True)

    # 5. Rigorous Export Logic
    if st.sidebar.button("Generate Executive Proposal"):
        analytics_block = generate_command_center_html(df_calls, export_mode=True)
        export_html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: 'Helvetica', sans-serif; padding: 40px; background: #fff; color: #333; }}
                h1 {{ color: #000; border-bottom: 2px solid #00D2FF; padding-bottom: 10px; }}
                .kpi-box {{ background: #f4f6f9; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
                .analytics-wrapper {{ background: #06060a; padding: 20px; border-radius: 12px; margin: 30px 0; }}
            </style>
        </head>
        <body>
            <h1>BRINC DFR Deployment Proposal: {st.session_state['active_city']}</h1>
            <div class="kpi-box">
                <strong>Projected Coverage:</strong> {call_cov_pct:.1f}% of calls | {area_cov_pct:.1f}% of land area<br>
                <strong>Proposed Fleet:</strong> {len(idx_r)} Responders, {len(idx_g)} Guardians
            </div>
            
            <h2>Historical Resource Analytics</h2>
            <div class="analytics-wrapper">
                {analytics_block}
            </div>

            <h2>Grant Narrative (Draft)</h2>
            <p>The {st.session_state['active_city']} Police Department seeks funding to establish a Drone as a First Responder (DFR) program. 
            Based on an analysis of {len(df_calls):,} calls for service, the proposed fleet of {len(idx_r)+len(idx_g)} BRINC systems 
            is strategically positioned to cover {call_cov_pct:.1f}% of critical historical incident locations...</p>
        </body>
        </html>
        """
        st.download_button("📥 Download HTML Proposal", export_html, file_name=f"BRINC_Proposal_{st.session_state['active_city']}.html", mime="text/html")
