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
# INITIALIZE SESSION STATE
# ============================================================
# This MUST be done before any part of the code tries to access these keys
session_defaults = {
    'csvs_ready': False, 
    'df_calls': None, 
    'df_stations': None, 
    'active_city': "Orlando", 
    'active_state': "FL", 
    'city_count': 1,
    '_opt_cache_key': None,
    'total_original_calls': 0,
    'user_name': "John Doe",
    'user_email': "john.doe@example.com"
}

for key, value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def generate_command_center_html(df, export_mode=False):
    if df is None or df.empty or 'date' not in df.columns:
        return "<div style='color:gray; padding:20px;'>Analytics unavailable. Missing date/time fields.</div>"
    
    # Standardize Data
    df['dt_obj'] = pd.to_datetime(df['date'] + ' ' + df.get('time', '00:00:00'), errors='coerce')
    df = df.dropna(subset=['dt_obj'])
    df['hour'] = df['dt_obj'].dt.hour
    df['dow'] = df['dt_obj'].dt.day_name()

    # Shift Windows (Rolling Sum)
    hourly_counts = df['hour'].value_counts().reindex(range(24), fill_value=0)
    shift_html = ""
    for win in [8, 10, 12]:
        best_v, best_s = 0, 0
        for s in range(24):
            v = sum(hourly_counts[(s + h) % 24] for h in range(win))
            if v > best_v: best_v, best_s = v, s
        pct = (best_v / len(df)) * 100
        shift_html += f"""
        <div style="display:flex; align-items:center; background:#0c0c12; border:1px solid #252535; padding:8px; margin-bottom:4px; border-radius:4px;">
            <div style="width:50px; font-weight:800; color:#fff;">{win}hr</div>
            <div style="width:130px; font-family:monospace; color:#00D2FF;">{best_s:02d}:00-{(best_s+win)%24:02d}:00</div>
            <div style="flex-grow:1; background:#1a1a26; height:8px; border-radius:4px; margin:0 15px; position:relative;">
                <div style="position:absolute; left:{(best_s/24)*100}%; width:{(win/24)*100}%; background:#00D2FF; height:100%; border-radius:4px; opacity:0.6;"></div>
            </div>
            <div style="width:60px; text-align:right; font-family:monospace; color:#00D2FF;">{pct:.1f}%</div>
        </div>"""

    return f"""
    <div style="background:#06060a; padding:20px; border-radius:8px; border:1px solid #1a1a26; font-family:sans-serif; color:white;">
        <div style="color:#00D2FF; font-weight:800; letter-spacing:2px; font-size:11px; text-transform:uppercase; margin-bottom:15px;">Historical Resource Analysis</div>
        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:15px; margin-bottom:20px;">
            <div style="background:#0c0c12; border-left:3px solid #00D2FF; padding:12px; border-radius:4px;">
                <div style="color:#00D2FF; font-size:24px; font-weight:bold;">{len(df):,}</div>
                <div style="color:#7777a0; font-size:10px; text-transform:uppercase;">Ingested Incidents</div>
            </div>
            <div style="background:#0c0c12; border-left:3px solid #F0B429; padding:12px; border-radius:4px;">
                <div style="color:#F0B429; font-size:24px; font-weight:bold;">{int(df['hour'].mode()[0]) if not df['hour'].mode().empty else 0}:00</div>
                <div style="color:#7777a0; font-size:10px; text-transform:uppercase;">Peak Activity Hour</div>
            </div>
        </div>
        {shift_html}
    </div>
    """

def aggressive_parse_calls(uploaded_files):
    all_calls_list = []
    CV = {
        'date': ['received date','incident date','call date','call creation date','calldatetime','call datetime','calltime','timestamp','date','datetime','dispatch date','time received','incdate'],
        'time': ['call creation time','call time','dispatch time','received time','time'],
        'priority': ['call priority','priority level','priority','pri','urgency'],
        'lat': ['latitude','lat','ycoor','addressy','y coord'],
        'lon': ['longitude','lon','long','xcoor','addressx','x coord']
    }

    def parse_priority(raw):
        s = str(raw).strip().upper()
        if re.search(r'\bHIGH\b|\bEMERG|\bCRITICAL\b|1', s): return 1
        if re.search(r'\bMED|2', s): return 2
        return 3

    for cfile in uploaded_files:
        try:
            content = cfile.getvalue().decode('utf-8', errors='ignore')
            raw_df = pd.read_csv(io.StringIO(content), dtype=str)
            raw_df.columns = [str(c).lower().strip() for c in raw_df.columns]
            res = pd.DataFrame()
            for field in ['lat', 'lon']:
                found = [c for c in raw_df.columns if any(s in c for s in CV[field])]
                if found: res[field] = pd.to_numeric(raw_df[found[0]], errors='coerce')
            p_found = [c for c in raw_df.columns if any(s in c for s in CV['priority'])]
            res['priority'] = raw_df[p_found[0]].apply(parse_priority) if p_found else 3
            d_found = [c for c in raw_df.columns if any(s in c for s in CV['date'])]
            t_found = [c for c in raw_df.columns if any(s in c for s in CV['time'])]
            if d_found:
                combined = raw_df[d_found[0]] + (' ' + raw_df[t_found[0]] if t_found and d_found[0]!=t_found[0] else '')
                dt_series = pd.to_datetime(combined, errors='coerce')
                res['date'] = dt_series.dt.strftime('%Y-%m-%d')
                res['time'] = dt_series.dt.strftime('%H:%M:%S')
            all_calls_list.append(res)
        except: continue
    return pd.concat(all_calls_list, ignore_index=True).dropna(subset=['lat', 'lon']) if all_calls_list else pd.DataFrame()

@st.cache_data
def reverse_geocode_state(lat, lon):
    url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&zoom=10"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'BRINC_Optimizer'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            addr = data.get('address', {})
            return addr.get('state', ''), addr.get('city', addr.get('town', 'Unknown City'))
    except: return "State", "Unknown City"

def get_circle_coords(lat, lon, r_mi=2.0):
    angles = np.linspace(0, 2*np.pi, 100)
    c_lats = lat + (r_mi/69.172) * np.sin(angles)
    c_lons = lon + (r_mi/(69.172 * np.cos(np.radians(lat)))) * np.cos(angles)
    return c_lats, c_lons

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="BRINC COS Optimizer", layout="wide")

# --- APP UI ---
if not st.session_state['csvs_ready']:
    st.title("🛰️ BRINC DFR Deployment Optimizer")
    st.markdown("Upload your CAD data to automatically generate station placements and shift analysis.")
    
    uploaded_files = st.file_uploader("Upload CAD CSV(s)", accept_multiple_files=True)
    if uploaded_files:
        with st.spinner("Processing Data..."):
            df_calls = aggressive_parse_calls(uploaded_files)
            if not df_calls.empty:
                st.session_state['df_calls'] = df_calls
                lat_c, lon_c = df_calls['lat'].mean(), df_calls['lon'].mean()
                
                # Auto-Stations via OSM
                osm_q = f"[out:json];(node['amenity'~'fire_station|police']({lat_c-0.2},{lon_c-0.2},{lat_c+0.2},{lon_c+0.2}););out;"
                try:
                    with urllib.request.urlopen(f"https://overpass-api.de/api/interpreter?data={osm_q}") as r:
                        data = json.loads(r.read().decode())
                        st_list = [{'name': e['tags'].get('name', 'Station'), 'lat': e['lat'], 'lon': e['lon']} for e in data['elements']]
                        st.session_state['df_stations'] = pd.DataFrame(st_list)
                except:
                    st.session_state['df_stations'] = pd.DataFrame([{'name': 'Hub 1', 'lat': lat_c, 'lon': lon_c}])
                
                st_full, city_name = reverse_geocode_state(lat_c, lon_c)
                st.session_state['active_city'], st.session_state['active_state'] = city_name, st_full
                st.session_state['csvs_ready'] = True
                st.rerun()

if st.session_state['csvs_ready']:
    df_calls, df_stations = st.session_state['df_calls'], st.session_state['df_stations']
    
    st.sidebar.title("Fleet Settings")
    k_resp = st.sidebar.slider("Responder Drones", 0, len(df_stations), 5)
    
    # Rigorous coverage math (Sample)
    st.title(f"📍 {st.session_state['active_city']} DFR Plan")
    
    fig = go.Figure()
    # Add incidents
    sample = df_calls.sample(min(2000, len(df_calls)))
    fig.add_trace(go.Scattermapbox(lat=sample.lat, lon=sample.lon, mode='markers', marker=dict(size=4, color='#00D2FF', opacity=0.4)))
    
    fig.update_layout(mapbox_style="carto-darkmatter", mapbox_center={"lat": df_calls.lat.mean(), "lon": df_calls.lon.mean()}, mapbox_zoom=11, margin={"r":0,"t":0,"l":0,"b":0}, height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # --- ANALYTICS DASHBOARD ---
    st.markdown("---")
    st.subheader("📊 Ingestion Analytics")
    analytics_html = generate_command_center_html(df_calls)
    components.html(analytics_html, height=450, scrolling=True)

    # --- EXPORT PROPOSAL ---
    if st.sidebar.button("Export Executive Summary"):
        analytics_block = generate_command_center_html(df_calls, export_mode=True)
        export_html = f"""
        <html>
        <body style="font-family: sans-serif; padding: 40px;">
            <h1>BRINC DFR Proposal: {st.session_state['active_city']}</h1>
            <div style="background:#06060a; padding:20px; border-radius:10px;">
                {analytics_block}
            </div>
            <h2>Grant Narrative</h2>
            <p>Based on {len(df_calls):,} historical calls, we recommend a phased rollout of {k_resp} stations...</p>
        </body>
        </html>
        """
        st.download_button("Download HTML Proposal", export_html, file_name=f"BRINC_{st.session_state['active_city']}.html", mime="text/html")
