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

def generate_command_center_html(df, export_mode=False, shift_hours=8):
    """Generates the full Command Center visual suite with interactive Calendar tooltips."""
    if df is None or df.empty or 'date' not in df.columns:
        return "<div style='color:gray; padding:20px;'>Analytics unavailable. Missing date/time fields.</div>"
    
    import calendar as _cal
    
    # 1. Standardize Data & Add Temporal Columns
    df_ana = df.copy()
    df_ana['dt_obj'] = pd.to_datetime(df_ana['date'] + ' ' + df_ana.get('time', '00:00:00'), errors='coerce')
    df_ana = df_ana.dropna(subset=['dt_obj'])
    if df_ana.empty: return "<div>No valid dates found in data.</div>"
    
    df_ana['hour'] = df_ana['dt_obj'].dt.hour
    df_ana['dow'] = df_ana['dt_obj'].dt.dayofweek # Mon=0, Sun=6
    df_ana['date_key'] = df_ana['dt_obj'].dt.strftime('%Y-%m-%d')
    df_ana['month_key'] = df_ana['dt_obj'].dt.strftime('%Y-%m')
    
    total_calls = len(df_ana)
    
    # 2. Prepare Tooltip Data (Daily Hourly Arrays)
    date_counts = df_ana['date_key'].value_counts().to_dict()
    date_hourly = {}
    for d, grp in df_ana.groupby('date_key'):
        date_hourly[d] = grp['hour'].value_counts().reindex(range(24), fill_value=0).tolist()
        
    # 3. Shift Windows (Rolling Sum)
    hourly_counts = df_ana['hour'].value_counts().reindex(range(24), fill_value=0).tolist()
    shift_html = ""
    for win in [8, 10, 12]:
        best_v, best_s = 0, 0
        for s in range(24):
            v = sum(hourly_counts[(s + h) % 24] for h in range(win))
            if v > best_v: best_v, best_s = v, s
        pct = (best_v / max(total_calls, 1)) * 100
        shift_html += f"""
        <div style="display:flex; align-items:center; background:#0c0c12; border:1px solid #252535; padding:8px; margin-bottom:5px; border-radius:4px;">
            <div style="width:50px; font-weight:800; color:#fff; font-size:13px;">{win}hr</div>
            <div style="width:110px; font-family:monospace; color:#00D2FF; font-size:12px;">{best_s:02d}:00 - {(best_s+win)%24:02d}:00</div>
            <div style="flex-grow:1; background:#1a1a26; height:8px; border-radius:4px; margin:0 15px; position:relative;">
                <div style="position:absolute; left:{(best_s/24)*100}%; width:{(win/24)*100}%; background:#00D2FF; height:100%; border-radius:4px; opacity:0.6;"></div>
            </div>
            <div style="width:60px; text-align:right; font-family:monospace; color:#00D2FF; font-size:13px;">{pct:.1f}%</div>
        </div>"""

    # 4. DOW Chart
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    dow_colors = ['#4ECDC4', '#45B7D1', '#F0B429', '#96CEB4', '#DDA0DD', '#FF9A8B', '#FF6B6B']
    dow_counts_list = df_ana['dow'].value_counts().reindex(range(7), fill_value=0).tolist()
    max_dow = max(dow_counts_list) if dow_counts_list else 1
    
    dow_html = "".join([f"""
        <div style="flex:1; display:flex; flex-direction:column; align-items:center;">
            <div style="background:#1a1a26; width:22px; height:80px; position:relative; border-radius:2px;">
                <div style="position:absolute; bottom:0; width:100%; height:{(v/max_dow)*100}%; background:{dow_colors[i]}; border-radius:2px;"></div>
            </div>
            <span style="font-size:10px; color:#7777a0; margin-top:6px; font-family:monospace;">{dow_names[i]}</span>
        </div>""" for i, v in enumerate(dow_counts_list)])

    # 5. Calendar Heatmap Setup
    month_keys = sorted(df_ana['month_key'].dropna().unique())
    cal_html = "<div style='display:grid; grid-template-columns:repeat(auto-fill, minmax(250px, 1fr)); gap:15px; margin-top:20px;'>"
    
    for mk in month_keys[:12]:
        yr, mo = int(mk.split('-')[0]), int(mk.split('-')[1])
        m_data = df_ana[df_ana['month_key'] == mk]
        m_max = max([date_counts.get(k, 0) for k in m_data['date_key'].unique()] + [1])
        
        cal_html += f"<div style='background:#0c0c12; border:1px solid #1a1a26; border-radius:6px; padding:12px;'>"
        cal_html += f"<div style='display:flex; justify-content:space-between; align-items:baseline; border-bottom:1px solid #252535; padding-bottom:6px; margin-bottom:8px;'><span style='color:#00D2FF; font-weight:800; font-size:12px; text-transform:uppercase; letter-spacing:1px;'>{_cal.month_name[mo]} {yr}</span><span style='color:#7777a0; font-size:10px; font-family:monospace;'>{len(m_data):,} calls</span></div>"
        
        # Header (Su -> Sa)
        cal_html += "<div style='display:grid; grid-template-columns:repeat(7, 1fr); gap:2px; margin-bottom:4px;'>"
        for i, dname in enumerate(['Su','Mo','Tu','We','Th','Fr','Sa']):
            c = ['#FF6B6B','#4ECDC4','#45B7D1','#F0B429','#96CEB4','#DDA0DD','#FF9A8B'][i]
            cal_html += f"<div style='font-size:9px; text-align:center; color:{c}; font-weight:600;'>{dname}</div>"
        cal_html += "</div>"
        
        # Grid Cells
        cal_html += "<div style='display:grid; grid-template-columns:repeat(7, 1fr); gap:2px;'>"
        first_dow = _cal.weekday(yr, mo, 1) # Mon=0, Sun=6
        first_dow_sun = (first_dow + 1) % 7
        last_day = _cal.monthrange(yr, mo)[1]
        
        for _ in range(first_dow_sun):
            cal_html += "<div></div>"
            
        for d in range(1, last_day + 1):
            dk = f"{yr}-{mo:02d}-{d:02d}"
            cnt = date_counts.get(dk, 0)
            ratio = cnt / m_max if m_max > 0 else 0
            
            if cnt == 0: bg, fc, cls = '#08080f', '#333', 'day-zero'
            elif ratio >= 0.85: bg, fc, cls = '#3d0a0a', '#ff4444', 'day-peak'
            elif ratio >= 0.55: bg, fc, cls = '#3d1a00', '#ff8c00', 'day-high'
            elif ratio >= 0.25: bg, fc, cls = '#2d2d00', '#d4c000', 'day-med'
            else: bg, fc, cls = '#0d3320', '#2ecc71', 'day-low'
            
            dow_idx = (_cal.weekday(yr, mo, d) + 1) % 7 # Map to Su=0
            stripe_color = ['#FF6B6B','#4ECDC4','#45B7D1','#F0B429','#96CEB4','#DDA0DD','#FF9A8B'][dow_idx]
            
            stripe_html = f"<div style='position:absolute; bottom:0; left:0; right:0; height:2px; background:{stripe_color}; opacity:0.7; border-radius:0 0 2px 2px;'></div>" if cnt > 0 else ""
            cnt_html = f"<span style='font-size:8px; opacity:0.7; margin-top:1px;'>{cnt}</span>" if cnt > 0 else ""
            
            # Hover properties
            cal_html += f"<div class='day-cell {cls}' data-date='{dk}' data-count='{cnt}' data-ratio='{ratio}' data-month='{_cal.month_name[mo]}' data-d='{d}' data-y='{yr}' data-dow='{dow_idx}' style='aspect-ratio:1; background:{bg}; border-radius:2px; display:flex; flex-direction:column; align-items:center; justify-content:center; color:{fc}; position:relative; font-family:monospace; cursor:default; border:1px solid transparent; transition:transform 0.1s;' onmouseover='showTooltip(this, event)' onmouseout='hideTooltip()'><span style='font-size:11px; z-index:1; font-weight:bold;'>{d}</span>{cnt_html}{stripe_html}</div>"
            
        cal_html += "</div></div>"
    cal_html += "</div>"

    # 6. Construct Final HTML Wrapper with JS Tooltip Logic
    full_html = f"""
    <div style="background:#000; color:#e8e8f2; font-family: 'Barlow', sans-serif; padding:15px; border-radius:8px;">
        <style>
            .day-cell:hover {{ transform: scale(1.15); z-index: 10; box-shadow: 0 4px 12px rgba(0,0,0,0.5); }}
            .day-peak {{ border-color: #cc0000 !important; font-weight: 700; }}
            #dfr-tooltip {{ position: fixed; z-index: 9999; background: #09090f; border: 1px solid #252535; border-radius: 6px; padding: 12px 16px; font-family: monospace; font-size: 11px; color: #e8e8f2; pointer-events: none; box-shadow: 0 6px 24px rgba(0,0,0,0.8); display: none; min-width: 220px; }}
        </style>
        
        <div id="dfr-tooltip"></div>
        
        <div style="color:#00D2FF; font-weight:900; letter-spacing:3px; font-size:14px; text-transform:uppercase; margin-bottom:20px; border-bottom:1px solid #1a1a26; padding-bottom:10px;">Data Ingestion Analytics</div>
        
        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:15px; margin-bottom:20px;">
            <div style="background:#0c0c12; border-left:4px solid #00D2FF; padding:15px; border-radius:4px; border-top:1px solid #1a1a26; border-right:1px solid #1a1a26; border-bottom:1px solid #1a1a26;">
                <div style="color:#00D2FF; font-size:26px; font-weight:900; font-family:monospace;">{total_calls:,}</div>
                <div style="color:#7777a0; font-size:10px; text-transform:uppercase; letter-spacing:1px; margin-top:4px;">Total Ingested Incidents</div>
            </div>
            <div style="background:#0c0c12; border-left:4px solid #F0B429; padding:15px; border-radius:4px; border-top:1px solid #1a1a26; border-right:1px solid #1a1a26; border-bottom:1px solid #1a1a26;">
                <div style="color:#F0B429; font-size:26px; font-weight:900; font-family:monospace;">{int(df_ana['hour'].mode()[0]) if not df_ana['hour'].mode().empty else 0}:00</div>
                <div style="color:#7777a0; font-size:10px; text-transform:uppercase; letter-spacing:1px; margin-top:4px;">Peak Activity Hour</div>
            </div>
        </div>
        
        <div style="display:grid; grid-template-columns: 3fr 2fr; gap:15px; margin-bottom:25px;">
            <div style="background:#06060a; border:1px solid #1a1a26; border-radius:6px; padding:15px;">
                <div style="margin-bottom:12px; font-size:10px; color:#7777a0; text-transform:uppercase; letter-spacing:1px; font-weight:bold;">Optimized DFR Shift Windows</div>
                {shift_html}
            </div>
            <div style="background:#06060a; border:1px solid #1a1a26; border-radius:6px; padding:15px; display:flex; flex-direction:column;">
                <div style="margin-bottom:12px; font-size:10px; color:#7777a0; text-transform:uppercase; letter-spacing:1px; font-weight:bold;">Call Volume by Day of Week</div>
                <div style="display:flex; justify-content:space-between; align-items:flex-end; flex-grow:1; padding:10px 5px 0;">
                    {dow_html}
                </div>
            </div>
        </div>
        
        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:5px; padding-top:15px; border-top:1px solid #1a1a26;">
            <div style="font-size:14px; font-weight:800; color:#fff; letter-spacing:1px; text-transform:uppercase;">DFR Deployment Calendar</div>
            <div style="display:flex; gap:12px; font-family:monospace; font-size:9px; color:#7777a0;">
                <div style="display:flex; align-items:center; gap:5px;"><div style="width:8px; height:8px; background:#2ecc71; border-radius:2px;"></div>LOW</div>
                <div style="display:flex; align-items:center; gap:5px;"><div style="width:8px; height:8px; background:#d4c000; border-radius:2px;"></div>MED</div>
                <div style="display:flex; align-items:center; gap:5px;"><div style="width:8px; height:8px; background:#ff8c00; border-radius:2px;"></div>HIGH</div>
                <div style="display:flex; align-items:center; gap:5px;"><div style="width:8px; height:8px; background:#ff4444; border-radius:2px;"></div>PEAK</div>
            </div>
        </div>
        
        {cal_html}
        
        <script>
            const dateHourly = {json.dumps(date_hourly)};
            const shiftHours = {shift_hours};
            const dowNames = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'];
            
            function showTooltip(el, ev) {{
                const cnt = parseInt(el.getAttribute('data-count'));
                if (cnt === 0) return;
                
                const dk = el.getAttribute('data-date');
                const ratio = parseFloat(el.getAttribute('data-ratio'));
                const mName = el.getAttribute('data-month');
                const d = el.getAttribute('data-d');
                const y = el.getAttribute('data-y');
                const dow = parseInt(el.getAttribute('data-dow'));
                
                let loadText = '';
                if (ratio >= 0.85) loadText = '<span style="color:#ff4444">■ PEAK</span> &mdash; Full crew';
                else if (ratio >= 0.55) loadText = '<span style="color:#ff8c00">■ HIGH</span> &mdash; Priority deploy';
                else if (ratio >= 0.25) loadText = '<span style="color:#d4c000">■ MEDIUM</span> &mdash; Standard ops';
                else loadText = '<span style="color:#2ecc71">■ LOW</span> &mdash; Light staffing';
                
                // Calculate best shift for THIS DAY
                const hrArr = dateHourly[dk] || Array(24).fill(0);
                let bestV = 0, bestS = 0;
                for (let s=0; s<24; s++) {{
                    let v = 0;
                    for (let h=0; h<shiftHours; h++) v += hrArr[(s+h)%24];
                    if (v > bestV) {{ bestV = v; bestS = s; }}
                }}
                const dayPct = Math.round((bestV / cnt) * 100);
                const eHr = (bestS + shiftHours) % 24;
                const fmt = (h) => (h%12 || 12) + (h<12 ? 'AM' : 'PM');
                
                const tt = document.getElementById('dfr-tooltip');
                tt.innerHTML = `
                    <div style="color:#00D2FF; margin-bottom:6px; font-size:12px; font-weight:bold; border-bottom:1px solid #252535; padding-bottom:4px;">${{mName}} ${{d}}, ${{y}} &middot; ${{dowNames[dow]}}</div>
                    <div style="margin-bottom:8px; font-size:13px;">Calls: <span style="color:#fff; font-weight:bold;">${{cnt}}</span> &nbsp;&middot;&nbsp; ${{loadText}}</div>
                    <div style="background:#1a1a26; padding:8px; border-radius:4px;">
                        <div style="color:#7777a0; font-size:9px; letter-spacing:1px; text-transform:uppercase; margin-bottom:4px;">Best ${{shiftHours}}hr Shift</div>
                        <div style="color:#00D2FF; font-size:14px; font-weight:bold; margin-bottom:2px;">${{fmt(bestS)}} &ndash; ${{fmt(eHr)}}</div>
                        <div style="color:#aaa; font-size:10px;">Covers <span style="color:#fff;">${{dayPct}}%</span> of daily volume</div>
                    </div>
                `;
                
                tt.style.display = 'block';
                
                // Adjust position to stay on screen
                let left = ev.clientX + 15;
                let top = ev.clientY - 20;
                if (left + 220 > window.innerWidth) left = ev.clientX - 235;
                if (top + 100 > window.innerHeight) top = ev.clientY - 110;
                
                tt.style.left = left + 'px';
                tt.style.top = top + 'px';
            }}
            
            function hideTooltip() {{
                document.getElementById('dfr-tooltip').style.display = 'none';
            }}
        </script>
    </div>
    """
    return full_html


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

def greedy_area_lazy(geos, total_area, max_st, progress_bar=None):
    if total_area <= 0: return [0.0]
    current_union = Polygon()
    curve = [0.0]
    geos_sub = geos[:max_st*2]
    pq = [(-g.area, i) for i, g in enumerate(geos_sub) if not g.is_empty]
    heapq.heapify(pq)
    for count in range(max_st):
        if not pq: break
        best_s, best_gain = -1, -1
        while pq:
            neg_gain, idx = heapq.heappop(pq)
            try: actual_gain = current_union.union(geos_sub[idx]).area - current_union.area
            except: actual_gain = 0
            if not pq or actual_gain >= -pq[0][0]:
                best_s, best_gain = idx, actual_gain
                break
            else: heapq.heappush(pq, (-actual_gain, idx))
        if best_s != -1 and best_gain > 0:
            current_union = current_union.union(geos_sub[best_s])
            curve.append((current_union.area / total_area) * 100)
        else: break
        if progress_bar: progress_bar.progress(int((count/max_st)*100))
    return curve

def solve_mclp_rigorous(resp_matrix, guard_matrix, k_resp, k_guard, allow_overlap):
    n_stations, n_calls = resp_matrix.shape
    prob = pulp.LpProblem("DFR_Optimizer", pulp.LpMaximize)
    x_r = pulp.LpVariable.dicts("st_resp", range(n_stations), 0, 1, pulp.LpBinary)
    x_g = pulp.LpVariable.dicts("st_guard", range(n_stations), 0, 1, pulp.LpBinary)
    y = pulp.LpVariable.dicts("call_covered", range(n_calls), 0, 1, pulp.LpBinary)
    prob += pulp.lpSum(y[j] for j in range(n_calls))
    prob += pulp.lpSum(x_r[i] for i in range(n_stations)) <= k_resp
    prob += pulp.lpSum(x_g[i] for i in range(n_stations)) <= k_guard
    if not allow_overlap:
        for i in range(n_stations): prob += x_r[i] + x_g[i] <= 1
    for j in range(n_calls):
        covering_stations = [x_r[i] for i in range(n_stations) if resp_matrix[i, j]] + [x_g[i] for i in range(n_stations) if guard_matrix[i, j]]
        if covering_stations: prob += y[j] <= pulp.lpSum(covering_stations)
        else: prob += y[j] == 0
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=20))
    res_r = [i for i in range(n_stations) if pulp.value(x_r[i]) > 0.5]
    res_g = [i for i in range(n_stations) if pulp.value(x_g[i]) > 0.5]
    return res_r, res_g

# ============================================================
# PAGE CONFIG & APP FLOW
# ============================================================
st.set_page_config(page_title="BRINC COS Optimizer", layout="wide")

st.sidebar.image("https://brincdrones.com/wp-content/uploads/2023/12/brinc_logo_white.png", width=150)
st.sidebar.markdown("---")

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
    k_guard = st.sidebar.slider("Guardian Drones", 0, len(df_stations), 2)
    allow_overlap = st.sidebar.checkbox("Allow Overlapping Coverage", value=True)

    with st.spinner("📡 Computing coverage matrices..."):
        utm_zone = int((df_calls['lon'].mean() + 180) / 6) + 1
        epsg = f"326{utm_zone}" if df_calls['lat'].mean() > 0 else f"327{utm_zone}"
        
        gdf_calls = gpd.GeoDataFrame(df_calls, geometry=gpd.points_from_xy(df_calls.lon, df_calls.lat), crs="EPSG:4326").to_crs(epsg)
        gdf_st = gpd.GeoDataFrame(df_stations, geometry=gpd.points_from_xy(df_stations.lon, df_stations.lat), crs="EPSG:4326").to_crs(epsg)
        
        radius_r = CONFIG["RESPONDER_RANGE_MI"] * 1609.34
        radius_g = 6.0 * 1609.34
        
        st_coords = np.array(list(zip(gdf_st.geometry.x, gdf_st.geometry.y)))
        call_coords = np.array(list(zip(gdf_calls.geometry.x, gdf_calls.geometry.y)))
        
        resp_matrix = np.zeros((len(st_coords), len(call_coords)), dtype=bool)
        guard_matrix = np.zeros((len(st_coords), len(call_coords)), dtype=bool)
        
        for i, s in enumerate(st_coords):
            dists = np.sqrt(np.sum((call_coords - s)**2, axis=1))
            resp_matrix[i] = dists <= radius_r
            guard_matrix[i] = dists <= radius_g

    idx_r, idx_g = solve_mclp_rigorous(resp_matrix, guard_matrix, k_resp, k_guard, allow_overlap)
    
    final_mask = np.zeros(len(df_calls), dtype=bool)
    if idx_r: final_mask |= resp_matrix[idx_r].any(axis=0)
    if idx_g: final_mask |= guard_matrix[idx_g].any(axis=0)
    call_cov_pct = (final_mask.sum() / len(df_calls)) * 100
    
    city_bounds = gdf_calls.unary_union.convex_hull
    active_geos = [gdf_st.iloc[i].geometry.buffer(radius_r) for i in idx_r] + \
                  [gdf_st.iloc[i].geometry.buffer(radius_g) for i in idx_g]
    area_cov_pct = (unary_union(active_geos).intersection(city_bounds).area / city_bounds.area) * 100 if active_geos else 0

    st.title(f"📍 {st.session_state['active_city']} DFR Deployment Plan")
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Call Coverage", f"{call_cov_pct:.1f}%")
    kpi2.metric("Land Area Covered", f"{area_cov_pct:.1f}%")
    kpi3.metric("Fleet CapEx", f"${(len(idx_r)*CONFIG['RESPONDER_COST'] + len(idx_g)*CONFIG['GUARDIAN_COST']):,}")

    fig = go.Figure()
    sample_calls = df_calls.sample(min(3000, len(df_calls)))
    fig.add_trace(go.Scattermapbox(lat=sample_calls.lat, lon=sample_calls.lon, mode='markers', marker=dict(size=3, color='#00D2FF', opacity=0.3), name="Incidents"))
    for i in idx_r:
        clat, clon = get_circle_coords(df_stations.iloc[i].lat, df_stations.iloc[i].lon, CONFIG['RESPONDER_RANGE_MI'])
        fig.add_trace(go.Scattermapbox(lat=clat, lon=clon, mode='lines', fill='toself', fillcolor='rgba(0,210,255,0.1)', line=dict(color='#00D2FF', width=2), name="Responder Range"))
    for i in idx_g:
        clat, clon = get_circle_coords(df_stations.iloc[i].lat, df_stations.iloc[i].lon, 6.0)
        fig.add_trace(go.Scattermapbox(lat=clat, lon=clon, mode='lines', fill='toself', fillcolor='rgba(240,180,41,0.1)', line=dict(color='#F0B429', width=2), name="Guardian Range"))
    
    fig.update_layout(mapbox_style="carto-darkmatter", mapbox_zoom=10, mapbox_center={"lat": df_calls.lat.mean(), "lon": df_calls.lon.mean()}, margin={"r":0,"t":0,"l":0,"b":0}, height=600)
    st.plotly_chart(fig, use_container_width=True)

    # ── COMMAND CENTER ANALYTICS DASHBOARD ──
    st.markdown("---")
    show_cad_analytics = st.toggle("📈 Show CAD Analytics Dashboard", value=True)
    analytics_html_block = generate_command_center_html(df_calls)
    
    if show_cad_analytics:
        # Render perfectly in Streamlit layout using components.html
        components.html(analytics_html_block, height=850, scrolling=True)

    # ── EXPORT PROPOSAL ──
    if st.sidebar.button("Generate Executive Proposal"):
        export_html = f"""
        <html>
        <head>
            <title>BRINC DFR Proposal</title>
            <style>
                body {{ font-family: 'Helvetica', sans-serif; padding: 40px; background: #fff; color: #333; }}
                h1 {{ color: #000; border-bottom: 2px solid #00D2FF; padding-bottom: 10px; }}
                .kpi-box {{ background: #f4f6f9; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <h1>BRINC DFR Deployment Proposal: {st.session_state['active_city']}</h1>
            <div class="kpi-box">
                <strong>Projected Coverage:</strong> {call_cov_pct:.1f}% of calls | {area_cov_pct:.1f}% of land area<br>
                <strong>Proposed Fleet:</strong> {len(idx_r)} Responders, {len(idx_g)} Guardians
            </div>
            
            <h2>Grant Narrative (Draft)</h2>
            <p>The {st.session_state['active_city']} Police Department seeks funding to establish a Drone as a First Responder (DFR) program. 
            Based on an analysis of {len(df_calls):,} calls for service, the proposed fleet of {len(idx_r)+len(idx_g)} BRINC systems 
            is strategically positioned to cover {call_cov_pct:.1f}% of critical historical incident locations. 
            This ensures rapid aerial response, enhancing officer safety and providing immediate situational awareness.</p>
            
            <p><strong>Potential Grant Funding Sources:</strong><br>
            • DOJ Byrne JAG — UAS and technology procurement eligible<br>
            • FEMA HSGP — CapEx offset for tactical deployments</p>
            
            <div style="margin-top: 50px;">
                <h2 style="margin-bottom:15px; color:#333;">Data Ingestion Analytics</h2>
                <div style="border-radius: 12px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.15);">
                    {analytics_html_block}
                </div>
            </div>
            
            <div style="margin-top:40px; padding-top:20px; border-top:2px solid #eee; text-align:center; font-size:13px; color:#555;">
                <div style="font-weight:bold; color:#111;">BRINC Drones, Inc.</div>
                <div>Leading the world in purpose-built Drone as a First Responder technology.</div>
                <div>brincdrones.com</div>
            </div>
        </body>
        </html>
        """
        st.download_button("📥 Download HTML Proposal", export_html, file_name=f"BRINC_Proposal_{st.session_state['active_city']}.html", mime="text/html")
