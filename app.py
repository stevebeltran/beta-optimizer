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
import urllib.parse
import zipfile
import streamlit.components.v1 as components
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import gspread
from google.oauth2.service_account import Credentials

# --- PAGE CONFIG & INITIALIZE SESSION STATE ---
st.set_page_config(page_title="BRINC COS Drone Optimizer", layout="wide", initial_sidebar_state="expanded")

# This MUST run before any st.session_state checks to prevent KeyError
defaults = {
    'csvs_ready': False, 'df_calls': None, 'df_stations': None,
    'active_city': "Orlando", 'active_state': "FL", 'estimated_pop': 316081,
    'k_resp': 2, 'k_guard': 0, 'r_resp': 2.0, 'r_guard': 8.0,
    'dfr_rate': 25, 'deflect_rate': 30, 'total_original_calls': 0,
    'onboarding_done': False, 'trigger_sim': False, 'city_count': 1,
    'brinc_user': 'steven.beltran'
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if 'target_cities' not in st.session_state:
    st.session_state['target_cities'] = [{"city": st.session_state.get('active_city', 'Orlando'), "state": st.session_state.get('active_state', 'FL')}]

def _notify_email(city, state, file_type, k_resp, k_guard, coverage, name, email):
    try:
        gmail_address  = st.secrets.get("GMAIL_ADDRESS", "")
        app_password   = st.secrets.get("GMAIL_APP_PASSWORD", "")
        notify_address = st.secrets.get("NOTIFY_EMAIL", gmail_address)
        if not gmail_address or not app_password: return
        emoji = {"HTML": "📄", "KML": "🌏", "BRINC": "💾"}.get(file_type, "📥")
        subject = f"{emoji} BRINC Download — {file_type} — {city}, {state}"
        body = f"""
        <html><body style="font-family:Arial,sans-serif;color:#333;padding:20px;">
        <div style="max-width:500px;margin:0 auto;border:1px solid #ddd;border-radius:8px;overflow:hidden;">
            <div style="background:#000;padding:16px 20px;border-bottom:3px solid #00D2FF;">
                <span style="color:#00D2FF;font-size:18px;font-weight:900;letter-spacing:2px;">BRINC</span>
                <span style="color:#888;font-size:12px;margin-left:8px;">Download Notification</span>
            </div>
            <div style="padding:20px;">
                <table style="width:100%;border-collapse:collapse;font-size:14px;">
                    <tr style="border-bottom:1px solid #f0f0f0;"><td style="padding:8px 4px;color:#888;width:40%;">File Type</td><td style="padding:8px 4px;font-weight:bold;">{emoji} {file_type}</td></tr>
                    <tr style="border-bottom:1px solid #f0f0f0;"><td style="padding:8px 4px;color:#888;">City</td><td style="padding:8px 4px;font-weight:bold;">{city}, {state}</td></tr>
                    <tr style="border-bottom:1px solid #f0f0f0;"><td style="padding:8px 4px;color:#888;">Fleet</td><td style="padding:8px 4px;">{k_resp} Responder · {k_guard} Guardian</td></tr>
                    <tr style="border-bottom:1px solid #f0f0f0;"><td style="padding:8px 4px;color:#888;">Call Coverage</td><td style="padding:8px 4px;">{coverage:.1f}%</td></tr>
                    <tr style="border-bottom:1px solid #f0f0f0;"><td style="padding:8px 4px;color:#888;">User Name</td><td style="padding:8px 4px;">{name if name else '—'}</td></tr>
                    <tr><td style="padding:8px 4px;color:#888;">User Email</td><td style="padding:8px 4px;">{f'<a href="mailto:{email}">{email}</a>' if email else '—'}</td></tr>
                </table>
                <div style="margin-top:16px;font-size:11px;color:#bbb;">{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} UTC</div>
            </div>
        </div>
        </body></html>
        """
        msg = MIMEMultipart("alternative")
        msg["Subject"], msg["From"], msg["To"] = subject, gmail_address, notify_address
        msg.attach(MIMEText(body, "html"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=8) as server:
            server.login(gmail_address, app_password)
            server.sendmail(gmail_address, notify_address, msg.as_string())
    except: pass

def _log_to_sheets(city, state, file_type, k_resp, k_guard, coverage, name, email):
    try:
        sheet_id = st.secrets.get("GOOGLE_SHEET_ID", "")
        creds_dict = st.secrets.get("gcp_service_account", {})
        if not sheet_id or not creds_dict: return
        scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_info(dict(creds_dict), scopes=scopes)
        client = gspread.authorize(creds)
        sheet = client.open_by_key(sheet_id).sheet1
        sheet.append_row([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), city, state, file_type, k_resp, k_guard, round(coverage, 1), name, email])
    except: pass

# --- GLOBAL CONFIGURATION ---
CONFIG = {"RESPONDER_COST": 80000, "GUARDIAN_COST": 160000, "RESPONDER_RANGE_MI": 2.0, "OFFICER_COST_PER_CALL": 82, "DRONE_COST_PER_CALL": 6, "DEFAULT_TRAFFIC_SPEED": 35.0, "RESPONDER_SPEED": 42.0, "GUARDIAN_SPEED": 60.0}
STATE_FIPS = {"AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08", "CT": "09", "DE": "10", "FL": "12", "GA": "13", "HI": "15", "ID": "16", "IL": "17", "IN": "18", "IA": "19", "KS": "20", "KY": "21", "LA": "22", "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27", "MS": "28", "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33", "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39", "OK": "40", "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46", "TN": "47", "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53", "WV": "54", "WI": "55", "WY": "56"}
US_STATES_ABBR = {"Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY"}
KNOWN_POPULATIONS = {"New York": 8336817, "Los Angeles": 3822238, "Chicago": 2665039, "Houston": 1304379, "Phoenix": 1644409, "Philadelphia": 1567258, "San Antonio": 2302878, "San Diego": 1472530, "Dallas": 1299544, "San Jose": 1381162, "Austin": 974447, "Jacksonville": 971319, "Fort Worth": 956709, "Columbus": 907971, "Indianapolis": 880621, "Charlotte": 897720, "San Francisco": 971233, "Seattle": 749256, "Denver": 713252, "Washington": 678972, "Nashville": 683622, "Oklahoma City": 694800, "El Paso": 694553, "Boston": 650706, "Portland": 635067, "Las Vegas": 656274, "Detroit": 620376, "Memphis": 633104, "Louisville": 628594, "Baltimore": 620961, "Milwaukee": 620251, "Albuquerque": 677122, "Tucson": 564559, "Fresno": 677102, "Sacramento": 808418, "Kansas City": 697738, "Mesa": 504258, "Atlanta": 499127, "Omaha": 508901, "Colorado Springs": 483956, "Raleigh": 476587, "Miami": 449514, "Virginia Beach": 455369, "Oakland": 530763, "Minneapolis": 563332, "Tulsa": 547239, "Arlington": 398654, "New Orleans": 562503, "Wichita": 402263, "Cleveland": 900000, "Tampa": 449514, "Orlando": 316081}
DEMO_CITIES = [("Las Vegas", "NV"), ("Austin", "TX"), ("Seattle", "WA"), ("Denver", "CO"), ("Nashville", "TN"), ("Columbus", "OH"), ("Detroit", "MI"), ("San Diego", "CA"), ("Charlotte", "NC"), ("Portland", "OR"), ("Memphis", "TN"), ("Louisville", "KY"), ("Baltimore", "MD"), ("Milwaukee", "WI"), ("Albuquerque", "NM"), ("Tucson", "AZ"), ("Fresno", "CA"), ("Sacramento", "CA"), ("Kansas City", "MO"), ("Mesa", "AZ"), ("Atlanta", "GA"), ("Omaha", "NE"), ("Colorado Springs", "CO"), ("Raleigh", "NC"), ("Miami", "FL"), ("Minneapolis", "MN"), ("Tulsa", "OK"), ("Arlington", "TX"), ("Tampa", "FL"), ("New Orleans", "LA"), ("Wichita", "KS"), ("Cleveland", "OH"), ("Virginia Beach", "VA"), ("Oakland", "CA"), ("Indianapolis", "IN"), ("Jacksonville", "FL"), ("Fort Worth", "TX"), ("Boston", "MA"), ("El Paso", "TX"), ("Oklahoma City", "OK"), ("Boise", "ID"), ("Richmond", "VA"), ("Spokane", "WA"), ("Tacoma", "WA"), ("Aurora", "CO"), ("Anaheim", "CA"), ("Bakersfield", "CA"), ("Riverside", "CA"), ("Stockton", "CA"), ("Corpus Christi", "TX"), ("Lexington", "KY"), ("Henderson", "NV"), ("Saint Paul", "MN"), ("Anchorage", "AK"), ("Plano", "TX"), ("Lincoln", "NE"), ("Buffalo", "NY"), ("Fort Wayne", "IN"), ("Jersey City", "NJ"), ("Chula Vista", "CA"), ("Orlando", "FL"), ("St. Louis", "MO"), ("Madison", "WI"), ("Durham", "NC"), ("Lubbock", "TX"), ("Winston-Salem", "NC"), ("Garland", "TX"), ("Glendale", "AZ"), ("Hialeah", "FL"), ("Scottsdale", "AZ"), ("Irving", "TX"), ("Fremont", "CA"), ("Baton Rouge", "LA"), ("Birmingham", "AL"), ("Rochester", "NY"), ("Des Moines", "IA"), ("Montgomery", "AL"), ("Modesto", "CA"), ("Fayetteville", "NC"), ("Shreveport", "LA"), ("Akron", "OH"), ("Grand Rapids", "MI"), ("Huntington Beach", "CA"), ("Little Rock", "AR")]
FAST_DEMO_CITIES = [("Henderson", "NV"), ("Lincoln", "NE"), ("Boise", "ID"), ("Des Moines", "IA"), ("Madison", "WI"), ("Colorado Springs", "CO"), ("Richmond", "VA"), ("Raleigh", "NC"), ("Durham", "NC"), ("Fort Wayne", "IN"), ("Omaha", "NE"), ("Wichita", "KS"), ("Tulsa", "OK"), ("Spokane", "WA"), ("Tacoma", "WA"), ("Aurora", "CO"), ("Las Vegas", "NV"), ("Nashville", "TN"), ("Columbus", "OH"), ("Charlotte", "NC"), ("Louisville", "KY"), ("Indianapolis", "IN"), ("Memphis", "TN"), ("Detroit", "MI"), ("Milwaukee", "WI"), ("Minneapolis", "MN"), ("Seattle", "WA"), ("Denver", "CO"), ("Portland", "OR"), ("Austin", "TX")]
FAA_CEILING_COLORS = {0: {"line": "rgba(255,  20,  20, 0.95)", "fill": "rgba(255,  20,  20, 0.20)"}, 50: {"line": "rgba(255, 120,   0, 0.95)", "fill": "rgba(255, 120,   0, 0.18)"}, 100: {"line": "rgba(255, 210,   0, 0.95)", "fill": "rgba(255, 210,   0, 0.18)"}, 200: {"line": "rgba(180, 230,   0, 0.95)", "fill": "rgba(180, 230,   0, 0.16)"}, 300: {"line": "rgba( 80, 200,  50, 0.95)", "fill": "rgba( 80, 200,  50, 0.16)"}, 400: {"line": "rgba(  0, 180, 100, 0.95)", "fill": "rgba(  0, 180, 100, 0.15)"}}
FAA_DEFAULT_COLOR = {"line": "rgba(150,150,150,0.8)", "fill": "rgba(150,150,150,0.10)"}
STATION_COLORS = ["#00D2FF", "#39FF14", "#FFD700", "#FF007F", "#FF4500", "#00FFCC", "#FF3333", "#7FFF00", "#00FFFF", "#FF9900"]

# --- THEME VARIABLES ---
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

HERO_MESSAGES = ["🚔 Building safer communities, one drone at a time…", "🛡️ Loading data because your officers deserve better tools…", "🫡 Honoring the men and women who answer the call every day…", "💙 Officers run toward danger so the rest of us don't have to…", "🚁 Optimizing so your team gets there first — every time…", "🌟 Every second we save is a life better protected…", "🤝 Technology in service of the community's greatest heroes…", "💪 Your officers deserve every advantage we can give them…", "🙏 Dedicated to the families who wait at home while heroes serve…", "🏅 Processing data worthy of those who wear the badge with pride…", "🌃 Mapping the city your officers protect through every shift…", "🔵 Building a network as reliable as the officers who depend on it…", "❤️ Because faster response means more lives saved…", "🌅 Creating tools that let officers come home safely every night…", "🦅 Guardian drones — always watching, always ready to assist…", "🏘️ Modeling coverage for the neighborhoods they protect and serve…", "📡 Connecting technology to the courage already on the streets…", "🧠 Smart systems for smarter, safer law enforcement…", "🌟 Every data point represents a community worth protecting…", "🚨 Fewer false alarms. More real backup. Better outcomes for all…"]
FAA_MESSAGES = ["✈️ Checking FAA airspace — keeping your drones and your pilots safe…", "🛫 Loading LAANC data — because safe skies mean more missions completed…", "🗺️ Mapping controlled airspace — so every flight is a legal, safe one…", "✈️ FAA compliance check in progress — protecting officers on the ground and drones in the air…", "🛡️ Pulling airspace boundaries — safe operations start before takeoff…", "🌐 Verifying flight corridors — your pilots deserve a clear path forward…", "📡 Syncing with FAA LAANC — because your department deserves zero surprises in the sky…", "🛩️ Loading aviation data — the same skies your officers look up to every night…"]
AIRFIELD_MESSAGES = ["🏗️ Locating nearby airfields — coordinating with the aviation community that shares your skies…", "📍 Mapping airports near each station — great neighbors make great operators…", "🛬 Finding local airfields — because your team coordinates with everyone keeping the community safe…", "✈️ Scanning for nearby aviation assets — your drones respect every aircraft they share the sky with…", "🗺️ Identifying airport proximity — so your officers always know what's overhead…", "🤝 Locating nearby airfields — collaboration between aviation and law enforcement saves lives…", "📡 Querying aviation infrastructure — the sky belongs to everyone who protects this community…"]
JURISDICTION_MESSAGES = ["🗺️ Identifying jurisdictions — every boundary represents a community counting on you…", "📐 Loading geographic boundaries — the lines officers cross every shift to keep people safe…", "🏙️ Mapping your jurisdiction — the streets your officers know better than anyone…", "🌆 Matching data to boundaries — every block is someone's home, someone's neighborhood…", "📍 Finding your coverage area — the community that trusts you with their safety…", "🗺️ Resolving jurisdictions — where every call for help deserves an answer…"]
SPATIAL_MESSAGES = ["⚡ Crunching coverage geometry — because your officers deserve precision, not guesswork…", "🧮 Computing spatial matrices — doing the math so your team can focus on what matters…", "📊 Building coverage model — every calculation brings faster response one step closer…", "🔬 Analyzing incident patterns — understanding the city so your officers can better protect it…", "💡 Optimizing station geometry — smart placement means no neighborhood is left behind…", "🧠 Modeling response zones — technology standing behind the officers who stand for us…"]

def get_hero_message(): return random.choice(HERO_MESSAGES)
def get_faa_message(): return random.choice(FAA_MESSAGES)
def get_airfield_message(): return random.choice(AIRFIELD_MESSAGES)
def get_jurisdiction_message(): return random.choice(JURISDICTION_MESSAGES)
def get_spatial_message(): return random.choice(SPATIAL_MESSAGES)

def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f: return base64.b64encode(f.read()).decode()
    except Exception: return None

# ============================================================
# COMMAND CENTER ANALYTICS GENERATOR
# ============================================================
def generate_command_center_html(df, total_orig_calls, export_mode=False, shift_hours=8):
    """Generates the full Command Center visual suite with interactive Calendar tooltips."""
    if df is None or df.empty or 'date' not in df.columns:
        return "<div style='color:gray; padding:20px;'>Analytics unavailable. Missing date/time fields.</div>"
    
    import calendar as _cal
    
    df_ana = df.copy()
    df_ana['dt_obj'] = pd.to_datetime(df_ana['date'] + ' ' + df_ana.get('time', '00:00:00'), errors='coerce')
    df_ana = df_ana.dropna(subset=['dt_obj'])
    if df_ana.empty: return "<div>No valid dates found in data.</div>"
    
    df_ana['hour'] = df_ana['dt_obj'].dt.hour
    df_ana['dow'] = df_ana['dt_obj'].dt.dayofweek # Mon=0, Sun=6
    df_ana['date_key'] = df_ana['dt_obj'].dt.strftime('%Y-%m-%d')
    df_ana['month_key'] = df_ana['dt_obj'].dt.strftime('%Y-%m')
    
    total_calls = len(df_ana)
    orig_calls_display = f"{total_orig_calls:,}" if total_orig_calls > 0 else f"{total_calls:,}"
    if total_orig_calls > total_calls:
        orig_calls_display += f" <br><span style='font-size:12px;color:#888;'>(Sampled: {total_calls:,})</span>"
    
    date_counts = df_ana['date_key'].value_counts().to_dict()
    date_hourly = {}
    for d, grp in df_ana.groupby('date_key'):
        date_hourly[d] = grp['hour'].value_counts().reindex(range(24), fill_value=0).tolist()
        
    hourly_counts = df_ana['hour'].value_counts().reindex(range(24), fill_value=0).tolist()
    shift_html = ""
    for win in [8, 10, 12]:
        best_v, best_s = 0, 0
        for s in range(24):
            v = sum(hourly_counts[(s + h) % 24] for h in range(win))
            if v > best_v: best_v, best_s = v, s
        pct = (best_v / max(total_calls, 1)) * 100
        
        is_active = (win == shift_hours)
        bg_color = "rgba(0,210,255,0.08)" if is_active else "#0c0c12"
        br_color = "#00D2FF" if is_active else "#252535"
        badge_html = "<div style='font-size:8px; color:#00D2FF; margin-left:10px; border:1px solid #00D2FF; padding:1px 4px; border-radius:2px;'>SELECTED</div>" if is_active else ""
        
        shift_html += f"""
        <div style="display:flex; align-items:center; background:{bg_color}; border:1px solid {br_color}; padding:8px; margin-bottom:5px; border-radius:4px; transition:all 0.2s;">
            <div style="width:50px; font-weight:800; color:#fff; font-size:13px;">{win}hr</div>
            <div style="width:110px; font-family:monospace; color:#00D2FF; font-size:12px;">{best_s:02d}:00 - {(best_s+win)%24:02d}:00</div>
            <div style="flex-grow:1; background:#1a1a26; height:8px; border-radius:4px; margin:0 15px; position:relative;">
                <div style="position:absolute; left:{(best_s/24)*100}%; width:{(win/24)*100}%; background:#00D2FF; height:100%; border-radius:4px; opacity:0.6;"></div>
            </div>
            <div style="width:50px; text-align:right; font-family:monospace; color:#00D2FF; font-size:13px;">{pct:.1f}%</div>
            {badge_html}
        </div>"""

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

    month_keys = sorted(df_ana['month_key'].dropna().unique())
    cal_html = "<div style='display:grid; grid-template-columns:repeat(auto-fill, minmax(250px, 1fr)); gap:15px; margin-top:20px;'>"
    
    for mk in month_keys[:12]:
        yr, mo = int(mk.split('-')[0]), int(mk.split('-')[1])
        m_data = df_ana[df_ana['month_key'] == mk]
        m_max = max([date_counts.get(k, 0) for k in m_data['date_key'].unique()] + [1])
        
        cal_html += f"<div style='background:#0c0c12; border:1px solid #1a1a26; border-radius:6px; padding:12px;'>"
        cal_html += f"<div style='display:flex; justify-content:space-between; align-items:baseline; border-bottom:1px solid #252535; padding-bottom:6px; margin-bottom:8px;'><span style='color:#00D2FF; font-weight:800; font-size:12px; text-transform:uppercase; letter-spacing:1px;'>{_cal.month_name[mo]} {yr}</span><span style='color:#7777a0; font-size:10px; font-family:monospace;'>{len(m_data):,} calls</span></div>"
        
        cal_html += "<div style='display:grid; grid-template-columns:repeat(7, 1fr); gap:2px; margin-bottom:4px;'>"
        for i, dname in enumerate(['Su','Mo','Tu','We','Th','Fr','Sa']):
            c = ['#FF6B6B','#4ECDC4','#45B7D1','#F0B429','#96CEB4','#DDA0DD','#FF9A8B'][i]
            cal_html += f"<div style='font-size:9px; text-align:center; color:{c}; font-weight:600;'>{dname}</div>"
        cal_html += "</div>"
        
        cal_html += "<div style='display:grid; grid-template-columns:repeat(7, 1fr); gap:2px;'>"
        first_dow = _cal.weekday(yr, mo, 1) # Mon=0, Sun=6
        first_dow_sun = (first_dow + 1) % 7
        last_day = _cal.monthrange(yr, mo)[1]
        
        for _ in range(first_dow_sun): cal_html += "<div></div>"
            
        for d in range(1, last_day + 1):
            dk = f"{yr}-{mo:02d}-{d:02d}"
            cnt = date_counts.get(dk, 0)
            ratio = cnt / m_max if m_max > 0 else 0
            
            if cnt == 0: bg, fc, cls = '#08080f', '#333', 'day-zero'
            elif ratio >= 0.85: bg, fc, cls = '#3d0a0a', '#ff4444', 'day-peak'
            elif ratio >= 0.55: bg, fc, cls = '#3d1a00', '#ff8c00', 'day-high'
            elif ratio >= 0.25: bg, fc, cls = '#2d2d00', '#d4c000', 'day-med'
            else: bg, fc, cls = '#0d3320', '#2ecc71', 'day-low'
            
            dow_idx = (_cal.weekday(yr, mo, d) + 1) % 7 
            stripe_color = ['#FF6B6B','#4ECDC4','#45B7D1','#F0B429','#96CEB4','#DDA0DD','#FF9A8B'][dow_idx]
            
            stripe_html = f"<div style='position:absolute; bottom:0; left:0; right:0; height:2px; background:{stripe_color}; opacity:0.7; border-radius:0 0 2px 2px;'></div>" if cnt > 0 else ""
            cnt_html = f"<span style='font-size:8px; opacity:0.7; margin-top:1px;'>{cnt}</span>" if cnt > 0 else ""
            
            cal_html += f"<div class='day-cell {cls}' data-date='{dk}' data-count='{cnt}' data-ratio='{ratio}' data-month='{_cal.month_name[mo]}' data-d='{d}' data-y='{yr}' data-dow='{dow_idx}' style='aspect-ratio:1; background:{bg}; border-radius:2px; display:flex; flex-direction:column; align-items:center; justify-content:center; color:{fc}; position:relative; font-family:monospace; cursor:default; border:1px solid transparent; transition:transform 0.1s;' onmouseover='showTooltip(this, event)' onmouseout='hideTooltip()'><span style='font-size:11px; z-index:1; font-weight:bold;'>{d}</span>{cnt_html}{stripe_html}</div>"
            
        cal_html += "</div></div>"
    cal_html += "</div>"

    brinc_info = """
    <div style="margin-bottom:20px; background:#0c0c12; border:1px solid #1a1a26; border-radius:4px; padding:18px 22px; display:grid; grid-template-columns:1fr auto; gap:18px; align-items:start;">
        <div style="font-family:monospace; font-size:11px; color:#7777a0; line-height:1.85;">
            <strong style="color:#00D2FF;font-weight:400;">BRINC</strong> is the largest drone manufacturer focused exclusively on public safety, trusted by <strong style="color:#00D2FF;font-weight:400;">900+ police and fire agencies across all 50 states</strong>. Founded by Blake Resnick and headquartered in <strong style="color:#00D2FF;font-weight:400;">Seattle, WA</strong>, BRINC designs NDAA & CJIS compliant systems built entirely in the USA. The Responder drone reaches 911 calls in under <strong style="color:#00D2FF;font-weight:400;">70 seconds</strong> — arriving before ground units to deliver live video, two-way communication, and real-time situational awareness. BRINC DFR integrates with CAD, RTCC, ALPR, gunshot detection, 911, and evidence management. Agencies resolve approximately <strong style="color:#00D2FF;font-weight:400;">25% of calls for service</strong> without dispatching officers.<br>
            <span style="color:#44445a;font-size:9px;">brincdrones.com · Seattle, WA · NDAA & CJIS Compliant · Made in USA · Backed by Sam Altman & Motorola Solutions</span>
        </div>
        <div style="display:flex; flex-direction:column; gap:5px;">
            <div style="font-family:monospace; font-size:9px; letter-spacing:1px; padding:3px 9px; border-radius:2px; text-transform:uppercase; white-space:nowrap; background:rgba(0,210,255,0.06); border:1px solid rgba(0,210,255,0.15); color:#00D2FF;">LEMUR 2 · Indoor Tactical</div>
            <div style="font-family:monospace; font-size:9px; letter-spacing:1px; padding:3px 9px; border-radius:2px; text-transform:uppercase; white-space:nowrap; background:rgba(0,210,255,0.06); border:1px solid rgba(0,210,255,0.15); color:#00D2FF;">RESPONDER · DFR Platform</div>
            <div style="font-family:monospace; font-size:9px; letter-spacing:1px; padding:3px 9px; border-radius:2px; text-transform:uppercase; white-space:nowrap; background:rgba(0,210,255,0.06); border:1px solid rgba(0,210,255,0.15); color:#00D2FF;">BRINC BALL · Compact Response</div>
            <div style="font-family:monospace; font-size:9px; letter-spacing:1px; padding:3px 9px; border-radius:2px; text-transform:uppercase; white-space:nowrap; background:rgba(0,210,255,0.06); border:1px solid rgba(0,210,255,0.15); color:#00D2FF;">GUARDIAN · Perimeter ISR</div>
            <div style="font-family:monospace; font-size:9px; letter-spacing:1px; padding:3px 9px; border-radius:2px; text-transform:uppercase; white-space:nowrap; background:rgba(0,210,255,0.06); border:1px solid rgba(0,210,255,0.15); color:#00D2FF;">LIVEOPS · Fleet Operations</div>
            <div style="font-family:monospace; font-size:9px; letter-spacing:1px; padding:3px 9px; border-radius:2px; text-transform:uppercase; white-space:nowrap; background:rgba(0,210,255,0.06); border:1px solid rgba(0,210,255,0.15); color:#00D2FF;">RESPONDER STATION · Auto Dock</div>
        </div>
    </div>
    """

    full_html = f"""
    <div style="background:#000; color:#e8e8f2; font-family: 'Barlow', sans-serif; padding:15px; border-radius:8px;">
        <style>
            .day-cell:hover {{ transform: scale(1.15); z-index: 10; box-shadow: 0 4px 12px rgba(0,0,0,0.5); }}
            .day-peak {{ border-color: #cc0000 !important; font-weight: 700; }}
            #dfr-tooltip {{ position: fixed; z-index: 9999; background: #09090f; border: 1px solid #252535; border-radius: 6px; padding: 12px 16px; font-family: monospace; font-size: 11px; color: #e8e8f2; pointer-events: none; box-shadow: 0 6px 24px rgba(0,0,0,0.8); display: none; min-width: 220px; }}
        </style>
        
        <div id="dfr-tooltip"></div>
        
        <div style="color:#00D2FF; font-weight:900; letter-spacing:3px; font-size:14px; text-transform:uppercase; margin-bottom:20px; border-bottom:1px solid #1a1a26; padding-bottom:10px;">Data Ingestion Analytics</div>
        {brinc_info}
        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:15px; margin-bottom:20px;">
            <div style="background:#0c0c12; border-left:4px solid #00D2FF; padding:15px; border-radius:4px; border-top:1px solid #1a1a26; border-right:1px solid #1a1a26; border-bottom:1px solid #1a1a26;">
                <div style="color:#00D2FF; font-size:26px; font-weight:900; font-family:monospace;">{orig_calls_display}</div>
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

# ============================================================
# AGGRESSIVE DATA PARSER
# ============================================================
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
        if not s or s == 'NAN': return None
        if re.search(r'\bNON[\-\s]?EMERG|\bROUTINE\b|\bINFORMATIONAL\b', s): return 4
        if re.search(r'\bHIGH\b|\bEMERG|\bCRITICAL\b|1', s): return 1
        if re.search(r'\bMED|2', s): return 2
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
            for field in ['lat', 'lon']:
                found = [c for c in raw_df.columns if any(s in c for s in CV[field])]
                if found: res[field] = pd.to_numeric(raw_df[found[0]], errors='coerce')
            
            p_found = [c for c in raw_df.columns if any(s in c for s in CV['priority'])]
            if p_found: res['priority'] = raw_df[p_found[0]].apply(parse_priority)
            else: res['priority'] = 3
            
            d_found = [c for c in raw_df.columns if any(s in c for s in CV['date'])]
            t_found = [c for c in raw_df.columns if any(s in c for s in CV['time'])]
            
            if d_found:
                if t_found and d_found[0] != t_found[0]:
                    dt_series = pd.to_datetime(raw_df[d_found[0]] + ' ' + raw_df[t_found[0]], errors='coerce')
                else:
                    dt_series = pd.to_datetime(raw_df[d_found[0]], errors='coerce')
                
                res['date'] = dt_series.dt.strftime('%Y-%m-%d')
                res['time'] = dt_series.dt.strftime('%H:%M:%S')
            
            all_calls_list.append(res)
        except: continue
        
    if not all_calls_list: return pd.DataFrame()
    return pd.concat(all_calls_list, ignore_index=True).dropna(subset=['lat', 'lon'])

def generate_stations_from_calls(df_calls, max_stations=100):
    """Auto-generate stations by querying OpenStreetMap."""
    lats = df_calls['lat'].dropna().values
    lons = df_calls['lon'].dropna().values
    if len(lats) == 0: return None, "No coordinates available to generate stations."

    q1_la, q3_la = np.percentile(lats, 25), np.percentile(lats, 75)
    q1_lo, q3_lo = np.percentile(lons, 25), np.percentile(lons, 75)
    iqr_la = q3_la - q1_la
    iqr_lo = q3_lo - q1_lo
    mask = (lats >= q1_la - 2.5 * iqr_la) & (lats <= q3_la + 2.5 * iqr_la) & (lons >= q1_lo - 2.5 * iqr_lo) & (lons <= q3_lo + 2.5 * iqr_lo)
    cen_lat, cen_lon = lats[mask].mean(), lons[mask].mean()

    R = 0.25 
    bbox = f"{cen_lat - R},{cen_lon - R},{cen_lat + R},{cen_lon + R}"
    query = (
        f'[out:json][timeout:25];'
        f'(node["amenity"="fire_station"]({bbox});'
        f'node["amenity"="police"]({bbox});'
        f'node["amenity"="school"]({bbox});'
        f'way["amenity"="fire_station"]({bbox});'
        f'way["amenity"="police"]({bbox});'
        f'way["amenity"="school"]({bbox});'
        f');out center;'
    )

    data = None
    for osm_url in ['https://overpass-api.de/api/interpreter', 'https://overpass.kumi.systems/api/interpreter']:
        try:
            req = urllib.request.Request(f"{osm_url}?data={urllib.parse.quote(query)}", headers={'User-Agent': 'BRINC_COS_Optimizer/1.0'})
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode('utf-8'))
            break
        except Exception: continue

    if data is None: return None, "OpenStreetMap query failed. Check your network connection."

    elements = data.get('elements', [])
    rows = []
    for el in elements:
        tags = el.get('tags', {})
        lat = el.get('lat') or (el.get('center') or {}).get('lat')
        lon = el.get('lon') or (el.get('center') or {}).get('lon')
        if lat is None or lon is None: continue
        
        amenity = tags.get('amenity', '')
        type_label = 'Fire' if amenity == 'fire_station' else 'Police' if amenity == 'police' else 'School'
        fac_name = tags.get('name', f"{type_label} Station")
        rows.append({'name': fac_name, 'lat': round(lat, 6), 'lon': round(lon, 6), 'type': type_label})

    if not rows: return None, "No police/fire/school stations found near this area on OpenStreetMap."

    df_s = pd.DataFrame(rows).drop_duplicates(subset=['lat', 'lon']).reset_index(drop=True)
    
    # ENFORCE UNIQUE NAMES FOR OSM DATA
    counts = {}
    new_names = []
    for n in df_s['name']:
        if n in counts:
            counts[n] += 1
            new_names.append(f"{n} ({counts[n]})")
        else:
            counts[n] = 0
            new_names.append(n)
    df_s['name'] = new_names

    if len(df_s) > max_stations:
        priority_order = {'Police': 0, 'Fire': 1, 'School': 2}
        df_s['_pri'] = df_s['type'].map(priority_order).fillna(3)
        df_s = df_s.sort_values('_pri').head(max_stations).drop(columns='_pri').reset_index(drop=True)

    return df_s, f"Auto-generated {len(df_s)} stations from OpenStreetMap."

# ============================================================
# CACHED DATA FUNCTIONS
# ============================================================
@st.cache_data
def reverse_geocode_state(lat, lon):
    url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&zoom=10&addressdetails=1"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'BRINC_COS_Optimizer/1.0'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            address = data.get('address', {})
            state = address.get('state', '')
            city = address.get('city', address.get('town', address.get('village', address.get('county', 'Unknown City'))))
            return state, city
    except Exception: return None, None

@st.cache_data
def fetch_census_population(state_fips, place_name):
    url = f"https://api.census.gov/data/2020/dec/pl?get=P1_001N,NAME&for=place:*&in=state:{state_fips}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))
            search_name = place_name.lower().strip()
            for row in data[1:]:
                place_full = row[1].lower().split(',')[0].strip()
                if place_full == search_name or place_full.startswith(search_name + " "): return int(row[0])
    except Exception: pass
    return None

SHAPEFILE_DIR = "jurisdiction_data"
if not os.path.exists(SHAPEFILE_DIR): os.makedirs(SHAPEFILE_DIR)

@st.cache_data
def fetch_tiger_city_shapefile(state_fips, city_name, output_dir):
    url = f"https://www2.census.gov/geo/tiger/TIGER2023/PLACE/tl_2023_{state_fips}_place.zip"
    try:
        req = urllib.request.urlopen(url, timeout=20)
        zip_file = zipfile.ZipFile(io.BytesIO(req.read()))
        temp_dir = os.path.join(output_dir, f"temp_tiger_{state_fips}")
        os.makedirs(temp_dir, exist_ok=True)
        zip_file.extractall(temp_dir)
        shp_path = glob.glob(os.path.join(temp_dir, "*.shp"))[0]
        gdf = gpd.read_file(shp_path)
        search_name = city_name.lower().strip()
        exact_mask = gdf['NAME'].str.lower().str.strip() == search_name
        if exact_mask.any(): city_gdf = gdf[exact_mask]
        else: city_gdf = gdf[gdf['NAME'].str.lower().str.contains(search_name, case=False, na=False)]
        if not city_gdf.empty:
            city_gdf = city_gdf.dissolve(by='NAME').reset_index()
            save_path = os.path.join(output_dir, f"{city_name.replace(' ', '_')}_{state_fips}.shp")
            city_gdf.to_file(save_path)
            return True, city_gdf
    except Exception as e: return False, None
    return False, None

def generate_mock_faa_grid(minx, miny, maxx, maxy):
    features = []
    x_steps = np.linspace(minx, maxx, 20)
    y_steps = np.linspace(miny, maxy, 20)
    mock_airports = [{"lon": minx + 0.3 * (maxx - minx), "lat": miny + 0.3 * (maxy - miny), "radius": 0.15, "name": "Mock Intl (MCK)"}]
    for i in range(len(x_steps) - 1):
        for j in range(len(y_steps) - 1):
            cell_poly = [[x_steps[i], y_steps[j]], [x_steps[i+1], y_steps[j]], [x_steps[i+1], y_steps[j+1]], [x_steps[i], y_steps[j+1]], [x_steps[i], y_steps[j]]]
            cell_center = Point((x_steps[i] + x_steps[i+1]) / 2, (y_steps[j] + y_steps[j+1]) / 2)
            ceiling, arpt_name = None, ""
            for ap in mock_airports:
                dist_ratio = cell_center.distance(Point(ap["lon"], ap["lat"])) / ap["radius"]
                if dist_ratio < 1.0:
                    if   dist_ratio < 0.15: ceiling, arpt_name = 0,   ap["name"]
                    elif dist_ratio < 0.35: ceiling, arpt_name = 50,  ap["name"]
                    elif dist_ratio < 0.55: ceiling, arpt_name = 100, ap["name"]
                    else:                   ceiling, arpt_name = 200, ap["name"]
                    break
            if ceiling is not None:
                features.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [cell_poly]}, "properties": {"CEILING": ceiling, "ARPT_Name": arpt_name}})
    return {"type": "FeatureCollection", "features": features}

@st.cache_data
def load_faa_parquet(minx, miny, maxx, maxy):
    if not os.path.exists("faa_uasfm.parquet"): return generate_mock_faa_grid(minx, miny, maxx, maxy)
    try:
        gdf = gpd.read_parquet("faa_uasfm.parquet")
        pad = 0.05
        filtered = gdf.cx[minx-pad:maxx+pad, miny-pad:maxy+pad]
        if filtered.empty: return {"type": "FeatureCollection", "features": []}
        return json.loads(filtered.to_json())
    except Exception as e: return generate_mock_faa_grid(minx, miny, maxx, maxy)

def add_faa_laanc_layer_to_plotly(fig, faa_geojson, is_dark=True):
    if not faa_geojson or not faa_geojson.get("features"): return
    text_lons, text_lats, text_strings, text_hovers = [], [], [], []
    for feature in faa_geojson.get("features", []):
        geom = feature.get("geometry")
        props = feature.get("properties", {})
        ceiling = props.get("CEILING")
        arpt = props.get("ARPT_Name") or props.get("ARPT_NAME") or "Unknown Airport"
        if ceiling is None or geom is None or geom.get("type") != "Polygon": continue
        snapped = min(FAA_CEILING_COLORS.keys(), key=lambda v: abs(v - ceiling))
        colors = FAA_CEILING_COLORS.get(snapped, FAA_DEFAULT_COLOR)
        coords = geom["coordinates"][0]
        bx, by = zip(*coords)
        fig.add_trace(go.Scattermapbox(mode="lines", lon=list(bx), lat=list(by), fill="toself", fillcolor=colors["fill"], line=dict(color=colors["line"], width=1.5), hoverinfo="text", text=f"<b>{ceiling} ft AGL</b><br>{arpt}", name=f"LAANC {ceiling}ft", showlegend=False))
        try:
            centroid = shape(geom).centroid
            text_lons.append(centroid.x); text_lats.append(centroid.y); text_strings.append(str(ceiling)); text_hovers.append(f"{ceiling} ft — {arpt}")
        except Exception: pass
    if text_lons:
        fig.add_trace(go.Scattermapbox(mode="text", lon=text_lons, lat=text_lats, text=text_strings, hovertext=text_hovers, hoverinfo="text", textfont=dict(size=10, color="#ffffff" if is_dark else "#000000"), showlegend=False, name="LAANC Labels"))

def get_station_faa_ceiling(lat, lon, faa_geojson):
    if not faa_geojson or 'features' not in faa_geojson: return "400 ft (Class G)"
    pt = Point(lon, lat)
    for feature in faa_geojson['features']:
        if 'geometry' in feature and feature['geometry']:
            try:
                s = shape(feature['geometry'])
                if s.contains(pt):
                    val = feature['properties'].get('CEILING')
                    if val is not None: return f"{val} ft (Controlled)"
            except Exception: pass
    return "400 ft (Class G)"

@st.cache_data
def fetch_airfields(minx, miny, maxx, maxy):
    pad = 0.2
    query = f"""[out:json];(node["aeroway"~"aerodrome|heliport"]({miny-pad},{minx-pad},{maxy+pad},{maxx+pad});way["aeroway"~"aerodrome|heliport"]({miny-pad},{minx-pad},{maxy+pad},{maxx+pad}););out center;"""
    try:
        req = urllib.request.Request("https://overpass-api.de/api/interpreter", data=query.encode('utf-8'), headers={'User-Agent': 'BRINC_Optimizer'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))
            airfields = []
            for el in data.get('elements', []):
                lat = el.get('lat') or el.get('center', {}).get('lat')
                lon = el.get('lon') or el.get('center', {}).get('lon')
                name = el.get('tags', {}).get('name', 'Unknown Airfield')
                if lat and lon: airfields.append({'name': name, 'lat': lat, 'lon': lon})
            return airfields
    except Exception: return []

def get_nearest_airfield(lat, lon, airfields):
    if not airfields: return "No data"
    min_dist = float('inf')
    best = None
    for af in airfields:
        lat1, lon1, lat2, lon2 = map(math.radians, [lat, lon, af['lat'], af['lon']])
        a = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
        dist = 3958.8 * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        if dist < min_dist:
            y = math.sin(lon2-lon1)*math.cos(lat2)
            x = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(lon2-lon1)
            bearing = (math.degrees(math.atan2(y, x)) + 360) % 360
            dirs = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW']
            min_dist = dist
            best = (af['name'], dist, dirs[int((bearing+11.25)/22.5) % 16])
    if best:
        n = best[0][:18] + ("..." if len(best[0]) > 18 else "")
        return f"{best[1]:.1f}mi {best[2]} ({n})"
    return "No data"

def generate_random_points_in_polygon(polygon, num_points):
    points = []
    minx, miny, maxx, maxy = polygon.bounds
    while len(points) < num_points:
        x_coords = np.random.uniform(minx, maxx, 1000)
        y_coords = np.random.uniform(miny, maxy, 1000)
        for x, y in zip(x_coords, y_coords):
            if len(points) >= num_points: break
            if polygon.contains(Point(x, y)): points.append((y, x))
    return points

def generate_clustered_calls(polygon, num_points):
    points = []
    minx, miny, maxx, maxy = polygon.bounds
    hotspots = []
    while len(hotspots) < random.randint(5, 15):
        hx, hy = random.uniform(minx, maxx), random.uniform(miny, maxy)
        if polygon.contains(Point(hx, hy)): hotspots.append((hx, hy))
    target_clustered = int(num_points * 0.75)
    while len(points) < target_clustered:
        hx, hy = random.choice(hotspots)
        px, py = np.random.normal(hx, 0.02), np.random.normal(hy, 0.02)
        if polygon.contains(Point(px, py)): points.append((py, px))
    while len(points) < num_points:
        px, py = random.uniform(minx, maxx), random.uniform(miny, maxy)
        if polygon.contains(Point(px, py)): points.append((py, px))
    np.random.shuffle(points)
    return points

def estimate_grants(population):
    if population > 1000000: return "$1.5M - $3.0M+"
    elif population > 500000: return "$500k - $1.5M"
    elif population > 250000: return "$250k - $500k"
    elif population > 100000: return "$100k - $250k"
    else: return "$25k - $100k"

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
        return f"{line1}<br>{rest}"
    if ',' in name_str:
        parts = name_str.split(',')
        if len(parts) >= 3:
            return f"{parts[0].strip()},<br>{parts[1].strip()},<br>{','.join(parts[2:]).strip()}"
    return name_str

def to_kml_color(hex_str):
    h = hex_str.lstrip('#')
    return f"ff{h[4:6]}{h[2:4]}{h[0:2]}" if len(h) == 6 else "ff0000ff"

def calculate_zoom(min_lon, max_lon, min_lat, max_lat):
    lon_diff = max_lon - min_lon
    lat_diff = max_lat - min_lat
    if lon_diff <= 0 or lat_diff <= 0: return 12
    return min(max(min(np.log2(360/lon_diff), np.log2(180/lat_diff)) + 1.6, 5), 18)

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
    fol_stations = kml.newfolder(name="Station Points")
    fol_rings = kml.newfolder(name="Coverage Rings")
    for d in active_drones:
        kml_c = to_kml_color(d['color'])
        pnt = fol_stations.newpoint(name=f"[{d['type'][:3]}] {d['name']}")
        pnt.coords = [(d['lon'], d['lat'])]
        pnt.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/paddle/blu-blank.png'
        lats, lons = get_circle_coords(d['lat'], d['lon'], r_mi=d['radius_m']/1609.34)
        ring_coords = list(zip(lons, lats))
        ring_coords.append(ring_coords[0])
        pol = fol_rings.newpolygon(name=f"Range: {d['name']}")
        pol.outerboundaryis = ring_coords
        pol.style.linestyle.color = kml_c
        pol.style.linestyle.width = 2
        pol.style.polystyle.color = simplekml.Color.changealphaint(60, kml_c)
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
                    name_col = next((c for c in ['NAME','DISTRICT','NAMELSAD'] if c in subset.columns), subset.columns[0])
                    subset['DISPLAY_NAME'] = subset[name_col].astype(str)
                    relevant_polys.append(subset)
        except Exception: continue
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
def precompute_spatial_data(df_calls, df_stations_all, _city_m, epsg_code, resp_radius_mi, guard_radius_mi, center_lat, center_lon, bounds_hash):
    gdf_calls = gpd.GeoDataFrame(df_calls, geometry=gpd.points_from_xy(df_calls.lon, df_calls.lat), crs="EPSG:4326")
    gdf_calls_utm = gdf_calls.to_crs(epsg=int(epsg_code))
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
    max_dist = max(((row['lon']-center_lon)**2 + (row['lat']-center_lat)**2)**0.5 for _, row in df_stations_all.iterrows()) or 1.0
    if not calls_in_city.empty:
        calls_array = np.array(list(zip(calls_in_city.geometry.x, calls_in_city.geometry.y)))
        for idx_pos, (i, row) in enumerate(df_stations_all.iterrows()):
            s_pt_m = gpd.GeoSeries([Point(row['lon'], row['lat'])], crs="EPSG:4326").to_crs(epsg=int(epsg_code)).iloc[0]
            dists = np.sqrt((calls_array[:,0]-s_pt_m.x)**2 + (calls_array[:,1]-s_pt_m.y)**2)
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
            dist_c = ((row['lon']-center_lon)**2 + (row['lat']-center_lat)**2)**0.5
            station_metadata.append({
                'name': row['name'], 'lat': row['lat'], 'lon': row['lon'],
                'clipped_2m': clipped_2m, 'clipped_guard': clipped_guard,
                'avg_dist_r': dists_mi[mask_r].mean() if mask_r.any() else resp_radius_mi*(2/3),
                'avg_dist_g': dists_mi[mask_g].mean() if mask_g.any() else guard_radius_mi*(2/3),
                'centrality': 1.0 - (dist_c / max_dist)
            })
    return calls_in_city, display_calls, resp_matrix, guard_matrix, dist_matrix_r, dist_matrix_g, station_metadata, total_calls

def solve_mclp(resp_matrix, guard_matrix, dist_r, dist_g, num_resp, num_guard, allow_redundancy, incremental=True):
    n_stations, n_calls = resp_matrix.shape
    if n_calls == 0 or (num_resp == 0 and num_guard == 0): return [], [], [], []
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
        penalty = 0.00001
        model += pulp.lpSum(y[i]*weights[i] for i in range(n_u)) - pulp.lpSum(
            x_r[s]*np.sum(u_dist_r[s,:])*penalty + x_g[s]*np.sum(u_dist_g[s,:])*penalty
            for s in range(n_stations))
        for i in range(n_u):
            cover = [x_r[s] for s in range(n_stations) if u_resp[s,i]] + [x_g[s] for s in range(n_stations) if u_guard[s,i]]
            if cover: model += y[i] <= pulp.lpSum(cover)
            else: model += y[i] == 0
        model.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=10, gapRel=0.0))
        return (
            [i for i in range(n_stations) if (pulp.value(x_r[i]) or 0) > 0.5],
            [i for i in range(n_stations) if (pulp.value(x_g[i]) or 0) > 0.5]
        )

    if not incremental:
        res_r, res_g = run_lp(num_resp, num_guard, [], [])
        return res_r, res_g, res_r, res_g
    curr_r, curr_g = [], []
    chrono_r, chrono_g = [], []
    for tg in range(1, num_guard+1):
        next_r, next_g = run_lp(0, tg, curr_r, curr_g)
        chrono_g.extend([x for x in next_g if x not in curr_g])
        curr_r, curr_g = next_r, next_g
    for tr in range(1, num_resp+1):
        next_r, next_g = run_lp(tr, num_guard, curr_r, curr_g)
        chrono_r.extend([x for x in next_r if x not in curr_r])
        curr_r, curr_g = next_r, next_g
    return curr_r, curr_g, chrono_r, chrono_g

@st.cache_resource
def compute_all_elbow_curves(n_calls, _resp_matrix, _guard_matrix, _geos_r, _geos_g, total_area, _bounds_hash, max_stations=30):
    n_st_calls = min(_resp_matrix.shape[0], max_stations)
    n_st_area  = min(_resp_matrix.shape[0], max_stations * 2)

    def greedy_calls(matrix):
        uncovered = np.ones(n_calls, dtype=bool)
        curve = [0.0]
        cov_count = 0
        import heapq as hq
        pq = [(-matrix[i].sum(), i) for i in range(n_st_calls)]
        hq.heapify(pq)
        for _ in range(n_st_calls):
            if not pq: break
            best_s, best_cov = -1, -1
            while pq:
                neg_gain, idx = hq.heappop(pq)
                actual_gain = (matrix[idx] & uncovered).sum()
                if not pq or actual_gain >= -pq[0][0]:
                    best_s, best_cov = idx, actual_gain
                    break
                else:
                    hq.heappush(pq, (-actual_gain, idx))
            if best_s != -1 and best_cov / max(1, n_calls) >= 0.005:
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
        import heapq as hq
        geos_sub = geos[:n_st_area]
        
        pq = [(-geos_sub[i].area, i) for i in range(len(geos_sub))]
        hq.heapify(pq)
        
        for _ in range(len(geos_sub)):
            if not pq: break
            best_s, best_gain = -1, -1
            
            while pq:
                neg_gain, idx = hq.heappop(pq)
                try:
                    actual_gain = current_union.union(geos_sub[idx]).area - current_union.area
                except Exception:
                    actual_gain = 0
                    
                if not pq or actual_gain >= -pq[0][0]:
                    best_s, best_gain = idx, actual_gain
                    break
                else:
                    hq.heappush(pq, (-actual_gain, idx))
                    
            if best_s != -1 and best_gain > 0:
                try:
                    current_union = current_union.union(geos_sub[best_s])
                    curve.append((current_union.area / total_area) * 100)
                except Exception:
                    pass
            else:
                break
        return curve

    with ThreadPoolExecutor() as executor:
        f_cr = executor.submit(greedy_calls, _resp_matrix[:n_st_calls])
        f_cg = executor.submit(greedy_calls, _guard_matrix[:n_st_calls])
        f_ar = executor.submit(greedy_area, _geos_r)
        f_ag = executor.submit(greedy_area, _geos_g)
        c_r, c_g, a_r, a_g = f_cr.result(), f_cg.result(), f_ar.result(), f_ag.result()

    max_len = max(len(c_r), len(c_g), len(a_r), len(a_g))
    def pad(c):
        r = list(c)
        while len(r) < max_len: r.append(np.nan)
        return r
    return pd.DataFrame({
        'Drones': range(max_len),
        'Responder (Calls)': pad(c_r),
        'Responder (Area)':  pad(a_r),
        'Guardian (Calls)':  pad(c_g),
        'Guardian (Area)':   pad(a_g)
    })

# --- PAGE CONFIG ---
# (Already defined at top)

if not st.session_state['csvs_ready']:

    st.markdown(f"""
    <style>
    @keyframes pulseGlow {{
        0%, 100% {{ opacity: 0.55; }}
        50%       {{ opacity: 1.0; }}
    }}
    @keyframes fadeUp {{
        from {{ opacity:0; transform:translateY(14px); }}
        to   {{ opacity:1; transform:translateY(0); }}
    }}
    .brinc-hero {{
        position: relative;
        text-align: center;
        padding: 52px 24px 40px;
        margin-bottom: 36px;
        border-radius: 12px;
        background: radial-gradient(ellipse at 50% 0%,
            rgba(0,210,255,0.13) 0%, rgba(0,0,0,0) 68%);
        border-bottom: 1px solid rgba(0,210,255,0.15);
        overflow: hidden;
        animation: fadeUp 0.5s ease both;
    }}
    .brinc-hero::before {{
        content: '';
        position: absolute; inset: 0;
        background:
            repeating-linear-gradient(0deg,
                transparent, transparent 39px,
                rgba(0,210,255,0.025) 39px,
                rgba(0,210,255,0.025) 40px),
            repeating-linear-gradient(90deg,
                transparent, transparent 79px,
                rgba(0,210,255,0.025) 79px,
                rgba(0,210,255,0.025) 80px);
        pointer-events: none;
    }}
    .brinc-eyebrow {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.62rem;
        font-weight: 700;
        letter-spacing: 4px;
        color: {accent_color};
        text-transform: uppercase;
        opacity: 0.7;
        margin-bottom: 12px;
    }}
    .brinc-h1 {{
        font-family: 'Manrope', sans-serif;
        font-size: clamp(2rem, 4vw, 3rem);
        font-weight: 900;
        color: #ffffff;
        letter-spacing: -0.5px;
        line-height: 1.08;
        margin-bottom: 12px;
    }}
    .brinc-h1 em {{
        font-style: normal;
        color: {accent_color};
    }}
    .brinc-tagline {{
        font-size: 0.88rem;
        color: #666;
        max-width: 500px;
        margin: 0 auto 22px;
        line-height: 1.65;
    }}
    .brinc-badges {{
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 8px;
        margin-top: 4px;
    }}
    .brinc-badge {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(0,210,255,0.07);
        border: 1px solid rgba(0,210,255,0.2);
        border-radius: 100px;
        padding: 4px 13px;
        font-size: 0.64rem;
        font-weight: 700;
        color: {accent_color};
        letter-spacing: 0.8px;
        text-transform: uppercase;
    }}
    .brinc-badge.pulse {{
        animation: pulseGlow 3s ease-in-out infinite;
    }}
    .path-card {{
        background: #080808;
        border: 1px solid #1c1c1c;
        border-radius: 10px;
        padding: 22px 18px 16px;
        position: relative;
        overflow: hidden;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }}
    .path-card::after {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: var(--accent);
        border-radius: 10px 10px 0 0;
    }}
    .path-card:hover {{
        border-color: rgba(255,255,255,0.12);
        box-shadow: 0 0 28px rgba(0,210,255,0.05);
    }}
    .pc-icon  {{ font-size: 1.5rem; display:block; margin-bottom:9px; }}
    .pc-tag   {{ font-size:0.55rem; font-weight:800; letter-spacing:2.5px;
                 text-transform:uppercase; color:var(--accent); margin-bottom:5px; }}
    .pc-title {{ font-size:1rem; font-weight:800; color:#fff;
                 line-height:1.25; margin-bottom:7px; }}
    .pc-desc  {{ font-size:0.7rem; color:#555; line-height:1.6; margin-bottom:0; }}
    .field-footnote {{
        font-size: 0.63rem; color: #3a3a3a; line-height: 1.75;
        margin-top: 10px; border-top: 1px solid #141414;
        padding-top: 10px;
    }}
    .demo-cities {{
        font-size: 0.65rem; color: #444; line-height: 1.9;
        margin-top: 10px;
    }}
    .demo-cities b {{ color: #555; }}
    .demo-check {{
        font-size: 0.63rem; color: #333; line-height: 1.8;
        margin-top: 12px; border-top: 1px solid #141414;
        padding-top: 10px;
    }}
    .demo-check span {{ color: {accent_color}; margin-right: 5px; }}
    </style>

    <div class="brinc-hero">
        <div class="brinc-eyebrow">BRINC Drones · DFR Platform</div>
        <div class="brinc-h1">
            Coverage. Operations.<br><em>Savings.</em>
        </div>
        <div class="brinc-tagline">
            Optimize drone-as-first-responder deployments for any US jurisdiction.
            Model coverage, forecast ROI, and generate grant-ready proposals in minutes.
        </div>
        <div class="brinc-badges">
            <div class="brinc-badge pulse">🛰 3D Swarm Simulation</div>
            <div class="brinc-badge">🗺 Census Boundaries</div>
            <div class="brinc-badge">📄 Grant Narrative Export</div>
            <div class="brinc-badge">✈️ FAA LAANC Overlay</div>
            <div class="brinc-badge">⚡ MCLP Optimizer</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    path_sim_col, path_upload_col, path_demo_col = st.columns(3, gap="medium")

    with path_sim_col:
        st.markdown(f"""
        <div class="path-card" style="--accent:{accent_color};">
            <span class="pc-icon">🗺</span>
            <div class="pc-tag">Path 01</div>
            <div class="pc-title">Simulate Any<br>US Region</div>
            <div class="pc-desc">No data needed. Real Census boundaries + realistic 911 call distribution generated automatically. Stack multiple jurisdictions in one run.</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        for i in range(st.session_state.city_count):
            c1, c2 = st.columns([3, 1])
            c_val = st.session_state['target_cities'][i]['city'] if i < len(st.session_state['target_cities']) else ""
            s_val = st.session_state['target_cities'][i]['state'] if i < len(st.session_state['target_cities']) else "FL"
            c_name = c1.text_input(
                f"City / Town {i+1}", value=c_val, key=f"c_{i}",
                placeholder="e.g. Orlando",
                help="Official municipality name used to fetch the Census boundary."
            )
            state_idx = list(STATE_FIPS.keys()).index(s_val) if s_val in STATE_FIPS else 8
            s_name = c2.selectbox(
                f"State {i+1}", list(STATE_FIPS.keys()), index=state_idx,
                key=f"s_{i}",
                label_visibility="collapsed" if i > 0 else "visible"
            )
            if i < len(st.session_state['target_cities']):
                st.session_state['target_cities'][i] = {"city": c_name, "state": s_name}
            else:
                st.session_state['target_cities'].append({"city": c_name, "state": s_name})

        col_add, col_run = st.columns([1, 1])
        if st.session_state.city_count < 10:
            if col_add.button("＋ City", use_container_width=True, key="add_city_btn"):
                st.session_state.city_count += 1
                st.rerun()
        submit_demo = col_run.button("▶ Run", use_container_width=True, key="run_sim_btn",
                                     help="Fetch boundaries and launch the simulation.")

    with path_upload_col:
        st.markdown(f"""
        <div class="path-card" style="--accent:#39FF14;">
            <span class="pc-icon">📂</span>
            <div class="pc-tag">Path 02</div>
            <div class="pc-title">Upload<br>Any CAD Data</div>
            <div class="pc-desc">
                Drop <b>any</b> CAD export CSV — no renaming needed.
                Optionally also drop a stations CSV; if omitted, stations are
                auto-generated from OpenStreetMap (police, fire, schools).
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "Drop your CAD export (+ optional stations CSV)",
            accept_multiple_files=True,
            label_visibility="collapsed",
            help="One file = raw CAD export (stations auto-generated from OSM). Two files = calls + stations. Column names are detected automatically."
        )

        st.markdown("""
        <div class="field-footnote">
            <b style='color:#555;'>1 file</b> — any CAD export; stations auto-built from OpenStreetMap<br>
            <b style='color:#555;'>2+ files</b> — calls + stations (any column names, any file names)<br>
            Max 25,000 calls · 100 stations · date/time/priority auto-detected
        </div>
        """, unsafe_allow_html=True)

        def _looks_like_stations(fname):
            n = fname.lower()
            return any(k in n for k in ['station','dept','agency','facility','police','fire','loc'])

        if uploaded_files and len(uploaded_files) >= 1:
            f_list = list(uploaded_files)
            call_files = []
            station_file = None

            for f in f_list:
                if _looks_like_stations(f.name):
                    station_file = f
                else:
                    call_files.append(f)
            
            if len(f_list) == 2 and not station_file:
                f0, f1 = f_list
                f0.seek(0); sz0 = len(f0.read()); f0.seek(0)
                f1.seek(0); sz1 = len(f1.read()); f1.seek(0)
                if sz0 >= sz1:
                    call_files = [f0]
                    station_file = f1
                else:
                    call_files = [f1]
                    station_file = f0

            if call_files:
                with st.spinner("🔍 Detecting column types in CAD export…"):
                    df_c = aggressive_parse_calls(call_files)

                if df_c is None or df_c.empty:
                    st.error("❌ Calls file error: Could not parse valid coordinates.")
                    st.stop()

                st.session_state['total_original_calls'] = len(df_c)
                
                # Sample if too large
                if len(df_c) > 25000:
                    df_c = df_c.sample(25000, random_state=42).reset_index(drop=True)
                    st.toast("⚠️ Sampled to 25,000 calls for performance.")
                else:
                    df_c = df_c.reset_index(drop=True)

                # ── Stations: load or auto-generate ───────
                if station_file is not None:
                    with st.spinner("🔍 Reading stations file…"):
                        try:
                            df_s = pd.read_csv(station_file)
                            df_s.columns = [str(c).lower().strip() for c in df_s.columns]
                            if 'latitude' in df_s.columns: df_s = df_s.rename(columns={'latitude':'lat'})
                            if 'longitude' in df_s.columns: df_s = df_s.rename(columns={'longitude':'lon'})
                            if 'station_name' in df_s.columns: df_s = df_s.rename(columns={'station_name':'name'})
                            if 'station_type' in df_s.columns: df_s = df_s.rename(columns={'station_type':'type'})
                            
                            # FORCE LAT/LON TO BE NUMBERS
                            if 'lat' in df_s.columns and 'lon' in df_s.columns:
                                df_s['lat'] = pd.to_numeric(df_s['lat'], errors='coerce')
                                df_s['lon'] = pd.to_numeric(df_s['lon'], errors='coerce')
                            else:
                                raise ValueError("Could not find lat/lon columns.")

                            if 'name' not in df_s.columns: df_s['name'] = [f"Station {i+1}" for i in range(len(df_s))]
                            if 'type' not in df_s.columns: df_s['type'] = 'Police'
                            df_s = df_s.dropna(subset=['lat', 'lon']).reset_index(drop=True)
                            osm_note = "Loaded stations from file."
                        except Exception as e:
                            df_s, osm_note = None, f"Failed: {e}"
                    if df_s is None or df_s.empty:
                        st.error(f"❌ Stations file error: {osm_note}")
                        st.stop()
                else:
                    with st.spinner("🌐 No stations file detected — querying OpenStreetMap for police, fire & schools…"):
                        df_s, osm_note = generate_stations_from_calls(df_c)
                    if df_s is None:
                        st.error(f"❌ Could not auto-generate stations: {osm_note}")
                        st.stop()
                    st.toast(f"✅ {osm_note}")

                if len(df_s) > 100:
                    df_s = df_s.sample(100, random_state=42).reset_index(drop=True)

                lat_min, lat_max = df_s['lat'].min(), df_s['lat'].max()
                lon_min, lon_max = df_s['lon'].min(), df_s['lon'].max()
                df_c = df_c[
                    (df_c['lat'] >= lat_min - 0.5) & (df_c['lat'] <= lat_max + 0.5) &
                    (df_c['lon'] >= lon_min - 0.5) & (df_c['lon'] <= lon_max + 0.5)
                ].reset_index(drop=True)

                st.session_state['df_calls']             = df_c
                st.session_state['df_stations']          = df_s

                with st.spinner(get_jurisdiction_message()):
                    detected_state_full, detected_city = reverse_geocode_state(
                        df_c['lat'].iloc[0], df_c['lon'].iloc[0]
                    )
                    if detected_state_full and detected_state_full in US_STATES_ABBR:
                        st.session_state['active_state'] = US_STATES_ABBR[detected_state_full]
                        if detected_city and detected_city != 'Unknown City':
                            st.session_state['active_city'] = detected_city
                        st.toast(f"📍 Detected: {st.session_state['active_city']}, {st.session_state['active_state']}")
                st.session_state['csvs_ready'] = True
                st.rerun()

    with path_demo_col:
        st.markdown(f"""
        <div class="path-card" style="--accent:#FFD700;">
            <span class="pc-icon">⚡</span>
            <div class="pc-tag">Path 03</div>
            <div class="pc-title">1-Click Demo<br>Large US City</div>
            <div class="pc-desc">Instantly spin up a fully pre-configured scenario for a major US city. Ideal for live stakeholder presentations and platform walkthroughs.</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        if st.button("⚡ Launch Random Demo City", use_container_width=True, key="demo_btn"):
            random.seed(datetime.datetime.now().microsecond + os.getpid())
            already_used = st.session_state.get('_last_demo_city', '')
            candidates = [c for c in FAST_DEMO_CITIES if c[0] != already_used]
            rcity, rstate = random.choice(candidates)
            st.session_state['_last_demo_city'] = rcity
            st.session_state['target_cities'] = [{"city": rcity, "state": rstate}]
            st.session_state.city_count = 1
            for i in range(10):
                st.session_state.pop(f"c_{i}", None)
                st.session_state.pop(f"s_{i}", None)
            st.session_state['trigger_sim'] = True
            st.rerun()

        city_chips = "  ·  ".join([f"{c}" for c, _ in DEMO_CITIES[:12]]) + "  · and more…"
        st.markdown(f"""
        <div class="demo-cities">
            <b>Available Cities</b><br>
            {city_chips}
        </div>
        <div class="demo-check">
            <span>✓</span>Real Census boundaries<br>
            <span>✓</span>Clustered 911 simulation<br>
            <span>✓</span>100 station candidates<br>
            <span>✓</span>Full optimization & export
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="text-align:center; margin-top:8px; font-size:0.63rem; color:#2a2a2a;">
        BRINC Drones, Inc. · <a href="https://brincdrones.com" target="_blank"
        style="color:#333; text-decoration:none;">brincdrones.com</a>
        · All coverage estimates are for planning purposes only.
    </div>
    """, unsafe_allow_html=True)

    if submit_demo or st.session_state.get('trigger_sim', False):
        if st.session_state.get('trigger_sim', False):
            st.session_state['trigger_sim'] = False

        active_targets = [loc for loc in st.session_state['target_cities'] if loc['city'].strip()]
        if not active_targets:
            st.error("Please enter at least one valid city name.")
            st.stop()

        if len(active_targets) == 1:
            st.session_state['active_city']  = active_targets[0]['city']
            st.session_state['active_state'] = active_targets[0]['state']
        else:
            st.session_state['active_city']  = f"{active_targets[0]['city']} & {len(active_targets)-1} others"
            st.session_state['active_state'] = active_targets[0]['state']

        prog = st.progress(0, text="🫡 Preparing tools worthy of those who serve…")
        all_gdfs = []
        total_estimated_pop = 0

        for i, loc in enumerate(active_targets):
            c_name = loc['city'].strip()
            s_name = loc['state']
            prog.progress(10 + int((i / len(active_targets)) * 20),
                          text=f"🗺️ Mapping {c_name}, {s_name} — because every block they patrol matters…")
            success, temp_gdf = fetch_tiger_city_shapefile(STATE_FIPS[s_name], c_name, SHAPEFILE_DIR)
            if success:
                all_gdfs.append(temp_gdf)
                pop = fetch_census_population(STATE_FIPS[s_name], c_name)
                if pop:
                    total_estimated_pop += pop
                    st.toast(f"✅ {c_name} population verified: {pop:,}")
                else:
                    gdf_proj   = temp_gdf.to_crs(epsg=3857)
                    area_sq_mi = gdf_proj.geometry.area.sum() / 2589988.11
                    est = KNOWN_POPULATIONS.get(c_name, int(area_sq_mi * 3500))
                    total_estimated_pop += est
                    st.toast(f"⚠️ {c_name} population estimated: {est:,}")
            else:
                st.warning(f"⚠️ Could not find a boundary for {c_name}, {s_name}. Try another city.")
                if st.session_state.get('_last_demo_city') == c_name:
                    random.seed(datetime.datetime.now().microsecond + os.getpid())
                    candidates = [c for c in DEMO_CITIES if c[0] != c_name]
                    rcity, rstate = random.choice(candidates)
                    st.session_state['_last_demo_city'] = rcity
                    st.session_state['target_cities'] = [{"city": rcity, "state": rstate}]
                    for i in range(10):
                        st.session_state.pop(f"c_{i}", None)
                        st.session_state.pop(f"s_{i}", None)
                    st.rerun()

        if not all_gdfs:
            prog.empty()
            st.error("❌ Could not find Census boundaries for any of the entered locations. Check spelling.")
            st.stop()

        prog.progress(35, text="💙 Boundaries loaded — honoring the officers who know every street…")
        active_city_gdf = pd.concat(all_gdfs, ignore_index=True)
        city_poly = active_city_gdf.geometry.union_all()
        st.session_state['estimated_pop'] = total_estimated_pop

        annual_cfs = int(total_estimated_pop * 0.6)
        st.session_state['total_original_calls'] = annual_cfs
        simulated_points_count = min(int(annual_cfs / 12), 25000)

        prog.progress(55, text="🚔 Modeling 911 calls — every one represents someone who needed help…")
        np.random.seed(42)
        call_points = generate_clustered_calls(city_poly, simulated_points_count)
        
        base_date = datetime.datetime.now() - datetime.timedelta(days=30)
        fake_dts = [(base_date + datetime.timedelta(days=random.randint(0, 30), hours=random.randint(0, 23), minutes=random.randint(0, 59))) for _ in range(simulated_points_count)]
        
        df_demo = pd.DataFrame({
            'lat':      [p[0] for p in call_points],
            'lon':      [p[1] for p in call_points],
            'priority': np.random.choice([1, 2, 3], simulated_points_count, p=[0.15, 0.35, 0.50]),
            'date':     [d.strftime('%Y-%m-%d') for d in fake_dts],
            'time':     [d.strftime('%H:%M:%S') for d in fake_dts]
        })
        st.session_state['df_calls'] = df_demo

        prog.progress(80, text="🏅 Placing stations — giving officers the best possible backup…")
        station_points = generate_random_points_in_polygon(city_poly, 100)
        types = ['Police', 'Fire', 'EMS'] * 34
        st.session_state['df_stations'] = pd.DataFrame({
            'name': [f'Station {i+1}' for i in range(len(station_points))],
            'lat':  [p[0] for p in station_points],
            'lon':  [p[1] for p in station_points],
            'type': types[:len(station_points)]
        })

        prog.progress(100, text="✅ Ready — built for the communities they protect and serve.")
        st.session_state['inferred_daily_calls_override'] = int(annual_cfs / 365)
        st.session_state['csvs_ready'] = True
        st.rerun()

# ============================================================
# MAIN MAP INTERFACE
# ============================================================
if st.session_state['csvs_ready']:
    components.html("<script>window._brincHasData = true;</script>", height=0)

    df_calls = st.session_state['df_calls'].copy()
    df_stations_all = st.session_state['df_stations'].copy()

    with st.spinner(get_jurisdiction_message()):
        master_gdf = find_relevant_jurisdictions(df_calls, df_stations_all, SHAPEFILE_DIR)

    if master_gdf is None or master_gdf.empty:
        min_lon, min_lat = df_calls['lon'].min(), df_calls['lat'].min()
        max_lon, max_lat = df_calls['lon'].max(), df_calls['lat'].max()
        lon_pad = (max_lon - min_lon) * 0.1
        lat_pad = (max_lat - min_lat) * 0.1
        poly = box(min_lon-lon_pad, min_lat-lat_pad, max_lon+lon_pad, max_lat+lat_pad)
        master_gdf = gpd.GeoDataFrame({'DISPLAY_NAME':['Auto-Generated Boundary'],'data_count':[len(df_calls)]}, geometry=[poly], crs="EPSG:4326")

    st.sidebar.markdown('<div class="sidebar-section-header">① Configure</div>', unsafe_allow_html=True)

    total_pts = master_gdf['data_count'].sum()
    master_gdf['LABEL'] = master_gdf['DISPLAY_NAME'] + " (" + (master_gdf['data_count']/total_pts*100).round(1).astype(str) + "%)"
    options_map = dict(zip(master_gdf['LABEL'], master_gdf['DISPLAY_NAME']))
    all_options = master_gdf['LABEL'].tolist()
    
    default_selection = [all_options[0]] if all_options else []
    selected_labels = st.sidebar.multiselect("Jurisdictions", options=all_options, default=default_selection,
                                             help="Select which geographic areas to include in coverage analysis.")

    if not selected_labels:
        st.warning("Please select at least one jurisdiction from the sidebar.")
        st.stop()
        
    selected_names = [options_map[l] for l in selected_labels]
    active_gdf = master_gdf[master_gdf['DISPLAY_NAME'].isin(selected_names)]
    if selected_names and st.session_state.get('active_city') == "Orlando":
        st.session_state['active_city'] = selected_names[0]

    filter_expander = st.sidebar.expander("⚙️ Data Filters", expanded=False)
    with filter_expander:
        if 'type' in df_stations_all.columns:
            all_types = sorted(df_stations_all['type'].dropna().astype(str).unique().tolist())
            if all_types:
                selected_types = st.multiselect("Facility Type", options=all_types, default=all_types,
                                                help="Filter which station types are eligible for drone deployment.")
                if not selected_types:
                    st.warning("Select at least one facility type.")
                    st.stop()
                df_stations_all = df_stations_all[df_stations_all['type'].astype(str).isin(selected_types)].copy().reset_index(drop=True)
                df_stations_all['name'] = "[" + df_stations_all['type'].astype(str) + "] " + df_stations_all['name'].astype(str)
        if 'priority' in df_calls.columns:
            all_priorities = sorted(df_calls['priority'].dropna().unique().tolist())
            if all_priorities:
                selected_priorities = st.multiselect("Incident Priority", options=all_priorities, default=all_priorities,
                                                     help="Filter which call priorities to include in coverage scoring.")
                if not selected_priorities:
                    st.warning("Select at least one priority level.")
                    st.stop()
                df_calls = df_calls[df_calls['priority'].isin(selected_priorities)].copy().reset_index(drop=True)

    if len(df_stations_all) == 0:
        st.error("No stations match the selected filters."); st.stop()
    if len(df_calls) == 0:
        st.error("No calls match the selected filters."); st.stop()

    disp_expander = st.sidebar.expander("👁️ Display Options", expanded=False)
    with disp_expander:
        show_boundaries = st.toggle("Jurisdiction Boundaries", value=True)
        show_heatmap    = st.toggle("911 Call Heatmap", value=False)
        show_health     = st.toggle("Health Score", value=False)
        show_satellite  = st.toggle("Satellite Imagery", value=False)
        show_cards      = st.toggle("Unit Economics Cards", value=True)
        show_faa        = st.toggle("FAA LAANC Airspace", value=False)
        simulate_traffic = st.toggle("Simulate Ground Traffic", value=False)
        traffic_level   = st.slider("Traffic Congestion", 0, 100, 40) if simulate_traffic else 40

    strat_expander = st.sidebar.expander("⚙️ Deployment Strategy", expanded=False)
    with strat_expander:
        incremental_build = st.toggle("Phased Rollout", value=True)
        allow_redundancy  = st.toggle("Allow Coverage Overlap", value=True)

    st.sidebar.markdown('<div class="sidebar-section-header">② Optimize Fleet</div>', unsafe_allow_html=True)

    opt_strategy_raw = st.sidebar.radio("Optimization Goal", ("Call Coverage", "Land Coverage"), horizontal=True)
    opt_strategy = "Maximize Call Coverage" if opt_strategy_raw == "Call Coverage" else "Maximize Land Coverage"

    minx, miny, maxx, maxy = active_gdf.to_crs(epsg=4326).total_bounds
    center_lon = (minx + maxx) / 2
    center_lat = (miny + maxy) / 2
    dynamic_zoom = calculate_zoom(minx, maxx, miny, maxy)
    utm_zone = int((center_lon + 180) / 6) + 1
    epsg_code = int(f"326{utm_zone}") if center_lat > 0 else int(f"327{utm_zone}")

    city_m = None
    city_boundary_geom = None
    try:
        active_utm = active_gdf.to_crs(epsg=epsg_code)
        full_boundary_utm = (active_utm.geometry.union_all() if hasattr(active_utm.geometry, 'union_all')
                             else active_utm.geometry.unary_union).buffer(0.1)
        full_boundary_utm = full_boundary_utm.buffer(-0.1)
        city_m = full_boundary_utm
        city_boundary_geom = gpd.GeoSeries([full_boundary_utm], crs=epsg_code).to_crs(epsg=4326).iloc[0]
    except Exception as e:
        st.error(f"Geometry Error: {e}"); st.stop()

    n = len(df_stations_all)

    area_sq_mi = city_m.area / 2589988.11 if city_m and not city_m.is_empty else 100.0
    r_resp_est = st.session_state.get('r_resp', 2.0)
    r_guard_est = st.session_state.get('r_guard', 8.0)
    
    # Dynamic Sliders based on Area Size
    area_sq_mi = city_m.area / 2589988.11 if city_m and not city_m.is_empty else 100.0
    r_resp_est = st.session_state.get('r_resp', 2.0)
    r_guard_est = st.session_state.get('r_guard', 8.0)
    
    max_resp_calc = min(n, int(math.ceil(area_sq_mi / (math.pi * (r_resp_est**2)))) + 5)
    max_guard_calc = min(n, int(math.ceil(area_sq_mi / (math.pi * (r_guard_est**2)))) + 5)

    # Safely pull the default values without exceeding the allowed maximums
    val_r = min(st.session_state.get('k_resp', 2), max_resp_calc)
    val_g = min(st.session_state.get('k_guard', 0), max_guard_calc)

    k_responder = st.sidebar.slider("🚁 Responder Count", 0, max(1, max_resp_calc), val_r,
                                    help="Short-range tactical drones (2-3mi radius).")
    k_guardian  = st.sidebar.slider("🦅 Guardian Count",  0, max(1, max_guard_calc), val_g,
                                    help="Long-range heavy-lift drones (up to 8mi radius).")
    
    # THESE TWO LINES WERE MISSING!
    resp_radius_mi  = st.sidebar.slider("🚁 Responder Range (mi)", 2.0, 3.0, st.session_state.get('r_resp', 2.0), step=0.5)
    guard_radius_mi = st.sidebar.slider("🦅 Guardian Range (mi)", 1, 8, int(st.session_state.get('r_guard', 8)))

    st.session_state.update({'k_resp': k_responder, 'k_guard': k_guardian, 'r_resp': resp_radius_mi, 'r_guard': guard_radius_mi})

    bounds_hash = f"{minx}_{miny}_{maxx}_{maxy}_{n}_{resp_radius_mi}_{guard_radius_mi}"

    prog2 = st.sidebar.empty()
    prog2.caption(get_spatial_message())
    calls_in_city, display_calls, resp_matrix, guard_matrix, dist_matrix_r, dist_matrix_g, station_metadata, total_calls = precompute_spatial_data(
        df_calls, df_stations_all, city_m, epsg_code, resp_radius_mi, guard_radius_mi, center_lat, center_lon, bounds_hash
    )
    df_curve = compute_all_elbow_curves(
        total_calls, resp_matrix, guard_matrix,
        [s['clipped_2m'] for s in station_metadata],
        [s['clipped_guard'] for s in station_metadata],
        city_m.area if city_m else 1.0, bounds_hash,
        max_stations=30
    )
    prog2.empty()

    def get_max_drones(col_name):
        series = df_curve[col_name].dropna()
        if len(series) == 0: return 1
        idx_99 = series[series >= 99.0].first_valid_index()
        fallback = series.index[-1]
        return int(df_curve.loc[idx_99 if idx_99 is not None else fallback, 'Drones'])

    with st.spinner(get_faa_message()):
        faa_geojson = load_faa_parquet(minx, miny, maxx, maxy)
    with st.spinner(get_airfield_message()):
        airfields = fetch_airfields(minx, miny, maxx, maxy)

    try:
        st.sidebar.image("logo.png", use_container_width=True)
    except FileNotFoundError:
        st.sidebar.markdown(f"<div style='font-size:2rem;font-weight:900;letter-spacing:4px;color:{accent_color};text-align:center;padding-bottom:10px;'>BRINC</div>", unsafe_allow_html=True)

    st.sidebar.markdown('<div class="sidebar-section-header">① Configure</div>', unsafe_allow_html=True)

    inferred_daily = st.session_state.get('inferred_daily_calls_override', max(1, int(total_calls/365)))
    calls_per_day = st.sidebar.slider("Total Daily Calls (citywide)", 1, max(100, inferred_daily*3), inferred_daily)

    st.sidebar.markdown(f"<div style='font-size:0.72rem; color:{text_muted}; margin-top:8px; margin-bottom:2px;'>DFR Dispatch Rate (%)</div>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<div style='font-size:0.65rem; color:#666; margin-bottom:4px;'>What % of in-range calls will the drone be sent to?</div>", unsafe_allow_html=True)
    dfr_dispatch_rate = st.sidebar.slider("DFR Dispatch Rate", 1, 100, st.session_state.get('dfr_rate',25), label_visibility="collapsed") / 100.0

    st.sidebar.markdown(f"<div style='font-size:0.72rem; color:{text_muted}; margin-top:8px; margin-bottom:2px;'>Calls Resolved Without Officer Dispatch (%)</div>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<div style='font-size:0.65rem; color:#666; margin-bottom:4px;'>Of drone-attended calls, what % close without a patrol car?</div>", unsafe_allow_html=True)
    deflection_rate = st.sidebar.slider("Resolution Rate", 0, 100, st.session_state.get('deflect_rate',30), label_visibility="collapsed") / 100.0

    st.session_state['dfr_rate']    = int(dfr_dispatch_rate * 100)
    st.session_state['deflect_rate'] = int(deflection_rate * 100)

    # ── OPTIMIZATION ──────────────────────────────────────────────────
    active_resp_names, active_guard_names = [], []
    active_resp_idx, active_guard_idx = [], []  # <-- This prevents the NameError!
    chrono_r, chrono_g = [], []
    best_combo = None

    opt_cache_key = f"{k_responder}_{k_guardian}_{resp_radius_mi}_{guard_radius_mi}_{opt_strategy}_{allow_redundancy}_{incremental_build}_{bounds_hash}"

    if k_responder + k_guardian > n:
        st.error("⚠️ Over-Deployment: Total drones exceed available stations.")
        active_resp_names, active_guard_names = [], []
        chrono_r, chrono_g = [], []
        best_combo = None
    elif k_responder == 0 and k_guardian == 0:
        active_resp_names, active_guard_names = [], []
        chrono_r, chrono_g = [], []
        best_combo = None
    else:
        if st.session_state.get('_opt_cache_key') != opt_cache_key:
            if opt_strategy == "Maximize Call Coverage":
                stage_bar = st.empty()
                stage_bar.info("🧠 Optimizing coverage — because smarter deployment means safer streets…")
                r_best, g_best, chrono_r, chrono_g = solve_mclp(
                    resp_matrix, guard_matrix, dist_matrix_r, dist_matrix_g,
                    k_responder, k_guardian, allow_redundancy, incremental=incremental_build
                )
                best_combo = (tuple(r_best), tuple(g_best))
                stage_bar.empty()
                st.toast("✅ Optimization complete — your officers just got powerful new backup!", icon="✅")
            else:
                def evaluate_combo(rg_combo):
                    r_combo, g_combo = rg_combo
                    if allow_redundancy:
                        score_area = (unary_union([station_metadata[i]['clipped_2m'] for i in r_combo]).area if r_combo else 0.0) + \
                                     (unary_union([station_metadata[i]['clipped_guard'] for i in g_combo]).area if g_combo else 0.0)
                    else:
                        geos = [station_metadata[i]['clipped_2m'] for i in r_combo] + [station_metadata[i]['clipped_guard'] for i in g_combo]
                        score_area = unary_union(geos).area if geos else 0.0
                    cov_r = resp_matrix[list(r_combo)].any(axis=0) if r_combo else np.zeros(total_calls, bool)
                    cov_g = guard_matrix[list(g_combo)].any(axis=0) if g_combo else np.zeros(total_calls, bool)
                    score_calls = np.logical_or(cov_r, cov_g).sum() if total_calls > 0 else 0
                    score_cent  = sum(station_metadata[i]['centrality'] for i in list(r_combo)+list(g_combo))
                    return (score_area, score_calls, score_cent, rg_combo)

                stage_bar = st.empty()
                stage_bar.info("🗺️ Maximizing land coverage — no neighborhood left behind…")
                if incremental_build:
                    locked_r, locked_g = (), ()
                    chrono_r, chrono_g = [], []
                    for _ in range(k_guardian):
                        best_pick = max(
                            [s for s in range(n) if s not in locked_g and (allow_redundancy or s not in locked_r)],
                            key=lambda s: evaluate_combo((locked_r, tuple(sorted(list(locked_g)+[s])))),
                            default=None
                        )
                        if best_pick is not None:
                            locked_g = tuple(sorted(list(locked_g)+[best_pick]))
                            chrono_g.append(best_pick)
                    for _ in range(k_responder):
                        best_pick = max(
                            [s for s in range(n) if s not in locked_r and (allow_redundancy or s not in locked_g)],
                            key=lambda s: evaluate_combo((tuple(sorted(list(locked_r)+[s])), locked_g)),
                            default=None
                        )
                        if best_pick is not None:
                            locked_r = tuple(sorted(list(locked_r)+[best_pick]))
                            chrono_r.append(best_pick)
                    best_combo = (locked_r, locked_g)
                else:
                    total_possible = math.comb(n, k_responder) * (math.comb(n-k_responder, k_guardian) if n >= k_responder else 1)
                    if total_possible > 3000:
                        combos = list(set(
                            (tuple(sorted(c[:k_responder])), tuple(sorted(c[k_responder:])))
                            for c in [np.random.choice(range(n), k_responder+k_guardian, replace=False) for _ in range(3000)]
                        ))
                    else:
                        combos = [(r_c, g_c) for r_c in itertools.combinations(range(n), k_responder)
                                  for g_c in (itertools.combinations([x for x in range(n) if x not in r_c], k_guardian) if k_guardian > 0 else [()])]
                    with ThreadPoolExecutor() as ex:
                        results = list(ex.map(evaluate_combo, combos))
                    best_combo = max(results, key=lambda x: x[:3])[3]
                    chrono_r, chrono_g = list(best_combo[0]), list(best_combo[1])
                stage_bar.empty()
                st.toast("✅ Coverage optimized — every corner of the city now has aerial support!", icon="✅")

            st.session_state['_opt_cache_key']  = opt_cache_key
            st.session_state['_opt_best_combo'] = best_combo
            st.session_state['_opt_chrono_r']   = chrono_r
            st.session_state['_opt_chrono_g']   = chrono_g
        else:
            best_combo = st.session_state.get('_opt_best_combo')
            chrono_r   = st.session_state.get('_opt_chrono_r', [])
            chrono_g   = st.session_state.get('_opt_chrono_g', [])

        if best_combo is not None:
            r_best, g_best = best_combo
            active_resp_names  = [station_metadata[i]['name'] for i in r_best]
            active_guard_names = [station_metadata[i]['name'] for i in g_best]
            active_resp_idx  = list(r_best)
            active_guard_idx = list(g_best)
        else:
            active_resp_names, active_guard_names = [], []
            active_resp_idx, active_guard_idx = [], []

    # ── METRICS ───────────────────────────────────────────────────────
    area_covered_perc = overlap_perc = calls_covered_perc = 0.0

    ordered_deployments_raw = []
    for idx in chrono_g:
        if idx in active_guard_idx: ordered_deployments_raw.append((idx,'GUARDIAN'))
    for idx in chrono_r:
        if idx in active_resp_idx: ordered_deployments_raw.append((idx,'RESPONDER'))
    for idx in active_resp_idx:
        if idx not in chrono_r: ordered_deployments_raw.append((idx,'RESPONDER'))
    for idx in active_guard_idx:
        if idx not in chrono_g: ordered_deployments_raw.append((idx,'GUARDIAN'))

    active_color_map = {}
    c_idx = 0
    for idx, d_type in ordered_deployments_raw:
        key = f"{idx}_{d_type}" # Using strictly unique index instead of name
        if key not in active_color_map:
            active_color_map[key] = STATION_COLORS[c_idx % len(STATION_COLORS)]
            c_idx += 1

    active_geos = [station_metadata[i]['clipped_2m'] for i in active_resp_idx] + \
                  [station_metadata[i]['clipped_guard'] for i in active_guard_idx]

    if active_geos and not city_m.is_empty:
        area_covered_perc = (unary_union(active_geos).area / city_m.area) * 100
    if active_geos and total_calls > 0:
        cov_r = resp_matrix[active_resp_idx].any(axis=0) if active_resp_idx else np.zeros(total_calls, bool)
        cov_g = guard_matrix[active_guard_idx].any(axis=0) if active_guard_idx else np.zeros(total_calls, bool)
        calls_covered_perc = (np.logical_or(cov_r, cov_g).sum() / total_calls) * 100
    if active_geos:
        inters = [active_geos[i].intersection(active_geos[j])
                  for i in range(len(active_geos)) for j in range(i+1, len(active_geos))
                  if not active_geos[i].intersection(active_geos[j]).is_empty]
        if inters and not city_m.is_empty:
            overlap_perc = (unary_union(inters).area / city_m.area) * 100

    # ── BUDGET CALCULATIONS ───────────────────────────────────────────
    actual_k_responder = len(active_resp_names)
    actual_k_guardian  = len(active_guard_names)
    capex_resp  = actual_k_responder * CONFIG["RESPONDER_COST"]
    capex_guard = actual_k_guardian  * CONFIG["GUARDIAN_COST"]
    fleet_capex = capex_resp + capex_guard

    annual_savings = 0
    break_even_text = "N/A"
    daily_drone_only_calls = 0
    covered_daily_calls = 0
    daily_dfr_responses = 0

    if fleet_capex > 0:
        covered_daily_calls    = calls_per_day * (calls_covered_perc / 100.0)
        daily_dfr_responses    = covered_daily_calls * dfr_dispatch_rate
        daily_drone_only_calls = daily_dfr_responses * deflection_rate
        if daily_drone_only_calls > 0:
            monthly_savings = (CONFIG["OFFICER_COST_PER_CALL"] - CONFIG["DRONE_COST_PER_CALL"]) * daily_drone_only_calls * 30.4
            annual_savings  = monthly_savings * 12
            break_even_text = f"{fleet_capex / monthly_savings:.1f} MONTHS"

    if fleet_capex > 0:
        st.sidebar.markdown(f"""
        <div style="background:{budget_box_bg}; border:1px solid {budget_box_border}; padding:12px; border-radius:4px;
             text-align:center; margin:8px 0 12px 0; box-shadow:0 2px 5px {budget_box_shadow};">
            <div style="font-size:0.7rem; color:{text_muted}; font-weight:600; text-transform:uppercase; letter-spacing:0.5px;">Annual Capacity Value</div>
            <div style="font-size:1.8rem; font-weight:900; color:{budget_box_border}; font-family:monospace;">${annual_savings:,.0f}</div>
            <div style="border-top:1px solid {card_border}; margin:8px 0;"></div>
            <div style="display:flex; justify-content:space-between; font-size:0.72rem; margin-bottom:3px;">
                <span style="color:{text_muted};">Calls in range:</span>
                <span style="color:{text_main}; font-weight:700;">{covered_daily_calls:.1f}/day</span>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.72rem; margin-bottom:3px;">
                <span style="color:{text_muted};">DFR flights ({int(dfr_dispatch_rate*100)}%):</span>
                <span style="color:{text_main}; font-weight:700;">{daily_dfr_responses:.1f}/day</span>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.72rem; margin-bottom:8px;">
                <span style="color:{text_muted};">Resolved no dispatch:</span>
                <span style="color:{text_main}; font-weight:700;">{daily_drone_only_calls:.1f}/day</span>
            </div>
            <div style="border-top:1px dashed {card_border}; margin:6px 0;"></div>
            <div style="display:flex; justify-content:space-between; font-size:0.72rem; margin-bottom:3px;">
                <span style="color:{text_muted};">Fleet CapEx:</span>
                <span style="color:{text_main}; font-weight:700;">${fleet_capex:,.0f}</span>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.72rem;">
                <span style="color:{text_muted};">Break-even:</span>
                <span style="color:{budget_box_border}; font-weight:700;">{break_even_text}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.info("👈 Set Responder/Guardian counts above to calculate budget impact.")

    # ── BUILD DRONE OBJECTS ───────────────────────────────────────────
    active_drones = []
    cumulative_mask = np.zeros(total_calls, dtype=bool) if total_calls > 0 else None
    step = 1
    for idx, d_type in ordered_deployments_raw:
        if d_type == 'RESPONDER':
            cov_array = resp_matrix[idx]; cost = CONFIG["RESPONDER_COST"]
            speed_mph = CONFIG["RESPONDER_SPEED"]; avg_dist = station_metadata[idx]['avg_dist_r']
            radius_m  = resp_radius_mi * 1609.34
        else:
            cov_array = guard_matrix[idx]; cost = CONFIG["GUARDIAN_COST"]
            speed_mph = CONFIG["GUARDIAN_SPEED"]; avg_dist = station_metadata[idx]['avg_dist_g']
            radius_m  = guard_radius_mi * 1609.34
        map_color    = active_color_map[f"{idx}_{d_type}"]
        avg_time_min = (avg_dist / speed_mph) * 60
        d_lat = station_metadata[idx]['lat']; d_lon = station_metadata[idx]['lon']

        d = {
            'idx': idx, 'name': station_metadata[idx]['name'],
            'lat': d_lat, 'lon': d_lon, 'type': d_type, 'cost': cost,
            'cov_array': cov_array, 'color': map_color,
            'deploy_step': step if (idx in chrono_r or idx in chrono_g) else "MANUAL",
            'avg_time_min': avg_time_min, 'speed_mph': speed_mph, 'radius_m': radius_m,
            'faa_ceiling': get_station_faa_ceiling(d_lat, d_lon, faa_geojson),
            'nearest_airport': get_nearest_airfield(d_lat, d_lon, airfields)
        }

        if total_calls > 0 and cumulative_mask is not None:
            marginal_mask    = cov_array & ~cumulative_mask
            marginal_historic = np.sum(marginal_mask)
            d['assigned_indices'] = np.where(marginal_mask)[0]
            cumulative_mask  = cumulative_mask | cov_array
            d['marginal_perc'] = marginal_historic / total_calls
            marginal_daily   = calls_per_day * d['marginal_perc']
            d['marginal_flights']   = marginal_daily * dfr_dispatch_rate
            d['marginal_deflected'] = d['marginal_flights'] * deflection_rate
            all_cov = np.vstack([resp_matrix[i] for i in active_resp_idx] + [guard_matrix[i] for i in active_guard_idx]) if (active_resp_idx or active_guard_idx) else np.zeros((1, total_calls), dtype=bool)
            shared_mask = d['cov_array'] & (all_cov.sum(axis=0) > 1)
            d['shared_flights']  = (np.sum(shared_mask) / total_calls) * calls_per_day * dfr_dispatch_rate
            d['monthly_savings'] = (CONFIG["OFFICER_COST_PER_CALL"] - CONFIG["DRONE_COST_PER_CALL"]) * d['marginal_deflected'] * 30.4
            d['annual_savings']  = d['monthly_savings'] * 12
            d['be_text'] = f"{d['cost']/d['monthly_savings']:.1f} MO" if d['monthly_savings'] > 0 else "N/A"
        else:
            d.update({'assigned_indices':[],'annual_savings':0,'marginal_flights':0,
                      'marginal_deflected':0,'shared_flights':0,'be_text':"N/A"})
        active_drones.append(d)
        step += 1

    pop_metric = st.session_state.get('estimated_pop', 250000)
    grant_bracket = estimate_grants(pop_metric)
    st.sidebar.markdown(f"""
    <div style="margin-top:12px; background:{card_bg}; border:1px solid {budget_box_border}; padding:10px; border-radius:4px; margin-bottom:10px;">
        <div style="font-size:0.68rem; color:{text_muted}; font-weight:bold; text-transform:uppercase;">Est. Grant Eligibility</div>
        <div style="font-size:1.1rem; color:{budget_box_border}; font-weight:bold; font-family:monospace;">{grant_bracket}</div>
    </div>
    <div style="font-size:0.73rem; color:{text_muted}; line-height:1.5; margin-bottom:10px;">
        <a href="https://bja.ojp.gov/program/jag/overview" target="_blank" style="color:{accent_color}; font-weight:bold;">DOJ Byrne JAG</a> — UAS procurement eligible<br>
        <a href="https://www.fema.gov/grants/preparedness/homeland-security" target="_blank" style="color:{accent_color}; font-weight:bold;">FEMA HSGP</a> — CapEx offset for tactical deployments
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    if show_health:
        norm_redundancy = min(overlap_perc/35.0, 1.0)*100
        health_score = (calls_covered_perc*0.50) + (area_covered_perc*0.35) + (norm_redundancy*0.15)
        h_color, h_label = (accent_color,"OPTIMAL") if health_score>=80 else ("#94c11f","GOOD") if health_score>=70 else ("#ffc107","MARGINAL") if health_score>=55 else ("#dc3545","ESSENTIAL")
        st.markdown(f"""<div style="background:{card_bg}; border-left:5px solid {h_color}; border:1px solid {card_border};
            padding:10px; border-radius:4px; color:{text_main}; margin-bottom:10px;
            display:flex; align-items:center; justify-content:space-between;">
            <span style="font-size:1.4em; font-weight:bold; color:{h_color};">Department Health Score: {health_score:.1f}%</span>
            <span style="font-size:1.2em; background:rgba(128,128,128,0.15); padding:2px 10px; border-radius:4px;">{h_label}</span>
            </div>""", unsafe_allow_html=True)

    if simulate_traffic:
        avg_ground_speed = CONFIG["DEFAULT_TRAFFIC_SPEED"] * (1 - traffic_level/100)
        eval_dist  = guard_radius_mi if active_guard_names else resp_radius_mi
        eval_speed = CONFIG["GUARDIAN_SPEED"] if active_guard_names else CONFIG["RESPONDER_SPEED"]
        if (active_resp_names or active_guard_names) and avg_ground_speed > 0:
            time_saved = ((eval_dist*1.4/avg_ground_speed) - (eval_dist/eval_speed)) * 60
            gain_val = f"{time_saved:.1f} min"
        else:
            gain_val = "N/A"
    else:
        gain_val = None

    orig_calls = st.session_state.get('total_original_calls', total_calls)
    call_str = f"{orig_calls:,}"
    if orig_calls > total_calls:
        call_str += f" <br><span style='font-size:0.55em;color:#888;'>(Sampled: {total_calls:,})</span>"

    # Calculate Date Range of CAD data (if available)
    date_range_str = "Simulated / Unknown"
    if 'date' in df_calls.columns:
        try:
            min_date = pd.to_datetime(df_calls['date']).min().strftime('%b %Y')
            max_date = pd.to_datetime(df_calls['date']).max().strftime('%b %Y')
            date_range_str = f"{min_date} – {max_date}" if min_date != max_date else min_date
        except:
            pass

    avg_resp_time = sum(d['avg_time_min'] for d in active_drones) / len(active_drones) if active_drones else 0.0

    # 1. THE SINGLE-LINE EXECUTIVE HEADER
    logo_b64 = get_base64_of_bin_file("logo.png")
    main_logo_html = f'<img src="data:image/png;base64,{logo_b64}" style="height:24px; vertical-align:middle; margin-right:15px;">' if logo_b64 else f'<span style="font-size:1.5rem; font-weight:900; letter-spacing:2px; color:{accent_color}; margin-right:15px;">BRINC</span>'

    header_html = f"""
    <div style="margin-top: 5px; margin-bottom: 15px; padding-bottom: 12px; border-bottom: 1px solid {card_border}; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
        <div style="display: flex; align-items: center; flex-wrap: wrap; font-size: 0.9rem;">
            {main_logo_html}
            <span style="color: {accent_color}; font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; letter-spacing: 1px; text-transform: uppercase; margin-right: 12px;">Strategic Deployment Plan</span>
            <span style="font-weight: 800; color: {text_main}; font-size: 1.1rem; margin-right: 12px;">{st.session_state.get('active_city', 'Unknown City')}, {st.session_state.get('active_state', 'US')}</span>
            <span style="color: {text_muted}; margin-right: 12px;">&bull; Serving {st.session_state.get('estimated_pop', 0):,} residents across ~{int(area_sq_mi):,} sq miles</span>
        </div>
        <div style="display: flex; align-items: center; font-size: 0.85rem; color: {text_muted}; gap: 15px;">
            <span>Data Period: <span style="color:#fff;">{date_range_str}</span></span>
            <span style="color:{card_border};">|</span>
            <span style="font-weight: 800; color: {text_main}; font-size: 0.95rem;">{actual_k_responder} <span style="color:#888; font-weight:normal;">Resp</span> &middot; {actual_k_guardian} <span style="color:#888; font-weight:normal;">Guard</span></span>
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

    # 2. THE STREAMLINED OPERATIONAL KPI BAR
    kpi_html = f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); background: {card_bg}; border: 1px solid {card_border}; border-radius: 8px; padding: 15px; margin-bottom: 15px; gap: 10px;">
        <div style="border-right: 1px solid #222; padding-right: 10px; text-align: center;">
            <div style="font-size: 0.65rem; color: {text_muted}; text-transform: uppercase; letter-spacing: 0.5px;">Total Incidents</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: {accent_color}; font-family: 'IBM Plex Mono', monospace;">{call_str}</div>
        </div>
        <div style="border-right: 1px solid #222; padding-right: 10px; text-align: center;">
            <div style="font-size: 0.65rem; color: {text_muted}; text-transform: uppercase; letter-spacing: 0.5px;">Call Coverage</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: {accent_color}; font-family: 'IBM Plex Mono', monospace;">{calls_covered_perc:.1f}%</div>
        </div>
        <div style="border-right: 1px solid #222; padding-right: 10px; text-align: center;">
            <div style="font-size: 0.65rem; color: {text_muted}; text-transform: uppercase; letter-spacing: 0.5px;">Land Covered</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: {accent_color}; font-family: 'IBM Plex Mono', monospace;">{area_covered_perc:.1f}%</div>
        </div>
        <div style="border-right: 1px solid #222; padding-right: 10px; text-align: center;">
            <div style="font-size: 0.65rem; color: {text_muted}; text-transform: uppercase; letter-spacing: 0.5px;">Overlap</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: {accent_color}; font-family: 'IBM Plex Mono', monospace;">{overlap_perc:.1f}%</div>
        </div>
        <div style="{'' if gain_val is None else 'border-right: 1px solid #222; padding-right: 10px;'} text-align: center;">
            <div style="font-size: 0.65rem; color: {text_muted}; text-transform: uppercase; letter-spacing: 0.5px;">Est. Avg Response</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: {accent_color}; font-family: 'IBM Plex Mono', monospace;">{avg_resp_time:.1f}m</div>
        </div>
    """
    
    if gain_val is not None:
        kpi_html += f"""
        <div style="text-align: center;">
            <div style="font-size: 0.65rem; color: {text_muted}; text-transform: uppercase; letter-spacing: 0.5px;">Time Saved ({eval_dist:.0f}mi)</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: {accent_color}; font-family: 'IBM Plex Mono', monospace;">{gain_val}</div>
        </div>"""
        
    kpi_html += "</div>"
    st.markdown(kpi_html, unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:0.65rem;color:gray;margin-top:-10px;margin-bottom:12px;text-align:right;'>(Optimization modeled via {total_calls:,} representative CAD samples)</div>", unsafe_allow_html=True)

    with map_col:
        fig = go.Figure()

        if show_boundaries and city_boundary_geom is not None and not city_boundary_geom.is_empty:
            geoms_to_draw = [city_boundary_geom] if isinstance(city_boundary_geom, Polygon) else list(city_boundary_geom.geoms)
            for gi, geom in enumerate(geoms_to_draw):
                bx, by = geom.exterior.coords.xy
                fig.add_trace(go.Scattermapbox(mode="lines", lon=list(bx), lat=list(by),
                    line=dict(color=map_boundary_color, width=2), name="Jurisdiction Boundary",
                    hoverinfo='skip', showlegend=(gi==0)))

        if show_heatmap and not display_calls.empty:
            fig.add_trace(go.Densitymapbox(lat=display_calls.geometry.y, lon=display_calls.geometry.x,
                z=np.ones(len(display_calls)), radius=12, colorscale='Inferno', opacity=0.6,
                showscale=False, name="Heatmap", hoverinfo='skip'))

        if not display_calls.empty:
            fig.add_trace(go.Scattermapbox(lat=display_calls.geometry.y, lon=display_calls.geometry.x,
                mode='markers', marker=dict(size=4, color=map_incident_color, opacity=0.4),
                name="Incident Data", hoverinfo='skip'))

        if show_faa and faa_geojson:
            add_faa_laanc_layer_to_plotly(fig, faa_geojson, is_dark=not show_satellite)

        for d in active_drones:
            clats, clons = get_circle_coords(d['lat'], d['lon'], r_mi=d['radius_m']/1609.34)
            lbl = f"{d['name'].split(',')[0]} ({'Resp' if d['type']=='RESPONDER' else 'Guard'})"
            fig.add_trace(go.Scattermapbox(
                lat=list(clats)+[None,d['lat']], lon=list(clons)+[None,d['lon']],
                mode='lines+markers',
                marker=dict(size=[0]*len(clats)+[0,20], color=d['color']),
                line=dict(color=d['color'], width=4.5),
                fill='toself', fillcolor='rgba(0,0,0,0)', name=lbl, hoverinfo='name'))

            # Guardian 5-mile rapid response focus ring
            if d['type'] == 'GUARDIAN' and d['radius_m']/1609.34 > 5.0:
                f_lats, f_lons = get_circle_coords(d['lat'], d['lon'], r_mi=5.0)
                fig.add_trace(go.Scattermapbox(
                    lat=list(f_lats), lon=list(f_lons),
                    mode='lines',
                    line=dict(color=d['color'], width=1.5),
                    opacity=0.5,
                    fill='toself',
                    fillcolor=f"rgba({int(d['color'][1:3],16)},{int(d['color'][3:5],16)},{int(d['color'][5:7],16)},0.06)",
                    name=f"Focus Zone 5mi · {d['name'].split(',')[0]}",
                    hoverinfo='text',
                    text=f"⚡ Rapid Response Focus Zone — 5mi<br>{d['name'].split(',')[0]}",
                    showlegend=False
                ))

            if simulate_traffic:
                t_color = "#28a745" if traffic_level<35 else "#ffc107" if traffic_level<75 else "#dc3545"
                t_fill  = f"rgba({'40,167,69' if traffic_level<35 else '255,193,7' if traffic_level<75 else '220,53,69'}, 0.15)"
                t_label = "Light" if traffic_level<35 else "Moderate" if traffic_level<75 else "Heavy"
                gs = CONFIG["DEFAULT_TRAFFIC_SPEED"]*(1-traffic_level/100)
                if gs > 0:
                    gr_mi = (gs/60) * (d['radius_m']/1609.34/d['speed_mph'])*60
                    ga = np.linspace(0,2*np.pi,9)
                    fig.add_trace(go.Scattermapbox(
                        lat=list(d['lat']+(gr_mi/69.172)*np.sin(ga)),
                        lon=list(d['lon']+(gr_mi/(69.172*np.cos(np.radians(d['lat']))))*np.cos(ga)),
                        mode='lines', line=dict(color=t_color, width=2.5),
                        fill='toself', fillcolor=t_fill,
                        name=f"Ground ({t_label})", hoverinfo='skip'))

        mapbox_cfg = dict(center=dict(lat=center_lat, lon=center_lon), zoom=dynamic_zoom, style=map_style)
        if show_satellite:
            mapbox_cfg["style"] = "carto-positron"
            mapbox_cfg["layers"] = [{"below":"traces","sourcetype":"raster",
                "sourceattribution":"Esri, Maxar, Earthstar Geographics",
                "source":["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"]}]

        fig.update_layout(uirevision="LOCKED_MAP", mapbox=mapbox_cfg,
            margin=dict(l=0,r=0,t=0,b=0), height=800, font=dict(size=18),
            showlegend=True,
            legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02,
                        bgcolor=legend_bg, bordercolor=accent_color, borderwidth=1,
                        font=dict(size=12, color=legend_text), itemclick="toggle"))

        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

    with stats_col:
        st.markdown(f"<h4 style='margin-top:0; border-bottom:1px solid {card_border}; padding-bottom:8px; color:{text_main};'>Coverage Curve</h4>", unsafe_allow_html=True)

        if not df_curve.empty:
            fig_curve = go.Figure()
            for col, color, dash in [('Responder (Calls)',accent_color,'solid'),('Guardian (Calls)','#FFD700','solid'),
                                      ('Responder (Area)',accent_color,'dash'),('Guardian (Area)','#FFD700','dash')]:
                y_data = df_curve[col].dropna()
                x_data = df_curve.loc[y_data.index,'Drones']
                if not y_data.empty:
                    fig_curve.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines+markers', name=col,
                        line=dict(color=color,width=2,dash=dash), marker=dict(size=4)))
                    if 'Calls' in col:
                        idx_90 = y_data[y_data >= 90.0].first_valid_index()
                        if idx_90 is not None:
                            fig_curve.add_trace(go.Scatter(x=[int(x_data.loc[idx_90])], y=[y_data.loc[idx_90]],
                                mode='markers', marker=dict(color=color,size=12,symbol='star',line=dict(color='white',width=1)),
                                showlegend=False, hoverinfo='skip'))
            fig_curve.update_layout(
                xaxis_title="Drones", yaxis_title="Coverage %",
                xaxis=dict(showgrid=True, gridcolor=card_border, tickfont=dict(color=text_muted)),
                yaxis=dict(showgrid=True, gridcolor=card_border, tickfont=dict(color=text_muted),
                           tickvals=[0,20,40,60,80,90,100], range=[0,105]),
                legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1,
                            font=dict(size=9,color=text_muted)),
                margin=dict(l=10,r=10,t=20,b=10), height=260,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_curve, use_container_width=True, config={'displayModeBar':False})

        if show_cards:
            st.markdown(f"<h4 style='margin-top:8px; border-bottom:1px solid {card_border}; padding-bottom:8px; color:{text_main};'>Unit Economics</h4>", unsafe_allow_html=True)
            if not active_drones:
                st.markdown(f"""
                <div style="background:{card_bg}; border:1px dashed {card_border}; border-radius:6px;
                     padding:24px; text-align:center; margin-top:10px;">
                    <div style="font-size:2rem; margin-bottom:8px;">🚁</div>
                    <div style="font-weight:700; color:{text_main}; margin-bottom:6px;">No drones deployed yet</div>
                    <div style="font-size:0.8rem; color:{text_muted};">
                        👈 Use the <b>Responder / Guardian Count</b> sliders in the <b>② Optimize Fleet</b> sidebar section to deploy drones and see per-unit economics here.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                for i in range(0, len(active_drones), 2):
                    cols = st.columns(2)
                    for j in range(2):
                        if i + j < len(active_drones):
                            d = active_drones[i + j]
                            short_name  = format_3_lines(d['name'])
                            d_color     = d['color']
                            d_type      = d['type']
                            d_step      = d['deploy_step']
                            d_savings   = d['annual_savings']
                            d_flights   = d['marginal_flights']
                            d_shared    = d['shared_flights']
                            d_deflected = d['marginal_deflected']
                            d_time      = d['avg_time_min']
                            d_faa       = d['faa_ceiling']
                            d_airport   = d['nearest_airport']
                            d_cost      = d['cost']
                            d_be        = d['be_text']
                            cols[j].markdown(f"""
<div style="background:{card_bg}; border-top:4px solid {d_color};
     border-left:1px solid {card_border}; border-right:1px solid {card_border};
     border-bottom:1px solid {card_border};
     border-radius:4px; padding:12px; margin-bottom:12px;">
    <div style="font-weight:700; font-size:0.73rem; color:{card_title}; margin-bottom:2px;">{short_name}</div>
    <div style="font-size:0.58rem; color:#888; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:8px;">{d_type} · Phase #{d_step}</div>
    <div style="background:rgba(0,210,255,0.07); border-radius:4px; padding:8px; text-align:center; margin-bottom:8px;">
        <div style="font-size:0.6rem; color:{text_muted}; text-transform:uppercase; letter-spacing:0.5px;">Annual Capacity Value</div>
        <div style="font-size:1.25rem; font-weight:900; color:{accent_color};">${d_savings:,.0f}</div>
    </div>
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:4px; font-size:0.62rem;">
        <div style="color:{text_muted};">Net Flights/day</div>
        <div style="text-align:right; font-weight:700; color:{accent_color};">{d_flights:.1f}</div>
        <div style="color:{text_muted};">Shared Flights/day</div>
        <div style="text-align:right; font-weight:700; color:{card_title};">{d_shared:.1f}</div>
        <div style="color:{text_muted};">Resolved/day</div>
        <div style="text-align:right; font-weight:700; color:{card_title};">{d_deflected:.1f}</div>
        <div style="color:{text_muted};">Avg Response</div>
        <div style="text-align:right; font-weight:700; color:{card_title};">{d_time:.1f} min</div>
        <div style="color:{text_muted};">FAA Ceiling</div>
        <div style="text-align:right; font-weight:700; color:{card_title};">{d_faa}</div>
        <div style="color:{text_muted};">Nearest Airfield</div>
        <div style="text-align:right; font-weight:700; color:{card_title}; font-size:0.55rem;">{d_airport}</div>
    </div>
    <div style="border-top:1px dashed {card_border}; margin-top:8px; padding-top:6px;
         display:grid; grid-template-columns:1fr 1fr; gap:4px; font-size:0.62rem;">
        <div style="color:{text_muted};">CapEx</div>
        <div style="text-align:right; font-weight:700; color:{card_title};">${d_cost:,.0f}</div>
        <div style="color:{text_muted};">ROI</div>
        <div style="text-align:right; font-weight:800; color:{accent_color};">{d_be}</div>
    </div>
</div>
""", unsafe_allow_html=True)

    # ── 3D SWARM SIMULATION ───────────────────────────────────────────
    if fleet_capex > 0:
        st.markdown("---")
        st.markdown(f"<h3 style='color:{text_main};'>🚁 3D Swarm Simulation</h3>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:0.82rem; color:{text_muted}; margin-bottom:10px;'>Animated deck.gl simulation of all DFR flights over a compressed 24-hour day. Use the speed slider to accelerate or slow the simulation. Great for council presentations.</div>", unsafe_allow_html=True)

        show_sim = st.toggle("🎬 Enable 3D Simulation", value=False)
        if show_sim:
            calls_lonlat = calls_in_city.to_crs(epsg=4326)
            calls_coords = np.column_stack((calls_lonlat.geometry.x, calls_lonlat.geometry.y))

            sim_assignments = {i:[] for i in range(len(active_drones))}
            for c_idx, cc in enumerate(calls_coords):
                best_d, best_dist = -1, float('inf')
                for d_idx, d in enumerate(active_drones):
                    if d['cov_array'][c_idx] if c_idx < len(d['cov_array']) else False:
                        dist = (cc[0]-d['lon'])**2 + (cc[1]-d['lat'])**2
                        if dist < best_dist:
                            best_dist, best_d = dist, d_idx
                if best_d != -1:
                    sim_assignments[best_d].append(c_idx)

            stations_json, flights_json, legend_html_sim = [], [], ""
            total_sim_flights = 0
            for d_idx, d in enumerate(active_drones):
                hex_c = d['color'].lstrip('#')
                rgb = [int(hex_c[j:j+2],16) for j in (0,2,4)]
                stations_json.append({"name":d['name'].split(',')[0][:30],"lon":d['lon'],"lat":d['lat'],"color":rgb,"radius":d['radius_m']})
                legend_html_sim += f'<div style="margin-bottom:3px;"><span style="display:inline-block;width:9px;height:9px;background:{d["color"]};margin-right:7px;border-radius:50%;"></span>{d["name"].split(",")[0][:28]} ({d["type"][:3]})</div>'
                frac = len(sim_assignments[d_idx])/len(calls_coords) if calls_coords.shape[0]>0 else 0
                monthly_for_drone = int(frac * calls_per_day * 30 * dfr_dispatch_rate)
                pool = sim_assignments[d_idx]

                if not pool:
                    sim_calls = []
                elif monthly_for_drone > len(pool):
                    sim_calls = random.choices(pool, k=monthly_for_drone)
                else:
                    sim_calls = random.sample(pool, monthly_for_drone)

                total_sim_flights += len(sim_calls)
                for ci in sim_calls:
                    lon1,lat1 = calls_coords[ci]
                    lon0,lat0 = d['lon'],d['lat']
                    dist_mi = math.sqrt((lon1-lon0)**2+(lat1-lat0)**2)*69.172
                    vis_time = max((dist_mi/d['speed_mph'])*3600*8, 240)
                    launch = random.randint(0, 2592000)
                    arc_h = min(max(dist_mi*90, 80), 400)
                    t0 = launch
                    t1 = launch + vis_time * 0.15
                    t2 = launch + vis_time * 0.40
                    t3 = launch + vis_time * 0.75
                    t4 = launch + vis_time * 0.90
                    t5 = launch + vis_time
                    mx1 = lon0 + 0.15*(lon1-lon0);  my1 = lat0 + 0.15*(lat1-lat0)
                    mx2 = lon0 + 0.35*(lon1-lon0);  my2 = lat0 + 0.35*(lat1-lat0)
                    mx3 = lon0 + 0.65*(lon1-lon0);  my3 = lat0 + 0.65*(lat1-lat0)
                    mx4 = lon0 + 0.85*(lon1-lon0);  my4 = lat0 + 0.85*(lat1-lat0)
                    flights_json.append({
                        "path": [
                            [lon0, lat0, 0],
                            [mx1,  my1,  arc_h*0.75],
                            [mx2,  my2,  arc_h],
                            [mx3,  my3,  arc_h],
                            [mx4,  my4,  arc_h*0.75],
                            [lon1, lat1, 0]
                        ],
                        "timestamps": [t0, t1, t2, t3, t4, t5],
                        "color": rgb
                    })

            warn_html_sim = ""
            if len(flights_json) > 3000:
                flights_json = random.sample(flights_json, 3000)
                warn_html_sim = f'<div style="background:#440000;border:1px solid #ff4b4b;color:#ffbbbb;padding:5px;font-size:10px;border-radius:4px;margin-bottom:8px;">⚠️ Capped at 3,000 flights for performance (actual: {total_sim_flights:,})</div>'

            drone_svg = "data:image/svg+xml;charset=utf-8,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='white'%3E%3Cpath d='M18 6a2 2 0 100-4 2 2 0 000 4zm-12 0a2 2 0 100-4 2 2 0 000 4zm12 12a2 2 0 100-4 2 2 0 000 4zm-12 0a2 2 0 100-4 2 2 0 000 4z'/%3E%3Cpath stroke='white' stroke-width='2' stroke-linecap='round' d='M8.5 8.5l7 7m0-7l-7 7'/%3E%3Ccircle cx='12' cy='12' r='2' fill='white'/%3E%3C/svg%3E"

            sim_html = f"""<!DOCTYPE html><html><head>
            <script src="https://unpkg.com/deck.gl@8.9.35/dist.min.js"></script>
            <script src="https://unpkg.com/maplibre-gl@3.0.0/dist/maplibre-gl.js"></script>
            <link href="https://unpkg.com/maplibre-gl@3.0.0/dist/maplibre-gl.css" rel="stylesheet"/>
            <style>
              body{{margin:0;padding:0;overflow:hidden;background:#000;font-family:Manrope,sans-serif;}}
              #map{{width:100vw;height:100vh;position:absolute;}}
              #ui{{position:absolute;top:16px;left:16px;background:rgba(17,17,17,0.92);padding:16px;border-radius:8px;
                   color:white;border:1px solid #333;z-index:10;box-shadow:0 4px 10px rgba(0,0,0,0.5);width:260px;}}
              button{{background:#00D2FF;color:black;border:none;padding:10px;cursor:pointer;font-weight:bold;
                      border-radius:4px;width:100%;font-size:13px;text-transform:uppercase;margin-bottom:8px;}}
              button:disabled{{background:#444;color:#888;cursor:not-allowed;}}
              #timeDisplay{{font-family:monospace;font-size:16px;color:#00ffcc;font-weight:bold;text-align:center;margin-bottom:8px;}}
            </style></head><body>
            <div id="ui">
              <h3 style="margin:0 0 8px;color:#00D2FF;font-size:14px;">DFR SWARM SIMULATION</h3>
              {warn_html_sim}
              <div style="font-size:11px;color:#aaa;margin-bottom:10px;">
                {total_sim_flights:,} flights over 30 days at {int(dfr_dispatch_rate*100)}% dispatch rate
              </div>
              <div style="margin-bottom:10px;">
                <label style="font-size:11px;color:#ccc;">Speed: <span id="speedLabel">1</span>x</label>
                <input type="range" id="speedSlider" min="1" max="100" value="1" style="width:100%;margin-top:4px;">
              </div>
              <button id="runBtn">▶ LAUNCH SWARM</button>
              <div id="timeDisplay">00:00</div>
              <div style="margin-top:10px;border-top:1px solid #333;padding-top:8px;">
                <div style="font-size:10px;color:#888;text-transform:uppercase;margin-bottom:5px;">Stations</div>
                <div style="font-size:10px;color:#ddd;max-height:100px;overflow-y:auto;">{legend_html_sim}</div>
              </div>
            </div>
            <div id="map"></div>
            <script>
              const stations={json.dumps(stations_json)};
              const flights={json.dumps(flights_json)};
              const speedSlider=document.getElementById('speedSlider');
              const speedLabel=document.getElementById('speedLabel');
              speedSlider.oninput=()=>speedLabel.innerText=speedSlider.value;
              const map=new deck.DeckGL({{
                container:'map',
                mapStyle:'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
                initialViewState:{{longitude:{center_lon},latitude:{center_lat},zoom:{dynamic_zoom},pitch:50,bearing:0}},
                controller:true
              }});
              let time=0,timer=null,lastTime=0;
              function render(){{
                map.setProps({{layers:[
                  new deck.ScatterplotLayer({{id:'rings',data:stations,getPosition:d=>[d.lon,d.lat],
                    getFillColor:d=>[d.color[0],d.color[1],d.color[2],25],
                    getLineColor:d=>[d.color[0],d.color[1],d.color[2],220],
                    lineWidthMinPixels:2,stroked:true,filled:true,getRadius:d=>d.radius}}),
                  new deck.ScatterplotLayer({{id:'pads',data:stations,getPosition:d=>[d.lon,d.lat],
                    getFillColor:d=>[d.color[0],d.color[1],d.color[2],120],getRadius:180}}),
                  new deck.IconLayer({{id:'icons',data:stations,
                    getIcon:d=>({{url:"{drone_svg}",width:24,height:24,anchorY:12}}),
                    getPosition:d=>[d.lon,d.lat],getSize:36,sizeScale:1}}),
                  new deck.TripsLayer({{id:'flights',data:flights,getPath:d=>d.path,
                    getTimestamps:d=>d.timestamps,getColor:d=>d.color,
                    opacity:0.85,widthMinPixels:5,trailLength:13500,currentTime:time,rounded:true}}),
                  new deck.ScatterplotLayer({{id:'landed',data:flights,getPosition:d=>d.path[5],
                    getFillColor:d=>time>=d.timestamps[5]?[d.color[0],d.color[1],d.color[2],255]:[0,0,0,0],
                    getRadius:25,radiusMinPixels:3,updateTriggers:{{getFillColor:time}}}})
                ]}});
                let day=Math.floor(time/86400)+1;
                let h=Math.floor((time%86400)/3600).toString().padStart(2,'0');
                let m=Math.floor((time%3600)/60).toString().padStart(2,'0');
                document.getElementById('timeDisplay').innerText=`Day ${{day}} · ${{h}}:${{m}}`;
              }}
              const animate=()=>{{
                let now=performance.now();
                let dt=Math.min(now-lastTime,100);
                lastTime=now;
                time+=dt/1000*43200*parseFloat(speedSlider.value);
                render();
                if(time<2592000){{timer=requestAnimationFrame(animate);}}
                else{{
                  document.getElementById('runBtn').disabled=false;
                  document.getElementById('runBtn').innerText='↺ RESTART';
                  time=0;
                }}
              }};
              document.getElementById('runBtn').onclick=()=>{{
                document.getElementById('runBtn').disabled=true;
                document.getElementById('runBtn').innerText='SIMULATING…';
                time=0;lastTime=performance.now();
                if(timer)cancelAnimationFrame(timer);
                animate();
              }};
              render();
            </script></body></html>"""

            components.html(sim_html, height=700)

    # ── COMMAND CENTER ANALYTICS DASHBOARD ──
    st.markdown("---")
    st.markdown(f"<h3 style='color:{text_main};'>📊 CAD Ingestion Analytics</h3>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:0.82rem; color:{text_muted}; margin-bottom:10px;'>Temporal patterns derived from your uploaded CAD data — hourly volumes, day-of-week distribution, optimal DFR shift windows, and an interactive call-volume calendar.</div>", unsafe_allow_html=True)

    ctrl_col1, ctrl_col2, _ = st.columns([1.5, 1.5, 2])
    with ctrl_col1:
        ana_shift = st.radio("Shift Length", [8, 10, 12], index=0, horizontal=True, format_func=lambda x: f"{x} HR")
    with ctrl_col2:
        ana_pri = st.radio("Priority Filter", ["ALL", "P1-3", "P1-2", "P1 ONLY"], index=0, horizontal=True)

    df_ana_display = df_calls.copy()
    if 'priority' in df_ana_display.columns:
        if ana_pri == "P1-3":
            df_ana_display = df_ana_display[df_ana_display['priority'] <= 3]
        elif ana_pri == "P1-2":
            df_ana_display = df_ana_display[df_ana_display['priority'] <= 2]
        elif ana_pri == "P1 ONLY":
            df_ana_display = df_ana_display[df_ana_display['priority'] == 1]

    show_cad_analytics = st.toggle("📈 Show Data Analytics Heatmaps", value=False)
    
    analytics_html_block = generate_command_center_html(df_ana_display, total_orig_calls=st.session_state.get('total_original_calls', total_calls), shift_hours=ana_shift)
    
    if show_cad_analytics:
        components.html(analytics_html_block, height=850, scrolling=True)

    # ── EXPORT BUTTONS ──
    if fleet_capex > 0:
        st.sidebar.markdown("---")
        
        brinc_user = st.sidebar.text_input("BRINC Email Prefix (first.last)", value=st.session_state.get('brinc_user', 'steven.beltran'), key='brinc_user', help="Enter 'first.last' to auto-generate your name and @brincdrones.com email address.")
        st.sidebar.caption("*(Press **Enter** after typing to apply changes)*")
        
        user_clean = brinc_user.strip()
        if not user_clean: user_clean = "steven.beltran"
        prop_email = f"{user_clean}@brincdrones.com"
        prop_name = " ".join([word.capitalize() for word in user_clean.split('.')])

        prop_city  = st.session_state.get('active_city', 'City')
        prop_state = st.session_state.get('active_state', 'FL')
        pop_metric = st.session_state.get('estimated_pop', 250000)
        current_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        safe_city = prop_city.replace(" ","_").replace("/","_")

        export_dict = {
            "city": prop_city, "state": prop_state,
            "k_resp": k_responder, "k_guard": k_guardian,
            "r_resp": resp_radius_mi, "r_guard": guard_radius_mi,
            "dfr_rate": int(dfr_dispatch_rate*100), "deflect_rate": int(deflection_rate*100),
            "calls_data": json.loads(st.session_state['df_calls'].replace({np.nan:None}).to_json(orient='records')) if st.session_state.get('df_calls') is not None else None,
            "stations_data": json.loads(st.session_state['df_stations'].replace({np.nan:None}).to_json(orient='records')) if st.session_state.get('df_stations') is not None else None,
            "faa_geojson": faa_geojson
        }

        avg_resp_time  = sum(d['avg_time_min'] for d in active_drones)/len(active_drones) if active_drones else 0.0
        avg_ground_speed = CONFIG["DEFAULT_TRAFFIC_SPEED"] * (1 - traffic_level/100)
        avg_time_saved = ((sum((d['radius_m']/1609.34*1.4/avg_ground_speed)*60 for d in active_drones)/len(active_drones)) - avg_resp_time) if active_drones and avg_ground_speed > 0 else 0.0

        fig_for_export = go.Figure()
        for d in active_drones:
            clats, clons = get_circle_coords(d['lat'], d['lon'], r_mi=d['radius_m']/1609.34)
            fig_for_export.add_trace(go.Scattermapbox(
                lat=list(clats)+[None,d['lat']], lon=list(clons)+[None,d['lon']],
                mode='lines+markers', line=dict(color=d['color'], width=3),
                marker=dict(size=[0]*len(clats)+[0,16], color=d['color']),
                fill='toself', fillcolor='rgba(0,0,0,0)', name=d['name'][:30]
            ))
        fig_for_export.update_layout(
            mapbox=dict(center=dict(lat=center_lat, lon=center_lon), zoom=dynamic_zoom, style="carto-darkmatter"),
            margin=dict(l=0,r=0,t=0,b=0), height=500, showlegend=True,
            legend=dict(bgcolor=legend_bg, font=dict(color=legend_text, size=11))
        )
        map_html_str = fig_for_export.to_html(full_html=False, include_plotlyjs='cdn', default_height='500px', default_width='100%')
        station_rows = "".join(f"<tr><td>{d['name']}</td><td>{d['type']}</td><td>{d['avg_time_min']:.1f} min</td><td>{d['faa_ceiling']}</td><td>${d['cost']:,}</td></tr>" for d in active_drones)

        logo_b64 = get_base64_of_bin_file("logo.png")
        logo_html_str = f'<img src="data:image/png;base64,{logo_b64}" style="height:32px;">' if logo_b64 else '<div style="font-size:24px;font-weight:900;letter-spacing:3px;color:#fff;">BRINC</div>'

        jurisdiction_list = ", ".join(selected_names) if selected_names else prop_city
        all_station_types = df_stations_all['type'].dropna().unique().tolist() if 'type' in df_stations_all.columns else []
        police_dept_names = [d['name'] for d in active_drones if '[Police]' in d['name']]
        fire_dept_names   = [d['name'] for d in active_drones if '[Fire]' in d['name']]
        ems_dept_names    = [d['name'] for d in active_drones if '[EMS]' in d['name']]

        police_stations = [d['name'] for d in active_drones if 'Police' in d.get('name','') or (
            'type' in df_stations_all.columns and
            'Police' in str(df_stations_all[df_stations_all['name'].str.contains(
                d['name'].split(']')[-1].strip(), na=False, regex=False
            )]['type'].values[:1])
        )]

        dept_summary_parts = []
        if police_dept_names: dept_summary_parts.append(f"{len(police_dept_names)} Police station{'s' if len(police_dept_names)>1 else ''}")
        if fire_dept_names:   dept_summary_parts.append(f"{len(fire_dept_names)} Fire station{'s' if len(fire_dept_names)>1 else ''}")
        if ems_dept_names:    dept_summary_parts.append(f"{len(ems_dept_names)} EMS station{'s' if len(ems_dept_names)>1 else ''}")
        dept_summary = ", ".join(dept_summary_parts) if dept_summary_parts else f"{len(active_drones)} municipal stations"
        police_names_str = (", ".join([n.replace('[Police] ','') for n in police_dept_names[:6]]) + ("..." if len(police_dept_names)>6 else "")) if police_dept_names else "municipal facilities"
        total_fleet = actual_k_responder + actual_k_guardian
        area_sq_mi_est = int((maxx - minx) * (maxy - miny) * 3280)

        # Build Analytics Block for Export (Ensure same filters are passed)
        analytics_html_export = generate_command_center_html(df_ana_display, total_orig_calls=st.session_state.get('total_original_calls', total_calls), export_mode=True, shift_hours=ana_shift)

        export_html = f"""<html><head><title>BRINC DFR Proposal — {prop_city}</title>
        <link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;800&display=swap" rel="stylesheet">
        <style>
        body {{ font-family: 'Manrope', Arial, sans-serif; color: #1e1e24; margin: 0; padding: 40px; background: #f4f6f9; }}
        .page {{ max-width: 1000px; margin: 0 auto; background: #fff; border-radius: 12px; box-shadow: 0 10px 40px rgba(0,0,0,0.08); overflow: hidden; }}
        .header {{ background: #06060a; color: #fff; padding: 40px 50px; border-bottom: 4px solid #00D2FF; display: flex; justify-content: space-between; align-items: center; }}
        .header h1 {{ margin: 0; font-size: 28px; font-weight: 800; letter-spacing: 0.5px; }}
        .header-sub {{ font-size: 14px; color: #aaa; margin-top: 8px; font-family: monospace; letter-spacing: 1px; }}
        .content {{ padding: 40px 50px; }}
        h2 {{ color: #111; font-size: 22px; font-weight: 800; margin-top: 40px; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 2px solid #eee; }}
        .kpi-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-bottom: 40px; }}
        .kpi-card {{ border: 1px solid #eaeaea; border-radius: 10px; padding: 25px; background: #fafafa; }}
        .kpi-card h3 {{ margin: 0 0 20px 0; font-size: 16px; color: #444; text-transform: uppercase; letter-spacing: 1px; }}
        .kpi-row {{ display: flex; justify-content: space-between; margin-bottom: 15px; padding-bottom: 15px; border-bottom: 1px solid #eee; }}
        .kpi-row:last-child {{ margin-bottom: 0; padding-bottom: 0; border-bottom: none; }}
        .kpi-lbl {{ font-size: 13px; color: #666; font-weight: 600; }}
        .kpi-val {{ font-size: 20px; font-weight: 800; color: #00D2FF; font-family: monospace; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 14px; margin-bottom: 30px; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.03); }}
        th, td {{ padding: 14px 15px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; font-weight: 800; text-transform: uppercase; font-size: 12px; letter-spacing: 0.5px; color: #555; }}
        .map-container {{ border: 1px solid #eee; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }}
        .disclaimer {{ background: #fff8e6; border-left: 4px solid #F0B429; padding: 15px 20px; font-size: 13px; color: #856404; border-radius: 0 8px 8px 0; margin-bottom: 25px; }}
        p {{ line-height: 1.8; color: #444; font-size: 15px; margin-bottom: 15px; }}
        .footer {{ background: #06060a; color: #888; padding: 40px 50px; text-align: center; font-size: 13px; }}
        .footer a {{ color: #00D2FF; text-decoration: none; font-weight: 600; }}
        </style>
        </head><body><div class="page">
        <div class="header">
            <div>{logo_html_str}</div>
            <div style="text-align:right;">
                <h1>DFR Deployment Proposal</h1>
                <div class="header-sub">{prop_city}, {prop_state} | Pop: {pop_metric:,}</div>
                <div class="header-sub" style="color:#00D2FF;">Prepared by: {prop_name}</div>
            </div>
        </div>
        <div class="content">
            <div class="kpi-grid">
                <div class="kpi-card">
                    <h3>Financial Impact</h3>
                    <div class="kpi-row"><span class="kpi-lbl">Fleet CapEx</span><span class="kpi-val">${fleet_capex:,.0f}</span></div>
                    <div class="kpi-row"><span class="kpi-lbl">Annual Savings Capacity</span><span class="kpi-val">${annual_savings:,.0f}</span></div>
                    <div class="kpi-row"><span class="kpi-lbl">Break-Even</span><span class="kpi-val" style="color:#F0B429;">{break_even_text}</span></div>
                </div>
                <div class="kpi-card">
                    <h3>Operational Impact</h3>
                    <div class="kpi-row"><span class="kpi-lbl">911 Call Coverage</span><span class="kpi-val">{calls_covered_perc:.1f}%</span></div>
                    <div class="kpi-row"><span class="kpi-lbl">Avg Response Time</span><span class="kpi-val">{avg_resp_time:.1f} min</span></div>
                    <div class="kpi-row"><span class="kpi-lbl">Time Saved vs Patrol</span><span class="kpi-val" style="color:#2ecc71;">{avg_time_saved:.1f} min</span></div>
                </div>
            </div>

            <h2>Proposed Fleet</h2>
            <table>
                <tr><th>Type</th><th>Qty</th><th>Range</th><th>Unit Cost</th></tr>
                <tr><td>BRINC Responder</td><td>{actual_k_responder}</td><td>{resp_radius_mi} mi</td><td>${CONFIG['RESPONDER_COST']:,}</td></tr>
                <tr><td>BRINC Guardian</td><td>{actual_k_guardian}</td><td>{guard_radius_mi} mi</td><td>${CONFIG['GUARDIAN_COST']:,}</td></tr>
            </table>

            <h2>Coverage Map</h2>
            <div class="map-container">{map_html_str}</div>

            <h2>Deployment Locations</h2>
            <table>
                <tr><th>Station</th><th>Type</th><th>Avg Response</th><th>FAA Ceiling</th><th>CapEx</th></tr>
                {station_rows}
            </table>

            <h2>Grant Narrative (AI Draft)</h2>
            <div class="disclaimer"><strong>DISCLAIMER:</strong> AI-generated draft. Must be reviewed, localized, and fact-checked by your grants administrator before submission. All statistics are model estimates.</div>
            <p><strong>Project Title:</strong> BRINC Drones Drone as a First Responder (DFR) Program — {jurisdiction_list}</p>
            <p><strong>Executive Summary:</strong> The {jurisdiction_list} respectfully submits this application requesting funding to establish a BRINC Drones-powered Drone as a First Responder (DFR) program. This initiative will deploy a fleet of {total_fleet} purpose-built BRINC Drones aerial systems — comprising {actual_k_responder} BRINC Responder and {actual_k_guardian} BRINC Guardian units — across {dept_summary} serving a combined population of {pop_metric:,} residents across approximately {area_sq_mi_est:,} square miles in {prop_city}, {prop_state}.</p>
            <p><strong>Statement of Need:</strong> The {jurisdiction_list} currently serves a population of {pop_metric:,} residents and responds to an estimated {st.session_state.get('total_original_calls', total_calls):,} calls for service annually. Ground-based patrol response times are constrained by traffic, geography, and unit availability. This proposal addresses a critical public safety gap: the need for immediate aerial situational awareness that arrives before ground units, enabling smarter, safer, and faster emergency response. BRINC Drones, the world leader in purpose-built DFR technology, provides the only fully integrated hardware, software, and operational support platform purpose-designed for law enforcement DFR deployment.</p>
            <p><strong>Geographic Scope & Participating Agencies:</strong> The proposed DFR network covers the jurisdictions of <strong>{jurisdiction_list}</strong> ({prop_state}). Drone stations will be hosted at {dept_summary}, including facilities operated by: <em>{police_names_str}</em>. The deployment area encompasses an estimated {area_sq_mi_est:,} square miles of mixed urban and suburban terrain, with BRINC Drones units positioned to achieve {calls_covered_perc:.1f}% coverage of historical incident locations and {area_covered_perc:.1f}% geographic area coverage.</p>
            <p><strong>Program Design:</strong> The proposed fleet consists of {actual_k_responder} <strong>BRINC Responder</strong> units (short-range tactical response, {resp_radius_mi}-mile operational radius) and {actual_k_guardian} <strong>BRINC Guardian</strong> units (long-range heavy-lift, {guard_radius_mi}-mile operational radius). All deployment sites have been pre-screened against FAA LAANC UAS Facility Maps. The BRINC Drones platform provides automated launch-on-dispatch, live-streaming HD/thermal video to dispatch and responding officers, and full chain-of-custody flight logging. Average aerial response time under this configuration is projected at <strong>{avg_resp_time:.1f} minutes</strong> — approximately <strong>{avg_time_saved:.1f} minutes faster</strong> than current vehicular patrol response for equivalent distances.</p>
            <p><strong>Fiscal Impact & Return on Investment:</strong> Total program capital expenditure is <strong>${fleet_capex:,.0f}</strong>. Based on a {int(dfr_dispatch_rate*100)}% DFR dispatch rate and {int(deflection_rate*100)}% call resolution rate, the program is projected to generate <strong>${annual_savings:,.0f} in annual operational savings</strong> through reduced officer dispatch on drone-resolved incidents, reaching full cost recovery in <strong>{break_even_text.lower()}</strong>. At ${CONFIG["DRONE_COST_PER_CALL"]}/drone response versus ${CONFIG["OFFICER_COST_PER_CALL"]}/officer dispatch, the BRINC Drones platform delivers a demonstrated cost-per-response reduction of over {int((1 - CONFIG["DRONE_COST_PER_CALL"]/CONFIG["OFFICER_COST_PER_CALL"])*100)}%.</p>
            <p><strong>About BRINC Drones:</strong> BRINC Drones, Inc. is the global leader in purpose-built Drone as a First Responder technology, with deployments across hundreds of law enforcement agencies in the United States. BRINC Drones designs, manufactures, and supports the only DFR platform built from the ground up for public safety — including the BRINC Responder for rapid tactical response and the BRINC Guardian for extended-range operations. BRINC provides full agency onboarding, FAA coordination support, pilot training, and ongoing operational guidance. Learn more at <a href="https://brincdrones.com" target="_blank">brincdrones.com</a>.</p>
            <p style="background: #f8f9fa; padding: 15px; border-radius: 6px; border: 1px solid #eee;"><strong>Potential Grant Funding Sources:</strong><br>
              <a href="https://bja.ojp.gov/program/jag/overview" target="_blank">DOJ Byrne JAG</a> — UAS and technology procurement eligible <br>
              <a href="https://www.fema.gov/grants/preparedness/homeland-security" target="_blank">FEMA HSGP</a> — CapEx offset for tactical deployments <br>
              <a href="https://cops.usdoj.gov/grants" target="_blank">DOJ COPS Office</a> — Law enforcement technology grants <br>
              <a href="https://www.transportation.gov/grants" target="_blank">DOT RAISE</a> — Regional infrastructure and safety
            </p>
            <div style="margin-top: 50px; font-family:'Manrope', Arial, sans-serif !important;">
                <div style="border-radius: 12px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.15);">
                    [ANALYTICS_HTML_EXPORT]
                </div>
            </div>
        <div class="footer">
            <div class="footer-logo">BRINC</div>
            <div style="font-weight:bold; font-size:15px; margin-bottom:8px; color:#fff;">BRINC Drones, Inc.</div>
            <div style="margin-bottom:15px;">Leading the world in purpose-built Drone as a First Responder technology.</div>
            <div style="margin-bottom:10px; color:#aaa;">Prepared by: {prop_name} | <a href="mailto:{prop_email}">{prop_email}</a></div>
            <div style="margin-bottom:15px;">
                <a href="https://brincdrones.com" target="_blank">brincdrones.com</a> | <a href="mailto:sales@brincdrones.com">sales@brincdrones.com</a> | +1 (855) 950-0226
            </div>
            <div style="color:#555;">
                <a href="https://www.linkedin.com/company/brincdrones" target="_blank">LinkedIn</a>  • 
                <a href="https://twitter.com/brincdrones" target="_blank">Twitter / X</a>  • 
                <a href="https://www.youtube.com/c/brincdrones" target="_blank">YouTube</a>
            </div>
        </div>
        </div></body></html>"""

        # Format safely using a plain text placeholder
        export_html = export_html.replace("[ANALYTICS_HTML_EXPORT]", analytics_html_export)

        if st.sidebar.download_button("💾 Save Deployment Plan", data=json.dumps(export_dict),
                                      file_name=f"Brinc_{safe_city}_{current_time_str}.brinc",
                                      mime="application/json", use_container_width=True):
            _notify_email(st.session_state.get('active_city',''), st.session_state.get('active_state',''),
                          "BRINC", k_responder, k_guardian, calls_covered_perc,
                          prop_name, prop_email)
            _log_to_sheets(st.session_state.get('active_city',''), st.session_state.get('active_state',''),
                           "BRINC", k_responder, k_guardian, calls_covered_perc,
                           prop_name, prop_email)

        if st.sidebar.download_button("📄 Executive Summary (HTML)", data=export_html,
                                      file_name=f"Brinc_{safe_city}_Proposal_{current_time_str}.html",
                                      mime="text/html", use_container_width=True):
            _notify_email(st.session_state.get('active_city',''), st.session_state.get('active_state',''),
                          "HTML", k_responder, k_guardian, calls_covered_perc,
                          prop_name, prop_email)
            _log_to_sheets(st.session_state.get('active_city',''), st.session_state.get('active_state',''),
                           "HTML", k_responder, k_guardian, calls_covered_perc,
                           prop_name, prop_email)

        if active_drones:
            if st.sidebar.download_button("🌏 Google Earth Briefing File", data=generate_kml(active_gdf, active_drones, calls_in_city),
                                          file_name="drone_deployment.kml", mime="application/vnd.google-earth.kml+xml",
                                          use_container_width=True):
                _notify_email(st.session_state.get('active_city',''), st.session_state.get('active_state',''),
                              "KML", k_responder, k_guardian, calls_covered_perc,
                              prop_name, prop_email)
                _log_to_sheets(st.session_state.get('active_city',''), st.session_state.get('active_state',''),
                               "KML", k_responder, k_guardian, calls_covered_perc,
                               prop_name, prop_email)
