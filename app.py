import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import plotly.graph_objects as go
from shapely.geometry import Point, Polygon, MultiPolygon, box, shape
from shapely.ops import unary_union
import os
import itertools
import glob
import math
import simplekml
import heapq
from concurrent.futures import ThreadPoolExecutor
import pulp
import re
import random
import json
import urllib.request
import urllib.parse
import zipfile
import io
import datetime
import base64
import streamlit.components.v1 as components
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import gspread
from google.oauth2.service_account import Credentials

# ============================================================
# 1. PAGE CONFIGURATION & SESSION STATE
# ============================================================
st.set_page_config(page_title="BRINC COS Drone Optimizer", layout="wide", initial_sidebar_state="expanded")

defaults = {
    'csvs_ready': False, 'df_calls': None, 'df_stations': None,
    'active_city': "Orlando", 'active_state': "FL", 'estimated_pop': 316081,
    'k_resp': 0, 'k_guard': 0, 'r_resp': 2.0, 'r_guard': 8.0,
    'dfr_rate': 25, 'deflect_rate': 30, 'total_original_calls': 0,
    'onboarding_done': False, 'trigger_sim': False, 'city_count': 1,
    'theme': "Dark Mode"
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if 'target_cities' not in st.session_state:
    st.session_state['target_cities'] = [{"city": st.session_state.get('active_city', 'Orlando'), "state": st.session_state.get('active_state', 'FL')}]

# ============================================================
# 2. GLOBAL CONFIGURATIONS
# ============================================================
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

DEMO_CITIES = [
    ("Las Vegas", "NV"), ("Austin", "TX"), ("Seattle", "WA"), 
    ("Denver", "CO"), ("Nashville", "TN"), ("Columbus", "OH"), 
    ("Detroit", "MI"), ("San Diego", "CA"), ("Charlotte", "NC"),
    ("Portland", "OR"), ("Memphis", "TN"), ("Louisville", "KY"),
    ("Baltimore", "MD"), ("Milwaukee", "WI"), ("Albuquerque", "NM"),
    ("Tucson", "AZ"), ("Fresno", "CA"), ("Sacramento", "CA")
]

FAST_DEMO_CITIES = [
    ("Henderson", "NV"), ("Lincoln", "NE"), ("Boise", "ID"),
    ("Des Moines", "IA"), ("Madison", "WI"), ("Colorado Springs", "CO"),
    ("Richmond", "VA"), ("Raleigh", "NC"), ("Durham", "NC")
]

FAA_CEILING_COLORS = {
    0:   {"line": "rgba(255,  20,  20, 0.95)", "fill": "rgba(255,  20,  20, 0.20)"},
    50:  {"line": "rgba(255, 120,   0, 0.95)", "fill": "rgba(255, 120,   0, 0.18)"},
    100: {"line": "rgba(255, 210,   0, 0.95)", "fill": "rgba(255, 210,   0, 0.18)"},
    200: {"line": "rgba(180, 230,   0, 0.95)", "fill": "rgba(180, 230,   0, 0.16)"},
    300: {"line": "rgba( 80, 200,  50, 0.95)", "fill": "rgba( 80, 200,  50, 0.16)"},
    400: {"line": "rgba(  0, 180, 100, 0.95)", "fill": "rgba(  0, 180, 100, 0.15)"},
}
FAA_DEFAULT_COLOR = {"line": "rgba(150,150,150,0.8)", "fill": "rgba(150,150,150,0.10)"}

STATION_COLORS = [
    "#00D2FF", "#39FF14", "#FFD700", "#FF007F", "#FF4500",
    "#00FFCC", "#FF3333", "#7FFF00", "#00FFFF", "#FF9900"
]

def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception:
        return None

def _notify_email(city, state, file_type, k_resp, k_guard, coverage, name, email):
    try:
        gmail_address  = st.secrets.get("GMAIL_ADDRESS", "")
        app_password   = st.secrets.get("GMAIL_APP_PASSWORD", "")
        notify_address = st.secrets.get("NOTIFY_EMAIL", gmail_address)
        if not gmail_address or not app_password:
            return
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
                    <tr style="border-bottom:1px solid #f0f0f0;">
                        <td style="padding:8px 4px;color:#888;width:40%;">File Type</td>
                        <td style="padding:8px 4px;font-weight:bold;">{emoji} {file_type}</td>
                    </tr>
                    <tr style="border-bottom:1px solid #f0f0f0;">
                        <td style="padding:8px 4px;color:#888;">City</td>
                        <td style="padding:8px 4px;font-weight:bold;">{city}, {state}</td>
                    </tr>
                    <tr style="border-bottom:1px solid #f0f0f0;">
                        <td style="padding:8px 4px;color:#888;">Fleet</td>
                        <td style="padding:8px 4px;">{k_resp} Responder · {k_guard} Guardian</td>
                    </tr>
                    <tr style="border-bottom:1px solid #f0f0f0;">
                        <td style="padding:8px 4px;color:#888;">Call Coverage</td>
                        <td style="padding:8px 4px;">{coverage:.1f}%</td>
                    </tr>
                    <tr style="border-bottom:1px solid #f0f0f0;">
                        <td style="padding:8px 4px;color:#888;">User Name</td>
                        <td style="padding:8px 4px;">{name if name else '—'}</td>
                    </tr>
                    <tr>
                        <td style="padding:8px 4px;color:#888;">User Email</td>
                        <td style="padding:8px 4px;">
                            {f'<a href="mailto:{email}">{email}</a>' if email else '—'}
                        </td>
                    </tr>
                </table>
                <div style="margin-top:16px;font-size:11px;color:#bbb;">
                    {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} UTC
                </div>
            </div>
        </div>
        </body></html>
        """
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = gmail_address
        msg["To"]      = notify_address
        msg.attach(MIMEText(body, "html"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=8) as server:
            server.login(gmail_address, app_password)
            server.sendmail(gmail_address, notify_address, msg.as_string())
    except Exception:
        pass


def _log_to_sheets(city, state, file_type, k_resp, k_guard, coverage, name, email):
    try:
        sheet_id   = st.secrets.get("GOOGLE_SHEET_ID", "")
        creds_dict = st.secrets.get("gcp_service_account", {})
        if not sheet_id or not creds_dict:
            return
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        creds  = Credentials.from_service_account_info(dict(creds_dict), scopes=scopes)
        client = gspread.authorize(creds)
        sheet  = client.open_by_key(sheet_id).sheet1
        sheet.append_row([
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            city, state, file_type,
            k_resp, k_guard,
            round(coverage, 1),
            name, email
        ])
    except Exception:
        pass

SHAPEFILE_DIR = "jurisdiction_data"
if not os.path.exists(SHAPEFILE_DIR):
    os.makedirs(SHAPEFILE_DIR)

# ============================================================
# 3. CSS & THEMING
# ============================================================
is_dark = st.session_state.get('theme', 'Dark Mode') == "Dark Mode"

bg_main = "#000000" if is_dark else "#ffffff"
bg_sidebar = "#111111" if is_dark else "#f8f9fa"
text_main = "#ffffff" if is_dark else "#222222"
text_muted = "#aaaaaa" if is_dark else "#666666"
accent_color = "#00D2FF" if is_dark else "#ff4b4b"
card_bg = "#111111" if is_dark else "#ffffff"
card_border = "#333333" if is_dark else "#e0e0e0"
card_text = "#eeeeee" if is_dark else "#222222"
card_title = "#ffffff" if is_dark else "#333333"
budget_box_bg = "#0a0a0a" if is_dark else "#ffffff"
budget_box_border = "#00D2FF" if is_dark else "#ff4b4b"
budget_box_shadow = "rgba(0, 210, 255, 0.15)" if is_dark else "rgba(0, 0, 0, 0.05)"
map_style = "carto-darkmatter" if is_dark else "open-street-map"
map_boundary_color = "#ffffff" if is_dark else "#222222"
map_incident_color = "#00D2FF" if is_dark else "#000080"
legend_bg = "rgba(0, 0, 0, 0.7)" if is_dark else "rgba(255, 255, 255, 0.9)"
legend_text = "#ffffff" if is_dark else "#333333"

theme_css = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;700&family=Manrope:wght@400;600;700&display=swap');
.stApp, .main {{ background-color: {bg_main} !important; }}
html, body, [class*="css"], p, label, li, h1, h2, h3, h4, h5, h6 {{ font-family: 'Manrope', sans-serif !important; color: {text_main} !important; font-size: 18px !important; }}
[data-testid="stSidebar"] {{ background-color: {bg_sidebar} !important; border-right: 1px solid {card_border}; }}
[data-testid="stSidebar"] img {{ filter: invert(1) brightness(2); }}
div[data-testid="stMetricValue"] {{ font-family: 'IBM Plex Mono', monospace !important; color: {accent_color} !important; }}
div[role="radiogroup"] label div {{ font-size: 20px !important; }}
.stRadio label p, .stMultiSelect label p, .stSlider label p, .stToggle label p, .stCheckbox label p {{ font-weight: 600 !important; font-size: 0.85rem !important; }}
[data-testid="stSidebar"] hr {{ margin-top: 0.25rem !important; margin-bottom: 0.25rem !important; }}
[data-testid="stSidebar"] h3 {{ padding-top: 0rem !important; margin-top: 0rem !important; padding-bottom: 0.25rem !important; }}
[data-testid="stSidebar"] .stMultiSelect, [data-testid="stSidebar"] .stSlider {{ margin-bottom: -0.75rem !important; }}
.sidebar-section-header {{ font-size: 0.65rem !important; font-weight: 800 !important; letter-spacing: 1.5px !important; text-transform: uppercase !important; color: {accent_color} !important; border-top: 1px solid {card_border}; padding-top: 12px; margin-top: 4px; margin-bottom: 8px; }}
@media print {{ section[data-testid="stSidebar"], header[data-testid="stHeader"] {{ display: none !important; }} }}
</style>
"""
st.markdown(theme_css, unsafe_allow_html=True)

# Prevent accidental navigation data loss
components.html("""<script>window.addEventListener('beforeunload', function(e) { if (window._brincHasData) { e.preventDefault(); e.returnValue = ''; } });</script>""", height=0)

# ============================================================
# 4. HTML SCHEDULER STRING
# ============================================================
SCHEDULER_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BRINC DFR · Operator Schedule Optimizer</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=Manrope:wght@400;500;600;700;800;900&display=swap');
:root { --bg: #000000; --bg2: #080808; --bg3: #0f0f0f; --border: #1e1e1e; --border2: #2a2a2a; --accent: #00D2FF; --gold: #FFD700; --green: #39FF14; --red: #FF3B3B; --text: #ffffff; --muted: #555555; --muted2: #888888; --card: #0a0a0a; }
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: var(--bg); color: var(--text); font-family: 'Manrope', sans-serif; min-height: 100vh; overflow-x: hidden; }
body::before { content: ''; position: fixed; inset: 0; pointer-events: none; z-index: 0; background: repeating-linear-gradient(0deg, transparent, transparent 39px, rgba(0,210,255,0.018) 39px, rgba(0,210,255,0.018) 40px), repeating-linear-gradient(90deg, transparent, transparent 79px, rgba(0,210,255,0.018) 79px, rgba(0,210,255,0.018) 80px); }
.wrap { position: relative; z-index: 1; max-width: 1280px; margin: 0 auto; padding: 0 24px 60px; }
header { display: flex; align-items: center; justify-content: space-between; padding: 20px 0 18px; border-bottom: 1px solid var(--border); margin-bottom: 36px; }
.logo { font-family: 'IBM Plex Mono', monospace; font-size: 1.1rem; font-weight: 700; letter-spacing: 3px; color: var(--accent); }
.logo span { color: var(--muted2); font-weight: 400; }
.header-tag { font-size: 0.6rem; font-weight: 800; letter-spacing: 2px; text-transform: uppercase; color: var(--accent); background: rgba(0,210,255,0.08); border: 1px solid rgba(0,210,255,0.2); padding: 4px 12px; border-radius: 100px; }
.drop-zone { border: 1.5px dashed var(--border2); border-radius: 12px; padding: 48px 32px; text-align: center; cursor: pointer; transition: border-color 0.2s, background 0.2s; background: var(--bg2); margin-bottom: 32px; position: relative; }
.drop-zone:hover, .drop-zone.drag-over { border-color: var(--accent); background: rgba(0,210,255,0.04); }
.drop-icon { font-size: 2.4rem; display: block; margin-bottom: 12px; opacity: 0.6; }
.drop-title { font-size: 1rem; font-weight: 800; color: var(--text); margin-bottom: 6px; }
.drop-sub { font-size: 0.75rem; color: var(--muted2); line-height: 1.6; }
.drop-sub code { background: #151515; border-radius: 3px; padding: 1px 6px; font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: #aaa; }
.file-input { position: absolute; inset: 0; opacity: 0; cursor: pointer; width: 100%; height: 100%; }
.controls { display: flex; align-items: center; gap: 16px; flex-wrap: wrap; margin-bottom: 28px; padding: 16px 20px; background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; }
.ctrl-label { font-size: 0.65rem; font-weight: 800; letter-spacing: 1.5px; text-transform: uppercase; color: var(--muted2); margin-right: 4px; }
.shift-btns { display: flex; gap: 6px; }
.shift-btn { background: var(--bg3); border: 1px solid var(--border2); color: var(--muted2); font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; font-weight: 700; padding: 7px 18px; border-radius: 4px; cursor: pointer; transition: all 0.15s; }
.shift-btn:hover { border-color: var(--accent); color: var(--accent); }
.shift-btn.active { background: var(--accent); border-color: var(--accent); color: #000; }
.divider { width: 1px; height: 28px; background: var(--border2); }
.min-priority-wrap { display: flex; align-items: center; gap: 8px; }
select { background: var(--bg3); border: 1px solid var(--border2); color: var(--text); font-family: 'Manrope', sans-serif; font-size: 0.8rem; padding: 7px 12px; border-radius: 4px; cursor: pointer; outline: none; }
select:focus { border-color: var(--accent); }
.btn-generate { margin-left: auto; background: var(--accent); border: none; color: #000; font-family: 'Manrope', sans-serif; font-weight: 800; font-size: 0.8rem; padding: 9px 24px; border-radius: 4px; cursor: pointer; letter-spacing: 0.5px; transition: background 0.15s; }
.btn-generate:hover { background: #00b8e0; }
.btn-generate:disabled { background: var(--border2); color: var(--muted); cursor: not-allowed; }
.status-bar { font-size: 0.72rem; color: var(--muted2); font-family: 'IBM Plex Mono', monospace; padding: 8px 0; margin-bottom: 24px; display: none; }
.status-bar.visible { display: block; }
.status-bar .hi { color: var(--accent); }
.kpi-row { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin-bottom: 28px; }
.kpi-card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 16px 14px; border-top: 3px solid var(--accent); }
.kpi-val { font-family: 'IBM Plex Mono', monospace; font-size: 1.5rem; font-weight: 700; color: var(--accent); display: block; margin-bottom: 4px; }
.kpi-lbl { font-size: 0.6rem; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; color: var(--muted2); }
.charts-row { display: grid; grid-template-columns: 2fr 1fr; gap: 16px; margin-bottom: 28px; }
.chart-card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 20px; }
.chart-title { font-size: 0.65rem; font-weight: 800; letter-spacing: 1.5px; text-transform: uppercase; color: var(--muted2); margin-bottom: 16px; }
.schedule-section { margin-bottom: 28px; }
.section-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px; }
.section-title { font-size: 0.65rem; font-weight: 800; letter-spacing: 1.5px; text-transform: uppercase; color: var(--muted2); }
.btn-export { background: var(--bg3); border: 1px solid var(--border2); color: var(--text); font-family: 'Manrope', sans-serif; font-weight: 700; font-size: 0.72rem; padding: 7px 16px; border-radius: 4px; cursor: pointer; transition: all 0.15s; }
.btn-export:hover { border-color: var(--accent); color: var(--accent); }
.sched-table { width: 100%; border-collapse: collapse; font-size: 0.78rem; }
.sched-table th { text-align: left; padding: 8px 12px; font-size: 0.58rem; font-weight: 800; letter-spacing: 1px; text-transform: uppercase; color: var(--muted2); border-bottom: 1px solid var(--border); font-family: 'IBM Plex Mono', monospace; }
.sched-table td { padding: 11px 12px; border-bottom: 1px solid var(--border); vertical-align: middle; }
.sched-table tr:hover td { background: rgba(255,255,255,0.02); }
.sched-table tr:last-child td { border-bottom: none; }
.shift-badge { display: inline-block; padding: 3px 10px; border-radius: 3px; font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.5px; }
.badge-day   { background: rgba(255,215,0,0.12); color: var(--gold); border: 1px solid rgba(255,215,0,0.25); }
.badge-eve   { background: rgba(0,210,255,0.1);  color: var(--accent); border: 1px solid rgba(0,210,255,0.25); }
.badge-night { background: rgba(57,255,20,0.08); color: var(--green); border: 1px solid rgba(57,255,20,0.2); }
.bar-wrap { display: flex; align-items: center; gap: 8px; }
.bar-bg { flex: 1; height: 6px; background: var(--border); border-radius: 3px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 3px; background: var(--accent); transition: width 0.4s; }
.bar-pct { font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: var(--muted2); min-width: 36px; text-align: right; }
.ops-badge { background: rgba(0,210,255,0.1); border: 1px solid rgba(0,210,255,0.2); color: var(--accent); font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; font-weight: 700; padding: 3px 10px; border-radius: 3px; }
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 28px; }
.scroll-table { max-height: 300px; overflow-y: auto; }
.scroll-table::-webkit-scrollbar { width: 4px; }
.scroll-table::-webkit-scrollbar-track { background: transparent; }
.scroll-table::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }
.hidden { display: none !important; }
.loading { text-align: center; padding: 60px 20px; font-family: 'IBM Plex Mono', monospace; color: var(--muted2); font-size: 0.8rem; }
.spinner { display: inline-block; width: 20px; height: 20px; border: 2px solid var(--border2); border-top-color: var(--accent); border-radius: 50%; animation: spin 0.8s linear infinite; margin-right: 10px; vertical-align: middle; }
@keyframes spin { to { transform: rotate(360deg); } }
footer { border-top: 1px solid var(--border); padding-top: 20px; margin-top: 20px; font-size: 0.65rem; color: var(--muted); text-align: center; line-height: 1.8; }
footer a { color: var(--accent); text-decoration: none; }
.ai-banner { display: none; align-items: center; gap: 12px; background: rgba(0,210,255,0.05); border: 1px solid rgba(0,210,255,0.2); border-radius: 8px; padding: 12px 16px; margin-bottom: 20px; font-size: 0.75rem; }
.ai-banner.visible { display: flex; }
.ai-banner .ai-text { flex: 1; color: var(--muted2); line-height: 1.5; }
.ai-banner .ai-text strong { color: var(--accent); }
.ai-mapping { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }
.ai-chip { background: var(--bg3); border: 1px solid var(--border2); border-radius: 3px; padding: 2px 8px; font-family: 'IBM Plex Mono', monospace; font-size: 0.62rem; color: var(--muted2); }
.ai-chip span { color: var(--accent); }
.api-row { display: flex; align-items: center; gap: 10px; margin-bottom: 20px; padding: 14px 16px; background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; }
.api-row label { font-size: 0.65rem; font-weight: 800; letter-spacing: 1px; text-transform: uppercase; color: var(--muted2); white-space: nowrap; }
.api-row input[type=password] { flex: 1; background: var(--bg3); border: 1px solid var(--border2); color: var(--text); font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; padding: 7px 12px; border-radius: 4px; outline: none; }
.api-row input[type=password]:focus { border-color: var(--accent); }
.api-row input[type=password]::placeholder { color: var(--muted); }
.api-note { font-size: 0.62rem; color: var(--muted); white-space: nowrap; }
.calendar-grid { display: flex; flex-wrap: wrap; gap: 20px; margin-top: 8px; }
.cal-month { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 16px; min-width: 224px; }
.cal-month-title { font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; font-weight: 700; color: var(--accent); margin-bottom: 10px; letter-spacing: 1px; text-transform: uppercase; }
.cal-dow-row { display: grid; grid-template-columns: repeat(7,1fr); gap: 2px; margin-bottom: 4px; }
.cal-dow { font-size: 0.55rem; font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase; color: var(--muted); text-align: center; padding: 2px 0; }
.cal-days { display: grid; grid-template-columns: repeat(7,1fr); gap: 2px; }
.cal-day { aspect-ratio: 1; border-radius: 3px; display: flex; flex-direction: column; align-items: center; justify-content: center; cursor: default; position: relative; transition: transform 0.1s; }
.cal-day:hover { transform: scale(1.18); z-index: 2; }
.cal-day.empty  { background: transparent; pointer-events: none; }
.cal-day.no-data { background: var(--bg3); color: var(--muted); font-size: 0.6rem; font-weight: 600; }
.cal-day.shift-day   { background: rgba(255,215,0,0.18);  color: var(--gold);   border: 1px solid rgba(255,215,0,0.3); }
.cal-day.shift-eve   { background: rgba(0,210,255,0.15);  color: var(--accent); border: 1px solid rgba(0,210,255,0.3); }
.cal-day.shift-night { background: rgba(57,255,20,0.1);   color: var(--green);  border: 1px solid rgba(57,255,20,0.25); }
.cal-day.hotday { box-shadow: 0 0 7px rgba(255,59,59,0.55); }
.cal-day .cal-num { font-size: 0.65rem; font-weight: 700; line-height: 1; }
.cal-day .cal-dot { width: 4px; height: 4px; border-radius: 50%; background: currentColor; margin-top: 2px; opacity: 0.7; }
.cal-legend { display: flex; gap: 14px; margin-bottom: 14px; flex-wrap: wrap; }
.cal-legend-item { display: flex; align-items: center; gap: 5px; font-size: 0.62rem; color: var(--muted2); }
.cal-legend-swatch { width: 10px; height: 10px; border-radius: 2px; }
@media print { body::before { display: none; } .drop-zone, .controls, .btn-export, .btn-generate, .api-row, footer { display: none !important; } .kpi-card { border: 1px solid #ccc !important; } }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="logo">BRINC <span>·</span> DFR</div>
    <div style="text-align:center;">
      <div style="font-size:1rem;font-weight:900;color:var(--text);letter-spacing:-0.3px;">Operator Schedule Optimizer</div>
      <div style="font-size:0.65rem;color:var(--muted2);margin-top:2px;">Upload CAD data · Analyze call patterns · Generate optimal shifts</div>
    </div>
    <div class="header-tag">DFR WORKFORCE PLANNING</div>
  </header>
  <div class="api-row">
    <label>🤖 Anthropic API Key</label>
    <input type="password" id="apiKey" placeholder="sk-ant-... (optional — enables smart column detection for non-standard CSVs)">
    <span class="api-note">Key stays in your browser · never transmitted except to api.anthropic.com</span>
  </div>
  <div class="ai-banner" id="aiBanner">
    <span style="font-size:1.3rem;">🤖</span>
    <div class="ai-text">
      <strong>AI Column Detection Active</strong> — Claude identified your CSV structure automatically.<br>
      <div class="ai-mapping" id="aiMappingChips"></div>
    </div>
  </div>
  <div class="drop-zone" id="dropZone">
    <input type="file" class="file-input" id="fileInput" accept=".csv">
    <span class="drop-icon">📂</span>
    <div class="drop-title">Drop your CAD calls CSV here</div>
    <div class="drop-sub">
      Expects columns: <code>Received Date</code> · <code>Nature</code> · <code>Priority</code> ·
      <code>X-Coordinate</code> · <code>Y-Coordinate</code> · <code>Address</code><br>
      Supports up to 500,000 rows · Processed entirely in your browser · No data uploaded
    </div>
  </div>
  <div class="controls" id="controls">
    <span class="ctrl-label">Shift Length</span>
    <div class="shift-btns">
      <button class="shift-btn" data-h="8"  onclick="setShift(8)">8 hr</button>
      <button class="shift-btn active" data-h="10" onclick="setShift(10)">10 hr</button>
      <button class="shift-btn" data-h="12" onclick="setShift(12)">12 hr</button>
    </div>
    <div class="divider"></div>
    <div class="min-priority-wrap">
      <span class="ctrl-label">Min Priority</span>
      <select id="minPriority" onchange="generate()">
        <option value="1">Priority 1 only</option>
        <option value="2" selected>Priority 1–2</option>
        <option value="3">Priority 1–3</option>
        <option value="4">Priority 1–4</option>
        <option value="99">All priorities</option>
      </select>
    </div>
    <div class="divider"></div>
    <div class="min-priority-wrap">
      <span class="ctrl-label">Operators / Shift</span>
      <select id="opsPerShift" onchange="generate()">
        <option value="1">1 operator</option>
        <option value="2" selected>2 operators</option>
        <option value="3">3 operators</option>
        <option value="4">4 operators</option>
      </select>
    </div>
    <button class="btn-generate" id="btnGenerate" onclick="generate()" disabled>Generate Schedule</button>
  </div>
  <div class="status-bar" id="statusBar"></div>
  <div class="loading hidden" id="loadingDiv">
    <span class="spinner"></span> Parsing CSV and computing schedules…
  </div>
  <div id="mainContent" class="hidden">
    <div class="kpi-row" id="kpiRow"></div>
    <div class="charts-row">
      <div class="chart-card">
        <div class="chart-title">Call Volume by Hour of Day — weighted by Priority</div>
        <canvas id="hourChart" height="120"></canvas>
      </div>
      <div class="chart-card">
        <div class="chart-title">Priority Distribution</div>
        <canvas id="priorityChart" height="120"></canvas>
      </div>
    </div>
    <div class="two-col">
      <div class="chart-card">
        <div class="chart-title">Top Call Types</div>
        <div class="scroll-table">
          <table class="sched-table" id="callTypeTable"></table>
        </div>
      </div>
      <div class="chart-card">
        <div class="chart-title">Busiest Hours (Priority 1–2)</div>
        <div class="scroll-table">
          <table class="sched-table" id="busyHoursTable"></table>
        </div>
      </div>
    </div>
    <div class="schedule-section" id="schedSection">
      <div class="section-header">
        <div class="section-title" id="schedTitle">Recommended Shift Schedule</div>
        <div style="display:flex;gap:8px;">
          <button class="btn-export" onclick="exportScheduleHTML()">📄 Export HTML Report</button>
          <button class="btn-export" onclick="exportScheduleCSV()">⬇ Export CSV</button>
        </div>
      </div>
      <div class="chart-card">
        <table class="sched-table" id="schedTable"></table>
      </div>
    </div>
    <div class="chart-card" style="margin-bottom:28px;">
      <div class="chart-title">Shift Coverage Overlay — call volume vs. assigned shifts</div>
      <canvas id="coverageChart" height="100"></canvas>
    </div>
    <div class="schedule-section" id="calendarSection">
      <div class="section-header">
        <div class="section-title">Daily Shift Calendar</div>
        <div style="font-size:0.65rem;color:var(--muted2);">Each day colored by dominant shift · 🔴 glow = highest call volume day in month</div>
      </div>
      <div class="chart-card">
        <div class="cal-legend">
          <div class="cal-legend-item"><div class="cal-legend-swatch" style="background:rgba(255,215,0,0.3);border:1px solid rgba(255,215,0,0.4)"></div>Day Shift</div>
          <div class="cal-legend-item"><div class="cal-legend-swatch" style="background:rgba(0,210,255,0.2);border:1px solid rgba(0,210,255,0.4)"></div>Evening Shift</div>
          <div class="cal-legend-item"><div class="cal-legend-swatch" style="background:rgba(57,255,20,0.15);border:1px solid rgba(57,255,20,0.3)"></div>Night Shift</div>
          <div class="cal-legend-item"><div class="cal-legend-swatch" style="background:var(--bg3)"></div>No Data</div>
        </div>
        <div class="calendar-grid" id="calendarGrid"></div>
      </div>
    </div>
  </div>
  <footer>
    BRINC Drones, Inc. · <a href="https://brincdrones.com" target="_blank">brincdrones.com</a>
    · DFR Operator Schedule Optimizer · All analysis performed locally in your browser · No data transmitted
  </footer>
</div>

<script>
let parsedData   = null;
let shiftHours   = 10;
let hourChart    = null;
let priorityChart = null;
let coverageChart = null;
let scheduleData  = [];

const dropZone  = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');

dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) processFile(file);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) processFile(fileInput.files[0]);
});

function smartSplit(line) {
  const result = []; let cur = ''; let inQ = false;
  for (let c of line) {
    if (c === '"') { inQ = !inQ; }
    else if (c === ',' && !inQ) { result.push(cur.trim()); cur = ''; }
    else { cur += c; }
  }
  result.push(cur.trim());
  return result;
}

function parseDate(s) {
  if (!s) return null;
  try {
    const d = new Date(s);
    if (!isNaN(d.getTime())) return d;
    const m = s.match(/(\d{1,2})\/(\d{1,2})\/(\d{2,4})\s+(\d{1,2}):(\d{2})/);
    if (m) {
      let yr = parseInt(m[3]);
      if (yr < 100) yr += 2000;
      return new Date(yr, parseInt(m[1])-1, parseInt(m[2]), parseInt(m[4]), parseInt(m[5]));
    }
  } catch(e) {}
  return null;
}

function renderAnalytics() {
  if (!parsedData) return;
  const total = parsedData.length;
  const p12   = parsedData.filter(r => r.priority <= 2).length;
  const p123  = parsedData.filter(r => r.priority <= 3).length;
  const withCoords = parsedData.filter(r => !isNaN(r.lat) && !isNaN(r.lon) && r.lat !== 0 && r.lon !== 0).length;
  const peakHour = getPeakHour();

  document.getElementById('kpiRow').innerHTML = `
    <div class="kpi-card"><span class="kpi-val">${total.toLocaleString()}</span><div class="kpi-lbl">Total Incidents</div></div>
    <div class="kpi-card" style="border-top-color:#FF3B3B"><span class="kpi-val" style="color:#FF3B3B">${p12.toLocaleString()}</span><div class="kpi-lbl">Priority 1–2 Calls</div></div>
    <div class="kpi-card" style="border-top-color:var(--gold)"><span class="kpi-val" style="color:var(--gold)">${p123.toLocaleString()}</span><div class="kpi-lbl">Priority 1–3 Calls</div></div>
    <div class="kpi-card" style="border-top-color:var(--green)"><span class="kpi-val" style="color:var(--green)">${withCoords.toLocaleString()}</span><div class="kpi-lbl">With Coordinates</div></div>
    <div class="kpi-card" style="border-top-color:#9b59b6"><span class="kpi-val" style="color:#9b59b6">${formatHour(peakHour)}</span><div class="kpi-lbl">Peak Hour</div></div>
  `;

  const hourCounts = Array(24).fill(0);
  const hourWeighted = Array(24).fill(0);
  parsedData.forEach(r => {
    hourCounts[r.hour]++;
    hourWeighted[r.hour] += priorityWeight(r.priority);
  });

  if (hourChart) hourChart.destroy();
  hourChart = new Chart(document.getElementById('hourChart'), {
    type: 'bar',
    data: {
      labels: Array.from({length:24}, (_,i) => formatHour(i)),
      datasets: [
        { label: 'Raw Count', data: hourCounts, backgroundColor: 'rgba(0,210,255,0.15)', borderColor: 'rgba(0,210,255,0.4)', borderWidth: 1, yAxisID: 'y' },
        { label: 'Priority-Weighted', data: hourWeighted, type: 'line', borderColor: '#FF3B3B', backgroundColor: 'rgba(255,59,59,0.08)', borderWidth: 2, pointRadius: 3, pointBackgroundColor: '#FF3B3B', fill: true, tension: 0.4, yAxisID: 'y1' }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: true,
      plugins: { legend: { labels: { color: '#888', font: { size: 11 } } } },
      scales: {
        x: { ticks: { color: '#666', font: { size: 10 } }, grid: { color: '#1a1a1a' } },
        y: { ticks: { color: '#666', font: { size: 10 } }, grid: { color: '#1a1a1a' }, title: { display: true, text: 'Call Count', color: '#555', font: { size: 10 } } },
        y1: { position: 'right', ticks: { color: '#666', font: { size: 10 } }, grid: { drawOnChartArea: false }, title: { display: true, text: 'Weighted Score', color: '#555', font: { size: 10 } } }
      }
    }
  });

  const priCounts = {};
  parsedData.forEach(r => { priCounts[r.priority] = (priCounts[r.priority]||0)+1; });
  const priLabels = Object.keys(priCounts).sort((a,b)=>a-b);
  const priColors = ['#FF3B3B','#FF8C00','#FFD700','#00D2FF','#39FF14','#888888','#555555'];

  if (priorityChart) priorityChart.destroy();
  priorityChart = new Chart(document.getElementById('priorityChart'), {
    type: 'doughnut',
    data: {
      labels: priLabels.map(p => `Priority ${p}`),
      datasets: [{ data: priLabels.map(p => priCounts[p]), backgroundColor: priLabels.map((_,i) => priColors[i]||'#444'), borderWidth: 0, hoverOffset: 4 }]
    },
    options: {
      responsive: true, maintainAspectRatio: true, cutout: '65%',
      plugins: { legend: { position: 'right', labels: { color: '#888', font: { size: 10 } } } }
    }
  });

  const natureCounts = {};
  parsedData.forEach(r => { natureCounts[r.nature] = (natureCounts[r.nature]||0)+1; });
  const sorted = Object.entries(natureCounts).sort((a,b)=>b[1]-a[1]).slice(0,20);
  const maxN = sorted[0][1];
  document.getElementById('callTypeTable').innerHTML = `
    <thead><tr><th>Call Type</th><th>Count</th><th>Share</th></tr></thead>
    <tbody>${sorted.map(([n,c]) => `
      <tr>
        <td style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;">${n}</td>
        <td style="font-family:'IBM Plex Mono',monospace;">${c.toLocaleString()}</td>
        <td><div class="bar-wrap"><div class="bar-bg"><div class="bar-fill" style="width:${(c/maxN*100).toFixed(1)}%"></div></div><span class="bar-pct">${(c/total*100).toFixed(1)}%</span></div></td>
      </tr>`).join('')}
    </tbody>`;

  const p12Counts = Array(24).fill(0);
  parsedData.filter(r=>r.priority<=2).forEach(r => p12Counts[r.hour]++);
  const p12Max = Math.max(...p12Counts);
  const p12Sorted = p12Counts.map((c,h)=>({h,c})).sort((a,b)=>b.c-a.c);
  document.getElementById('busyHoursTable').innerHTML = `
    <thead><tr><th>Hour</th><th>P1-2 Calls</th><th>Intensity</th></tr></thead>
    <tbody>${p12Sorted.slice(0,15).map(({h,c}) => `
      <tr>
        <td style="font-family:'IBM Plex Mono',monospace;">${formatHour(h)}</td>
        <td style="font-family:'IBM Plex Mono',monospace;color:var(--red);">${c.toLocaleString()}</td>
        <td><div class="bar-wrap"><div class="bar-bg"><div class="bar-fill" style="width:${(c/p12Max*100).toFixed(1)}%;background:#FF3B3B"></div></div><span class="bar-pct">${(c/p12Max*100).toFixed(0)}%</span></div></td>
      </tr>`).join('')}
    </tbody>`;

  document.getElementById('mainContent').classList.remove('hidden');
}

function setShift(h) {
  shiftHours = h;
  document.querySelectorAll('.shift-btn').forEach(b => b.classList.toggle('active', parseInt(b.dataset.h) === h));
  if (parsedData) generate();
}

function findOptimalShifts(scores, shiftLen, numShifts) {
  const windows = [];
  for (let start = 0; start < 24; start++) {
    let score = 0;
    const hours = [];
    for (let h = 0; h < shiftLen; h++) {
      const hr = (start + h) % 24;
      score += scores[hr];
      hours.push(hr);
    }
    windows.push({ start, end: (start + shiftLen - 1) % 24, hours, score });
  }

  const step = Math.floor(24 / numShifts);
  const result = [];
  for (let i = 0; i < numShifts; i++) {
    const base = (i * step) % 24;
    let best = windows[base];
    for (let d = -2; d <= 2; d++) {
      const idx = ((base + d) + 24) % 24;
      if (windows[idx].score > best.score) best = windows[idx];
    }
    result.push(best);
  }
  return result;
}

function renderSchedule(hourScore, totalScore) {
  const maxPct = Math.max(...scheduleData.map(s => s.pct));
  document.getElementById('schedTable').innerHTML = `
    <thead>
      <tr>
        <th>Shift</th>
        <th>Hours</th>
        <th>Type</th>
        <th>Operators</th>
        <th>Call Volume Coverage</th>
        <th>P1–2 Coverage</th>
        <th>Top Call Type</th>
      </tr>
    </thead>
    <tbody>
      ${scheduleData.map(s => `
        <tr>
          <td style="font-family:'IBM Plex Mono',monospace;font-weight:700;">Shift ${s.shift}</td>
          <td style="font-family:'IBM Plex Mono',monospace;color:var(--accent);">${formatHour(s.start)} – ${formatHour(s.end + 1 >= 24 ? 0 : s.end + 1)}</td>
          <td><span class="shift-badge ${shiftClass(s.start)}">${s.label}</span></td>
          <td><span class="ops-badge">${s.ops}x</span></td>
          <td>
            <div class="bar-wrap">
              <div class="bar-bg"><div class="bar-fill" style="width:${(s.pct/maxPct*100).toFixed(1)}%"></div></div>
              <span class="bar-pct">${s.pct.toFixed(1)}%</span>
            </div>
          </td>
          <td>
            <div class="bar-wrap">
              <div class="bar-bg"><div class="bar-fill" style="width:${s.p12pct.toFixed(1)}%;background:#FF3B3B"></div></div>
              <span class="bar-pct" style="color:#FF3B3B">${s.p12pct.toFixed(1)}%</span>
            </div>
          </td>
          <td style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;color:var(--muted2);">${s.topNature}</td>
        </tr>
      `).join('')}
    </tbody>`;
}

function renderCoverageChart(hourScore, shifts) {
  const labels = Array.from({length:24}, (_,i) => formatHour(i));
  const shiftOverlay = Array(24).fill(0);
  shifts.forEach((s, i) => s.hours.forEach(h => { shiftOverlay[h] = i+1; }));

  const shiftColors = ['rgba(0,210,255,0.7)','rgba(255,215,0,0.7)','rgba(57,255,20,0.7)','rgba(255,107,107,0.7)'];
  const barColors = shiftOverlay.map((s, i) => s > 0 ? shiftColors[(s-1) % shiftColors.length] : 'rgba(50,50,50,0.5)');

  if (coverageChart) coverageChart.destroy();
  coverageChart = new Chart(document.getElementById('coverageChart'), {
    type: 'bar',
    data: {
      labels,
      datasets: [{ label: 'Weighted Call Score', data: hourScore, backgroundColor: barColors, borderWidth: 0 }]
    },
    options: {
      responsive: true, maintainAspectRatio: true,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => {
              const s = shiftOverlay[ctx.dataIndex];
              return ` Score: ${ctx.parsed.y.toFixed(0)} · ${s > 0 ? 'Shift '+s : 'Uncovered'}`;
            }
          }
        }
      },
      scales: {
        x: { ticks: { color: '#666', font: { size: 10 } }, grid: { color: '#1a1a1a' } },
        y: { ticks: { color: '#666', font: { size: 10 } }, grid: { color: '#1a1a1a' } }
      }
    }
  });
}

function exportScheduleCSV() {
  if (!scheduleData.length) return;
  const rows = [['Shift','Start Hour','End Hour','Type','Operators','Volume Coverage %','P1-2 Coverage %','Top Call Type']];
  scheduleData.forEach(s => rows.push([
    'Shift '+s.shift, formatHour(s.start),
    formatHour((s.start + shiftHours) % 24),
    s.label, s.ops, s.pct.toFixed(1), s.p12pct.toFixed(1), s.topNature
  ]));
  const csv = rows.map(r => r.map(v => `"${v}"`).join(',')).join('\n');
  download('DFR_Operator_Schedule.csv', csv, 'text/csv');
}

function exportScheduleHTML() {
  if (!scheduleData.length) return;
  const date = new Date().toLocaleDateString('en-US', {year:'numeric',month:'long',day:'numeric'});
  const tableRows = scheduleData.map(s => `
    <tr>
      <td><strong>Shift ${s.shift}</strong></td>
      <td style="font-family:monospace;">${formatHour(s.start)} – ${formatHour((s.start+shiftHours)%24)}</td>
      <td>${s.label}</td>
      <td>${s.ops} operator${s.ops>1?'s':''}</td>
      <td>${s.pct.toFixed(1)}%</td>
      <td style="color:#cc0000;">${s.p12pct.toFixed(1)}%</td>
      <td style="font-family:monospace;font-size:12px;">${s.topNature}</td>
    </tr>`).join('');

  const html = `<!DOCTYPE html><html><head><title>BRINC DFR Operator Schedule</title>
  <style>
    body{font-family:'Helvetica Neue',Arial,sans-serif;color:#333;padding:40px;background:#f4f6f9;margin:0;}
    .page{max-width:960px;margin:0 auto;background:#fff;padding:48px;border-radius:8px;box-shadow:0 4px 15px rgba(0,0,0,0.06);}
    .header{border-bottom:2px solid #00D2FF;padding-bottom:16px;margin-bottom:28px;display:flex;justify-content:space-between;align-items:flex-end;}
    .logo{font-size:22px;font-weight:900;letter-spacing:2px;color:#000;}
    h2{font-size:16px;color:#444;margin-top:24px;border-bottom:1px solid #eee;padding-bottom:6px;}
    table{width:100%;border-collapse:collapse;font-size:13px;margin-top:10px;}
    th{background:#f1f1f1;padding:8px 12px;text-align:left;font-size:11px;text-transform:uppercase;color:#666;letter-spacing:0.5px;}
    td{padding:10px 12px;border-bottom:1px solid #eee;}
    .kpis{display:flex;gap:16px;margin-bottom:24px;}
    .kpi{flex:1;border:1px solid #eaeaea;border-radius:6px;padding:16px;border-top:3px solid #00D2FF;}
    .kpi-val{font-size:20px;font-weight:700;color:#00D2FF;}
    .kpi-lbl{font-size:10px;text-transform:uppercase;color:#999;font-weight:700;letter-spacing:0.5px;margin-top:2px;}
    .disclaimer{background:#fff3cd;border-left:4px solid #ffc107;padding:10px 14px;font-size:11px;color:#856404;margin-bottom:20px;}
    footer{margin-top:32px;text-align:center;font-size:11px;color:#aaa;border-top:1px solid #eee;padding-top:16px;}
    footer a{color:#00D2FF;text-decoration:none;}
  </style></head><body><div class="page">
  <div class="header">
    <div><div class="logo">BRINC</div><div style="font-size:11px;color:#999;margin-top:2px;">Drone as a First Responder</div></div>
    <div style="text-align:right;"><div style="font-size:18px;font-weight:700;">DFR Operator Schedule</div><div style="font-size:12px;color:#888;margin-top:2px;">Generated ${date} · ${shiftHours}-Hour Shifts</div></div>
  </div>
  <div class="disclaimer"><strong>Note:</strong> This schedule was generated algorithmically based on historic call volume and priority weighting. Review with your operations team before implementation.</div>
  <div class="kpis">
    <div class="kpi"><div class="kpi-val">${parsedData.length.toLocaleString()}</div><div class="kpi-lbl">Total Incidents Analyzed</div></div>
    <div class="kpi"><div class="kpi-val">${parsedData.filter(r=>r.priority<=2).length.toLocaleString()}</div><div class="kpi-lbl">Priority 1–2 Calls</div></div>
    <div class="kpi"><div class="kpi-val">${shiftHours}h</div><div class="kpi-lbl">Shift Length</div></div>
    <div class="kpi"><div class="kpi-val">${Math.floor(24/shiftHours)}</div><div class="kpi-lbl">Shifts Per Day</div></div>
  </div>
  <h2>Recommended Shift Schedule</h2>
  <table><thead><tr><th>Shift</th><th>Hours</th><th>Type</th><th>Staffing</th><th>Volume Coverage</th><th>P1–2 Coverage</th><th>Top Call Type</th></tr></thead>
  <tbody>${tableRows}</tbody></table>
  <footer>BRINC Drones, Inc. · <a href="https://brincdrones.com">brincdrones.com</a> · DFR Operator Schedule Optimizer</footer>
  </div></body></html>`;
  download('BRINC_DFR_Operator_Schedule.html', html, 'text/html');
}

function download(filename, content, mime) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([content], {type: mime}));
  a.download = filename; a.click();
}

async function detectColumnsWithAI(headerRow, sampleRows) {
  const apiKey = document.getElementById('apiKey').value.trim();
  if (!apiKey) return null;

  const prompt = `You are analyzing a CSV file from a law enforcement CAD system.
Here is the header row and 3 sample data rows:

HEADER: ${headerRow.join(', ')}
ROW 1: ${sampleRows[0]?.join(', ') || 'N/A'}
ROW 2: ${sampleRows[1]?.join(', ') || 'N/A'}
ROW 3: ${sampleRows[2]?.join(', ') || 'N/A'}

Identify which column INDEX (0-based integer) corresponds to each of these fields. If a field cannot be found, return -1.

Return ONLY valid JSON, no markdown, no explanation:
{
  "date": <index of the call received/dispatched datetime column>,
  "nature": <index of call type/nature/incident type column>,
  "priority": <index of priority/urgency level column>,
  "x": <index of longitude/x-coordinate column>,
  "y": <index of latitude/y-coordinate column>,
  "address": <index of address/location column>,
  "callnum": <index of unique call/incident number column>,
  "notes": "<brief one-sentence description>"
}`;

  try {
    const res = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'x-api-key': apiKey, 'anthropic-version': '2023-06-01' },
      body: JSON.stringify({ model: 'claude-sonnet-4-20250514', max_tokens: 400, messages: [{ role: 'user', content: prompt }] })
    });
    if (!res.ok) return null;
    const data = await res.json();
    const text = data.content?.[0]?.text || '';
    const clean = text.replace(/```json|```/g, '').trim();
    return JSON.parse(clean);
  } catch(e) {
    console.warn('AI detection failed:', e);
    return null;
  }
}

function showAIBanner(mapping, header) {
  const fieldLabels = { date:'Date', nature:'Call Type', priority:'Priority', x:'Longitude', y:'Latitude', address:'Address', callnum:'Call Number' };
  const chips = Object.entries(fieldLabels).filter(([k]) => mapping[k] >= 0).map(([k, label]) => `<div class="ai-chip">${label} → <span>${header[mapping[k]] || '?'}</span></div>`).join('');
  document.getElementById('aiMappingChips').innerHTML = chips;
  document.getElementById('aiBanner').classList.add('visible');
}

async function processFile(file) {
  const apiKey = document.getElementById('apiKey').value.trim();
  document.getElementById('loadingDiv').classList.remove('hidden');
  document.getElementById('mainContent').classList.add('hidden');
  document.getElementById('aiBanner').classList.remove('visible');
  document.getElementById('btnGenerate').disabled = true;

  const reader = new FileReader();
  reader.onload = async e => {
    try {
      const text = e.target.result;
      const normalized = text.replace(/\r\n/g,'\n').replace(/\r/g,'\n');
      const lines = normalized.trim().split('\n').filter(l=>l.trim());
      if (lines.length < 2) { alert('CSV appears empty.'); document.getElementById('loadingDiv').classList.add('hidden'); return; }

      const headerRow   = smartSplit(lines[0]).map(h=>h.trim().replace(/^"|"$/g,''));
      const sampleRows  = lines.slice(1,4).map(l=>smartSplit(l));

      let aiMapping = null;
      if (apiKey) {
        setStatus('🤖 Asking Claude to identify columns…');
        aiMapping = await detectColumnsWithAI(headerRow, sampleRows);
        if (aiMapping) showAIBanner(aiMapping, headerRow);
      }

      parseCSVWithMapping(text, file.name, aiMapping);
    } catch(err) {
      alert('Error parsing CSV: ' + err.message);
      document.getElementById('loadingDiv').classList.add('hidden');
    }
  };
  reader.readAsText(file);
}

function parseCSVWithMapping(text, filename, aiMapping) {
  const normalized = text.replace(/\r\n/g,'\n').replace(/\r/g,'\n');
  const lines = normalized.trim().split('\n').filter(l=>l.trim().length>0);
  if (lines.length < 2) { alert('CSV appears to be empty or has only a header row.'); document.getElementById('loadingDiv').classList.add('hidden'); return; }

  const header = smartSplit(lines[0]).map(h=>h.trim().replace(/^"|"$/g,'').toLowerCase());

  let iDate, iNature, iPriority, iX, iY, iAddress, iCallNum;

  if (aiMapping) {
    iDate = aiMapping.date ?? -1; iNature = aiMapping.nature ?? -1; iPriority = aiMapping.priority ?? -1;
    iX = aiMapping.x ?? -1; iY = aiMapping.y ?? -1; iAddress = aiMapping.address ?? -1; iCallNum = aiMapping.callnum ?? -1;
  } else {
    const col = name => {
      const variants = {
        'date':     ['received date','date','call date','datetime','received','incident date','dispatch time','time received'],
        'nature':   ['nature','call type','type','incident type','nature of call','call_type','calltype','description','problem'],
        'priority': ['priority','pri','call priority','priority level','urgency'],
        'x':        ['x-coordinate','x coordinate','longitude','lon','long','x_coord','xcoordi','x-coord','lng'],
        'y':        ['y-coordinate','y coordinate','latitude','lat','y_coord','ycoordi','y-coord'],
        'address':  ['address','location','street','incident address','block address'],
        'callnum':  ['call number','call_number','incident','incident number','callnum','call no','cad number','event number','master incident'],
      };
      const opts = variants[name] || [name];
      return header.findIndex(h => opts.some(o => h.includes(o)));
    };
    iDate=col('date'); iNature=col('nature'); iPriority=col('priority');
    iX=col('x'); iY=col('y'); iAddress=col('address'); iCallNum=col('callnum');
  }

  if (iDate < 0) { alert('Could not find a date column.\n\nFound columns:\n' + header.join(', ') + '\n\nTip: Add an Anthropic API key above to enable AI-powered column detection.'); document.getElementById('loadingDiv').classList.add('hidden'); return; }

  const seen=[]; const records=[]; let skipped=0;

  for (let i=1; i<lines.length; i++) {
    const line=lines[i].trim(); if(!line) continue;
    let row; try { row=smartSplit(line); } catch(e) { skipped++; continue; }
    if(!row||row.length<2) { skipped++; continue; }
    const get=idx=>(idx>=0&&idx<row.length&&row[idx]!=null)?String(row[idx]).trim():'';

    const callNum=get(iCallNum);
    if(callNum&&seen.includes(callNum)) continue;
    if(callNum) seen.push(callNum);

    const dt=parseDate(get(iDate)); if(!dt) { skipped++; continue; }
    const priority=parseInt(get(iPriority))||9;
    const hour=dt.getHours();
    const lat=get(iY)?parseFloat(get(iY)):NaN;
    const lon=get(iX)?parseFloat(get(iX)):NaN;
    const nature=(get(iNature).replace(/^"|"$/g,'')||'UNKNOWN');

    records.push({ hour, priority, nature, lat, lon, address:get(iAddress), dateStr:get(iDate), callNum, dt });
  }

  if(records.length===0) { alert(`No valid records found.\n\nSkipped: ${skipped} rows\n\nTip: Add an Anthropic API key to enable AI column detection.`); document.getElementById('loadingDiv').classList.add('hidden'); return; }

  parsedData=records;
  document.getElementById('btnGenerate').disabled=false;
  setStatus(`✓ Loaded <span class="hi">${records.length.toLocaleString()}</span> unique incidents from <span class="hi">${filename}</span>${skipped>0?` · <span style="color:var(--gold)">${skipped} rows skipped</span>`:''}`);
  document.getElementById('loadingDiv').classList.add('hidden');
  renderAnalytics();
  generate();
  renderCalendar();
}

function renderCalendar() {
  if (!parsedData || !scheduleData.length) return;
  const dayMap = {};
  parsedData.forEach(r => {
    if (!r.dt) return;
    const key = `${r.dt.getFullYear()}-${String(r.dt.getMonth()+1).padStart(2,'0')}-${String(r.dt.getDate()).padStart(2,'0')}`;
    if (!dayMap[key]) dayMap[key] = { total: 0, shiftScores: {} };
    dayMap[key].total += priorityWeight(r.priority);
    const bestShift = getBestShiftForHour(r.hour);
    if (bestShift !== null) {
      dayMap[key].shiftScores[bestShift] = (dayMap[key].shiftScores[bestShift]||0) + priorityWeight(r.priority);
    }
  });

  if (!Object.keys(dayMap).length) return;
  const monthMap = {};
  Object.entries(dayMap).forEach(([key, val]) => {
    const [y, m] = key.split('-');
    const mk = `${y}-${m}`;
    if (!monthMap[mk]) monthMap[mk] = {};
    monthMap[mk][parseInt(key.split('-')[2])] = val;
  });

  const sortedMonths = Object.keys(monthMap).sort();
  const container = document.getElementById('calendarGrid');
  container.innerHTML = '';
  const MONTH_NAMES = ['January','February','March','April','May','June','July','August','September','October','November','December'];
  const DOW = ['Su','Mo','Tu','We','Th','Fr','Sa'];

  sortedMonths.forEach(mk => {
    const [y, m] = mk.split('-').map(Number);
    const monthData = monthMap[mk];
    const maxScore = Math.max(...Object.values(monthData).map(d=>d.total));

    const monthDiv = document.createElement('div');
    monthDiv.className = 'cal-month';

    const title = document.createElement('div');
    title.className = 'cal-month-title';
    title.textContent = `${MONTH_NAMES[m-1]} ${y}`;
    monthDiv.appendChild(title);

    const dowRow = document.createElement('div');
    dowRow.className = 'cal-dow-row';
    DOW.forEach(d => { const el=document.createElement('div'); el.className='cal-dow'; el.textContent=d; dowRow.appendChild(el); });
    monthDiv.appendChild(dowRow);

    const daysDiv = document.createElement('div');
    daysDiv.className = 'cal-days';

    const firstDow = new Date(y, m-1, 1).getDay();
    const daysInMonth = new Date(y, m, 0).getDate();

    for (let i=0; i<firstDow; i++) {
      const el = document.createElement('div'); el.className='cal-day empty'; daysDiv.appendChild(el);
    }

    for (let d=1; d<=daysInMonth; d++) {
      const el = document.createElement('div');
      el.className = 'cal-day';
      const dayData = monthData[d];

      if (!dayData) {
        el.classList.add('no-data');
        el.innerHTML = `<div class="cal-num">${d}</div>`;
      } else {
        const dominant = Object.entries(dayData.shiftScores).sort((a,b)=>b[1]-a[1])[0];
        if (dominant) {
          const shiftIdx = parseInt(dominant[0]);
          const sd = scheduleData[shiftIdx];
          if (sd) {
            const cls = sd.label === 'DAY' ? 'shift-day' : sd.label === 'EVE' ? 'shift-eve' : 'shift-night';
            el.classList.add(cls);
            el.title = `${sd.label} Shift · ${sd.score ? dayData.total.toFixed(0)+' weighted calls' : ''}`;
          }
        }
        if (dayData.total >= maxScore * 0.9) el.classList.add('hotday');
        el.innerHTML = `<div class="cal-num">${d}</div><div class="cal-dot"></div>`;
      }
      daysDiv.appendChild(el);
    }
    monthDiv.appendChild(daysDiv);
    container.appendChild(monthDiv);
  });
}

function getBestShiftForHour(hour) {
  if (!scheduleData.length) return null;
  let best = 0, bestScore = -1;
  scheduleData.forEach((s, i) => {
    if (s.hours && s.hours.includes(hour)) {
      if (s.score > bestScore) { bestScore = s.score; best = i; }
    }
  });
  return best;
}

function generate() {
  if (!parsedData) return;
  const minPri   = parseInt(document.getElementById('minPriority').value);
  const opsCount = parseInt(document.getElementById('opsPerShift').value);
  const hourScore = Array(24).fill(0);
  parsedData.forEach(r => {
    if (r.priority <= minPri || minPri === 99) hourScore[r.hour] += priorityWeight(r.priority);
  });
  const totalScore = hourScore.reduce((a,b)=>a+b,0);
  if (totalScore === 0) { setStatus('No calls match the selected priority filter.'); return; }
  const numShifts = Math.floor(24 / shiftHours);
  const shifts    = findOptimalShifts(hourScore, shiftHours, numShifts);

  scheduleData = shifts.map((s, i) => {
    const covered   = s.hours.reduce((acc,h)=>acc+hourScore[h],0);
    const p12covered= s.hours.reduce((acc,h)=>acc+parsedData.filter(r=>r.hour===h&&r.priority<=2).length,0);
    const p12total  = parsedData.filter(r=>r.priority<=2).length;
    const label     = shiftLabel(s.start, shiftHours);
    return { shift:i+1, label, start:s.start, end:s.end, hours:s.hours,
             score:covered, pct:(covered/totalScore*100),
             p12:p12covered, p12pct:(p12total>0?p12covered/p12total*100:0),
             ops:opsCount, topNature:getTopNatureForHours(s.hours) };
  });

  renderSchedule(hourScore, totalScore);
  renderCoverageChart(hourScore, scheduleData);
  renderCalendar();
  document.getElementById('schedTitle').textContent = `Recommended ${shiftHours}-Hour Shift Schedule (${numShifts} shifts · ${opsCount} operator${opsCount>1?'s':''}/shift)`;
  setStatus(`✓ Schedule generated — <span class="hi">${numShifts} shifts</span> of <span class="hi">${shiftHours}h</span> · covering <span class="hi">${scheduleData.reduce((a,s)=>a+s.p12,0).toLocaleString()}</span> Priority 1–2 calls`);
}
function priorityWeight(p) { const w = {1:10, 2:7, 3:5, 4:3, 5:2, 6:1, 7:1}; return w[p] || 1; }
function getPeakHour() {
  if (!parsedData) return 0;
  const counts = Array(24).fill(0);
  parsedData.filter(r=>r.priority<=2).forEach(r=>counts[r.hour]++);
  return counts.indexOf(Math.max(...counts));
}
function getTopNatureForHours(hours) {
  const counts = {};
  parsedData.filter(r => hours.includes(r.hour)).forEach(r => { counts[r.nature] = (counts[r.nature]||0)+1; });
  const top = Object.entries(counts).sort((a,b)=>b[1]-a[1])[0];
  return top ? top[0] : 'N/A';
}
function formatHour(h) {
  const hr = ((h % 24) + 24) % 24;
  const ampm = hr < 12 ? 'AM' : 'PM';
  const disp = hr === 0 ? 12 : hr > 12 ? hr-12 : hr;
  return `${disp}:00 ${ampm}`;
}
function shiftLabel(start, len) {
  const end = (start + len) % 24;
  if (start >= 6  && end <= 16) return 'DAY';
  if (start >= 14 && end <= 24) return 'EVE';
  if (start >= 22 || end <= 6)  return 'NIGHT';
  if (start >= 6)  return 'DAY';
  return 'NIGHT';
}
function shiftClass(start) {
  const lbl = shiftLabel(start, shiftHours);
  return lbl === 'DAY' ? 'badge-day' : lbl === 'EVE' ? 'badge-eve' : 'badge-night';
}
function setStatus(msg) {
  const el = document.getElementById('statusBar');
  el.innerHTML = msg;
  el.classList.add('visible');
}
</script>
</body>
</html>
"""

# ============================================================
# 7. APP NAVIGATION LOGIC
# ============================================================
try:
    if os.path.exists("logo.png"):
        st.sidebar.image("logo.png", use_container_width=True)
    else:
        st.sidebar.markdown(f"<div style='font-size:1.4rem;font-weight:900;letter-spacing:3px;color:{accent_color};padding:10px 0;'>BRINC</div>", unsafe_allow_html=True)
except Exception:
    pass

st.sidebar.markdown("<h3 style='margin-bottom:0px; margin-top:0px;'>📍 App Navigation</h3>", unsafe_allow_html=True)
app_mode = st.sidebar.radio("Navigation", ["🚁 Fleet Deployment", "📅 Operator Scheduler"], label_visibility="collapsed")
st.sidebar.markdown("---")

if app_mode == "📅 Operator Scheduler":
    st.markdown(f"<div style='font-size:1.8rem;font-weight:900;color:{text_main};'>📅 DFR Operator Schedule Optimizer</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:0.9rem;color:{text_muted};margin-bottom:20px;'>Use this standalone tool to analyze historical call volume and generate optimal staffing shifts.</div>", unsafe_allow_html=True)
    components.html(SCHEDULER_HTML, height=1200, scrolling=True)

elif app_mode == "🚁 Fleet Deployment":
    
    if not st.session_state['csvs_ready']:
        with st.sidebar.expander("💾 Load Saved Scenario", expanded=False):
            uploaded_scenario = st.file_uploader("Load .brinc file", type=['brinc','json'], label_visibility="collapsed")
            if uploaded_scenario is not None and st.session_state.get('last_loaded_scenario') != uploaded_scenario.file_id:
                try:
                    scenario_data = json.loads(uploaded_scenario.getvalue().decode("utf-8"))
                    for k in ['active_city','active_state','k_resp','k_guard','r_resp','r_guard','dfr_rate','deflect_rate']:
                        if k in scenario_data: st.session_state[k] = scenario_data[k]
                    calls_data = scenario_data.get('calls_data')
                    stations_data = scenario_data.get('stations_data')
                    st.session_state['last_loaded_scenario'] = uploaded_scenario.file_id
                    if calls_data and stations_data:
                        st.session_state['df_calls'] = pd.DataFrame(calls_data)
                        st.session_state['df_stations'] = pd.DataFrame(stations_data)
                        st.session_state['total_original_calls'] = len(calls_data)
                        st.session_state['csvs_ready'] = True
                        st.toast(f"✅ Loaded scenario for {st.session_state['active_city']}!")
                        st.rerun()
                    else:
                        st.session_state['trigger_sim'] = True
                        st.toast(f"✅ Loaded synthetic scenario for {st.session_state['active_city']}!")
                        st.rerun()
                except Exception:
                    st.error("Failed to load file — it may be corrupted or incorrectly formatted.")
                    st.session_state['last_loaded_scenario'] = uploaded_scenario.file_id

    # ============================================================
    # ONBOARDING / LANDING PAGE 
    # ============================================================
    if not st.session_state['csvs_ready']:

        st.markdown(f"""
        <div style="text-align:center; padding: 20px 0 10px 0;">
            <div style="font-size:2.2rem; font-weight:900; letter-spacing:2px; color:{accent_color};">🛰️ BRINC COS</div>
            <div style="font-size:1.1rem; color:{text_muted}; margin-top:4px;">Drone Optimizer — Coverage, Operations & Savings</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #0a0a0a 0%, #001a22 100%);
             border:1px solid {card_border}; border-left: 4px solid {accent_color};
             border-radius:8px; padding:14px 18px; margin-bottom:24px; display:flex; align-items:center; gap:16px;">
            <div style="font-size:2.5rem;">🚁</div>
            <div>
                <div style="font-weight:800; color:{accent_color}; font-size:0.9rem;">3D SWARM SIMULATION INCLUDED</div>
                <div style="font-size:0.78rem; color:{text_muted}; margin-top:2px;">
                    Deploy a fleet to unlock a live animated 3D simulation showing every DFR flight over a 24-hour day. Export grant-ready HTML proposals directly from your simulated operations.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown(f"<h3 style='color:{text_main}; margin-bottom:4px;'>🚀 Simulate Any US Region</h3>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:0.8rem; color:{text_muted}; margin-bottom:14px;'>No data needed — we fetch the real Census boundaries and generate a highly realistic 911 call distribution automatically. Combine multiple towns!</div>", unsafe_allow_html=True)

            st.markdown(f"<div style='border:1px solid {card_border}; padding:15px; border-radius:8px;'>", unsafe_allow_html=True)
            for i in range(st.session_state.city_count):
                c1, c2 = st.columns([3, 1])
                c_val = st.session_state['target_cities'][i]['city'] if i < len(st.session_state['target_cities']) else ""
                s_val = st.session_state['target_cities'][i]['state'] if i < len(st.session_state['target_cities']) else "FL"
                
                c_name = c1.text_input(f"City/Town {i+1}", value=c_val, key=f"c_{i}", help="Enter the official name of the municipality to fetch its boundary.")
                state_idx = list(STATE_FIPS.keys()).index(s_val) if s_val in STATE_FIPS else 8
                s_name = c2.selectbox(f"State {i+1}", list(STATE_FIPS.keys()), index=state_idx, key=f"s_{i}", label_visibility="collapsed" if i>0 else "visible", help="Select the state abbreviation.")
                
                if i < len(st.session_state['target_cities']):
                    st.session_state['target_cities'][i] = {"city": c_name, "state": s_name}
                else:
                    st.session_state['target_cities'].append({"city": c_name, "state": s_name})

            if st.session_state.city_count < 10:
                if st.button("➕ Add another city/town", use_container_width=True):
                    st.session_state.city_count += 1
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            submit_demo = st.button("🚀 Run Simulation", use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🎲 1-Click Demo (Random Large City)", use_container_width=True):
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

        with col_right:
            st.markdown(f"<h3 style='color:{text_main}; margin-bottom:4px;'>📁 Upload Real Data</h3>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-size:0.8rem; color:{text_muted}; margin-bottom:12px;">
                Upload your own <b>calls.csv</b> and <b>stations.csv</b> (Requires <code>lat</code> and <code>lon</code> columns). The jurisdiction boundary will auto-detect from your coordinates.
            </div>
            """, unsafe_allow_html=True)
            uploaded_files = st.file_uploader("Drop calls.csv & stations.csv here", accept_multiple_files=True, label_visibility="collapsed")
            call_file, station_file = None, None
            if uploaded_files:
                for f in uploaded_files:
                    fname = f.name.lower()
                    if fname == "calls.csv": call_file = f
                    elif fname == "stations.csv": station_file = f
                if call_file and station_file:
                    df_c = pd.read_csv(call_file)
                    df_c.columns = [str(c).lower().strip() for c in df_c.columns]
                    df_c = df_c.rename(columns={'latitude':'lat','longitude':'lon'})
                    if 'lat' not in df_c.columns or 'lon' not in df_c.columns:
                        st.error(f"❌ calls.csv must have lat/lon columns. Found: {', '.join(df_c.columns)}")
                        st.stop()
                    keep_c = ['lat','lon'] + (['priority'] if 'priority' in df_c.columns else [])
                    df_c = df_c[keep_c].dropna(subset=['lat','lon']).reset_index(drop=True)
                    st.session_state['total_original_calls'] = len(df_c)
                    if len(df_c) > 25000:
                        df_c = df_c.sample(25000, random_state=42).reset_index(drop=True)
                        st.toast("⚠️ Sampled to 25,000 calls for performance.")
                    st.session_state['df_calls'] = df_c
                    df_s = pd.read_csv(station_file)
                    df_s.columns = [str(c).lower().strip() for c in df_s.columns]
                    df_s = df_s.rename(columns={'latitude':'lat','longitude':'lon'})
                    if 'lat' not in df_s.columns or 'lon' not in df_s.columns:
                        st.error(f"❌ stations.csv must have lat/lon columns. Found: {', '.join(df_s.columns)}")
                        st.stop()
                    keep_s = ['lat','lon'] + [c for c in ['name','type'] if c in df_s.columns]
                    df_s = df_s[keep_s].dropna(subset=['lat','lon']).reset_index(drop=True)
                    if 'name' not in df_s.columns:
                        df_s['name'] = [f"Station {i+1}" for i in range(len(df_s))]
                    if len(df_s) > 100:
                        df_s = df_s.sample(100, random_state=42).reset_index(drop=True)
                    st.session_state['df_stations'] = df_s
                    with st.spinner("🌍 Thank you for keeping our communities safe! Auto-detecting jurisdiction..."):
                        detected_state_full, detected_city = reverse_geocode_state(df_c['lat'].iloc[0], df_c['lon'].iloc[0])
                        if detected_state_full and detected_state_full in US_STATES_ABBR:
                            st.session_state['active_state'] = US_STATES_ABBR[detected_state_full]
                            if detected_city and detected_city != 'Unknown City':
                                st.session_state['active_city'] = detected_city
                            st.toast(f"📍 Detected: {st.session_state['active_city']}, {st.session_state['active_state']}")
                    st.session_state['csvs_ready'] = True
                    st.rerun()
                elif call_file or station_file:
                    missing = "stations.csv" if call_file else "calls.csv"
                    st.warning(f"⚠️ Also upload **{missing}** to continue.")

        if submit_demo or st.session_state.get('trigger_sim', False):
            if st.session_state.get('trigger_sim', False):
                st.session_state['trigger_sim'] = False
                
            active_targets = [loc for loc in st.session_state['target_cities'] if loc['city'].strip()]
            if not active_targets:
                st.error("Please enter at least one valid city name.")
                st.stop()

            if len(active_targets) == 1:
                st.session_state['active_city'] = active_targets[0]['city']
                st.session_state['active_state'] = active_targets[0]['state']
            else:
                st.session_state['active_city'] = f"{active_targets[0]['city']} & {len(active_targets)-1} others"
                st.session_state['active_state'] = active_targets[0]['state'] 

            prog = st.progress(0, text="Starting simulation... We appreciate your service!")
            all_gdfs = []
            total_estimated_pop = 0

            for i, loc in enumerate(active_targets):
                c_name = loc['city'].strip()
                s_name = loc['state']
                
                prog.progress(10 + int((i/len(active_targets))*20), text=f"📡 Fetching boundary for {c_name}, {s_name}... Equipping your department for safer responses!")
                success, temp_gdf = fetch_tiger_city_shapefile(STATE_FIPS[s_name], c_name, SHAPEFILE_DIR)

                if success:
                    all_gdfs.append(temp_gdf)
                    pop = fetch_census_population(STATE_FIPS[s_name], c_name)
                    if pop:
                        total_estimated_pop += pop
                        st.toast(f"✅ {c_name} population verified: {pop:,}")
                    else:
                        gdf_proj = temp_gdf.to_crs(epsg=3857)
                        area_sq_mi = gdf_proj.geometry.area.sum() / 2589988.11
                        est = KNOWN_POPULATIONS.get(c_name, int(area_sq_mi * 3500))
                        total_estimated_pop += est
                        st.toast(f"⚠️ {c_name} population estimated: {est:,}")
                else:
                    st.warning(f"⚠️ Could not find a boundary for {c_name}, {s_name}. Skipping.")

            if not all_gdfs:
                prog.empty()
                st.error("❌ Could not find Census boundaries for any of the entered locations. Check spelling.")
                st.stop()

            prog.progress(35, text="✅ Boundaries loaded. Working to get your department safer!")
            active_city_gdf = pd.concat(all_gdfs, ignore_index=True)
            city_poly = active_city_gdf.geometry.union_all()
            
            st.session_state['estimated_pop'] = total_estimated_pop

            annual_cfs = int(total_estimated_pop * 0.6)
            st.session_state['total_original_calls'] = annual_cfs
            simulated_points_count = min(int(annual_cfs / 12), 25000)

            prog.progress(55, text="📍 Generating 911 calls... Thank you for your dedication!")
            np.random.seed(42)
            call_points = generate_clustered_calls(city_poly, simulated_points_count)
            st.session_state['df_calls'] = pd.DataFrame({
                'lat': [p[0] for p in call_points],
                'lon': [p[1] for p in call_points],
                'priority': np.random.choice(['High','Medium','Low'], simulated_points_count)
            })

            prog.progress(80, text="🏢 Distributing infrastructure... Building a safer community together!")
            station_points = generate_random_points_in_polygon(city_poly, 100)
            types = ['Police','Fire','EMS'] * 34
            st.session_state['df_stations'] = pd.DataFrame({
                'name': [f'Station {i+1}' for i in range(len(station_points))],
                'lat': [p[0] for p in station_points],
                'lon': [p[1] for p in station_points],
                'type': types[:len(station_points)]
            })

            prog.progress(100, text="✅ Simulation complete! Thank you for your commitment to public safety!")
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

        with st.spinner("🌍 Identifying jurisdictions... We appreciate the sacrifices you make for us!"):
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
        selected_labels = st.sidebar.multiselect("Jurisdictions", options=all_options, default=all_options,
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
            show_boundaries = st.toggle("Jurisdiction Boundaries", value=True, help="Toggle the display of the official city/county borders.")
            show_heatmap = st.toggle("911 Call Heatmap", value=False, help="Overlay a density heatmap of historic 911 incidents.")
            show_health = st.toggle("Health Score", value=False, help="Display the overall department operational health metric.")
            show_satellite = st.toggle("Satellite Imagery", value=False, help="Switch the basemap to high-resolution satellite imagery.")
            show_cards = st.toggle("Unit Economics Cards", value=True, help="Show detailed financial and operational breakdowns for each deployed drone.")
            show_faa = st.toggle("FAA LAANC Airspace", value=False, help="Overlay FAA UAS Facility Map ceiling grids (Procedural estimation for speed).")
            simulate_traffic = st.toggle("Simulate Ground Traffic", value=False, help="Calculate vehicular patrol response times based on traffic congestion.")
            traffic_level = st.slider("Traffic Congestion", 0, 100, 40, help="Adjust ground traffic severity. 0=Empty roads, 100=Gridlock.") if simulate_traffic else 40

        strat_expander = st.sidebar.expander("⚙️ Deployment Strategy", expanded=False)
        with strat_expander:
            incremental_build = st.toggle("Phased Rollout", value=True,
                                          help="Locks previously deployed stations in place as new drones are added — mirrors real procurement phases.")
            allow_redundancy = st.toggle("Allow Coverage Overlap", value=True,
                                         help="Permits rings to overlap in high-density areas. Disable to force maximum geographic spread.")

        st.sidebar.markdown('<div class="sidebar-section-header">② Optimize Fleet</div>', unsafe_allow_html=True)

        opt_strategy_raw = st.sidebar.radio("Optimization Goal", ("Call Coverage", "Land Coverage"), horizontal=True,
                                            help="Prioritize covering the most 911 incidents (Call) or the most land area (Land).")
        opt_strategy = "Maximize Call Coverage" if opt_strategy_raw == "Call Coverage" else "Maximize Land Coverage"

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
            full_boundary_utm = (active_utm.geometry.union_all() if hasattr(active_utm.geometry, 'union_all')
                                 else active_utm.geometry.unary_union).buffer(0.1)
            full_boundary_utm = full_boundary_utm.buffer(-0.1)
            city_m = full_boundary_utm
            city_boundary_geom = gpd.GeoSeries([full_boundary_utm], crs=epsg_code).to_crs(epsg=4326).iloc[0]
        except Exception as e:
            st.error(f"Geometry Error: {e}"); st.stop()

        n = len(df_stations_all)
        bounds_hash = f"{minx}_{miny}_{maxx}_{maxy}_{n}"

        prog2 = st.sidebar.empty()
        prog2.caption("⚡ Precomputing spatial matrices... Enhancing your operational safety!")
        calls_in_city, display_calls, resp_matrix, guard_matrix, dist_matrix_r, dist_matrix_g, station_metadata, total_calls = precompute_spatial_data(
            df_calls, df_stations_all, city_m, epsg_code,
            st.session_state.get('r_resp', 2.0), st.session_state.get('r_guard', 8.0),
            center_lat, center_lon, bounds_hash
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
            return int(df_curve.loc[idx_99 if idx_99 is not None else series.last_valid_index(), 'Drones'])

        max_r = min(max(1, get_max_drones('Responder (Calls)') + 4), n)
        max_g = min(max(1, get_max_drones('Guardian (Calls)') + 4), n)

        k_responder = st.sidebar.slider("🚁 Responder Count", 0, max_r, min(st.session_state.get('k_resp',0), max_r),
                                        help="Short-range tactical drones (2-3mi radius).")
        k_guardian  = st.sidebar.slider("🦅 Guardian Count",  0, max_g, min(st.session_state.get('k_guard',0), max_g),
                                        help="Long-range heavy-lift drones (up to 8mi radius).")
        resp_radius_mi  = st.sidebar.slider("🚁 Responder Range (mi)", 2.0, 3.0, st.session_state.get('r_resp', 2.0), step=0.5)
        guard_radius_mi = st.sidebar.slider("🦅 Guardian Range (mi)", 1, 8, int(st.session_state.get('r_guard', 8)))
        st.session_state.update({'k_resp': k_responder, 'k_guard': k_guardian, 'r_resp': resp_radius_mi, 'r_guard': guard_radius_mi})

        bounds_hash = f"{minx}_{miny}_{maxx}_{maxy}_{n}_{resp_radius_mi}_{guard_radius_mi}"
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

        faa_geojson = load_faa_parquet(minx, miny, maxx, maxy)
        airfields = fetch_airfields(minx, miny, maxx, maxy)

        st.sidebar.markdown('<div class="sidebar-section-header">③ Budget & Export</div>', unsafe_allow_html=True)

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

        export_placeholder = st.sidebar.container()

        # ── OPTIMIZATION ──────────────────────────────────────────────────
        active_resp_names, active_guard_names = [], []
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
                    stage_bar.info("🧠 Running optimizer... Thank you for answering the call to protect and serve! (may take 10-20s)")
                    r_best, g_best, chrono_r, chrono_g = solve_mclp(
                        resp_matrix, guard_matrix, dist_matrix_r, dist_matrix_g,
                        k_responder, k_guardian, allow_redundancy, incremental=incremental_build
                    )
                    best_combo = (tuple(r_best), tuple(g_best))
                    stage_bar.empty()
                    st.toast("✅ Optimization complete!", icon="✅")
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
                    stage_bar.info("🗺️ Optimizing coverage... Helping you stay safe out there!")
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
                    st.toast("✅ Optimization complete!", icon="✅")

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
            else:
                active_resp_names, active_guard_names = [], []

        # ── METRICS ───────────────────────────────────────────────────────
        area_covered_perc = overlap_perc = calls_covered_perc = 0.0
        
        if not 'active_resp_names' in locals(): active_resp_names = []
        if not 'active_guard_names' in locals(): active_guard_names = []
        
        if k_responder > 0 or k_guardian > 0:
            active_resp_names = best_resp_names
            active_guard_names = best_guard_names
            
        active_resp_idx  = [i for i,s in enumerate(station_metadata) if s['name'] in active_resp_names]
        active_guard_idx = [i for i,s in enumerate(station_metadata) if s['name'] in active_guard_names]

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
            key = f"{station_metadata[idx]['name']}_{d_type}"
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
            
            if actual_k_responder > 0:
                st.sidebar.markdown(f"""
                <div style="background-color: {card_bg}; border: 1px solid {card_border}; padding: 10px; border-radius: 4px; margin-bottom: 8px;">
                    <h5 style="color: {text_main}; margin: 0 0 4px 0; font-size: 0.85rem;">RESPONDER <span style="color:{text_muted}; font-weight:normal;">(x{actual_k_responder})</span></h5>
                    <div style="color: {text_muted}; font-size: 0.75rem;">COVERAGE: <span style="color:{text_main}; font-weight:600;">{resp_radius_mi} MI RADIUS</span></div>
                    <div style="color: {text_muted}; font-size: 0.75rem;">UNIT CAPEX: <span style="color:{text_main}; font-weight:600;">${CONFIG["RESPONDER_COST"]:,.0f}</span></div>
                    <div style="color: {text_muted}; font-size: 0.75rem; margin-top: 4px; border-top: 1px solid {card_border}; padding-top: 4px;">SUBTOTAL: <span style="color:{text_main}; font-weight:600;">${capex_responder_total:,.0f}</span></div>
                </div>
                """, unsafe_allow_html=True)
                
            if actual_k_guardian > 0:
                st.sidebar.markdown(f"""
                <div style="background-color: {card_bg}; border: 1px solid {card_border}; padding: 10px; border-radius: 4px; margin-bottom: 8px;">
                    <h5 style="color: {text_main}; margin: 0 0 4px 0; font-size: 0.85rem;">GUARDIAN <span style="color:{text_muted}; font-weight:normal;">(x{actual_k_guardian})</span></h5>
                    <div style="color: {text_muted}; font-size: 0.75rem;">COVERAGE: <span style="color:{text_main}; font-weight:600;">{guard_radius_mi} MI RADIUS</span></div>
                    <div style="color: {text_muted}; font-size: 0.75rem;">UNIT CAPEX: <span style="color:{text_main}; font-weight:600;">${CONFIG["GUARDIAN_COST"]:,.0f}</span></div>
                    <div style="color: {text_muted}; font-size: 0.75rem; margin-top: 4px; border-top: 1px solid {card_border}; padding-top: 4px;">SUBTOTAL: <span style="color:{text_main}; font-weight:600;">${capex_guardian_total:,.0f}</span></div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.sidebar.info("👈 Set Responder/Guardian counts above to calculate budget impact.")

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
            map_color    = active_color_map[f"{station_metadata[idx]['name']}_{d_type}"]
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

        # ── EXPORT BUTTONS ────────────────────────────────────────────────
        if fleet_capex > 0:
            st.sidebar.markdown("---")
            col_n, col_e = st.sidebar.columns(2)
            prop_name  = col_n.text_input("Your Name",  value=st.session_state.get('user_name', 'John Doe'), key='user_name')
            prop_email = col_e.text_input("Your Email", value=st.session_state.get('user_email', 'john.doe@example.com'), key='user_email')
            st.sidebar.caption("*(Press **Enter** after typing to apply changes to your document)*")

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
            logo_html_str = f'<img src="data:image/png;base64,{logo_b64}" style="height:40px;">' if logo_b64 else '<div style="font-size:28px;font-weight:900;letter-spacing:3px;color:#111;">BRINC</div>'
            
            # ── GRANT NARRATIVE VARIABLES ─────────────────────────────────
            jurisdiction_list = ", ".join(selected_names) if selected_names else prop_city
            
            # Safely group stations by checking their names
            police_dept_names = [d['name'] for d in active_drones if 'Police' in str(d.get('name', ''))]
            fire_dept_names   = [d['name'] for d in active_drones if 'Fire' in str(d.get('name', ''))]
            ems_dept_names    = [d['name'] for d in active_drones if 'EMS' in str(d.get('name', ''))]
            
            dept_summary_parts = []
            if police_dept_names: dept_summary_parts.append(f"{len(police_dept_names)} Police station{'s' if len(police_dept_names)>1 else ''}")
            if fire_dept_names:   dept_summary_parts.append(f"{len(fire_dept_names)} Fire station{'s' if len(fire_dept_names)>1 else ''}")
            if ems_dept_names:    dept_summary_parts.append(f"{len(ems_dept_names)} EMS station{'s' if len(ems_dept_names)>1 else ''}")
            dept_summary = ", ".join(dept_summary_parts) if dept_summary_parts else f"{len(active_drones)} municipal stations"
            
            police_names_str = (", ".join([n.replace('[Police] ','') for n in police_dept_names[:6]]) + ("..." if len(police_dept_names)>6 else "")) if police_dept_names else "municipal facilities"
            total_fleet = actual_k_responder + actual_k_guardian
            area_sq_mi_est = int((maxx - minx) * (maxy - miny) * 3280)
            
            export_html = f"""<html><head><title>BRINC DFR Proposal — {prop_city}</title>
            <style>
            body{{font-family:'Helvetica Neue',Arial,sans-serif;color:#333;margin:0;padding:40px;background:#f4f6f9;}}
            .page{{max-width:1000px;margin:0 auto;background:#fff;padding:50px;border-radius:8px;box-shadow:0 4px 15px rgba(0,0,0,0.05);}}
            .header{{display:flex;justify-content:space-between;align-items:flex-end;border-bottom:2px solid #00D2FF;padding-bottom:15px;margin-bottom:30px;}}
            h1{{color:#000;margin:0;font-size:24px;}} h2{{color:#444;margin-top:30px;font-size:18px;border-bottom:1px solid #ddd;padding-bottom:5px;}}
            table{{width:100%;border-collapse:collapse;margin-top:10px;font-size:13px;}}
            th,td{{padding:8px 12px;text-align:left;border-bottom:1px solid #ddd;}}
            th{{background:#f1f1f1;font-size:12px;text-transform:uppercase;color:#555;}}
            .map-container{{border:1px solid #ddd;border-radius:8px;overflow:hidden;margin-top:10px;}}
            .footer{{margin-top:40px;padding-top:20px;border-top:2px solid #eee;text-align:center;font-size:13px;color:#555;line-height:1.6;}}
            .footer a{{color:#00D2FF;text-decoration:none;font-weight:bold;}}
            .kpi-grid{{display:flex;gap:20px;margin-bottom:30px;}}
            .kpi-box{{flex:1;border:1px solid #eaeaea;border-radius:8px;padding:20px;background:#fafafa;}}
            .kpi-box h2{{margin-top:0;}}
            .kpi-val{{font-size:22px;font-weight:bold;color:#00D2FF;}}
            .kpi-lbl{{font-size:11px;font-weight:bold;color:#888;text-transform:uppercase;}}
            .disclaimer{{background:#fff3cd;border-left:4px solid #ffeeba;padding:12px;margin-bottom:16px;font-size:12px;color:#856404;}}
            </style></head><body><div class="page">
            <div class="header"><div>{logo_html_str}</div>
            <div style="text-align:right;"><h1>DFR Deployment Proposal</h1>
            <div style="font-size:14px;color:#666;margin-top:5px;">For: {prop_city}, {prop_state} | Pop: {pop_metric:,}</div>
            <div style="font-size:14px;color:#666;margin-top:3px;">By: {prop_name} | {prop_email}</div></div></div>
            <div class="kpi-grid">
            <div class="kpi-box"><h2>Financial</h2>
              <div class="kpi-lbl">Fleet CapEx</div><div class="kpi-val">${fleet_capex:,.0f}</div>
              <div class="kpi-lbl" style="margin-top:12px;">Annual Savings Capacity</div><div class="kpi-val">${annual_savings:,.0f}</div>
              <div class="kpi-lbl" style="margin-top:12px;">Break-Even</div><div class="kpi-val">{break_even_text}</div>
            </div>
            <div class="kpi-box"><h2>Operational</h2>
              <div class="kpi-lbl">911 Call Coverage</div><div class="kpi-val">{calls_covered_perc:.1f}%</div>
              <div class="kpi-lbl" style="margin-top:12px;">Avg Response Time</div><div class="kpi-val">{avg_resp_time:.1f} min</div>
              <div class="kpi-lbl" style="margin-top:12px;">Time Saved vs Patrol</div><div class="kpi-val">{avg_time_saved:.1f} min</div>
            </div></div>
            <h2>Proposed Fleet</h2>
            <table><tr><th>Type</th><th>Qty</th><th>Range</th><th>Unit Cost</th></tr>
            <tr><td>BRINC Responder</td><td>{actual_k_responder}</td><td>{resp_radius_mi} mi</td><td>${CONFIG['RESPONDER_COST']:,}</td></tr>
            <tr><td>BRINC Guardian</td><td>{actual_k_guardian}</td><td>{guard_radius_mi} mi</td><td>${CONFIG['GUARDIAN_COST']:,}</td></tr></table>
            <h2>Coverage Map</h2>
            <div class="map-container">{map_html_str}</div>
            <h2>Deployment Locations</h2>
            <table><tr><th>Station</th><th>Type</th><th>Avg Response</th><th>FAA Ceiling</th><th>CapEx</th></tr>{station_rows}</table>
            <h2>Grant Narrative (AI Draft)</h2>
            <div class="disclaimer"><strong>DISCLAIMER:</strong> AI-generated draft. Must be reviewed, localized, and fact-checked by your grants administrator before submission. All statistics are model estimates.</div>

            <p><strong>Project Title:</strong> BRINC Drones Drone as a First Responder (DFR) Program — {jurisdiction_list}</p>

            <p><strong>Executive Summary:</strong> The {jurisdiction_list} respectfully submits this application requesting funding to establish a BRINC Drones-powered Drone as a First Responder (DFR) program. This initiative will deploy a fleet of {total_fleet} purpose-built BRINC Drones aerial systems — comprising {actual_k_responder} BRINC Responder and {actual_k_guardian} BRINC Guardian units — across {dept_summary} serving a combined population of {pop_metric:,} residents across approximately {area_sq_mi_est:,} square miles in {prop_city}, {prop_state}.</p>

            <p><strong>Statement of Need:</strong> The {jurisdiction_list} currently serves a population of {pop_metric:,} residents and responds to an estimated {st.session_state.get('total_original_calls', total_calls):,} calls for service annually. Ground-based patrol response times are constrained by traffic, geography, and unit availability. This proposal addresses a critical public safety gap: the need for immediate aerial situational awareness that arrives before ground units, enabling smarter, safer, and faster emergency response. BRINC Drones, the world leader in purpose-built DFR technology, provides the only fully integrated hardware, software, and operational support platform purpose-designed for law enforcement DFR deployment.</p>

            <p><strong>Geographic Scope & Participating Agencies:</strong> The proposed DFR network covers the jurisdictions of <strong>{jurisdiction_list}</strong> ({prop_state}). Drone stations will be hosted at {dept_summary}, including facilities operated by: <em>{police_names_str}</em>. The deployment area encompasses an estimated {area_sq_mi_est:,} square miles of mixed urban and suburban terrain, with BRINC Drones units positioned to achieve {calls_covered_perc:.1f}% coverage of historical incident locations and {area_covered_perc:.1f}% geographic area coverage.</p>

            <p><strong>Program Design:</strong> The proposed fleet consists of {actual_k_responder} <strong>BRINC Responder</strong> units (short-range tactical response, {resp_radius_mi}-mile operational radius) and {actual_k_guardian} <strong>BRINC Guardian</strong> units (long-range heavy-lift, {guard_radius_mi}-mile operational radius). All deployment sites have been pre-screened against FAA LAANC UAS Facility Maps. The BRINC Drones platform provides automated launch-on-dispatch, live-streaming HD/thermal video to dispatch and responding officers, and full chain-of-custody flight logging. Average aerial response time under this configuration is projected at <strong>{avg_resp_time:.1f} minutes</strong> — approximately <strong>{avg_time_saved:.1f} minutes faster</strong> than current vehicular patrol response for equivalent distances.</p>

            <p><strong>Fiscal Impact & Return on Investment:</strong> Total program capital expenditure is <strong>${fleet_capex:,.0f}</strong>. Based on a {int(dfr_dispatch_rate*100)}% DFR dispatch rate and {int(deflection_rate*100)}% call resolution rate, the program is projected to generate <strong>${annual_savings:,.0f} in annual operational savings</strong> through reduced officer dispatch on drone-resolved incidents, reaching full cost recovery in <strong>{break_even_text.lower()}</strong>. At ${CONFIG["DRONE_COST_PER_CALL"]}/drone response versus ${CONFIG["OFFICER_COST_PER_CALL"]}/officer dispatch, the BRINC Drones platform delivers a demonstrated cost-per-response reduction of over {int((1 - CONFIG["DRONE_COST_PER_CALL"]/CONFIG["OFFICER_COST_PER_CALL"])*100)}%.</p>

            <p><strong>About BRINC Drones:</strong> BRINC Drones, Inc. is the global leader in purpose-built Drone as a First Responder technology, with deployments across hundreds of law enforcement agencies in the United States. BRINC Drones designs, manufactures, and supports the only DFR platform built from the ground up for public safety — including the BRINC Responder for rapid tactical response and the BRINC Guardian for extended-range operations. BRINC provides full agency onboarding, FAA coordination support, pilot training, and ongoing operational guidance. Learn more at <a href="https://brincdrones.com" target="_blank">brincdrones.com</a>.</p>

            <p><strong>Potential Grant Funding Sources:</strong>
              <a href="https://bja.ojp.gov/program/jag/overview" target="_blank">DOJ Byrne JAG</a> — UAS and technology procurement eligible  • 
              <a href="https://www.fema.gov/grants/preparedness/homeland-security" target="_blank">FEMA HSGP</a> — CapEx offset for tactical deployments  • 
              <a href="https://cops.usdoj.gov/grants" target="_blank">DOJ COPS Office</a> — Law enforcement technology grants  • 
              <a href="https://www.transportation.gov/grants" target="_blank">DOT RAISE</a> — Regional infrastructure and safety
            </p>
            <div class="footer">
              <div style="font-size:20px;font-weight:900;letter-spacing:2px;color:#111;margin-bottom:4px;">BRINC</div>
              <div style="font-weight:bold;margin-bottom:4px;">BRINC Drones, Inc.</div>
              <div style="margin-bottom:8px;">Leading the world in purpose-built Drone as a First Responder technology.</div>
              <div style="margin-bottom:8px;font-weight:bold;">Prepared by: {prop_name} | <a href="mailto:{prop_email}">{prop_email}</a></div>
              <div style="margin-bottom:8px;">
                <a href="https://brincdrones.com" target="_blank">brincdrones.com</a> | <a href="mailto:sales@brincdrones.com">sales@brincdrones.com</a> | +1 (855) 950-0226
              </div>
              <div>
                <a href="https://www.linkedin.com/company/brincdrones" target="_blank">LinkedIn</a> • 
                <a href="https://twitter.com/brincdrones" target="_blank">Twitter / X</a> • 
                <a href="https://www.youtube.com/c/brincdrones" target="_blank">YouTube</a>
              </div>
            </div></div></body></html>"""

            if st.sidebar.download_button("💾 Save Deployment Plan", data=json.dumps(export_dict),
                                          file_name=f"Brinc_{safe_city}_{current_time_str}.brinc",
                                          mime="application/json", use_container_width=True):
                _notify_email(st.session_state.get('active_city',''), st.session_state.get('active_state',''),
                              "BRINC", k_responder, k_guardian, calls_covered_perc,
                              st.session_state.get('user_name',''), st.session_state.get('user_email',''))
                _log_to_sheets(st.session_state.get('active_city',''), st.session_state.get('active_state',''),
                               "BRINC", k_responder, k_guardian, calls_covered_perc,
                               st.session_state.get('user_name',''), st.session_state.get('user_email',''))

            if st.sidebar.download_button("📄 Executive Summary (HTML)", data=export_html,
                                          file_name=f"Brinc_{safe_city}_Proposal_{current_time_str}.html",
                                          mime="text/html", use_container_width=True):
                _notify_email(st.session_state.get('active_city',''), st.session_state.get('active_state',''),
                              "HTML", k_responder, k_guardian, calls_covered_perc,
                              st.session_state.get('user_name',''), st.session_state.get('user_email',''))
                _log_to_sheets(st.session_state.get('active_city',''), st.session_state.get('active_state',''),
                               "HTML", k_responder, k_guardian, calls_covered_perc,
                               st.session_state.get('user_name',''), st.session_state.get('user_email',''))

            if active_drones:
                if st.sidebar.download_button("🌏 Google Earth Briefing File", data=generate_kml(active_gdf, active_drones, calls_in_city),
                                              file_name="drone_deployment.kml", mime="application/vnd.google-earth.kml+xml",
                                              use_container_width=True):
                    _notify_email(st.session_state.get('active_city',''), st.session_state.get('active_state',''),
                                  "KML", k_responder, k_guardian, calls_covered_perc,
                                  st.session_state.get('user_name',''), st.session_state.get('user_email',''))
                    _log_to_sheets(st.session_state.get('active_city',''), st.session_state.get('active_state',''),
                                   "KML", k_responder, k_guardian, calls_covered_perc,
                                   st.session_state.get('user_name',''), st.session_state.get('user_email',''))

        # Grant eligibility
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

        # ── MAIN CONTENT ──────────────────────────────────────────────────
        st.markdown("---")

        # Health score
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

        # KPI bar
        if simulate_traffic:
            avg_ground_speed = CONFIG["DEFAULT_TRAFFIC_SPEED"] * (1 - traffic_level/100)
            eval_dist = guard_radius_mi if active_guard_names else resp_radius_mi
            eval_speed = CONFIG["GUARDIAN_SPEED"] if active_guard_names else CONFIG["RESPONDER_SPEED"]
            if (active_resp_names or active_guard_names) and avg_ground_speed > 0:
                time_saved = ((eval_dist*1.4/avg_ground_speed) - (eval_dist/eval_speed)) * 60
                gain_val = f"{time_saved:.1f} min"
            else:
                gain_val = "N/A"
        else:
            gain_val = None

        kpi_html = f"""
        <div style="display:flex; justify-content:space-around; background:{card_bg}; border:1px solid {card_border}; border-radius:8px; padding:15px; margin-bottom:15px; flex-wrap:wrap; gap:10px;">
            <div style="text-align:center;"><div style="font-size:0.75rem; color:{text_muted}; text-transform:uppercase;">Total Incidents</div><div style="font-size:1.6rem; font-weight:800; color:{accent_color}; font-family:'IBM Plex Mono', monospace;">{st.session_state.get('total_original_calls',total_calls):,}</div></div>
            <div style="text-align:center;"><div style="font-size:0.75rem; color:{text_muted}; text-transform:uppercase;">Response Capacity</div><div style="font-size:1.6rem; font-weight:800; color:{accent_color}; font-family:'IBM Plex Mono', monospace;">{calls_covered_perc:.1f}%</div></div>
            <div style="text-align:center;"><div style="font-size:0.75rem; color:{text_muted}; text-transform:uppercase;">Land Covered</div><div style="font-size:1.6rem; font-weight:800; color:{accent_color}; font-family:'IBM Plex Mono', monospace;">{area_covered_perc:.1f}%</div></div>
            <div style="text-align:center;"><div style="font-size:0.75rem; color:{text_muted}; text-transform:uppercase;">Overlap</div><div style="font-size:1.6rem; font-weight:800; color:{accent_color}; font-family:'IBM Plex Mono', monospace;">{overlap_perc:.1f}%</div></div>
        """
        if gain_val is not None:
            kpi_html += f"""<div style="text-align:center;"><div style="font-size:0.75rem; color:{text_muted}; text-transform:uppercase;">Time Saved ({eval_dist:.0f}mi)</div><div style="font-size:1.6rem; font-weight:800; color:{accent_color}; font-family:'IBM Plex Mono', monospace;">{gain_val}</div></div>"""
        
        kpi_html += "</div>"
        st.markdown(kpi_html, unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:0.65rem;color:gray;margin-top:-12px;margin-bottom:12px;text-align:center;'>(Optimized via {total_calls:,} representative sample)</div>", unsafe_allow_html=True)

        # ── MAP + STATS COLUMNS ───────────────────────────────────────────
        map_col, stats_col = st.columns([4.2, 1.8])

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

                # ── Guardian 5-mile rapid response focus ring ─────────────
                if d['type'] == 'GUARDIAN' and d['radius_m']/1609.34 > 5.0:
                    f_lats, f_lons = get_circle_coords(d['lat'], d['lon'], r_mi=5.0)
                    fig.add_trace(go.Scattermapbox(
                        lat=list(f_lats), lon=list(f_lons),
                        mode='lines',
                        line=dict(color=d['color'], width=1.5, dash='dot' if False else None),
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

            # --- GRANT PROPOSAL GENERATOR ---
            with export_placeholder:
                st.markdown("---")
                st.markdown(f"<h4 style='margin-top:0px;'>📄 Grant Proposal Generator</h4>", unsafe_allow_html=True)
                st.write("Generate a customized narrative leveraging the data on this dashboard for federal grant applications.")
                
                col_a, col_b = st.columns(2)
                acct_name = col_a.text_input("Account Holder Name", placeholder="Jane Doe")
                acct_email = col_b.text_input("Account Holder Email", placeholder="jane.doe@brinc.com")
                
                if fleet_capex > 0:
                    prop_city = st.session_state.get('active_city', 'City')
                    
                    narrative = f"""PROJECT TITLE: Implementation of BRINC Drones for First Responder (DFR) Operations

PRIMARY CONTACT: {acct_name} | {acct_email}

1. PROJECT SUMMARY
This proposal seeks funding to establish a Drone as a First Responder (DFR) program utilizing advanced BRINC Drones. By deploying {actual_k_responder} BRINC Responder drones and {actual_k_guardian} BRINC Guardian drones across strategic municipal infrastructure, this initiative will dramatically reduce response times, enhance situational awareness, and improve safety for both first responders and the public.

2. GEOGRAPHIC & DEMOGRAPHIC TARGET
The proposed BRINC drone network optimally utilizes {len(active_drones)} municipal facilities as launch nodes in {prop_city}. This configuration achieves a {calls_covered_perc:.1f}% coverage rate of historical 911 calls for service, covering {area_covered_perc:.1f}% of the physical jurisdiction.

3. DATA-DRIVEN IMPACT & GRANT ALIGNMENT
Based on historical incident data ({total_calls:,} total evaluated calls), the BRINC DFR fleet is projected to directly respond to {daily_dfr_responses:.1f} calls per day. 
Crucially, {daily_drone_only_calls:.1f} of these daily calls can be fully deflected—resolved entirely by the BRINC drone's telepresence and sensor suite—eliminating the need to dispatch a physical officer or apparatus. 

This translates to an estimated capacity savings of ${annual_savings:,.0f} annually, strongly aligning with federal grant priorities for operational efficiency, inter-agency operability, and life-safety enhancement.

4. BUDGET & ROI JUSTIFICATION
The total capital expenditure for the BRINC DFR hardware is ${fleet_capex:,.0f}. Given the projected operational deflection rate, the system achieves a full return on investment in {break_even_text}, freeing up critical personnel to focus on high-priority community policing and emergency response.
"""
                    st.download_button(
                        label="📥 Export Grant Proposal",
                        data=narrative,
                        file_name="BRINC_Grant_Proposal.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                else:
                    st.info("Deploy drones using the Optimizer Controls to generate a customized grant proposal.")

        # ── 3D SWARM SIMULATION ───────────────────────────────────────────
        if fleet_capex > 0:
            st.markdown("---")
            st.markdown(f"<h3 style='color:{text_main};'>🚁 3D Swarm Simulation</h3>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:0.82rem; color:{text_muted}; margin-bottom:10px;'>Animated deck.gl simulation of all DFR flights over a compressed 24-hour day. Use the speed slider to accelerate or slow the simulation. Great for council presentations.</div>", unsafe_allow_html=True)

            show_sim = st.toggle("🎬 Enable 3D Simulation", value=False, help="Launch a dynamic 3D rendering of daily drone flights.")
            if show_sim:
                calls_coords = np.column_stack((calls_in_city['lon'], calls_in_city['lat']))
                
                # --- RESTRICTED NEAREST NEIGHBOR DISPATCH SIMULATION ---
                sim_assignments = {i:[] for i in range(len(active_drones))}
                for c_idx, cc in enumerate(calls_coords):
                    best_d, best_dist = -1, float('inf')
                    for d_idx, d in enumerate(active_drones):
                        if d['cov_array'][c_idx]:
                            dist = (cc[0]-d['lon'])**2 + (cc[1]-d['lat'])**2
                            if dist < best_dist:
                                best_dist, best_d = dist, d_idx
                    if best_d != -1:
                        sim_assignments[best_d].append(c_idx)

                stations_json, flights_json, legend_html_sim = [], [], ""
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
                    
                    legend_html_sim += f'<div style="margin-bottom:3px;"><span style="display:inline-block;width:10px;height:10px;background-color:{d["color"]};margin-right:8px;border-radius:50%;"></span>{short_name}</div>'
                    
                    assigned_calls = sim_assignments[d_idx]
                    num_to_simulate = int(len(assigned_calls) * dfr_dispatch_rate)
                    if num_to_simulate > 0:
                        assigned_calls = random.sample(list(assigned_calls), min(num_to_simulate, len(assigned_calls)))
                    else:
                        assigned_calls = []

                    total_sim_flights += len(assigned_calls)

                    for call_idx in assigned_calls:
                        lon1, lat1 = calls_coords[call_idx]
                        lon0, lat0 = d['lon'], d['lat']
                        
                        dist_mi = math.sqrt((lon1-lon0)**2+(lat1-lat0)**2)*69.172
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
                
                sim_html = f"""
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
                        <div style="font-size: 13px; color: #aaa; margin-bottom: 15px;">Simulating {len(flights_json)} flights (approx {int(dfr_dispatch_rate*100)}% dispatch rate) over a 24-hour cycle.</div>
                        <div style="margin-bottom: 15px;">
                            <label style="font-size: 12px; color: #ccc;">Time Speed Multiplier: <span id="speedLabel">1</span>x</label>
                            <input type="range" id="speedSlider" min="1" max="100" value="1" style="width: 100%;">
                        </div>
                        <button id="runBtn">LAUNCH SWARM</button>
                        <div id="timeDisplay" style="font-family: monospace; font-size: 18px; color: #00ffcc; font-weight: bold; text-align: center;">00:00:00</div>
                        <div style="margin-top: 15px; border-top: 1px solid #333; padding-top: 10px;">
                            <h4 style="margin: 0 0 5px 0; color: #aaa; font-size: 11px; text-transform: uppercase;">Active Stations</h4>
                            <div style="font-size: 11px; color: #ddd; max-height: 120px; overflow-y: auto;">
                                {legend_html_sim}
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
                                longitude: {center_lon},
                                latitude: {center_lat},
                                zoom: {dynamic_zoom},
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
                components.html(sim_html, height=700)
