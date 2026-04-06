import streamlit as st
import os, math, urllib.parse

st.set_page_config(
    page_title="BRINC DFR — Mobile Summary",
    page_icon="🚁",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Hide sidebar nav and sidebar entirely on mobile summary page
st.markdown("""
<style>
[data-testid="stSidebarNav"]     { display: none !important; }
[data-testid="stSidebar"]        { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── Read URL params ────────────────────────────────────────────────────────
p        = st.query_params
city     = str(p.get("city",  "Your City")).title()
state    = str(p.get("state", ""))
pop      = int(p.get("pop",   0) or 0)
cov      = float(p.get("cov", 0) or 0)
resp     = float(p.get("resp", 0) or 0)
saves    = int(p.get("saves", 0) or 0)
capex    = int(p.get("capex", 0) or 0)
k_r      = int(p.get("r",     0) or 0)
k_g      = int(p.get("g",     0) or 0)
calls    = int(p.get("calls", 0) or 0)
area     = int(p.get("area",  0) or 0)
tsav     = float(p.get("tsav", 0) or 0)

# Derived
roi_mult  = round(saves / max(capex, 1), 2) if capex > 0 else 0
be_months = round(capex / max(saves / 12, 1)) if saves > 0 else 0
fleet_txt = []
if k_g: fleet_txt.append(f"{k_g} Guardian{'s' if k_g != 1 else ''}")
if k_r: fleet_txt.append(f"{k_r} Responder{'s' if k_r != 1 else ''}")
fleet_str = " + ".join(fleet_txt) if fleet_txt else "No drones deployed"
no_data   = (cov == 0 and saves == 0 and k_r == 0 and k_g == 0)

# ── CSS ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700;900&family=DM+Mono:wght@500&display=swap');

[data-testid="stAppViewContainer"] {
  background: #080c14 !important;
}
[data-testid="stMain"] {
  background: #080c14 !important;
}
[data-testid="block-container"] {
  padding: 1rem 1rem 3rem !important;
  max-width: 480px !important;
  margin: 0 auto !important;
}
[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

.mob-header {
  text-align: center;
  padding: 24px 0 20px;
  border-bottom: 1px solid rgba(0,210,255,0.2);
  margin-bottom: 24px;
}
.mob-logo {
  font-size: 2rem;
  font-weight: 900;
  color: #00D2FF;
  letter-spacing: 3px;
  font-family: 'DM Sans', sans-serif;
  line-height: 1;
}
.mob-logo span { color: #ffffff; }
.mob-city {
  font-size: 1.35rem;
  font-weight: 700;
  color: #f0f0f0;
  margin-top: 8px;
  font-family: 'DM Sans', sans-serif;
}
.mob-tagline {
  font-size: 0.75rem;
  color: #666;
  text-transform: uppercase;
  letter-spacing: 1.5px;
  margin-top: 4px;
}
.mob-fleet {
  display: inline-block;
  background: rgba(0,210,255,0.1);
  border: 1px solid rgba(0,210,255,0.35);
  border-radius: 20px;
  padding: 4px 14px;
  font-size: 0.72rem;
  color: #00D2FF;
  font-weight: 700;
  margin-top: 8px;
  letter-spacing: 0.3px;
}
.kpi-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
  margin-bottom: 14px;
}
.kpi-card {
  background: #111520;
  border: 1px solid #1e2535;
  border-radius: 12px;
  padding: 16px 12px 14px;
  text-align: center;
}
.kpi-card.accent { border-top: 3px solid #00D2FF; }
.kpi-card.green  { border-top: 3px solid #22c55e; }
.kpi-card.gold   { border-top: 3px solid #f59e0b; }
.kpi-card.red    { border-top: 3px solid #ef4444; }
.kpi-val {
  font-size: 1.95rem;
  font-weight: 900;
  color: #f0f0f0;
  font-family: 'DM Mono', monospace;
  line-height: 1.1;
}
.kpi-val.cyan  { color: #00D2FF; }
.kpi-val.green { color: #22c55e; }
.kpi-val.gold  { color: #f59e0b; }
.kpi-val.red   { color: #ef4444; }
.kpi-label {
  font-size: 0.60rem;
  color: #667;
  text-transform: uppercase;
  letter-spacing: 0.8px;
  margin-top: 4px;
  font-family: 'DM Sans', sans-serif;
}
.section-head {
  font-size: 0.65rem;
  color: #00D2FF;
  text-transform: uppercase;
  letter-spacing: 1.8px;
  font-weight: 700;
  margin: 20px 0 10px;
  font-family: 'DM Sans', sans-serif;
  border-left: 3px solid #00D2FF;
  padding-left: 8px;
}
.roi-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 9px 0;
  border-bottom: 1px solid #1e2535;
  font-size: 0.78rem;
  font-family: 'DM Sans', sans-serif;
}
.roi-row:last-child { border-bottom: none; }
.roi-label { color: #8899aa; }
.roi-val   { color: #f0f0f0; font-family: 'DM Mono', monospace; font-weight: 600; font-size: 0.75rem; }
.roi-val.g { color: #22c55e; }
.roi-panel {
  background: #111520;
  border: 1px solid #1e2535;
  border-radius: 12px;
  padding: 14px 16px;
  margin-bottom: 14px;
}
.disclaimer {
  font-size: 0.62rem;
  color: #445;
  line-height: 1.55;
  text-align: center;
  margin-top: 28px;
  padding-top: 14px;
  border-top: 1px solid #1a1f2e;
}
.brinc-footer {
  text-align: center;
  margin-top: 20px;
  padding-top: 16px;
}
.no-data-msg {
  background: rgba(0,210,255,0.05);
  border: 1px dashed rgba(0,210,255,0.3);
  border-radius: 12px;
  padding: 28px 20px;
  text-align: center;
  color: #556;
  font-size: 0.85rem;
  line-height: 1.7;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────
location_str = f"{city}, {state}" if state else city
pop_str = f"{pop:,} residents" if pop > 0 else ""

st.markdown(f"""
<div class="mob-header">
  <div class="mob-logo">BRINC<span> DFR</span></div>
  <div class="mob-city">{location_str}</div>
  {"<div class='mob-tagline'>" + pop_str + "</div>" if pop_str else ""}
  {"<div class='mob-fleet'>" + fleet_str + "</div>" if not no_data else ""}
</div>
""", unsafe_allow_html=True)

# ── No data state ─────────────────────────────────────────────────────────
if no_data:
    st.markdown("""
    <div class="no-data-msg">
      <div style="font-size:2rem;margin-bottom:10px;">🚁</div>
      <strong style="color:#aaa;">Report not yet generated</strong><br>
      Open the full optimizer on a desktop browser, configure your deployment,
      then scan the QR code again to view your personalized summary here.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="brinc-footer">
      <div style="font-size:0.65rem;color:#334;letter-spacing:1.2px;text-transform:uppercase;">
        BRINC Drones, Inc. · brincdrones.com
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Coverage KPIs ─────────────────────────────────────────────────────────
st.markdown('<div class="section-head">Coverage & Response</div>', unsafe_allow_html=True)

resp_display = f"{resp:.1f} min" if resp > 0 else "N/A"
tsav_display = f"{tsav:.1f} min" if tsav > 0 else "N/A"
calls_display = f"{calls:,}" if calls > 0 else "N/A"
area_display = f"{area:,} mi²" if area > 0 else "N/A"

st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-card accent">
    <div class="kpi-val cyan">{cov:.1f}%</div>
    <div class="kpi-label">Call Coverage</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-val">{resp_display}</div>
    <div class="kpi-label">Avg Aerial Response</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-val">{calls_display}</div>
    <div class="kpi-label">Annual Calls</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-val">{area_display}</div>
    <div class="kpi-label">Coverage Area</div>
  </div>
</div>
""", unsafe_allow_html=True)

if tsav > 0:
    st.markdown(f"""
    <div class="roi-panel" style="margin-bottom:6px;">
      <div style="font-size:0.68rem;color:#556;text-align:center;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.8px;">Time Saved Per Incident</div>
      <div style="text-align:center;font-size:2.2rem;font-weight:900;color:#00D2FF;font-family:'DM Mono',monospace;">{tsav_display}</div>
      <div style="text-align:center;font-size:0.65rem;color:#556;margin-top:4px;">faster than traditional ground response</div>
    </div>
    """, unsafe_allow_html=True)

# ── Financial KPIs ────────────────────────────────────────────────────────
st.markdown('<div class="section-head">Financial Impact</div>', unsafe_allow_html=True)

roi_display = f"{roi_mult:.1f}×" if roi_mult > 0 else "N/A"
be_display  = f"{be_months} mo" if be_months > 0 else "N/A"

st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-card green">
    <div class="kpi-val green">${saves:,.0f}</div>
    <div class="kpi-label">Annual Savings</div>
  </div>
  <div class="kpi-card gold">
    <div class="kpi-val gold">{roi_display}</div>
    <div class="kpi-label">ROI Multiple</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-val">${capex:,.0f}</div>
    <div class="kpi-label">Fleet CapEx</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-val">{be_display}</div>
    <div class="kpi-label">Break-Even</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── ROI Detail ────────────────────────────────────────────────────────────
st.markdown('<div class="section-head">Deployment Summary</div>', unsafe_allow_html=True)

monthly_saves = saves // 12 if saves > 0 else 0

rows = []
if k_g > 0:
    rows.append(("Guardian Drones", str(k_g), ""))
if k_r > 0:
    rows.append(("Responder Drones", str(k_r), ""))
rows.append(("Annual Operational Savings", f"${saves:,.0f}", "g"))
rows.append(("Monthly Savings", f"${monthly_saves:,.0f}", "g"))
rows.append(("Total Fleet CapEx", f"${capex:,.0f}", ""))
if be_months > 0:
    rows.append(("Break-Even Timeline", f"{be_months} months", ""))
if roi_mult > 0:
    rows.append(("Return on Investment", f"{roi_mult:.2f}× annually", "g"))

roi_html = '<div class="roi-panel">'
for label, val, color_cls in rows:
    val_class = f'roi-val {color_cls}'.strip()
    roi_html += f'<div class="roi-row"><span class="roi-label">{label}</span><span class="{val_class}">{val}</span></div>'
roi_html += '</div>'

st.markdown(roi_html, unsafe_allow_html=True)

# ── Disclaimer + Footer ───────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
  All figures are model estimates based on deployment parameters, national DFR benchmark rates,
  and CAD data. Response times, ROI, and outcomes are projections — not guarantees.
  Actual results depend on staffing, policy, FAA authorization, and operational execution.
</div>
<div class="brinc-footer">
  <div style="font-size:0.75rem;font-weight:900;color:#00D2FF;letter-spacing:3px;">BRINC</div>
  <div style="font-size:0.60rem;color:#334;letter-spacing:1px;text-transform:uppercase;margin-top:2px;">
    Drone as First Responder · brincdrones.com
  </div>
</div>
""", unsafe_allow_html=True)
