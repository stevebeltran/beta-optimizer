# BRINC Drone Deployment Optimizer V2

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import plotly.graph_objects as go
import simplekml
import pulp

st.set_page_config(page_title="BRINC Drone Optimizer V2", layout="wide")

RESP_METERS = 3218.69
GUARD_METERS = 12874.75

def get_circle_coords(lat, lon, r_mi=2):

```
angles = np.linspace(0, 2*np.pi, 120)

c_lats = lat + (r_mi/69.172) * np.sin(angles)

c_lons = lon + (r_mi/(69.172*np.cos(np.radians(lat)))) * np.cos(angles)

return c_lats, c_lons
```

def generate_kml(stations, responders, guardians):

```
kml = simplekml.Kml()

fol_stations = kml.newfolder(name="Stations")
fol_coverage = kml.newfolder(name="Coverage")

for i, row in stations.iterrows():

    p = fol_stations.newpoint(name=row["name"])
    p.coords = [(row["lon"], row["lat"])]

    if i in responders:
        radius = 2
        color = simplekml.Color.blue
    elif i in guardians:
        radius = 8
        color = simplekml.Color.orange
    else:
        continue

    lats, lons = get_circle_coords(row["lat"], row["lon"], radius)
    ring = list(zip(lons, lats))
    ring.append(ring[0])

    poly = fol_coverage.newpolygon(name=row["name"])
    poly.outerboundaryis = ring
    poly.style.linestyle.color = color
    poly.style.polystyle.color = simplekml.Color.changealphaint(60, color)

return kml.kml()
```

@st.cache_resource
def precompute(df_calls, df_stations):

```
calls = gpd.GeoDataFrame(
    df_calls,
    geometry=gpd.points_from_xy(df_calls.lon, df_calls.lat),
    crs="EPSG:4326"
).to_crs(3857)

stations = gpd.GeoDataFrame(
    df_stations,
    geometry=gpd.points_from_xy(df_stations.lon, df_stations.lat),
    crs="EPSG:4326"
).to_crs(3857)

calls_xy = np.vstack([calls.geometry.x, calls.geometry.y]).T
stations_xy = np.vstack([stations.geometry.x, stations.geometry.y]).T

dx = calls_xy[:, 0][:, None] - stations_xy[:, 0]
dy = calls_xy[:, 1][:, None] - stations_xy[:, 1]

dist = np.sqrt(dx**2 + dy**2)

resp_matrix = (dist <= RESP_METERS).T
guard_matrix = (dist <= GUARD_METERS).T

return resp_matrix, guard_matrix
```

def solve_mclp(resp_matrix, guard_matrix, num_resp, num_guard):

```
n_stations = resp_matrix.shape[0]
n_calls = resp_matrix.shape[1]

model = pulp.LpProblem("DroneCoverage", pulp.LpMaximize)

x = pulp.LpVariable.dicts("station", range(n_stations), 0, 1, pulp.LpBinary)
y = pulp.LpVariable.dicts("call", range(n_calls), 0, 1, pulp.LpBinary)

model += pulp.lpSum(y[i] for i in range(n_calls))

model += pulp.lpSum(x[i] for i in range(n_stations)) <= (num_resp + num_guard)

for call in range(n_calls):

    cover = []

    for s in range(n_stations):

        if resp_matrix[s][call] or guard_matrix[s][call]:
            cover.append(x[s])

    if cover:
        model += y[call] <= pulp.lpSum(cover)

model.solve(pulp.PULP_CBC_CMD(msg=0))

selected = [i for i in range(n_stations) if pulp.value(x[i]) == 1]

responders = selected[:num_resp]
guardians = selected[num_resp:num_resp + num_guard]

return responders, guardians
```

def render_map(df_calls, df_stations, responders, guardians):

```
fig = go.Figure()

fig.add_trace(go.Scattermapbox(
    lat=df_calls.lat,
    lon=df_calls.lon,
    mode="markers",
    marker=dict(size=4),
    name="Calls"
))

for i, row in df_stations.iterrows():

    if i in responders:
        color = "blue"
        radius = 2
    elif i in guardians:
        color = "orange"
        radius = 8
    else:
        continue

    fig.add_trace(go.Scattermapbox(
        lat=[row.lat],
        lon=[row.lon],
        mode="markers",
        marker=dict(size=12, color=color),
        name=row.name
    ))

    lats, lons = get_circle_coords(row.lat, row.lon, radius)

    fig.add_trace(go.Scattermapbox(
        lat=lats,
        lon=lons,
        mode="lines",
        line=dict(width=2, color=color),
        showlegend=False
    ))

fig.update_layout(
    mapbox_style="open-street-map",
    mapbox_zoom=10,
    mapbox_center=dict(lat=df_calls.lat.mean(), lon=df_calls.lon.mean()),
    height=700
)

return fig
```

st.title("Drone Deployment Optimizer V2")

calls_file = st.file_uploader("Upload calls.csv")
stations_file = st.file_uploader("Upload stations.csv")

if calls_file and stations_file:

```
df_calls = pd.read_csv(calls_file)
df_stations = pd.read_csv(stations_file)

st.success("Files loaded")

num_resp = st.slider("Responder Drones", 1, 20, 3)
num_guard = st.slider("Guardian Drones", 1, 20, 2)

resp_matrix, guard_matrix = precompute(df_calls, df_stations)

if st.button("Run Optimizer"):

    responders, guardians = solve_mclp(
        resp_matrix,
        guard_matrix,
        num_resp,
        num_guard
    )

    st.write("Responder Stations:", responders)
    st.write("Guardian Stations:", guardians)

    fig = render_map(df_calls, df_stations, responders, guardians)
    st.plotly_chart(fig, use_container_width=True)

    kml = generate_kml(df_stations, responders, guardians)

    st.download_button(
        "Download Google Earth KML",
        kml,
        "drone_coverage.kml"
    )
```
