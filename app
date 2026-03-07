# Save this as: brinc_drone_optimizer.py

import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import plotly.graph_objects as go
from shapely.geometry import Point, Polygon, MultiPolygon
import shapely.wkt
from shapely.ops import unary_union
import os
import itertools
import glob
import math
import simplekml
from concurrent.futures import ThreadPoolExecutor

st.set_page_config(page_title="brinc COS Drone Optimizer", layout="wide")

SHAPEFILE_DIR = "jurisdiction_data"
os.makedirs(SHAPEFILE_DIR, exist_ok=True)

STATION_COLORS = [
"#E6194B","#3CB44B","#4363D8","#F58231","#911EB4",
"#800000","#333333","#000075","#808000","#9A6324"
]

def get_circle_coords(lat, lon, r_mi=2.0):
angles = np.linspace(0,2*np.pi,100)
c_lats = lat + (r_mi/69.172)*np.sin(angles)
c_lons = lon + (r_mi/(69.172*np.cos(np.radians(lat))))*np.cos(angles)
return c_lats,c_lons

def generate_kml(active_gdf,df_stations_all,resp_names,guard_names,calls_gdf):

```
kml = simplekml.Kml()

fol_bounds = kml.newfolder(name="Jurisdictions")

for _,row in active_gdf.iterrows():

    geom = row.geometry

    if isinstance(geom,Polygon):
        geoms=[geom]

    elif isinstance(geom,MultiPolygon):
        geoms=list(geom.geoms)

    else:
        continue

    for g in geoms:

        pol=fol_bounds.newpolygon(name=row.get('DISPLAY_NAME','Boundary'))
        pol.outerboundaryis=list(g.exterior.coords)

        pol.style.linestyle.color=simplekml.Color.red
        pol.style.linestyle.width=3
        pol.style.polystyle.color=simplekml.Color.changealphaint(30,simplekml.Color.red)

fol_stations=kml.newfolder(name="Stations")
fol_rings=kml.newfolder(name="Coverage")

def add_station(row,radius,color,label):

    p=fol_stations.newpoint(name=f"{label} {row['name']}")
    p.coords=[(row['lon'],row['lat'])]

    lats,lons=get_circle_coords(row['lat'],row['lon'],radius)
    ring=list(zip(lons,lats))
    ring.append(ring[0])

    poly=fol_rings.newpolygon(name=row['name'])
    poly.outerboundaryis=ring
    poly.style.linestyle.color=color
    poly.style.polystyle.color=simplekml.Color.changealphaint(60,color)

for _,row in df_stations_all[df_stations_all.name.isin(resp_names)].iterrows():
    add_station(row,2.0,simplekml.Color.blue,"Responder")

for _,row in df_stations_all[df_stations_all.name.isin(guard_names)].iterrows():
    add_station(row,8.0,simplekml.Color.orange,"Guardian")

return kml.kml()
```

@st.cache_resource
def precompute_spatial_data(df_calls,df_stations_all,city_wkt,epsg):

```
city=shapely.wkt.loads(city_wkt)

gdf_calls=gpd.GeoDataFrame(
    df_calls,
    geometry=gpd.points_from_xy(df_calls.lon,df_calls.lat),
    crs="EPSG:4326"
)

gdf_calls=gdf_calls.to_crs(epsg)

calls_in_city=gdf_calls[gdf_calls.within(city)]

total_calls=len(calls_in_city)

radius_resp=3218.69
radius_guard=12874.75

stations=gpd.GeoDataFrame(
    df_stations_all,
    geometry=gpd.points_from_xy(df_stations_all.lon,df_stations_all.lat),
    crs="EPSG:4326"
).to_crs(epsg)

calls_xy=np.array(list(zip(calls_in_city.geometry.x,calls_in_city.geometry.y)))
stations_xy=np.array(list(zip(stations.geometry.x,stations.geometry.y)))

dx=calls_xy[:,0][:,None]-stations_xy[:,0]
dy=calls_xy[:,1][:,None]-stations_xy[:,1]

dists=np.sqrt(dx**2+dy**2)

resp_matrix=(dists<=radius_resp).T
guard_matrix=(dists<=radius_guard).T

station_metadata=[]

for i,row in stations.iterrows():

    pt=row.geometry

    buf2=pt.buffer(radius_resp)
    buf8=pt.buffer(radius_guard)

    try:
        buf2=buf2.intersection(city)
        buf8=buf8.intersection(city)
    except:
        pass

    station_metadata.append({
        "name":row["name"],
        "lat":df_stations_all.iloc[i].lat,
        "lon":df_stations_all.iloc[i].lon,
        "clipped_2m":buf2,
        "clipped_8m":buf8
    })

display_calls=calls_in_city.sample(min(5000,total_calls)).to_crs(4326)

return calls_in_city,display_calls,resp_matrix,guard_matrix,station_metadata,total_calls
```

def compute_overlap(active_geos,city):

```
if not active_geos:
    return 0

union=unary_union(active_geos)

sum_area=sum(g.area for g in active_geos)

overlap=sum_area-union.area

return (overlap/city.area)*100
```

st.title("🛰️ BRINC COS Drone Optimizer")

uploaded=st.file_uploader("Upload calls.csv and stations.csv",accept_multiple_files=True)

if uploaded:

```
call_file=None
station_file=None

for f in uploaded:
    if f.name.lower()=="calls.csv":
        call_file=f
    if f.name.lower()=="stations.csv":
        station_file=f

if call_file and station_file:

    df_calls=pd.read_csv(call_file)
    df_stations=pd.read_csv(station_file)

    if not {"lat","lon"}.issubset(df_calls.columns):
        st.error("calls.csv must contain lat, lon")
        st.stop()

    if not {"lat","lon","name"}.issubset(df_stations.columns):
        st.error("stations.csv must contain name, lat, lon")
        st.stop()

    st.success("Files loaded successfully.")
```
