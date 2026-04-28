"""Coverage optimization and elbow curve analysis."""
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import heapq
from concurrent.futures import ThreadPoolExecutor
import pulp
import streamlit as st
from modules.config import CONFIG
from modules.geospatial import build_display_calls

def mean_covered_distance_miles(dist_matrix, cover_matrix, station_idx, fallback_miles=0.0):
    """Return the mean covered distance for one station from the live coverage matrices."""
    try:
        idx = int(station_idx)
        if dist_matrix is None or cover_matrix is None:
            return float(fallback_miles or 0.0)
        if idx < 0 or idx >= len(cover_matrix) or idx >= len(dist_matrix):
            return float(fallback_miles or 0.0)
        mask = np.asarray(cover_matrix[idx], dtype=bool)
        if mask.size == 0 or not mask.any():
            return float(fallback_miles or 0.0)
        distances = np.asarray(dist_matrix[idx], dtype=float)[mask]
        if distances.size == 0:
            return float(fallback_miles or 0.0)
        finite = distances[np.isfinite(distances)]
        if finite.size == 0:
            return float(fallback_miles or 0.0)
        return float(finite.mean())
    except Exception:
        return float(fallback_miles or 0.0)


def bounded_station_avg_distance_miles(
    dist_matrix,
    cover_matrix,
    station_idx,
    fallback_miles=0.0,
    max_radius_miles=None,
):
    """Return a stable average covered distance for one station.

    In normal operation, the mean covered distance cannot exceed the station's
    coverage radius. If the live matrix/index path drifts and produces an
    impossible value, fall back to the precomputed station average instead of
    collapsing station capacity to zero.
    """
    fallback = float(fallback_miles or 0.0)
    radius = float(max_radius_miles or 0.0)
    avg_miles = mean_covered_distance_miles(
        dist_matrix,
        cover_matrix,
        station_idx,
        fallback_miles=fallback,
    )
    if not np.isfinite(avg_miles) or avg_miles < 0:
        avg_miles = fallback
    if radius > 0:
        if not np.isfinite(fallback) or fallback < 0:
            fallback = min(max(avg_miles, 0.0), radius)
        if avg_miles > (radius + 1e-6):
            avg_miles = min(max(fallback, 0.0), radius)
        else:
            avg_miles = min(avg_miles, radius)
    return float(max(avg_miles, 0.0))

@st.cache_resource
def precompute_spatial_data(df_calls, df_calls_full, df_stations_all, _city_m, epsg_code, resp_radius_mi, guard_radius_mi, center_lat, center_lon, bounds_hash):
    gdf_calls = gpd.GeoDataFrame(df_calls, geometry=gpd.points_from_xy(df_calls.lon, df_calls.lat), crs="EPSG:4326")
    gdf_calls_utm = gdf_calls.to_crs(epsg=int(epsg_code))
    try:
        # Use same 300 m buffer as build_display_calls so coverage denominator
        # matches the call dots shown on the map (avoids false 100% when fringe
        # calls outside the exact boundary are visible but not counted).
        _clip_geom = _city_m.buffer(300) if _city_m is not None else _city_m
        calls_in_city = gdf_calls_utm[gdf_calls_utm.within(_clip_geom)]
    except Exception:
        calls_in_city = gdf_calls_utm
    radius_resp_m = resp_radius_mi * 1609.34
    radius_guard_m = guard_radius_mi * 1609.34
    station_metadata = []
    total_calls = len(calls_in_city)
    n = len(df_stations_all)
    resp_matrix = np.zeros((n, total_calls), dtype=bool)
    guard_matrix = np.zeros((n, total_calls), dtype=bool)
    dist_matrix_r = np.zeros((n, total_calls))
    dist_matrix_g = np.zeros((n, total_calls))
    display_calls = build_display_calls(df_calls_full if df_calls_full is not None else df_calls, _city_m, epsg_code, max_points=300000, bounds_hash=bounds_hash)
    max_dist = max(
        (((row['lon'] - center_lon) ** 2 + (row['lat'] - center_lat) ** 2) ** 0.5 for _, row in df_stations_all.iterrows()),
        default=1.0
    ) or 1.0
    calls_array = None
    calls_lat_arr = None
    calls_lon_arr = None
    if total_calls > 0:
        calls_array = np.array(list(zip(calls_in_city.geometry.x, calls_in_city.geometry.y)))
        calls_lat_arr = calls_in_city['lat'].values
        calls_lon_arr = calls_in_city['lon'].values

    def _haversine_mi(slat, slon, lats, lons):
        R = 3958.8
        phi1 = np.radians(slat)
        dphi = np.radians(lats - slat)
        dlam = np.radians(lons - slon)
        a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(np.radians(lats)) * np.sin(dlam / 2) ** 2
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    for idx_pos, (_, row) in enumerate(df_stations_all.iterrows()):
        s_pt_m = gpd.GeoSeries([Point(row['lon'], row['lat'])], crs="EPSG:4326").to_crs(epsg=int(epsg_code)).iloc[0]

        if calls_array is not None:
            dists = np.sqrt((calls_array[:, 0] - s_pt_m.x) ** 2 + (calls_array[:, 1] - s_pt_m.y) ** 2)
            dists_mi = dists / 1609.34
            mask_r = dists <= radius_resp_m
            mask_g = dists <= radius_guard_m
            resp_matrix[idx_pos, :] = mask_r
            guard_matrix[idx_pos, :] = mask_g
            dist_matrix_r[idx_pos, :] = dists_mi
            dist_matrix_g[idx_pos, :] = dists_mi
            avg_dist_r = dists_mi[mask_r].mean() if mask_r.any() else resp_radius_mi * (2 / 3)
            avg_dist_g = dists_mi[mask_g].mean() if mask_g.any() else guard_radius_mi * (2 / 3)
            # Haversine-based raw counts for accurate per-unit display (UTM Euclidean
            # overshoots at large radii and can saturate to the full city total).
            hav_dists = _haversine_mi(row['lat'], row['lon'], calls_lat_arr, calls_lon_arr)
            raw_calls_r = int(np.sum(hav_dists <= resp_radius_mi))
            raw_calls_g = int(np.sum(hav_dists <= guard_radius_mi))
        else:
            avg_dist_r = resp_radius_mi * (2 / 3)
            avg_dist_g = guard_radius_mi * (2 / 3)
            raw_calls_r = 0
            raw_calls_g = 0

        full_buf_2m = s_pt_m.buffer(radius_resp_m)
        try:
            clipped_2m = full_buf_2m.intersection(_city_m)
        except Exception:
            clipped_2m = full_buf_2m
        full_buf_guard = s_pt_m.buffer(radius_guard_m)
        try:
            clipped_guard = full_buf_guard.intersection(_city_m)
        except Exception:
            clipped_guard = full_buf_guard
        dist_c = ((row['lon'] - center_lon) ** 2 + (row['lat'] - center_lat) ** 2) ** 0.5
        station_metadata.append({
            'name': row['name'], 'lat': row['lat'], 'lon': row['lon'],
            'clipped_2m': clipped_2m, 'clipped_guard': clipped_guard,
            'avg_dist_r': avg_dist_r,
            'avg_dist_g': avg_dist_g,
            'centrality': 1.0 - (dist_c / max_dist),
            'raw_calls_r': raw_calls_r,
            'raw_calls_g': raw_calls_g,
        })

    return calls_in_city, display_calls, resp_matrix, guard_matrix, dist_matrix_r, dist_matrix_g, station_metadata, total_calls

def solve_mclp(
    resp_matrix,
    guard_matrix,
    dist_r,
    dist_g,
    num_resp,
    num_guard,
    allow_redundancy,
    incremental=True,
    forced_r=None,
    forced_g=None,
    forbidden_r=None,
    forbidden_g=None,
    incompatible_rr=None,
    incompatible_gg=None,
    incompatible_rg=None,
):
    """MCLP optimizer.

    forced_r / forced_g are station indices that must be included.
    forbidden_r / forbidden_g are station indices that cannot be selected.
    incompatible_* hold station-index pairs that may not be active together.
    """
    forced_r = list(forced_r or [])
    forced_g = list(forced_g or [])
    forbidden_r = set(forbidden_r or []) - set(forced_r)
    forbidden_g = set(forbidden_g or []) - set(forced_g)
    incompatible_rr = list(incompatible_rr or [])
    incompatible_gg = list(incompatible_gg or [])
    incompatible_rg = list(incompatible_rg or [])
    forced_r_set = set(forced_r)
    forced_g_set = set(forced_g)
    n_stations, n_calls = resp_matrix.shape
    if n_calls == 0 or (num_resp == 0 and num_guard == 0):
        return [], [], [], []

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
        for r in locked_r:
            model += x_r[r] == 1
        for g in locked_g:
            model += x_g[g] == 1
        for r in forbidden_r:
            model += x_r[r] == 0
        for g in forbidden_g:
            model += x_g[g] == 0
        if not allow_redundancy:
            for s in range(n_stations):
                model += x_r[s] + x_g[s] <= 1
        for a, b in incompatible_rr:
            if a in forced_r_set and b in forced_r_set:
                continue
            model += x_r[a] + x_r[b] <= 1
        for a, b in incompatible_gg:
            if a in forced_g_set and b in forced_g_set:
                continue
            model += x_g[a] + x_g[b] <= 1
        for r_idx, g_idx in incompatible_rg:
            if r_idx in forced_r_set and g_idx in forced_g_set:
                continue
            model += x_r[r_idx] + x_g[g_idx] <= 1
        y = pulp.LpVariable.dicts("cl", range(n_u), 0, 1, pulp.LpBinary)
        penalty = 0.00001
        model += pulp.lpSum(y[i] * weights[i] for i in range(n_u)) - pulp.lpSum(
            x_r[s] * np.sum(u_dist_r[s, :]) * penalty + x_g[s] * np.sum(u_dist_g[s, :]) * penalty
            for s in range(n_stations)
        )
        for i in range(n_u):
            cover = [x_r[s] for s in range(n_stations) if u_resp[s, i]] + [x_g[s] for s in range(n_stations) if u_guard[s, i]]
            if cover:
                model += y[i] <= pulp.lpSum(cover)
            else:
                model += y[i] == 0
        model.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=10, gapRel=0.0))
        if pulp.LpStatus.get(model.status) not in {"Optimal", "Feasible"}:
            return list(locked_r), list(locked_g)
        return (
            [i for i in range(n_stations) if (pulp.value(x_r[i]) or 0) > 0.5],
            [i for i in range(n_stations) if (pulp.value(x_g[i]) or 0) > 0.5],
        )

    if not incremental:
        res_r, res_g = run_lp(num_resp, num_guard, forced_r, forced_g)
        return res_r, res_g, res_r, res_g

    curr_r, curr_g = list(forced_r), list(forced_g)
    chrono_r, chrono_g = list(forced_r), list(forced_g)
    for tg in range(len(forced_g) + 1, num_guard + 1):
        next_r, next_g = run_lp(0, tg, curr_r, curr_g)
        chrono_g.extend([x for x in next_g if x not in curr_g])
        curr_r, curr_g = next_r, next_g
    for tr in range(len(forced_r) + 1, num_resp + 1):
        next_r, next_g = run_lp(tr, num_guard, curr_r, curr_g)
        chrono_r.extend([x for x in next_r if x not in curr_r])
        curr_r, curr_g = next_r, next_g
    return curr_r, curr_g, chrono_r, chrono_g

@st.cache_resource
def compute_all_elbow_curves(n_calls, _resp_matrix, _guard_matrix, _geos_r, _geos_g, total_area, _bounds_hash, max_stations=100):
    n_st_calls = min(_resp_matrix.shape[0], max_stations)
    n_st_area_r = min(len(_geos_r), 25)
    n_st_area_g = min(len(_geos_g), 25)

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
            if best_s != -1 and best_cov > 0:
                uncovered = uncovered & ~matrix[best_s]
                cov_count += best_cov
                curve.append((cov_count / max(1, n_calls)) * 100)
                if cov_count == n_calls: break
            else:
                break
        return curve

    def greedy_area(geos, limit):
        if total_area <= 0 or limit <= 0: return [0.0]
        current_union = Polygon()
        curve = [0.0]
        import heapq as hq
        geos_sub = geos[:limit]
        
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
        f_ar = executor.submit(greedy_area, _geos_r, n_st_area_r)
        f_ag = executor.submit(greedy_area, _geos_g, n_st_area_g)
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

