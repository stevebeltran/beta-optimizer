"""Helpers for loading Metra route geometry."""

from typing import Optional

import geopandas as gpd
import requests
from shapely.geometry import LineString
from shapely.ops import linemerge, unary_union

METRA_ROUTE_OPTIONS = [
    ("BNSF", "BNSF"),
    ("Heritage Corridor", "HC"),
    ("Metra Electric", "ME"),
    ("Milwaukee District North", "MD-N"),
    ("Milwaukee District West", "MD-W"),
    ("North Central Service", "NCS"),
    ("Rock Island", "RI"),
    ("SouthWest Service", "SWS"),
    ("Union Pacific North", "UP-N"),
    ("Union Pacific Northwest", "UP-NW"),
    ("Union Pacific West", "UP-W"),
]

METRA_ROUTE_LABELS = [f"Metra: {name}" for name, _ in METRA_ROUTE_OPTIONS]
_METRA_LABEL_TO_CODE = {f"Metra: {name}": code for name, code in METRA_ROUTE_OPTIONS}
_METRA_CODE_TO_LABEL = {code: f"Metra: {name}" for name, code in METRA_ROUTE_OPTIONS}

_OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]


def is_metra_route(route_name: str) -> bool:
    return str(route_name or "").strip().startswith("Metra:")


def metra_label_from_code(route_code: str) -> str:
    route_code = str(route_code or "").strip().upper()
    return _METRA_CODE_TO_LABEL.get(route_code, f"Metra: {route_code}")


def metra_code_from_label(route_label: str) -> Optional[str]:
    route_label = str(route_label or "").strip()
    if route_label in _METRA_LABEL_TO_CODE:
        return _METRA_LABEL_TO_CODE[route_label]
    code = route_label.upper()
    if code in _METRA_CODE_TO_LABEL:
        return code
    return None


def _route_ref_pattern(route_code: str) -> str:
    code = str(route_code or "").strip().upper()
    if not code:
        return ""
    return rf"(^|;\s*){code}(\s*;|$)"


def _query_overpass(route_code: str):
    ref_pattern = _route_ref_pattern(route_code)
    if not ref_pattern:
        return None

    query = (
        '[out:json][timeout:60];\n'
        '(\n'
        f'  relation["route"="train"]["operator"="Metra"]["ref"~"{ref_pattern}"](41.2,-88.6,42.6,-87.2);\n'
        f'  relation["route"="train"]["operator"="Metra"]["name"~"{route_code}",i](41.2,-88.6,42.6,-87.2);\n'
        ');\n'
        'out body;\n'
        '>;\n'
        'out geom qt;\n'
    )

    headers = {"User-Agent": "BRINC-Frankenstein/1.0"}
    for url in _OVERPASS_URLS:
        try:
            resp = requests.post(url, data={"data": query}, headers=headers, timeout=90)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            continue
    return None


def _geometry_from_overpass_elements(elements: list[dict]):
    lines = []
    for el in elements:
        if el.get("type") != "way":
            continue
        geom = el.get("geometry") or []
        coords = [(pt["lon"], pt["lat"]) for pt in geom if "lon" in pt and "lat" in pt]
        if len(coords) >= 2:
            lines.append(LineString(coords))

    if not lines:
        return None

    merged = linemerge(unary_union(lines))
    return merged


def fetch_metra_geometry(route_label: str):
    """Fetch Metra route geometry from OpenStreetMap route relations."""
    route_code = metra_code_from_label(route_label)
    if not route_code:
        return None

    try:
        payload = _query_overpass(route_code)
        if not payload:
            return None

        elements = payload.get("elements", [])
        if not elements:
            return None

        geom = _geometry_from_overpass_elements(elements)
        if geom is None:
            return None

        return gpd.GeoDataFrame(
            {"route_label": [metra_label_from_code(route_code)]},
            geometry=[geom],
            crs="EPSG:4326",
        )
    except Exception:
        return None
