"""Public report path and link helpers."""

from __future__ import annotations

import datetime
import hashlib
import hmac
import json
import os
import re
import tempfile
import urllib.parse
from pathlib import Path

import streamlit as st


APP_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PUBLIC_REPORT_WEBAPP_URL = (
    "https://script.google.com/macros/s/"
    "AKfycbxrj2C6as_yDDSdIbUF17m9CDEaABlYAAGNeTY6EnnaCjxB2caoJtMqC44aLUNQV-lY/exec"
)
DEFAULT_PUBLIC_REPORT_RETENTION_HOURS = 0.5
_PUBLIC_REPORT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _resolve_public_reports_dir():
    for _candidate in (APP_DIR / "public_reports", Path(tempfile.gettempdir()) / "frankenstein_public_reports"):
        try:
            _candidate.mkdir(parents=True, exist_ok=True)
            if _candidate.is_dir():
                return _candidate
        except OSError:
            continue
    raise OSError("Unable to initialize a writable public reports directory.")


PUBLIC_REPORTS_DIR = _resolve_public_reports_dir()


def _get_public_report_retention_hours():
    try:
        _raw = str(st.secrets.get("PUBLIC_REPORT_RETENTION_HOURS", "") or "").strip()
    except Exception:
        _raw = ""
    if not _raw:
        _raw = str(os.environ.get("PUBLIC_REPORT_RETENTION_HOURS", "")).strip()
    try:
        _hours = float(_raw) if _raw else DEFAULT_PUBLIC_REPORT_RETENTION_HOURS
    except Exception:
        _hours = DEFAULT_PUBLIC_REPORT_RETENTION_HOURS
    return max(0.0, _hours)


def _parse_public_report_timestamp(value):
    raw = str(value or "").strip()
    if not raw:
        return None
    candidates = [raw, raw.replace("Z", "+00:00")]
    for candidate in candidates:
        try:
            parsed = datetime.datetime.fromisoformat(candidate)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=datetime.timezone.utc)
            else:
                parsed = parsed.astimezone(datetime.timezone.utc)
            return parsed
        except Exception:
            continue
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            parsed = datetime.datetime.strptime(raw, fmt).replace(tzinfo=datetime.timezone.utc)
            return parsed
        except Exception:
            continue
    return None


def _cleanup_public_reports(max_age_hours=None):
    """Delete stale generated public report artifacts."""
    if max_age_hours is None:
        age_hours = _get_public_report_retention_hours()
    else:
        try:
            age_hours = max(0.0, float(max_age_hours))
        except Exception:
            age_hours = _get_public_report_retention_hours()
    if age_hours <= 0:
        return 0

    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=age_hours)
    removed = 0
    seen_stems = set()
    candidates = list(PUBLIC_REPORTS_DIR.glob("*.json")) + list(PUBLIC_REPORTS_DIR.glob("*.html"))

    for path in candidates:
        stem = path.stem
        if stem in seen_stems:
            continue
        seen_stems.add(stem)

        try:
            _validate_public_report_id(stem)
        except ValueError:
            continue

        json_path = _public_report_metadata_path(stem)
        html_path = _public_report_html_path(stem)
        timestamp = None

        try:
            if json_path.exists():
                metadata = json.loads(json_path.read_text(encoding="utf-8"))
                if isinstance(metadata, dict):
                    timestamp = _parse_public_report_timestamp(
                        metadata.get("updated_at")
                        or metadata.get("created_at")
                        or metadata.get("session_start")
                    )
        except Exception:
            timestamp = None

        if timestamp is None:
            try:
                timestamp = datetime.datetime.fromtimestamp(path.stat().st_mtime, tz=datetime.timezone.utc)
            except Exception:
                continue

        if timestamp >= cutoff:
            continue

        for stale_path in (html_path, json_path):
            try:
                if stale_path.exists():
                    stale_path.unlink()
                    removed += 1
            except Exception:
                pass

    return removed


def _get_query_params_dict():
    try:
        return {str(k): str(v) for k, v in dict(st.query_params).items()}
    except Exception:
        return {}


def _slugify(value):
    return re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-") or "report"


def _validate_public_report_id(report_id):
    value = str(report_id or "").strip()
    if not _PUBLIC_REPORT_ID_RE.fullmatch(value):
        raise ValueError("Invalid public report ID.")
    return value


def _get_document_jurisdiction_name(session_state, selected_names=None, fallback="City"):
    _active_name = str(session_state.get("active_city", fallback) or fallback).strip() or fallback
    _boundary_kind = str(session_state.get("boundary_kind", "") or "").strip().lower()
    _use_county_boundary = bool(session_state.get("use_county_boundary", False))
    _selected = [str(name).strip() for name in (selected_names or []) if str(name).strip()]

    if _use_county_boundary or _boundary_kind == "county":
        if _selected:
            return ", ".join(_selected)
        return _active_name

    return _active_name


def _get_request_base_url():
    try:
        _host = st.context.headers.get("host", "") or st.context.headers.get("Host", "")
        if _host:
            _proto = "https" if ("streamlit.app" in _host or "share" in _host) else "http"
            return f"{_proto}://{_host}"
    except Exception:
        pass
    _port = str(
        st.get_option("server.port")
        or os.environ.get("STREAMLIT_SERVER_PORT")
        or os.environ.get("PORT")
        or ""
    ).strip()
    if _port:
        return f"http://localhost:{_port}"
    return "http://localhost:8501"


def _is_local_or_private_base_url(base_url):
    try:
        parsed = urllib.parse.urlsplit(str(base_url or "").strip())
        host = str(parsed.hostname or "").strip().lower()
    except Exception:
        host = ""
    if not host:
        return True
    if host in {"localhost", "127.0.0.1", "::1"}:
        return True
    if host.endswith(".local"):
        return True
    if host.startswith("10.") or host.startswith("192.168."):
        return True
    if host.startswith("172."):
        try:
            second = int(host.split(".", 2)[1])
            if 16 <= second <= 31:
                return True
        except Exception:
            pass
    return False


def _get_public_report_secret():
    try:
        if "PUBLIC_REPORT_SECRET" in st.secrets:
            return str(st.secrets["PUBLIC_REPORT_SECRET"])
        if "auth" in st.secrets and isinstance(st.secrets["auth"], dict):
            _cookie_secret = st.secrets["auth"].get("cookie_secret")
            if _cookie_secret:
                return str(_cookie_secret)
    except Exception:
        pass
    env_secret = os.environ.get("PUBLIC_REPORT_SECRET", "")
    if env_secret:
        return str(env_secret)

    # Local/dev fallback: stable for this installation, but not a universal
    # hardcoded signing secret shared by every checkout of the app.
    fallback_material = f"{APP_DIR.resolve()}:{os.environ.get('USERNAME') or os.environ.get('USER') or 'local'}"
    return hashlib.sha256(fallback_material.encode("utf-8")).hexdigest()


def _sign_public_report_id(report_id):
    safe_report_id = _validate_public_report_id(report_id)
    return hmac.new(
        _get_public_report_secret().encode("utf-8"),
        safe_report_id.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _build_public_report_url(report_id):
    _sig = _sign_public_report_id(report_id)
    try:
        _public_webapp_url = str(st.secrets.get("PUBLIC_REPORT_WEBAPP_URL", "")).strip()
    except Exception:
        _public_webapp_url = ""
    if _public_webapp_url:
        _city = str(st.session_state.get("active_city", "") or "").strip()
        _state = str(st.session_state.get("active_state", "") or "").strip()
        _dept = str(st.session_state.get("active_dept_name", "") or "").strip()
        _rep_name = str(st.session_state.get("google_user_name", "") or "").strip()
        _rep_email = str(st.session_state.get("google_user_email", "") or "").strip()
        _brinc_user = str(st.session_state.get("brinc_user", "") or "").strip()
        _summary_text = str(st.session_state.get("qr_summary_text", "") or "").strip()
        _fleet_summary = str(st.session_state.get("qr_fleet_summary", "") or "").strip()
        _annual_savings = str(st.session_state.get("qr_annual_savings", "") or "").strip()
        _call_coverage = str(st.session_state.get("qr_call_coverage", "") or "").strip()
        _avg_response = str(st.session_state.get("qr_avg_response", "") or "").strip()
        _avg_time_saved = str(st.session_state.get("qr_avg_time_saved", "") or "").strip()
        _covered_calls = str(st.session_state.get("qr_covered_calls", "") or "").strip()
        _station_count = str(st.session_state.get("qr_station_count", "") or "").strip()
        _stations_json = str(st.session_state.get("qr_stations_json", "") or "").strip()
        _query = urllib.parse.urlencode({
            "report_id": report_id,
            "public_report": report_id,
            "sig": _sig,
            "city": _city,
            "state": _state,
            "department": _dept,
            "rep_name": _rep_name,
            "rep_email": _rep_email,
            "brinc_user": _brinc_user,
            "session_id": str(st.session_state.get("session_id", "") or "").strip(),
            "session_start": str(st.session_state.get("session_start", "") or "").strip(),
            "summary_text": _summary_text,
            "fleet_summary": _fleet_summary,
            "annual_savings": _annual_savings,
            "call_coverage": _call_coverage,
            "avg_response": _avg_response,
            "avg_time_saved": _avg_time_saved,
            "covered_calls": _covered_calls,
            "station_count": _station_count,
            "stations_json": _stations_json,
        })
        _sep = "&" if "?" in _public_webapp_url else "?"
        return f"{_public_webapp_url}{_sep}{_query}"


def _public_report_html_path(report_id):
    safe_report_id = _validate_public_report_id(report_id)
    root = PUBLIC_REPORTS_DIR.resolve()
    path = (root / f"{safe_report_id}.html").resolve()
    if not path.is_relative_to(root):
        raise ValueError("Invalid public report path.")
    return path


def _public_report_metadata_path(report_id):
    safe_report_id = _validate_public_report_id(report_id)
    root = PUBLIC_REPORTS_DIR.resolve()
    path = (root / f"{safe_report_id}.json").resolve()
    if not path.is_relative_to(root):
        raise ValueError("Invalid public report metadata path.")
    return path


def _publish_public_report_html(report_id, html_text, metadata=None):
    _cleanup_public_reports()
    _html_path = _public_report_html_path(report_id)
    _html_path.write_text(str(html_text or ""), encoding="utf-8")
    if metadata is not None:
        _public_report_metadata_path(report_id).write_text(
            json.dumps(metadata, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
    return _html_path
