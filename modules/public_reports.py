"""Public report path and link helpers."""

from __future__ import annotations

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
    return "http://localhost:8501"


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
        _query = urllib.parse.urlencode({"report_id": report_id, "sig": _sig})
        _sep = "&" if "?" in _public_webapp_url else "?"
        return f"{_public_webapp_url}{_sep}{_query}"
    return f"{_get_request_base_url()}/?public_report={report_id}&sig={_sig}"


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
    _html_path = _public_report_html_path(report_id)
    _html_path.write_text(str(html_text or ""), encoding="utf-8")
    if metadata is not None:
        _public_report_metadata_path(report_id).write_text(
            json.dumps(metadata, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
    return _html_path
