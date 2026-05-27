"""
Lightweight compliance guardrails for internal and customer-facing exports.
"""

EXPORT_DISCLAIMER = (
    "This executive summary may include sample data, simulated data, and model "
    "estimates based on user inputs and publicly available data. No guarantees "
    "or warranties are made or implied. Not a legal recommendation, binding "
    "proposal, contract, or guarantee. Deployments require FAA authorization "
    "and formal procurement."
)

DATA_CLASS_OPTIONS = (
    "Simulated / public data only",
    "Internal business data",
    "Sensitive personal or incident data",
    "CJIS / criminal-justice data",
    "Regulated data (HIPAA / financial / other)",
)

DEFAULTS = {
    "compliance_data_classification": DATA_CLASS_OPTIONS[1],
    "compliance_authorization_ack": False,
    "compliance_retention_ack": False,
    "compliance_external_review_ack": False,
    "compliance_accessibility_ack": False,
    "compliance_export_controls_ack": False,
    "compliance_supply_chain_ack": False,
}


def init_compliance_state(session_state) -> None:
    """Install defaults for the sidebar compliance checklist."""
    for key, value in DEFAULTS.items():
        if key not in session_state:
            session_state[key] = value


def get_export_disclaimer_text() -> str:
    return EXPORT_DISCLAIMER


def _is_sensitive_classification(label: str) -> bool:
    return label in {
        "Sensitive personal or incident data",
        "CJIS / criminal-justice data",
        "Regulated data (HIPAA / financial / other)",
    }


def render_compliance_sidebar(st, *, city: str = "", state: str = "") -> dict:
    """
    Render a compact sidebar checklist and return the current completion state.
    """
    with st.sidebar.expander("Compliance & release checks", expanded=False):
        st.caption(
            "Use this before sharing exports or proposal material outside the company."
        )
        data_classification = st.selectbox(
            "Data classification",
            DATA_CLASS_OPTIONS,
            key="compliance_data_classification",
            help="Choose the highest sensitivity class that applies to the current run.",
        )

        if data_classification == "CJIS / criminal-justice data":
            st.error(
                "CJIS data requires the strictest handling. Confirm approved access, logging, retention, and distribution rules before export."
            )
        elif _is_sensitive_classification(data_classification):
            st.warning(
                "Sensitive or regulated data detected. Confirm authorization, retention, and disclosure rules before exporting."
            )
        else:
            st.info(
                "This run is marked as lower sensitivity, but customer-facing outputs still need human review."
            )

        st.checkbox(
            "I am authorized to use this data and I have minimized what is uploaded, logged, and exported.",
            key="compliance_authorization_ack",
        )
        st.checkbox(
            "I reviewed retention and deletion expectations and will not keep unnecessary copies or downloads.",
            key="compliance_retention_ack",
        )
        st.checkbox(
            "I will review the executive summary or other customer-facing output before sharing it externally.",
            key="compliance_external_review_ack",
        )
        st.checkbox(
            "I checked that the export is readable and accessible enough for the intended audience.",
            key="compliance_accessibility_ack",
        )
        st.checkbox(
            "I reviewed export-control, encryption, and cross-border access implications if they apply.",
            key="compliance_export_controls_ack",
        )
        st.checkbox(
            "I confirmed dependency, open-source, and license review is complete for this release.",
            key="compliance_supply_chain_ack",
        )

        ready = all(
            bool(st.session_state.get(key, False))
            for key in (
                "compliance_authorization_ack",
                "compliance_retention_ack",
                "compliance_external_review_ack",
                "compliance_accessibility_ack",
                "compliance_export_controls_ack",
                "compliance_supply_chain_ack",
            )
        )

        if ready:
            st.success("Compliance checks complete for export.")
        else:
            st.warning("Complete the checklist above before sharing customer-facing exports.")

        st.caption(
            "These checks do not replace legal review. They are a release gate for ordinary internal use."
        )
        if city or state:
            st.caption(f"Current jurisdiction context: {city}{', ' if city and state else ''}{state}")

    return {
        "ready": ready,
        "data_classification": data_classification,
        "sensitive": _is_sensitive_classification(data_classification),
    }
