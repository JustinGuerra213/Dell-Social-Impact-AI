"""Streamlit interface for local flood impact triage."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from PIL import Image

from flood_app.analyzer import LocalQwenAnalyzer
from flood_app.dataset import DatasetRepository, filter_pairs_by_locality
from flood_app.models import DamageAssessment, ImagePair
from flood_app.research import AreaResearch, AreaResearchService


APP_ROOT = Path(__file__).parent
DATA_CACHE = Path(os.getenv("FLOOD_DATA_CACHE", APP_ROOT / "data" / "hf-cache"))


@st.cache_resource(show_spinner="Connecting to the local UTSA image metadata…")
def dataset_repository() -> DatasetRepository:
    """Create one dataset repository per Streamlit process."""
    return DatasetRepository(DATA_CACHE)


@st.cache_data(show_spinner=False)
def available_pairs() -> list[ImagePair]:
    """Cache pair metadata after the first Hub download."""
    return dataset_repository().list_pairs()


@st.cache_resource
def qwen_analyzer(temperature: float, max_tokens: int) -> LocalQwenAnalyzer:
    """Create the local Qwen API client."""
    return LocalQwenAnalyzer(
        base_url=os.getenv("QWEN_BASE_URL", "http://127.0.0.1:8000/v1"),
        model=os.getenv("QWEN_MODEL", "Qwen/Qwen3-VL-8B-Instruct"),
        temperature=temperature,
        max_tokens=max_tokens,
    )


@st.cache_data(show_spinner="Loading selected dataset images…")
def load_dataset_image(image_id: str, image_path: str) -> Image.Image:
    """Load a dataset image and make a Streamlit-safe cached copy."""
    from flood_app.models import ImageRecord

    record = ImageRecord(
        image_id=image_id,
        image_path=image_path,
        pair_id="",
        paired_image_id="",
        image_role="",
        event_name="",
        acquisition_date="",
        locality="",
        source_agency="",
        media_type="",
        attribution="",
    )
    return dataset_repository().load_image(record)


def main() -> None:
    """Render the app and handle a single comparison workflow."""
    st.set_page_config(
        page_title="Saving Texas · Flood impact triage",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("🌊 Saving Texas")
    st.caption("Before-and-after flood imagery · Local Qwen vision analysis")
    st.info(
        "Triage support only. Image analysis cannot confirm structural safety, "
        "power status, drainage function, human injury, or actual repair costs. "
        "Verify hazards on the ground with qualified responders.",
        icon="⚠️",
    )

    with st.sidebar:
        st.header("Model settings")
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05,
            help="Lower values make responses more consistent; higher values allow more variation.",
        )
        max_tokens = st.number_input(
            "Maximum response tokens",
            min_value=256,
            max_value=4096,
            value=2000,
            step=256,
            help="Limits the length of the model response. Larger values can take longer.",
        )
        st.header("Choose imagery")
        source = st.radio(
            "Image source",
            ["UTSA dataset pair", "Upload my own pair"],
            help="Dataset images are fetched only when selected.",
        )
        pair: ImagePair | None = None
        before_image: Image.Image | None = None
        after_image: Image.Image | None = None

        if source == "UTSA dataset pair":
            try:
                pairs = available_pairs()
                if not pairs:
                    st.error("No linked before/response pairs were found in the dataset metadata.")
                else:
                    localities = sorted(
                        {item.before.locality for item in pairs if item.before.locality}
                    )
                    city_area = st.selectbox(
                        "City / area",
                        ["All cities and areas", *localities],
                        help="Uses the dataset's locality labels; some are neighborhoods, rivers, or regions rather than incorporated cities.",
                    )
                    filtered_pairs = filter_pairs_by_locality(
                        pairs,
                        None if city_area == "All cities and areas" else city_area,
                    )
                    pair = st.selectbox(
                        "Location / event",
                        filtered_pairs,
                        format_func=lambda item: item.label,
                    )
                    before_image = load_dataset_image(pair.before.image_id, pair.before.image_path)
                    after_image = load_dataset_image(pair.after.image_id, pair.after.image_path)
                    st.caption(
                        f"Before: {pair.before.acquisition_date or 'date unavailable'} · "
                        f"After: {pair.after.acquisition_date or 'date unavailable'}"
                    )
            except Exception as exc:  # Network and dataset access errors need a clear UI message.
                st.error(f"Could not load the dataset: {exc}")
        else:
            before_file = st.file_uploader(
                "Before image", type=["jpg", "jpeg", "png", "webp"], key="before"
            )
            after_file = st.file_uploader(
                "After image", type=["jpg", "jpeg", "png", "webp"], key="after"
            )
            if before_file and after_file:
                before_image = Image.open(before_file).convert("RGB")
                after_image = Image.open(after_file).convert("RGB")

        st.divider()
        st.caption("Model endpoint: " + os.getenv("QWEN_BASE_URL", "http://127.0.0.1:8000/v1"))
        analyze = st.button(
            "Analyze changes",
            type="primary",
            use_container_width=True,
            disabled=before_image is None or after_image is None,
        )

    if before_image is None or after_image is None:
        st.subheader("Compare a location across time")
        st.write(
            "Choose a linked image pair from the UTSA flood dataset or upload your own. "
            "The app sends both images to the configured local Qwen model."
        )
        st.markdown("**The report covers** power infrastructure · road hazards · drainage · landscape · soil erosion · cost range")
        st.markdown("**Personal injury:** aerial and scene images cannot reliably identify or assess injured people.")
        return

    before_column, after_column = st.columns(2)
    with before_column:
        st.subheader("Before")
        st.image(before_image, use_container_width=True)
    with after_column:
        st.subheader("After / response")
        st.image(after_image, use_container_width=True)

    if pair:
        details = " · ".join(
            value for value in [pair.before.locality, pair.before.event_name, pair.before.source_agency] if value
        )
        if details:
            st.caption(f"Dataset context: {details}")

    if analyze:
        with st.spinner("Qwen is comparing both images locally…"):
            try:
                assessment = qwen_analyzer(temperature, int(max_tokens)).analyze(
                    before_image, after_image
                )
                st.session_state["assessment"] = assessment
                st.session_state["assessment_pair"] = pair.label if pair else "Uploaded images"
                st.session_state["model_settings"] = {
                    "temperature": temperature,
                    "max_tokens": int(max_tokens),
                }
                if pair:
                    with st.spinner("Checking official alerts and area history…"):
                        st.session_state["area_research"] = AreaResearchService().research(pair)
                else:
                    st.session_state.pop("area_research", None)
            except Exception as exc:
                st.error(
                    "Analysis failed. Confirm that the local Qwen OpenAI-compatible "
                    f"server is running and the model name is correct. Details: {exc}"
                )

    assessment = st.session_state.get("assessment")
    if assessment:
        render_assessment(
            assessment,
            st.session_state.get("assessment_pair", ""),
            st.session_state.get("area_research"),
            st.session_state.get("model_settings", {}),
        )


def render_assessment(
    assessment: DamageAssessment,
    pair_label: str,
    research: AreaResearch | None = None,
    model_settings: dict | None = None,
) -> None:
    """Render model output as readable impact categories."""
    st.divider()
    st.header("Impact triage report")
    if pair_label:
        st.caption(pair_label)
    st.write(assessment.summary)
    if model_settings:
        st.caption(
            f"Qwen settings: temperature {model_settings['temperature']:.2f} · "
            f"maximum {model_settings['max_tokens']} tokens"
        )

    citizen_tab, city_tab, response_tab, sources_tab = st.tabs(
        ["Citizen impact", "City impact", "Emergency response", "Sources & history"]
    )
    with citizen_tab:
        render_impact_section(
            "Citizen impact",
            assessment.citizen_impact,
            ["housing_water_impact", "vehicles_visible", "utility_damage_visible"],
        )
        st.markdown(
            "Housing and vehicle repair, insurance claims, mold remediation, relocation, "
            "contamination, and utility outage costs require verified claims, inspections, "
            "or incident reports. Aerial imagery alone cannot establish these costs."
        )
        st.markdown(
            "[FEMA NFIP redacted claims data](https://www.fema.gov/openfema-data-page/fima-nfip-redacted-claims-v2) · "
            "[FEMA/CDC mold cleanup guide](https://www.fema.gov/sites/default/files/documents/fema_hm-cdc-homeowners-and-renters-mold-guide_english.pdf) · "
            "[FEMA disaster assistance](https://www.disasterassistance.gov/)"
        )
        render_cost_categories(
            assessment.cost_estimates_usd,
            {
                "housing_and_insurance": "Housing and insurance",
                "vehicles": "Vehicles",
                "mold_cleanup": "Mold cleanup",
                "relocation_and_shelter": "Relocation and shelter",
                "contamination": "Contamination response",
                "utility_and_energy": "Household utilities",
            },
        )
        if research and research.historical_disasters:
            st.markdown("**Historical county disaster declarations**")
            render_findings(research.historical_disasters)
    with city_tab:
        render_impact_section(
            "City impact",
            assessment.city_impact,
            ["infrastructure_visible", "drainage_or_erosion_visible"],
        )
        st.markdown(
            "The total cost is not defensible without measured quantities and current local unit "
            "prices. Utility/energy loss, road repairs, levee or dam restructuring, drainage "
            "work, landscape restoration, and reinsurance are kept unpriced unless a cited "
            "source provides a location-specific amount."
        )
        st.markdown(
            "[HCFCD drainage network](https://www.hcfcd.org/About/Flooding-and-Floodplains/Drainage-Network) · "
            "[FEMA Public Assistance project data](https://www.fema.gov/openfema-data-page/public-assistance-funded-projects-details-v2)"
        )
        render_cost_categories(
            assessment.cost_estimates_usd,
            {
                "utility_and_energy": "Utilities and energy",
                "roads_and_public_infrastructure": "Roads and public infrastructure",
                "erosion_and_landscape": "Erosion and landscape",
                "dams_and_levees": "Dams and levees",
                "drainage": "Drainage",
                "reinsurance": "Reinsurance",
                "total": "Final total",
            },
        )
        if research and research.regional_reports:
            render_findings(research.regional_reports)
    with response_tab:
        visible_hazards = [
            ("Power infrastructure", assessment.power_grid_damage),
            ("Roads and access", assessment.public_road_hazards),
            ("Drainage", assessment.drainage_status),
            ("Erosion / unstable ground", assessment.soil_erosion),
        ]
        shown = False
        for label, item in visible_hazards:
            status = str(item.get("status", "unknown")).lower()
            if status == "observed":
                st.warning(f"Possible image-based hazard — {label}: {item.get('evidence', 'Inspect on site.')}")
                shown = True
        if not shown:
            st.caption("No specific emergency hazard was confidently identified in the image report. Absence of a visual finding does not establish safety.")
        st.markdown(
            "Use current official warnings and local evacuation instructions. Do not route around "
            "floodwater using this image report; road closures and water depth change quickly. "
            "Call 911 for an immediate life-threatening emergency."
        )
        if research and research.active_alerts:
            st.markdown("**Active NWS alerts at the image coordinates**")
            render_findings(research.active_alerts)
        else:
            st.info("No active NWS alerts were returned for this point at the time checked. This does not establish that conditions are safe.")
        if research and research.historical_disasters:
            past_fire = [item for item in research.historical_disasters if "fire" in item.title.lower()]
            if past_fire:
                st.markdown("**Past county fire declarations (historical context, not a current incident)**")
                render_findings(tuple(past_fire))
        st.markdown(
            "**Evacuation planning:** check the linked county evacuation map; identify an "
            "accessible shelter and an out-of-area contact; plan for mobility, medical, and "
            "vehicle needs; monitor road closures; never drive or walk through floodwater. "
            "Hazardous structures, chemical releases, fires, collapses, landslides, and drowning "
            "risks are listed only when an official report or visible evidence supports them."
        )
        st.markdown("[Ready Harris evacuation information](https://www.readyharris.org/Stay-Safe/Evacuation) · [NWS active alerts](https://www.weather.gov/alerts) · [Report a Texas spill to TCEQ](https://www.tceq.texas.gov/response/spills)")
    with sources_tab:
        if research:
            st.caption(f"Area: {research.county} · Checked: {research.checked_at}")
            if research.errors:
                for error in research.errors:
                    st.warning(error)
            if research.historical_disasters:
                st.markdown("**FEMA county disaster history**")
                render_findings(research.historical_disasters)
            if research.regional_reports:
                st.markdown("**Historical regional reports**")
                render_findings(research.regional_reports)
        else:
            st.info("Area-specific external research is available for UTSA dataset pairs with location metadata.")

    cost = assessment.estimated_cost_usd
    low, high = cost.get("low"), cost.get("high")
    if isinstance(low, (int, float)) and isinstance(high, (int, float)):
        st.metric("Illustrative total cost range", f"${low:,.0f}–${high:,.0f} USD")
    else:
        st.metric("Estimated financial cost", "Insufficient evidence")
    if cost.get("basis"):
        st.caption(f"Cost basis: {cost['basis']}")
    assumptions = cost.get("assumptions", [])
    if assumptions:
        st.caption("Assumptions: " + "; ".join(map(str, assumptions)))

    st.subheader("Observed changes")
    categories = [
        ("Power grid", assessment.power_grid_damage),
        ("Public roads", assessment.public_road_hazards),
        ("Drainage", assessment.drainage_status),
        ("Landscape", assessment.landscape_damage),
        ("Soil erosion", assessment.soil_erosion),
        ("Personal injury", assessment.personal_injury),
    ]
    columns = st.columns(3)
    for index, (title, finding) in enumerate(categories):
        with columns[index % len(columns)]:
            status = str(finding.get("status", "unknown")).replace("_", " ").title()
            severity = finding.get("severity")
            st.markdown(f"**{title}** · {status}")
            if severity:
                st.caption(f"Severity: {severity}")
            st.write(finding.get("evidence", "No visual evidence description provided."))

    if assessment.limitations:
        st.subheader("Limitations")
        for limitation in assessment.limitations:
            st.markdown(f"- {limitation}")
    st.caption(f"Model confidence: {assessment.confidence}. This is not an engineering inspection or official loss estimate.")
    st.download_button(
        "Download report as JSON",
        data=__import__("json").dumps(assessment.__dict__, indent=2),
        file_name="saving-texas-assessment.json",
        mime="application/json",
    )
    st.caption("Dataset: UTSA Flood Images, CC BY-NC 4.0. Use is subject to the dataset’s non-commercial license.")


def render_impact_section(title: str, impact: dict, keys: list[str]) -> None:
    """Render image-based citizen or city observations from Qwen."""
    st.markdown(f"**{title} · image evidence**")
    for key in keys:
        label = key.replace("_", " ").title()
        st.markdown(f"**{label}:** {impact.get(key) or 'Not established from these images.'}")
    limitations = impact.get("limitations", [])
    if limitations:
        st.caption("Limitations: " + "; ".join(map(str, limitations)))


def render_findings(findings: tuple) -> None:
    """Render external research with direct source links and dates."""
    for finding in findings:
        st.markdown(f"**[{finding.title}]({finding.url})**")
        st.write(finding.detail)
        suffix = " · ".join(value for value in [finding.source_name, finding.observed_at] if value)
        if suffix:
            st.caption(suffix)


def render_cost_categories(costs: dict, categories: dict[str, str]) -> None:
    """Show sourced cost ranges while keeping unknown amounts explicit."""
    st.markdown("**Cost assessment · USD**")
    rows = []
    for key, label in categories.items():
        item = costs.get(key, {})
        low, high = item.get("low"), item.get("high")
        if isinstance(low, (int, float)) and isinstance(high, (int, float)):
            amount = f"${low:,.0f}–${high:,.0f}"
        else:
            amount = "Not established"
        basis = item.get("basis", "")
        rows.append({"Category": label, "Range": amount, "Basis": basis})
    st.dataframe(rows, hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
