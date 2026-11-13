"""Streamlit interface for image-based flood impact triage."""

from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st
from PIL import Image

from flood_app.analyzer import LocalQwenAnalyzer
from flood_app.dataset import DatasetRepository, filter_pairs_by_locality
from flood_app.models import DamageAssessment, ImagePair


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
    """Create a Qwen client configured with the selected generation settings."""
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
        "Image-based triage only. Visual analysis cannot confirm structural safety, "
        "utility service, or human injury.",
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
            help="Limits response length. Larger values can take longer.",
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
                        help="Uses dataset locality labels; some are neighborhoods, rivers, or regions rather than incorporated cities.",
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
            except Exception as exc:
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
            "Choose a linked UTSA image pair or upload your own. The report uses only "
            "visible evidence from the two images."
        )
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
            st.session_state.get("model_settings", {}),
        )


def render_assessment(
    assessment: DamageAssessment,
    pair_label: str,
    model_settings: dict | None = None,
) -> None:
    """Render the requested impact, observation, and limitation tabs."""
    st.divider()
    st.header("Triage report")
    if pair_label:
        st.caption(pair_label)
    with st.container(border=True):
        st.write(assessment.summary)
    if model_settings:
        st.caption(
            f"Qwen settings: temperature {model_settings['temperature']:.2f} · "
            f"maximum {model_settings['max_tokens']} tokens"
        )

    impact_tab, observed_tab, limitations_tab = st.tabs(
        ["Impact", "Observed changes", "Limitations"]
    )
    with impact_tab:
        st.markdown("#### Citizen impact")
        render_impact_section(
            assessment.citizen_impact,
            ["housing_water_impact", "vehicles_visible", "utility_damage_visible"],
        )
        st.markdown("#### City impact")
        render_impact_section(
            assessment.city_impact,
            ["infrastructure_visible", "drainage_or_erosion_visible"],
        )
    with observed_tab:
        categories = [
            ("Power grid", assessment.power_grid_damage),
            ("Public roads", assessment.public_road_hazards),
            ("Drainage", assessment.drainage_status),
            ("Landscape", assessment.landscape_damage),
            ("Soil erosion", assessment.soil_erosion),
            ("Personal injury", assessment.personal_injury),
        ]
        for title, finding in categories:
            with st.container(border=True):
                status = str(finding.get("status", "unknown")).replace("_", " ").title()
                st.markdown(f"**{title} · {status}**")
                if finding.get("severity"):
                    st.caption(f"Severity: {finding['severity']}")
                st.write(finding.get("evidence", "Not established from these images."))
    with limitations_tab:
        limitations = [item[:180] for item in assessment.limitations[:3]]
        if limitations:
            for limitation in limitations:
                with st.container(border=True):
                    st.write(limitation)
        else:
            with st.container(border=True):
                st.write(
                    "Image resolution, view angle, and occlusion may hide damage. "
                    "The photos cannot verify conditions inside structures or below floodwater."
                )

    st.download_button(
        "Download report as JSON",
        data=json.dumps(assessment.__dict__, indent=2),
        file_name="saving-texas-assessment.json",
        mime="application/json",
    )
    st.caption("Dataset: UTSA Flood Images, CC BY-NC 4.0.")


def render_impact_section(impact: dict, keys: list[str]) -> None:
    """Render image-based citizen or city observations from Qwen."""
    for key in keys:
        label = key.replace("_", " ").title()
        with st.container(border=True):
            st.markdown(f"**{label}**")
            st.write(impact.get(key) or "Not established from these images.")


if __name__ == "__main__":
    main()
