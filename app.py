"""Streamlit interface for local flood impact triage."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from PIL import Image

from flood_app.analyzer import LocalQwenAnalyzer
from flood_app.dataset import DatasetRepository
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
def qwen_analyzer() -> LocalQwenAnalyzer:
    """Create the local Qwen API client."""
    return LocalQwenAnalyzer(
        base_url=os.getenv("QWEN_BASE_URL", "http://127.0.0.1:8000/v1"),
        model=os.getenv("QWEN_MODEL", "Qwen/Qwen3-VL-8B-Instruct"),
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
        page_title="FloodLens · Flood impact triage",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("🌊 FloodLens")
    st.caption("Before-and-after flood imagery · Local Qwen vision analysis")
    st.info(
        "Triage support only. Image analysis cannot confirm structural safety, "
        "power status, drainage function, human injury, or actual repair costs. "
        "Verify hazards on the ground with qualified responders.",
        icon="⚠️",
    )

    with st.sidebar:
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
                    pair = st.selectbox("Location / event", pairs, format_func=lambda item: item.label)
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
                assessment = qwen_analyzer().analyze(before_image, after_image)
                st.session_state["assessment"] = assessment
                st.session_state["assessment_pair"] = pair.label if pair else "Uploaded images"
            except Exception as exc:
                st.error(
                    "Analysis failed. Confirm that the local Qwen OpenAI-compatible "
                    f"server is running and the model name is correct. Details: {exc}"
                )

    assessment = st.session_state.get("assessment")
    if assessment:
        render_assessment(assessment, st.session_state.get("assessment_pair", ""))


def render_assessment(assessment: DamageAssessment, pair_label: str) -> None:
    """Render model output as readable impact categories."""
    st.divider()
    st.header("Impact triage report")
    if pair_label:
        st.caption(pair_label)
    st.write(assessment.summary)

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
        file_name="floodlens-assessment.json",
        mime="application/json",
    )
    st.caption("Dataset: UTSA Flood Images, CC BY-NC 4.0. Use is subject to the dataset’s non-commercial license.")


if __name__ == "__main__":
    main()
