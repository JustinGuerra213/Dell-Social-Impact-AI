"""Streamlit interface for image-based flood impact triage."""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from flood_app.analyzer import LocalQwenAnalyzer
from flood_app.dataset import DatasetRepository, filter_pairs_by_locality
from flood_app.models import DamageAssessment, ImagePair
from flood_app.weather import HistoricalWeatherProvider


APP_ROOT = Path(__file__).parent
DATA_CACHE = Path(os.getenv("FLOOD_DATA_CACHE", APP_ROOT / "data" / "hf-cache"))
ASSESSMENT_PROMPT_VERSION = "metrics-report-v3"


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


@st.cache_data(ttl=21600, show_spinner=False)
def historical_weather_context(
    latitude: float | None,
    longitude: float | None,
    image_date: str,
) -> dict | None:
    """Cache daily weather context for the selected image-date window."""
    return HistoricalWeatherProvider().get_context(latitude, longitude, image_date)


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
    st.caption(
        "Compare before-and-after flood images to summarize visible damage and community impacts "
        "with a local Qwen vision model."
    )
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
            value=3072,
            step=256,
            help="Limits response length. Larger values can take longer.",
        )
        st.header("UTSA dataset pair")
        pair: ImagePair | None = None
        before_image: Image.Image | None = None
        after_image: Image.Image | None = None

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
        st.write("Choose a UTSA dataset pair. The report uses only visible evidence from the two images.")
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
        with st.spinner("Loading optional weather context and comparing both images with local Qwen…"):
            try:
                weather_observations = None
                if pair:
                    try:
                        latitude, longitude = pair_coordinates(pair)
                        weather_observations = historical_weather_context(
                            latitude,
                            longitude,
                            pair.after.acquisition_date or pair.before.acquisition_date,
                        )
                    except Exception:
                        # Weather is optional; the model still analyzes both images alone.
                        weather_observations = None
                assessment = qwen_analyzer(temperature, int(max_tokens)).analyze(
                    before_image,
                    after_image,
                    weather_context=weather_observations,
                )
                if weather_observations:
                    assessment.metrics["weather_context"] = {
                        key: weather_observations[key]
                        for key in (
                            "source",
                            "start_date",
                            "end_date",
                            "mean_temperature_c",
                            "total_rain_mm",
                            "rainy_days",
                            "maximum_daily_rain_mm",
                        )
                    }
                st.session_state["weather_context_included"] = bool(weather_observations)
                st.session_state["assessment"] = assessment
                st.session_state["assessment_pair"] = pair.label if pair else "Uploaded images"
                st.session_state["model_settings"] = {
                    "temperature": temperature,
                    "max_tokens": int(max_tokens),
                }
                st.session_state["assessment_prompt_version"] = ASSESSMENT_PROMPT_VERSION
            except Exception as exc:
                st.error(
                    "Analysis failed. Confirm that the local Qwen OpenAI-compatible "
                    f"server is running and the model name is correct. Details: {exc}"
                )

    assessment = st.session_state.get("assessment")
    if (
        assessment
        and st.session_state.get("assessment_prompt_version") != ASSESSMENT_PROMPT_VERSION
    ):
        st.info("The report was generated with an earlier prompt. Select “Analyze changes” to create the expanded report.")
        assessment = None
    if assessment:
        render_assessment(
            assessment,
            st.session_state.get("assessment_pair", ""),
            st.session_state.get("model_settings", {}),
            weather_context=(
                (
                    pair_coordinates(pair),
                    pair.after.acquisition_date or pair.before.acquisition_date,
                    pair.after.locality or pair.before.locality,
                )
                if pair
                else None
            ),
            weather_context_included=st.session_state.get("weather_context_included", False),
        )


def render_assessment(
    assessment: DamageAssessment,
    pair_label: str,
    model_settings: dict | None = None,
    weather_context: tuple[tuple[float | None, float | None], str, str] | None = None,
    weather_context_included: bool = False,
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

    impact_tab, observed_tab, limitations_tab, metrics_tab, area_map_tab = st.tabs(
        ["Impact", "Observed changes", "Limitations", "Metrics", "Area map"]
    )
    with impact_tab:
        st.caption(
            "Impact statements describe possible effects supported by visible evidence in this image pair. "
            "They do not confirm injuries, service outages, or damage hidden from view."
        )
        st.markdown("#### Citizen impact")
        render_impact_section(
            assessment.citizen_impact,
            [
                ("housing_water_impact", "Water exposure to homes"),
                ("vehicles_visible", "Vehicles in the image"),
                ("utility_damage_visible", "Visible utility infrastructure"),
            ],
        )
        st.markdown("#### City impact")
        render_impact_section(
            assessment.city_impact,
            [
                ("infrastructure_visible", "Public infrastructure"),
                ("drainage_or_erosion_visible", "Drainage, erosion, and landscape"),
            ],
        )
    with observed_tab:
        st.caption(
            "Each finding compares visible features across the before and response images. "
            "“Not observed” means no clear evidence appears in the visible area; it does not mean the area is safe."
        )
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
                raw_status = str(finding.get("status", "unknown")).lower()
                status = {
                    "observed": "Visible evidence",
                    "not_observed": "No clear evidence in view",
                    "unknown": "Cannot determine from these images",
                    "not_assessable_from_images": "Not assessable from images",
                }.get(raw_status, raw_status.replace("_", " ").title())
                st.markdown(f"**{title} · {status}**")
                if finding.get("severity"):
                    st.caption(f"Visible extent: {finding['severity']}")
                st.write(
                    finding.get("evidence")
                    or "The images do not provide enough clear visual evidence to describe this condition."
                )
    with limitations_tab:
        st.caption(
            "These image-specific limits affect confidence in the visual comparison. They describe what this pair cannot establish."
        )
        limitations = assessment.limitations[:4]
        if limitations:
            for limitation in limitations:
                with st.container(border=True):
                    st.write(limitation)
        else:
            with st.container(border=True):
                st.write(
                    "The pair shows only two capture times, so it cannot establish exactly when a visible change occurred or prove what caused it. "
                    "Aerial views may hide damage beneath water, vegetation, shadows, or structures; interior conditions and utility service status are not visible."
                )
    with metrics_tab:
        render_impact_metrics(assessment)
        st.divider()
        st.markdown("#### Temperature and rainfall context")
        if weather_context_included:
            st.caption(
                "Historical temperature and rainfall data for the image-date window was supplied to Qwen as context. "
                "The area and road percentages remain visual estimates from the images, not weather-derived measurements."
            )
        else:
            st.caption(
                "No historical weather data was supplied during this analysis. Metrics use only the before and response images. "
                "The chart below may still load weather data for reference, but it did not affect this prediction."
            )
        if weather_context:
            with st.container(border=True):
                coordinates, acquisition_date, locality = weather_context
                render_weather_explorer(coordinates, acquisition_date, locality)
        else:
            st.info("No weather data is available for this pair; the image-based metrics use only the before and response images.")
    with area_map_tab:
        if weather_context:
            coordinates, _, locality = weather_context
            render_area_map(coordinates, locality)
        else:
            st.info("Map data is unavailable for this report.")
    st.download_button(
        "Download report as JSON",
        data=json.dumps(assessment.__dict__, indent=2),
        file_name="saving-texas-assessment.json",
        mime="application/json",
    )
    st.caption("Dataset: UTSA Flood Images, CC BY-NC 4.0.")


def pair_coordinates(pair: ImagePair) -> tuple[float | None, float | None]:
    """Use response coordinates, falling back to before-image coordinates."""
    after_coordinates = (pair.after.latitude, pair.after.longitude)
    if None not in after_coordinates:
        return after_coordinates
    return pair.before.latitude, pair.before.longitude


def render_impact_section(impact: dict, fields: list[tuple[str, str]]) -> None:
    """Render image-based citizen or city observations from Qwen."""
    for key, label in fields:
        with st.container(border=True):
            st.markdown(f"**{label}**")
            st.write(
                impact.get(key)
                or "Not established from these images; this pair does not show enough clear evidence to describe this impact."
            )


def render_impact_metrics(assessment: DamageAssessment) -> None:
    """Render bounded visual estimates and the model's confidence rationale."""
    metrics = assessment.metrics
    flood_range = _format_percent_range(
        metrics.get("flood_area_percent_low"), metrics.get("flood_area_percent_high")
    )
    road_range = _format_percent_range(
        metrics.get("road_affected_percent_low"), metrics.get("road_affected_percent_high")
    )
    confidence = assessment.confidence.strip().title()
    if confidence.lower() not in {"low", "medium", "high"}:
        confidence = "Unavailable"

    st.caption(
        "Percentages refer only to the visible image footprint and visible road segments. "
        "They are approximate visual estimates, not surveyed measurements or totals for the surrounding area."
    )
    area_column, roads_column, confidence_column = st.columns(3)
    area_column.metric("Visible flood-affected scene", flood_range or "Not estimable")
    roads_column.metric("Visible roads affected", road_range or "Not estimable")
    confidence_column.metric("Prediction confidence", confidence)

    with st.container(border=True):
        st.markdown("**Evidence and estimate basis**")
        st.write(
            "**Visible scene:** "
            + str(metrics.get("flood_area_basis") or "The model did not provide a defensible area estimate.")
        )
        st.write(
            "**Visible roads:** "
            + str(metrics.get("road_affected_basis") or "The model did not provide a defensible road estimate.")
        )
        st.write(
            "**Confidence:** "
            + str(
                metrics.get("confidence_rationale")
                or "No confidence explanation was returned; review the image-specific limitations and observed evidence."
            )
        )


def _format_percent_range(low: object, high: object) -> str | None:
    """Format a valid 0–100 percent interval from model output."""
    try:
        low_value = float(low)
        high_value = float(high)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(low_value) and math.isfinite(high_value)):
        return None
    if not (0 <= low_value <= high_value <= 100):
        return None
    return f"{low_value:g}–{high_value:g}%"


def render_weather_explorer(
    coordinates: tuple[float | None, float | None],
    acquisition_date: str,
    locality: str,
) -> None:
    """Embed the interactive JavaScript historical weather explorer."""
    st.subheader("Temperature and rainfall distribution")
    if None in coordinates:
        st.info("Weather data is unavailable because this dataset pair has no coordinates.")
        return

    latitude, longitude = coordinates
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        st.info("Weather data is unavailable because this pair has invalid coordinates.")
        return
    date_match = re.search(r"\d{4}-\d{2}-\d{2}", acquisition_date or "")
    if not date_match:
        st.info("Weather data is unavailable because this pair has no recorded image date.")
        return

    weather_html = (APP_ROOT / "flood_app" / "weather_explorer.html").read_text(encoding="utf-8")
    weather_html = weather_html.replace("__LATITUDE__", json.dumps(latitude))
    weather_html = weather_html.replace("__LONGITUDE__", json.dumps(longitude))
    weather_html = weather_html.replace("__DATE__", json.dumps(date_match.group(0)))
    locality_literal = json.dumps(locality or "Selected location").replace("<", "\\u003c")
    weather_html = weather_html.replace("__LOCALITY__", locality_literal)
    components.html(weather_html, height=760, scrolling=False)


def render_area_map(
    coordinates: tuple[float | None, float | None],
    locality: str,
) -> None:
    """Embed an interactive JavaScript map centered on the selected pair."""
    st.subheader("Selected image location")
    if None in coordinates:
        st.info("Map data is unavailable because this dataset pair has no coordinates.")
        return

    latitude, longitude = coordinates
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        st.info("Map data is unavailable because this pair has invalid coordinates.")
        return

    map_html = (APP_ROOT / "flood_app" / "area_map.html").read_text(encoding="utf-8")
    map_html = map_html.replace("__LATITUDE__", json.dumps(latitude))
    map_html = map_html.replace("__LONGITUDE__", json.dumps(longitude))
    locality_literal = json.dumps(locality or "Selected image location").replace("<", "\\u003c")
    map_html = map_html.replace("__LOCALITY__", locality_literal)
    components.html(map_html, height=560, scrolling=False)


if __name__ == "__main__":
    main()
