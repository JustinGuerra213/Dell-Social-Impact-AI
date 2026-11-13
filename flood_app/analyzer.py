"""Local Qwen vision-language model client and prompt construction."""

from __future__ import annotations

import base64
import json

import requests
from PIL import Image

from flood_app.dataset import image_to_bytes
from flood_app.models import DamageAssessment


class LocalQwenAnalyzer:
    """Send paired images to an OpenAI-compatible local Qwen endpoint."""

    PROMPT_VERSION = 2

    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 3072,
        timeout_seconds: int = 240,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds

    def analyze(self, before: Image.Image, after: Image.Image) -> DamageAssessment:
        """Compare two images and return structured, cautious findings."""
        before_data = self._data_url(before)
        after_data = self._data_url(after)
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a careful visual flood-damage triage assistant. Compare only "
                        "visible evidence in the paired images and describe it precisely. Do not "
                        "claim a condition is safe based on absence of evidence. Do not infer "
                        "people, injuries, power outages, hidden drainage conditions, or costs "
                        "from aerial images. Mark unobservable items unknown or not assessable. "
                        "Never treat a visual difference as flood-caused unless the images "
                        "directly support that conclusion. Return only valid JSON matching the "
                        "requested schema."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._prompt()},
                        {"type": "text", "text": "BEFORE image:"},
                        {"type": "image_url", "image_url": {"url": before_data}},
                        {"type": "text", "text": "AFTER / RESPONSE image:"},
                        {"type": "image_url", "image_url": {"url": after_data}},
                    ],
                },
            ],
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return DamageAssessment.from_payload(self._parse_json(content))

    @staticmethod
    def _data_url(image: Image.Image) -> str:
        encoded = base64.b64encode(image_to_bytes(image)).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    @staticmethod
    def _parse_json(content: str) -> dict:
        text = content.strip()
        if text.startswith("```"):
            text = text.removeprefix("```json").removeprefix("```")
            text = text.removesuffix("```").strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            raise ValueError("Qwen did not return a JSON assessment.")
        payload = json.loads(text[start : end + 1])
        if not isinstance(payload, dict):
            raise ValueError("Qwen returned an unexpected assessment format.")
        return payload

    @staticmethod
    def _prompt() -> str:
        return """Return JSON with this shape:
{
  "summary": "2-3 sentence overview of the clearest visible changes and how confidently the pair can be compared",
  "power_grid_damage": {"status": "observed|not_observed|unknown", "evidence": "", "severity": "low|moderate|high|unknown"},
  "public_road_hazards": {"status": "observed|not_observed|unknown", "evidence": "", "severity": "low|moderate|high|unknown"},
  "drainage_status": {"status": "observed|not_observed|unknown", "evidence": "", "severity": "low|moderate|high|unknown"},
  "landscape_damage": {"status": "observed|not_observed|unknown", "evidence": "", "severity": "low|moderate|high|unknown"},
  "soil_erosion": {"status": "observed|not_observed|unknown", "evidence": "", "severity": "low|moderate|high|unknown"},
  "personal_injury": {"status": "not_assessable_from_images", "evidence": ""},
  "citizen_impact": {"housing_water_impact": "", "vehicles_visible": "", "utility_damage_visible": ""},
  "city_impact": {"infrastructure_visible": "", "drainage_or_erosion_visible": ""},
  "confidence": "low|medium|high",
  "limitations": []
}
Write in clear language for a non-specialist. Be specific, comparative, and evidence-led; do not pad the answer or repeat generic warnings. For each impact field and each observed-category evidence field, write 2-3 concise sentences when the images support that detail. Identify the image (before or response), the visible feature and its approximate location or extent, and the change between images. Separate direct observation from a cautious possible implication. If a feature is not visible or the pair cannot support a conclusion, say so briefly instead of guessing.

For citizen_impact, describe only visible water exposure to homes, vehicles actually visible, and utility infrastructure physically shown. For city_impact, describe only visible public infrastructure, drainage, erosion, and landscape changes. Do not infer service outages, contamination, structural failure, costs, or injury.

Use status=observed only for clear visual evidence. Use not_observed only when the relevant area is visible enough to make that limited statement; it does not mean safe or undamaged. Use unknown when imagery is insufficient. Severity describes visible extent only: low is small/localized, moderate is clearly affecting multiple features or a larger area, high is extensive visible impact; otherwise use unknown. Do not assign severity based on hypothetical danger.

Limit limitations to at most four short, image-specific items, each one or two sentences. Explain how resolution, image angle, occlusion, incomplete scene coverage, different sensors, or the before/response date gap affects what can be concluded. Keep limitations grounded in the images and do not introduce outside reports or sources. This dataset may contain aerial imagery from different dates and sensors; visual differences can reflect capture conditions rather than flood damage. Never diagnose individual injury or declare infrastructure safe."""
