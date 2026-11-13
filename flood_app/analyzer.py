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

    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
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
                        "You are a careful visual flood-damage triage assistant. "
                        "Compare only visible evidence in the paired images. "
                        "Do not claim a condition is safe based on absence of evidence. "
                        "Do not infer people, injuries, power outages, hidden drainage "
                        "conditions, or exact repair costs from aerial images. Mark "
                        "unobservable items as unknown/not assessable. Cost estimates "
                        "must be broad illustrative ranges, never authoritative, and "
                        "should be null when scale or unit costs are unknown. Return "
                        "only valid JSON matching the requested schema."
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
  "summary": "short comparison",
  "power_grid_damage": {"status": "observed|not_observed|unknown", "evidence": "", "severity": "low|moderate|high|unknown"},
  "public_road_hazards": {"status": "observed|not_observed|unknown", "evidence": "", "severity": "low|moderate|high|unknown"},
  "drainage_status": {"status": "observed|not_observed|unknown", "evidence": "", "severity": "low|moderate|high|unknown"},
  "landscape_damage": {"status": "observed|not_observed|unknown", "evidence": "", "severity": "low|moderate|high|unknown"},
  "soil_erosion": {"status": "observed|not_observed|unknown", "evidence": "", "severity": "low|moderate|high|unknown"},
  "personal_injury": {"status": "not_assessable_from_images", "evidence": ""},
  "estimated_cost_usd": {"low": null, "high": null, "basis": "", "assumptions": []},
  "citizen_impact": {"housing_water_impact": "", "vehicles_visible": "", "utility_damage_visible": "", "limitations": []},
  "city_impact": {"infrastructure_visible": "", "drainage_or_erosion_visible": "", "costs": null, "limitations": []},
  "cost_estimates_usd": {
    "housing_and_insurance": {"low": null, "high": null, "basis": "", "source_url": ""},
    "vehicles": {"low": null, "high": null, "basis": "", "source_url": ""},
    "mold_cleanup": {"low": null, "high": null, "basis": "", "source_url": ""},
    "relocation_and_shelter": {"low": null, "high": null, "basis": "", "source_url": ""},
    "contamination": {"low": null, "high": null, "basis": "", "source_url": ""},
    "utility_and_energy": {"low": null, "high": null, "basis": "", "source_url": ""},
    "roads_and_public_infrastructure": {"low": null, "high": null, "basis": "", "source_url": ""},
    "erosion_and_landscape": {"low": null, "high": null, "basis": "", "source_url": ""},
    "dams_and_levees": {"low": null, "high": null, "basis": "", "source_url": ""},
    "drainage": {"low": null, "high": null, "basis": "", "source_url": ""},
    "reinsurance": {"low": null, "high": null, "basis": "", "source_url": ""},
    "total": {"low": null, "high": null, "basis": "", "source_url": ""}
  },
  "confidence": "low|medium|high",
  "limitations": []
}
For citizen_impact, describe only visible housing water impact, vehicles actually visible, and utility damage visibly evidenced. Mold, insurance coverage/claims, displacement, encampments, contamination, and utility service loss require external verified sources or ground inspection; mark them unknown rather than infer. For city_impact, describe only visible public infrastructure, drainage, erosion, and landscape changes; do not infer dam/levee failure or repair prices from aerial imagery. For every cost_estimates_usd category, populate low/high only when measured quantities and a credible source with applicable local unit costs support a calculation; otherwise keep them null and say what evidence is missing. Do not double-count categories; total and estimated_cost_usd must match and remain null if category amounts are missing or overlap. Distinguish observed changes from interpretation. State when imagery resolution, date mismatch, view angle, or missing ground-level evidence makes a conclusion uncertain. Use USD. This dataset may contain aerial imagery from different dates and sensors; visual differences can reflect capture conditions rather than flood damage. Never diagnose individual injury or declare infrastructure safe."""
