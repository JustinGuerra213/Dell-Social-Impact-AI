"""Typed data structures used by the flood assessment workflow."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ImageRecord:
    """A single image and its dataset metadata."""

    image_id: str
    image_path: str
    pair_id: str
    paired_image_id: str
    image_role: str
    event_name: str
    acquisition_date: str
    locality: str
    source_agency: str
    media_type: str
    attribution: str


@dataclass(frozen=True)
class ImagePair:
    """Before and response imagery for one dataset location."""

    pair_id: str
    before: ImageRecord
    after: ImageRecord

    @property
    def label(self) -> str:
        """Return a readable location label for the pair picker."""
        locality = self.before.locality or "Unknown location"
        event = self.before.event_name or "Flood event"
        return f"{self.pair_id} · {locality} · {event}"


@dataclass(frozen=True)
class DamageAssessment:
    """Structured findings returned by the local vision model."""

    summary: str
    power_grid_damage: dict[str, Any]
    public_road_hazards: dict[str, Any]
    drainage_status: dict[str, Any]
    landscape_damage: dict[str, Any]
    soil_erosion: dict[str, Any]
    personal_injury: dict[str, Any]
    estimated_cost_usd: dict[str, Any]
    confidence: str
    limitations: list[str]

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "DamageAssessment":
        """Build a safe assessment from possibly incomplete model output."""
        return cls(
            summary=str(payload.get("summary", "No summary provided.")),
            power_grid_damage=_as_dict(payload.get("power_grid_damage")),
            public_road_hazards=_as_dict(payload.get("public_road_hazards")),
            drainage_status=_as_dict(payload.get("drainage_status")),
            landscape_damage=_as_dict(payload.get("landscape_damage")),
            soil_erosion=_as_dict(payload.get("soil_erosion")),
            personal_injury=_as_dict(payload.get("personal_injury")),
            estimated_cost_usd=_as_dict(payload.get("estimated_cost_usd")),
            confidence=str(payload.get("confidence", "low")),
            limitations=_as_list(payload.get("limitations")),
        )


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]
