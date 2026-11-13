"""Retrieve image-date weather context for paired flood imagery."""

from __future__ import annotations

from datetime import date, timedelta
import math
from typing import Any

import requests


class HistoricalWeatherProvider:
    """Fetch a compact daily weather window from Open-Meteo's archive API."""

    API_URL = "https://archive-api.open-meteo.com/v1/archive"
    WINDOW_DAYS = 30

    def __init__(self, timeout_seconds: int = 20) -> None:
        self.timeout_seconds = timeout_seconds

    def get_context(
        self,
        latitude: float | None,
        longitude: float | None,
        image_date: str,
    ) -> dict[str, Any] | None:
        """Return daily temperature and rainfall around the image date, if available."""
        if latitude is None or longitude is None:
            return None
        if not (math.isfinite(latitude) and math.isfinite(longitude)):
            return None
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            return None
        try:
            center_date = date.fromisoformat(image_date[:10])
        except (TypeError, ValueError):
            return None

        half_window = self.WINDOW_DAYS // 2
        start_date = center_date - timedelta(days=half_window)
        end_date = center_date + timedelta(days=self.WINDOW_DAYS - half_window - 1)
        response = requests.get(
            self.API_URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "daily": "temperature_2m_mean,rain_sum",
                "timezone": "auto",
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        return self._parse_daily_data(payload, start_date, end_date)

    @staticmethod
    def _parse_daily_data(
        payload: dict[str, Any],
        start_date: date,
        end_date: date,
    ) -> dict[str, Any] | None:
        """Validate daily values and add simple distribution summaries."""
        daily = payload.get("daily")
        if not isinstance(daily, dict) or not isinstance(daily.get("time"), list):
            return None

        dates = daily["time"]
        temperatures = daily.get("temperature_2m_mean", [])
        rainfall = daily.get("rain_sum", [])
        observations: list[dict[str, Any]] = []
        for index, day in enumerate(dates):
            temperature = _finite_number(_at(temperatures, index))
            rain = _finite_number(_at(rainfall, index))
            observations.append({"date": str(day), "temperature_c": temperature, "rain_mm": rain})

        temperatures_only = [row["temperature_c"] for row in observations if row["temperature_c"] is not None]
        rainfall_only = [row["rain_mm"] for row in observations if row["rain_mm"] is not None]
        if not observations:
            return None

        return {
            "source": "Open-Meteo Historical Weather API (default Best Match gridded reanalysis)",
            "latitude": _finite_number(payload.get("latitude")),
            "longitude": _finite_number(payload.get("longitude")),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "observations": observations,
            "mean_temperature_c": (
                round(sum(temperatures_only) / len(temperatures_only), 1)
                if temperatures_only
                else None
            ),
            "total_rain_mm": round(sum(rainfall_only), 1) if rainfall_only else None,
            "rainy_days": sum(value > 0 for value in rainfall_only),
            "maximum_daily_rain_mm": round(max(rainfall_only), 1) if rainfall_only else None,
        }


def _at(values: Any, index: int) -> Any:
    return values[index] if isinstance(values, list) and index < len(values) else None


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
