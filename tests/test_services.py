"""Unit tests for dataset pairing and model-response parsing."""

from __future__ import annotations

import csv
from datetime import date
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

from flood_app.analyzer import LocalQwenAnalyzer
from flood_app.dataset import DatasetRepository, filter_pairs_by_locality
from flood_app.models import DamageAssessment
from flood_app.weather import HistoricalWeatherProvider


class DatasetRepositoryTests(unittest.TestCase):
    """Verify the metadata-driven pair workflow."""

    def test_list_pairs_uses_paired_image_id_and_roles(self) -> None:
        fields = [
            "image_id",
            "image_path",
            "pair_id",
            "paired_image_id",
            "image_role",
            "event_name",
            "acquisition_date",
            "locality",
            "source_agency",
            "media_type",
            "attribution",
        ]
        rows = [
            {
                "image_id": "loc_before",
                "image_path": "images/loc_before.png",
                "pair_id": "loc",
                "paired_image_id": "loc_after",
                "image_role": "before",
                "event_name": "Flood event",
                "acquisition_date": "2020-01-01",
                "locality": "Test location",
            },
            {
                "image_id": "loc_after",
                "image_path": "images/loc_after.png",
                "pair_id": "loc",
                "paired_image_id": "loc_before",
                "image_role": "response",
                "event_name": "Flood event",
                "acquisition_date": "2020-01-02",
                "locality": "Test location",
            },
            {
                "image_id": "unpaired",
                "image_path": "images/unpaired.png",
                "image_role": "before",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            metadata = Path(directory) / "image_metadata.csv"
            with metadata.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

            repository = DatasetRepository(Path(directory) / "cache")
            repository._download = lambda _: metadata  # type: ignore[method-assign]
            pairs = repository.list_pairs()

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].before.image_id, "loc_before")
        self.assertEqual(pairs[0].after.image_id, "loc_after")
        self.assertEqual(pairs[0].label, "loc · Test location · Flood event")
        self.assertEqual(filter_pairs_by_locality(pairs, "Test location"), pairs)
        self.assertEqual(filter_pairs_by_locality(pairs, "Unknown locality"), [])
        self.assertEqual(filter_pairs_by_locality(pairs, None), pairs)


class LocalQwenAnalyzerTests(unittest.TestCase):
    """Verify prompt guidance and tolerant parsing of model output."""

    def test_prompt_requests_comparative_detail_and_specific_limitations(self) -> None:
        prompt = LocalQwenAnalyzer._prompt()

        self.assertIn("2-3 concise sentences", prompt)
        self.assertIn("visible feature and its approximate location or extent", prompt)
        self.assertIn("Limit limitations to at most four short, image-specific items", prompt)
        self.assertIn("Use not_observed only when the relevant area is visible enough", prompt)
        self.assertIn("road_affected_percent_low", prompt)

    def test_prompt_uses_weather_as_context_only_when_supplied(self) -> None:
        prompt = LocalQwenAnalyzer._prompt({"source": "weather API", "rainy_days": 5})

        self.assertIn("rainfall as proof", prompt)
        self.assertIn('"rainy_days":5', prompt)
        self.assertIn("No weather data was supplied", LocalQwenAnalyzer._prompt())

    def test_assessment_preserves_image_metrics(self) -> None:
        assessment = DamageAssessment.from_payload(
            {
                "metrics": {
                    "flood_area_percent_low": 15,
                    "flood_area_percent_high": 25,
                    "road_affected_percent_low": None,
                    "flood_area_basis": "Visible water covers a portion of the scene.",
                }
            }
        )

        self.assertEqual(assessment.metrics["flood_area_percent_low"], 15)
        self.assertIsNone(assessment.metrics["road_affected_percent_low"])

    def test_parse_json_accepts_markdown_fence(self) -> None:
        payload = LocalQwenAnalyzer._parse_json('```json\n{"summary":"change"}\n```')
        self.assertEqual(payload, {"summary": "change"})

    def test_parse_json_rejects_non_json(self) -> None:
        with self.assertRaises(ValueError):
            LocalQwenAnalyzer._parse_json("No structured answer")

    @patch("flood_app.analyzer.requests.post")
    def test_analyze_sends_selected_generation_settings(self, post: Mock) -> None:
        response = Mock()
        response.json.return_value = {
            "choices": [{"message": {"content": '{"summary":"ok"}'}}]
        }
        post.return_value = response
        analyzer = LocalQwenAnalyzer(
            "http://localhost:8000/v1",
            "local-qwen",
            temperature=0.65,
            max_tokens=768,
        )

        analyzer.analyze(
            Image.new("RGB", (2, 2)),
            Image.new("RGB", (2, 2)),
            weather_context={"source": "Open-Meteo", "rainy_days": 4},
        )

        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["temperature"], 0.65)
        self.assertEqual(payload["max_tokens"], 768)
        prompt_text = payload["messages"][1]["content"][0]["text"]
        self.assertIn('"rainy_days":4', prompt_text)


class HistoricalWeatherProviderTests(unittest.TestCase):
    """Verify safe summaries of historical weather API responses."""

    def test_parse_daily_data_calculates_temperature_and_rainfall_summaries(self) -> None:
        parsed = HistoricalWeatherProvider._parse_daily_data(
            {
                "latitude": 29.7,
                "longitude": -95.3,
                "daily": {
                    "time": ["2020-01-01", "2020-01-02", "2020-01-03"],
                    "temperature_2m_mean": [20, None, 22],
                    "rain_sum": [0, 10, 5],
                },
            },
            date(2020, 1, 1),
            date(2020, 1, 3),
        )

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["mean_temperature_c"], 21)
        self.assertEqual(parsed["total_rain_mm"], 15)
        self.assertEqual(parsed["rainy_days"], 2)

    def test_invalid_coordinates_skip_weather_lookup(self) -> None:
        self.assertIsNone(HistoricalWeatherProvider().get_context(None, -95.3, "2020-01-01"))


if __name__ == "__main__":
    unittest.main()
