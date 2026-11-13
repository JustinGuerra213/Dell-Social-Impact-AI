"""Unit tests for dataset pairing and model-response parsing."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

from flood_app.analyzer import LocalQwenAnalyzer
from flood_app.dataset import DatasetRepository


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


class LocalQwenAnalyzerTests(unittest.TestCase):
    """Verify tolerant parsing of the local model's JSON output."""

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

        analyzer.analyze(Image.new("RGB", (2, 2)), Image.new("RGB", (2, 2)))

        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["temperature"], 0.65)
        self.assertEqual(payload["max_tokens"], 768)


if __name__ == "__main__":
    unittest.main()
