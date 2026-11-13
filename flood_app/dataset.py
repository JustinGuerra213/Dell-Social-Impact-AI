"""Hugging Face dataset access and before/after pair discovery."""

from __future__ import annotations

import csv
import io
from pathlib import Path

from huggingface_hub import hf_hub_download
from PIL import Image

from flood_app.models import ImagePair, ImageRecord


class DatasetRepository:
    """Loads metadata and selected images from the UTSA flood dataset."""

    REPO_ID = "jennifer-fowler/UTSA-flood-images"

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def list_pairs(self) -> list[ImagePair]:
        """Return the dataset's verified before/response pairs."""
        csv_path = Path(
            hf_hub_download(
                repo_id=self.REPO_ID,
                repo_type="dataset",
                filename="image_metadata.csv",
                cache_dir=str(self.cache_dir),
            )
        )
        records: dict[str, ImageRecord] = {}
        with csv_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                record = self._record_from_row(row)
                records[record.image_id] = record

        pairs: list[ImagePair] = []
        seen: set[str] = set()
        for record in records.values():
            if record.image_role != "before" or not record.paired_image_id:
                continue
            after = records.get(record.paired_image_id)
            if after is None or after.image_role != "response":
                continue
            key = record.pair_id or record.image_id
            if key in seen:
                continue
            pairs.append(ImagePair(pair_id=key, before=record, after=after))
            seen.add(key)
        return sorted(pairs, key=lambda pair: pair.label.lower())

    def load_image(self, record: ImageRecord) -> Image.Image:
        """Fetch one dataset image on demand and return it as RGB."""
        local_path = hf_hub_download(
            repo_id=self.REPO_ID,
            repo_type="dataset",
            filename=record.image_path,
            cache_dir=str(self.cache_dir),
        )
        with Image.open(local_path) as image:
            return image.convert("RGB")

    @staticmethod
    def _record_from_row(row: dict[str, str]) -> ImageRecord:
        return ImageRecord(
            image_id=row.get("image_id", ""),
            image_path=row.get("image_path", ""),
            pair_id=row.get("pair_id", ""),
            paired_image_id=row.get("paired_image_id", ""),
            image_role=row.get("image_role", ""),
            event_name=row.get("event_name", ""),
            acquisition_date=row.get("acquisition_date", ""),
            locality=row.get("locality", ""),
            source_agency=row.get("source_agency", ""),
            media_type=row.get("media_type", ""),
            attribution=row.get("attribution", ""),
        )


def image_to_bytes(image: Image.Image, image_format: str = "JPEG") -> bytes:
    """Encode an image for transport to the local model endpoint."""
    buffer = io.BytesIO()
    image.save(buffer, format=image_format, quality=90)
    return buffer.getvalue()
