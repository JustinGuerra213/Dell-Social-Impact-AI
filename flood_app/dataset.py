"""Hugging Face dataset access and before/after pair discovery."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from urllib.parse import quote

from PIL import Image
import requests

from flood_app.models import ImagePair, ImageRecord


class DatasetRepository:
    """Loads metadata and selected images from the UTSA flood dataset."""

    REPO_ID = "jennifer-fowler/UTSA-flood-images"

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def list_pairs(self) -> list[ImagePair]:
        """Return the dataset's verified before/response pairs."""
        csv_path = self._download("image_metadata.csv")
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
        local_path = self._download(record.image_path)
        with Image.open(local_path) as image:
            return image.convert("RGB")

    def _download(self, filename: str) -> Path:
        """Return a locally cached dataset file, downloading it when needed."""
        safe_path = Path(filename)
        if safe_path.is_absolute() or ".." in safe_path.parts:
            raise ValueError("Dataset file path is not safe.")
        target = self.cache_dir / safe_path
        if target.is_file():
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        url = (
            f"https://huggingface.co/datasets/{self.REPO_ID}/resolve/main/"
            f"{quote(filename, safe='/')}"
        )
        response = requests.get(url, timeout=(15, 180), stream=True)
        response.raise_for_status()
        temporary = target.with_suffix(target.suffix + ".part")
        try:
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        return target

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
