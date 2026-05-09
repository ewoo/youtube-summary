"""
Parse a DayOne JSON export.

DayOne exports a .zip containing:
  Journal.json  – all entries
  photos/       – image attachments (jpeg/png/gif)

Usage:
    entries = parse_dayone_export("/path/to/Journal.zip")
    # or an already-unzipped directory:
    entries = parse_dayone_export("/path/to/Journal/")
"""
import json
import os
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

from ..models import JournalEntry


def _parse_date(date_str: str) -> datetime:
    # DayOne uses ISO-8601: "2023-01-15T10:30:00Z"
    date_str = date_str.replace("Z", "+00:00")
    return datetime.fromisoformat(date_str).astimezone(tz=None).replace(tzinfo=None)


def _entry_title(text: str, date: datetime) -> str:
    """Use first non-empty line as title, fall back to date string."""
    for line in text.splitlines():
        line = line.strip().lstrip("#").strip()
        if line:
            return line[:80]
    return date.strftime("%Y-%m-%d") if date else "Untitled"


def _extract_photo_paths(entry: dict, media_dir: Path) -> list[str]:
    paths = []
    for photo in entry.get("photos", []):
        identifier = photo.get("identifier", "")
        ext = photo.get("type", "jpeg")
        candidate = media_dir / f"{identifier}.{ext}"
        if candidate.exists():
            paths.append(str(candidate))
    return paths


def parse_dayone_export(source: Union[str, Path]) -> list[JournalEntry]:
    source = Path(source)
    tmp_dir = None

    if source.suffix.lower() == ".zip":
        import tempfile
        tmp_dir = tempfile.mkdtemp(prefix="dayone_")
        with zipfile.ZipFile(source) as zf:
            zf.extractall(tmp_dir)
        # Find the Journal.json – may be nested inside a subdirectory
        matches = list(Path(tmp_dir).rglob("Journal.json"))
        if not matches:
            raise FileNotFoundError("Journal.json not found inside the DayOne zip.")
        journal_path = matches[0]
        media_dir = journal_path.parent / "photos"
    else:
        # Assume unzipped directory
        journal_path = source / "Journal.json"
        if not journal_path.exists():
            # Try one level deeper (DayOne sometimes nests)
            matches = list(source.rglob("Journal.json"))
            if not matches:
                raise FileNotFoundError(f"Journal.json not found under {source}")
            journal_path = matches[0]
        media_dir = journal_path.parent / "photos"

    with open(journal_path, encoding="utf-8") as f:
        data = json.load(f)

    entries = []
    for raw in data.get("entries", []):
        text = raw.get("text", "")
        date = _parse_date(raw["creationDate"]) if "creationDate" in raw else None
        title = _entry_title(text, date)
        tags = raw.get("tags", [])
        attachments = _extract_photo_paths(raw, media_dir) if media_dir.exists() else []

        entries.append(JournalEntry(
            title=title,
            date=date,
            content=text,
            tags=tags,
            source="dayone",
            attachments=attachments,
        ))

    entries.sort(key=lambda e: e.date or datetime.min)
    return entries
