"""
Parse a Bear markdown export.

Bear exports either:
  - A folder of .md files (one per note)
  - A .zip of the above

Each filename is the note title. Bear embeds tags inline as #tag or #nested/tag.
Dates are inferred from file modification time or a 'date:' line at the top.

Usage:
    entries = parse_bear_export("/path/to/Bear Export/")
    entries = parse_bear_export("/path/to/Bear Export.zip")
"""
import os
import re
import zipfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Union

from ..models import JournalEntry


_DATE_HEADER_RE = re.compile(
    r"^date:\s*(\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?)", re.IGNORECASE
)
_BEAR_TAG_RE = re.compile(r"(?<!\w)#([\w/.-]+)(?!\w)")


def _extract_inline_tags(text: str) -> list[str]:
    return list(dict.fromkeys(_BEAR_TAG_RE.findall(text)))  # preserve order, dedupe


def _parse_date_from_content(text: str) -> datetime | None:
    for line in text.splitlines()[:10]:
        m = _DATE_HEADER_RE.match(line.strip())
        if m:
            raw = m.group(1).replace("T", " ")
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
                try:
                    return datetime.strptime(raw, fmt)
                except ValueError:
                    continue
    return None


def _mtime(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime)


def _parse_md_file(path: Path) -> JournalEntry:
    text = path.read_text(encoding="utf-8", errors="replace")
    date = _parse_date_from_content(text) or _mtime(path)
    title = path.stem  # filename without extension
    tags = _extract_inline_tags(text)

    return JournalEntry(
        title=title,
        date=date,
        content=text,
        tags=tags,
        source="bear",
    )


def parse_bear_export(source: Union[str, Path]) -> list[JournalEntry]:
    source = Path(source)

    if source.suffix.lower() == ".zip":
        tmp_dir = tempfile.mkdtemp(prefix="bear_")
        with zipfile.ZipFile(source) as zf:
            zf.extractall(tmp_dir)
        source = Path(tmp_dir)

    md_files = sorted(source.rglob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No .md files found under {source}")

    entries = [_parse_md_file(f) for f in md_files]
    entries.sort(key=lambda e: e.date or datetime.min)
    return entries
