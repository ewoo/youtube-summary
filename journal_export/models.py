from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class JournalEntry:
    title: str
    date: Optional[datetime]
    content: str               # markdown body
    tags: list[str] = field(default_factory=list)
    source: str = ""           # "dayone" or "bear"
    attachments: list[str] = field(default_factory=list)  # local file paths

    def to_markdown(self) -> str:
        """Return full markdown file content with YAML frontmatter."""
        lines = ["---"]
        if self.date:
            lines.append(f"date: {self.date.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"title: {self.title!r}")
        if self.tags:
            tags_yaml = ", ".join(f'"{t}"' for t in self.tags)
            lines.append(f"tags: [{tags_yaml}]")
        if self.source:
            lines.append(f"source: {self.source}")
        lines.append("---")
        lines.append("")
        lines.append(self.content)
        return "\n".join(lines)

    def filename(self) -> str:
        """Filesystem-safe filename."""
        date_str = self.date.strftime("%Y-%m-%d_%H-%M-%S") if self.date else "undated"
        safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in self.title)
        safe_title = safe_title.strip().replace(" ", "-")[:60]
        return f"{date_str}_{safe_title}.md" if safe_title else f"{date_str}.md"

    def drive_folder_path(self) -> list[str]:
        """Returns folder hierarchy: ['Journals', 'DayOne', '2023', '01']"""
        source_name = "DayOne" if self.source == "dayone" else "Bear"
        if self.date:
            return ["Journals", source_name, self.date.strftime("%Y"), self.date.strftime("%m")]
        return ["Journals", source_name, "Undated"]
