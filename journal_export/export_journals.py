#!/usr/bin/env python3
"""
Export DayOne / Bear journal entries to Google Drive as markdown files.

Examples:
  # Parse DayOne export and preview (no upload)
  python export_journals.py --dayone ~/Downloads/Journal.zip --dry-run

  # Parse Bear export and upload to Drive
  python export_journals.py --bear ~/Downloads/Bear\ Export/ --drive

  # Both sources, upload to a specific Drive folder
  python export_journals.py \\
      --dayone ~/Downloads/Journal.zip \\
      --bear ~/Downloads/BearExport.zip \\
      --drive \\
      --drive-folder-id 1AbCdEfGhIjK

  # Write markdown files locally instead of (or before) uploading
  python export_journals.py --dayone ~/Downloads/Journal.zip --output ./output
"""
import argparse
import os
import sys
from pathlib import Path

from .parsers.dayone import parse_dayone_export
from .parsers.bear import parse_bear_export
from .models import JournalEntry


def write_locally(entries: list[JournalEntry], output_dir: Path) -> None:
    for entry in entries:
        folder = output_dir.joinpath(*entry.drive_folder_path())
        folder.mkdir(parents=True, exist_ok=True)
        dest = folder / entry.filename()
        dest.write_text(entry.to_markdown(), encoding="utf-8")
    print(f"  Written {len(entries)} files to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export DayOne/Bear journals to Google Drive as markdown.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dayone", metavar="PATH", help="Path to DayOne export (.zip or folder)")
    parser.add_argument("--bear", metavar="PATH", help="Path to Bear export (.zip or folder)")
    parser.add_argument("--output", metavar="DIR", help="Write markdown files locally to DIR")
    parser.add_argument("--drive", action="store_true", help="Upload to Google Drive")
    parser.add_argument(
        "--drive-folder-id",
        metavar="ID",
        default=None,
        help="Google Drive folder ID to use as root (defaults to My Drive root)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and list entries without writing or uploading anything",
    )
    args = parser.parse_args()

    if not args.dayone and not args.bear:
        parser.error("Provide at least one of --dayone or --bear.")

    entries: list[JournalEntry] = []

    if args.dayone:
        print(f"Parsing DayOne export: {args.dayone}")
        dayone_entries = parse_dayone_export(args.dayone)
        print(f"  Found {len(dayone_entries)} DayOne entries.")
        entries.extend(dayone_entries)

    if args.bear:
        print(f"Parsing Bear export: {args.bear}")
        bear_entries = parse_bear_export(args.bear)
        print(f"  Found {len(bear_entries)} Bear entries.")
        entries.extend(bear_entries)

    print(f"\nTotal entries: {len(entries)}")

    if args.dry_run:
        print("\n--- Dry run: listing entries ---")
        for e in entries:
            date_str = e.date.strftime("%Y-%m-%d") if e.date else "undated"
            tags_str = f"  tags: {e.tags}" if e.tags else ""
            print(f"  [{e.source:6}] {date_str}  {e.title[:60]}{tags_str}")
        return

    if args.output:
        output_dir = Path(args.output)
        print(f"\nWriting markdown files to {output_dir} ...")
        write_locally(entries, output_dir)

    if args.drive:
        from .drive.uploader import DriveUploader
        print("\nUploading to Google Drive ...")
        uploader = DriveUploader(root_folder_id=args.drive_folder_id)
        uploader.upload_entries(entries, verbose=True)
        print("Upload complete.")

    if not args.output and not args.drive:
        print("\nNothing to do — add --output <dir> or --drive to export.")


if __name__ == "__main__":
    main()
