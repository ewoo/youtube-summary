"""
Upload journal entries to Google Drive.

Setup (one-time):
  1. Go to https://console.cloud.google.com/
  2. Create a project → Enable "Google Drive API"
  3. OAuth 2.0 → Desktop app → Download credentials.json
  4. Place credentials.json in journal_export/credentials/
  5. First run will open a browser for authorization; token.json is saved for reuse.

Folder structure created on Drive:
  Journals/
    DayOne/
      2023/
        01/
          2023-01-15_10-30-00_Entry-title.md
    Bear/
      2023/
        ...
"""
import io
import os
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

from ..models import JournalEntry

SCOPES = ["https://www.googleapis.com/auth/drive.file"]
CREDENTIALS_DIR = Path(__file__).parent.parent / "credentials"


def _get_credentials() -> Credentials:
    token_path = CREDENTIALS_DIR / "token.json"
    creds_path = CREDENTIALS_DIR / "credentials.json"

    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not creds_path.exists():
                raise FileNotFoundError(
                    f"credentials.json not found at {creds_path}\n"
                    "Download it from Google Cloud Console → APIs & Services → Credentials."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), SCOPES)
            creds = flow.run_local_server(port=0)
        token_path.write_text(creds.to_json())

    return creds


class DriveUploader:
    def __init__(self, root_folder_id: str | None = None):
        """
        root_folder_id: Google Drive folder ID to use as the upload root.
                        If None, uploads to My Drive root.
        """
        creds = _get_credentials()
        self._service = build("drive", "v3", credentials=creds)
        self._root_id = root_folder_id or "root"
        self._folder_cache: dict[tuple, str] = {}  # path tuple → folder ID

    def _get_or_create_folder(self, name: str, parent_id: str) -> str:
        """Return Drive folder ID, creating it if it doesn't exist."""
        query = (
            f"name={name!r} and mimeType='application/vnd.google-apps.folder' "
            f"and '{parent_id}' in parents and trashed=false"
        )
        results = self._service.files().list(q=query, fields="files(id)").execute()
        files = results.get("files", [])
        if files:
            return files[0]["id"]

        metadata = {
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_id],
        }
        folder = self._service.files().create(body=metadata, fields="id").execute()
        return folder["id"]

    def _resolve_folder_path(self, path_parts: list[str]) -> str:
        """Walk/create the full folder path, return the leaf folder ID."""
        current_id = self._root_id
        for i, part in enumerate(path_parts):
            cache_key = tuple(path_parts[: i + 1])
            if cache_key in self._folder_cache:
                current_id = self._folder_cache[cache_key]
            else:
                current_id = self._get_or_create_folder(part, current_id)
                self._folder_cache[cache_key] = current_id
        return current_id

    def upload_entry(self, entry: JournalEntry) -> str:
        """Upload one entry. Returns the Drive file ID."""
        folder_id = self._resolve_folder_path(entry.drive_folder_path())
        content = entry.to_markdown().encode("utf-8")
        media = MediaIoBaseUpload(io.BytesIO(content), mimetype="text/plain")
        metadata = {"name": entry.filename(), "parents": [folder_id]}

        # Avoid duplicates: update if file with same name already exists
        query = f"name={entry.filename()!r} and '{folder_id}' in parents and trashed=false"
        existing = self._service.files().list(q=query, fields="files(id)").execute().get("files", [])

        if existing:
            file_id = existing[0]["id"]
            self._service.files().update(fileId=file_id, media_body=media).execute()
        else:
            file = self._service.files().create(
                body=metadata, media_body=media, fields="id"
            ).execute()
            file_id = file["id"]

        return file_id

    def upload_entries(self, entries: list[JournalEntry], verbose: bool = True) -> list[str]:
        """Upload a list of entries, return list of file IDs."""
        ids = []
        for i, entry in enumerate(entries, 1):
            file_id = self.upload_entry(entry)
            ids.append(file_id)
            if verbose:
                print(f"  [{i}/{len(entries)}] {entry.filename()}")
        return ids
