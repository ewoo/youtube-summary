# journal-to-drive

Export **DayOne** and **Bear** journal entries to **Google Drive** as organized markdown files.

## Install

```bash
pip install -r journal_export/requirements.txt
```

## Google Drive setup (one-time)

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a project → **Enable** the "Google Drive API"
3. **APIs & Services → Credentials → Create → OAuth 2.0 Client ID → Desktop app**
4. Download `credentials.json` → place it in `journal_export/credentials/`
5. First run opens a browser for authorization; `token.json` is saved automatically.

## Usage

Run from the repo root as a module:

```bash
# Preview entries without uploading
python -m journal_export.export_journals --dayone ~/Downloads/Journal.zip --dry-run

# Write locally
python -m journal_export.export_journals \
    --dayone ~/Downloads/Journal.zip \
    --bear ~/Downloads/BearExport.zip \
    --output ./output

# Upload to Google Drive (My Drive root)
python -m journal_export.export_journals \
    --dayone ~/Downloads/Journal.zip \
    --bear ~/Downloads/BearExport.zip \
    --drive

# Upload to a specific Drive folder
python -m journal_export.export_journals \
    --dayone ~/Downloads/Journal.zip \
    --drive \
    --drive-folder-id 1AbCdEfGhIjKlMnOpQ
```

## Drive folder structure

```
Journals/
├── DayOne/
│   └── 2023/
│       └── 01/
│           └── 2023-01-15_10-30-00_My-entry-title.md
└── Bear/
    └── 2023/
        └── 03/
            └── 2023-03-20_14-00-00_Note-title.md
```

Each file has YAML frontmatter:

```markdown
---
date: 2023-01-15 10:30:00
title: 'My entry title'
tags: ["personal", "travel"]
source: dayone
---

Entry content here...
```

## Exporting from DayOne

**DayOne → File → Export → JSON** — produces a `.zip` with `Journal.json` + `photos/`.

## Exporting from Bear

**Bear → File → Export Notes → Markdown** — produces a folder of `.md` files (or zip it yourself).
