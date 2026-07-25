# Google Sheets Setup for Hermes Agent (Headless / Server-Side)

## Problem Diagnosis

The service account (`hermes@protean-theater-497013-j9.iam.gserviceaccount.com`) cannot **create** spreadsheets. We confirmed this empirically:

- **Drive API**: `403 storageQuotaExceeded` — "The user's Drive storage quota has been exceeded"
- **Sheets API create**: `403` — "The caller does not have permission"
- **Drive about().get()**: `storageQuota.limit = "0"` — **zero bytes** allocated
- **Can create folders**: Yes (folders don't consume storage quota)
- **Cannot create**: Spreadsheets, forms, or any file (zero-byte quota)
- **Cannot create shared drives**: `403` — "The authenticated user cannot create new shared drives"

### Root Cause

**Service accounts on consumer (non-Workspace) Google Cloud projects get 0 bytes of Drive storage by default.** Google does not provide free Drive storage to service accounts. This is by design — service accounts are meant to *access* data shared with them, not to *own* files.

There is no setting to change, no API to call, and no billing toggle that fixes this. The quota is structurally zero for consumer-project service accounts.

---

## Solution: Two-Phase Authentication

### Phase 1 — One-Time Setup: OAuth User Token (for creating spreadsheets)

Use the **OAuth desktop client** (already exists in this project) to perform a **one-time** browser-based authorization. This produces a `refresh_token` that can create spreadsheets indefinitely (refresh tokens don't expire unless revoked).

**The OAuth client secret already exists:**
- File: `~/.hermes/workspace/bookclub-podcast/client_secret_455087558752-fn8023f9ehcpugm2qlv07aei4n2vo1r6.apps.googleusercontent.com.json`
- Project: `protean-theater-497013-j9` (same project as the service account)
- Type: `installed` (desktop OAuth client)

**Steps:**

1. Copy the client secret to the profile's secrets directory:
```bash
cp ~/.hermes/workspace/bookclub-podcast/client_secret_455087558752-fn8023f9ehcpugm2qlv07aei4n2vo1r6.apps.googleusercontent.com.json \
   ~/.hermes/profiles/oracle/.secrets/google_client_secret.json
```

2. Run the one-time OAuth flow (requires a browser, done once):
```bash
cd ~/.hermes/hermes-agent/venv && source bin/activate

python3 -c "
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive',
]

flow = InstalledAppFlow.from_client_secrets_file(
    '/Users/clawdio/.hermes/profiles/oracle/.secrets/google_client_secret.json',
    scopes=SCOPES,
)
creds = flow.run_local_server(port=0, access_type='offline', prompt='consent')

import json
with open('/Users/clawdio/.hermes/profiles/oracle/.secrets/google_oauth_token.json', 'w') as f:
    f.write(creds.to_json())
print('OAuth token saved. Refresh token present:', bool(creds.refresh_token))
"
```

This opens a browser window, asks you to log in with a Google account that has Drive storage, and saves a refresh token. **This is the only step that requires a browser.** The refresh token works indefinitely from a headless server.

3. Ensure Google Sheets API and Drive API are enabled on the project (already confirmed enabled for `protean-theater-497013-j9`).

### Phase 2 — Runtime: Use Service Account for Read/Write

Once a spreadsheet exists (created via the OAuth token or manually in Google Sheets web UI), share it with the service account email and use the service account for all ongoing read/write operations. The service account needs no browser, no OAuth dance, and never expires.

**Share the spreadsheet with:**
```
hermes@protean-theater-497013-j9.iam.gserviceaccount.com
```
Give it **Editor** access.

---

## Working Python Code

### Complete Solution: Create + Share + Write

```python
#!/usr/bin/env python3
"""Google Sheets helper for Hermes Agent (Oracle profile).

Two-mode operation:
  create  — uses OAuth user token (needs refresh token from one-time browser auth)
  read/write — uses service account (headless, no browser, never expires)

Usage:
  python3 google_sheets_helper.py create "My Spreadsheet"
  python3 google_sheets_helper.py write SPREADSHEET_ID "A1:B2" '[["a","b"],["c","d"]]'
  python3 google_sheets_helper.py read SPREADSHEET_ID "A1:Z100"
  python3 google_sheets_helper.py share SPREADSHEET_ID
"""

import json
import os
import sys
from pathlib import Path

# --- Paths ---
PROFILE_DIR = Path.home() / ".hermes" / "profiles" / "oracle"
SECRETS_DIR = PROFILE_DIR / ".secrets"
SA_KEY_PATH = SECRETS_DIR / "google-service-account.json"
OAUTH_CLIENT_SECRET_PATH = SECRETS_DIR / "google_client_secret.json"
OAUTH_TOKEN_PATH = SECRETS_DIR / "google_oauth_token.json"

SA_EMAIL = "hermes@protean-theater-497013-j9.iam.gserviceaccount.com"

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


def get_oauth_credentials():
    """Load OAuth user credentials (has Drive storage, can create files)."""
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request

    if not OAUTH_TOKEN_PATH.exists():
        print(f"ERROR: OAuth token not found at {OAUTH_TOKEN_PATH}", file=sys.stderr)
        print("Run the one-time OAuth flow first. See google-sheets-setup.md", file=sys.stderr)
        sys.exit(1)

    creds = Credentials.from_authorized_user_file(str(OAUTH_TOKEN_PATH), SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        OAUTH_TOKEN_PATH.write_text(
            json.dumps(json.loads(creds.to_json()), indent=2)
        )
    if not creds.valid:
        print("ERROR: OAuth token is invalid. Re-run the OAuth flow.", file=sys.stderr)
        sys.exit(1)
    return creds


def get_sa_credentials():
    """Load service account credentials (headless, for read/write on shared sheets)."""
    from google.oauth2.service_account import Credentials

    if not SA_KEY_PATH.exists():
        print(f"ERROR: Service account key not found at {SA_KEY_PATH}", file=sys.stderr)
        sys.exit(1)

    return Credentials.from_service_account_file(str(SA_KEY_PATH), scopes=SCOPES)


def create_spreadsheet(title: str) -> dict:
    """Create a spreadsheet using OAuth user token (has Drive storage).
    
    Then shares it with the service account so future reads/writes
    can use the service account (headless, no browser).
    """
    from googleapiclient.discovery import build

    creds = get_oauth_credentials()
    sheets_service = build("sheets", "v4", credentials=creds)
    
    spreadsheet = {
        "properties": {"title": title},
        "sheets": [{"properties": {"title": "Sheet1"}}]
    }
    result = sheets_service.spreadsheets().create(
        body=spreadsheet,
        fields="spreadsheetId,spreadsheetUrl,properties.title"
    ).execute()
    
    sheet_id = result["spreadsheetId"]
    sheet_url = result["spreadsheetUrl"]
    
    # Share with service account so it can read/write headlessly
    drive_service = build("drive", "v3", credentials=creds)
    drive_service.permissions().create(
        fileId=sheet_id,
        body={
            "type": "user",
            "role": "writer",
            "emailAddress": SA_EMAIL,
        },
        sendNotificationEmail=False,
    ).execute()
    
    return {
        "status": "created",
        "spreadsheetId": sheet_id,
        "spreadsheetUrl": sheet_url,
        "title": result["properties"]["title"],
        "sharedWith": SA_EMAIL,
    }


def share_spreadsheet(sheet_id: str) -> dict:
    """Share an existing spreadsheet with the service account (using OAuth)."""
    from googleapiclient.discovery import build

    creds = get_oauth_credentials()
    drive_service = build("drive", "v3", credentials=creds)
    
    drive_service.permissions().create(
        fileId=sheet_id,
        body={
            "type": "user",
            "role": "writer",
            "emailAddress": SA_EMAIL,
        },
        sendNotificationEmail=False,
    ).execute()
    
    return {
        "status": "shared",
        "spreadsheetId": sheet_id,
        "sharedWith": SA_EMAIL,
        "role": "writer",
    }


def write_data(sheet_id: str, range_name: str, values_json: str) -> dict:
    """Write data using service account (headless, no browser needed)."""
    from googleapiclient.discovery import build

    creds = get_sa_credentials()
    service = build("sheets", "v4", credentials=creds)
    
    values = json.loads(values_json)
    body = {"values": values}
    
    result = service.spreadsheets().values().update(
        spreadsheetId=sheet_id,
        range=range_name,
        valueInputOption="USER_ENTERED",
        body=body,
    ).execute()
    
    return {
        "status": "updated",
        "updatedCells": result.get("updatedCells", 0),
        "updatedRange": result.get("updatedRange", ""),
    }


def read_data(sheet_id: str, range_name: str) -> list:
    """Read data using service account (headless)."""
    from googleapiclient.discovery import build

    creds = get_sa_credentials()
    service = build("sheets", "v4", credentials=creds)
    
    result = service.spreadsheets().values().get(
        spreadsheetId=sheet_id,
        range=range_name,
    ).execute()
    
    return result.get("values", [])


def append_rows(sheet_id: str, range_name: str, values_json: str) -> dict:
    """Append rows using service account (headless)."""
    from googleapiclient.discovery import build

    creds = get_sa_credentials()
    service = build("sheets", "v4", credentials=creds)
    
    values = json.loads(values_json)
    body = {"values": values}
    
    result = service.spreadsheets().values().append(
        spreadsheetId=sheet_id,
        range=range_name,
        valueInputOption="USER_ENTERED",
        insertDataOption="INSERT_ROWS",
        body=body,
    ).execute()
    
    return {
        "status": "appended",
        "updatedCells": result.get("updates", {}).get("updatedCells", 0),
    }


# --- gspread alternative (simpler API, same auth) ---

def gspread_example(sheet_id: str):
    """Example using gspread library with service account for read/write."""
    import gspread
    from google.oauth2.service_account import Credentials
    
    creds = Credentials.from_service_account_file(
        str(SA_KEY_PATH),
        scopes=SCOPES,
    )
    gc = gspread.authorize(creds)
    
    # Open by ID (most reliable)
    sh = gc.open_by_key(sheet_id)
    ws = sh.sheet1
    
    # Write
    ws.update("A1:C3", [
        ["Name", "Status", "Date"],
        ["Task 1", "Done", "2026-07-14"],
        ["Task 2", "Pending", "2026-07-15"],
    ])
    
    # Read
    data = ws.get_all_values()
    return data


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    if cmd == "create":
        title = sys.argv[2]
        print(json.dumps(create_spreadsheet(title), indent=2))
    
    elif cmd == "share":
        sheet_id = sys.argv[2]
        print(json.dumps(share_spreadsheet(sheet_id), indent=2))
    
    elif cmd == "write":
        sheet_id = sys.argv[2]
        range_name = sys.argv[3]
        values_json = sys.argv[4]
        print(json.dumps(write_data(sheet_id, range_name, values_json), indent=2))
    
    elif cmd == "read":
        sheet_id = sys.argv[2]
        range_name = sys.argv[3]
        data = read_data(sheet_id, range_name)
        print(json.dumps(data, indent=2, ensure_ascii=False))
    
    elif cmd == "append":
        sheet_id = sys.argv[2]
        range_name = sys.argv[3]
        values_json = sys.argv[4]
        print(json.dumps(append_rows(sheet_id, range_name, values_json), indent=2))
    
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
```

---

## Quick Start (Step by Step)

### Step 1: One-time OAuth flow (requires browser, done once)

```bash
# Copy the existing OAuth client secret to the oracle profile
cp ~/.hermes/workspace/bookclub-podcast/client_secret_455087558752-fn8023f9ehcpugm2qlv07aei4n2vo1r6.apps.googleusercontent.com.json \
   ~/.hermes/profiles/oracle/.secrets/google_client_secret.json

# Run the one-time OAuth flow
cd ~/.hermes/hermes-agent/venv && source bin/activate
python3 -c "
from google_auth_oauthlib.flow import InstalledAppFlow
import json

SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']

flow = InstalledAppFlow.from_client_secrets_file(
    '/Users/clawdio/.hermes/profiles/oracle/.secrets/google_client_secret.json',
    scopes=SCOPES,
)
creds = flow.run_local_server(port=0, access_type='offline', prompt='consent')

with open('/Users/clawdio/.hermes/profiles/oracle/.secrets/google_oauth_token.json', 'w') as f:
    f.write(creds.to_json())
print('OAuth token saved. Refresh token:', bool(creds.refresh_token))
"
```

### Step 2: Create a spreadsheet (uses OAuth token, auto-shares with SA)

```bash
python3 google_sheets_helper.py create "Spark LLM Optimization Results"
# Output: {"spreadsheetId": "1abc...", "spreadsheetUrl": "https://...", "sharedWith": "hermes@..."}
```

### Step 3: Write data (uses service account — headless, no browser)

```bash
python3 google_sheets_helper.py write SPREADSHEET_ID "A1:C3" '[["Model","Speed","Quality"],["Qwen-122B","31 tok/s","High"],["DS4","26 tok/s","Medium"]]'
```

### Step 4: Read data (uses service account — headless)

```bash
python3 google_sheets_helper.py read SPREADSHEET_ID "A1:C3"
```

### Alternative: Manual spreadsheet creation

If you don't want to use the OAuth flow at all:
1. Go to [sheets.google.com](https://sheets.google.com) in a browser
2. Create a new spreadsheet manually
3. Click **Share** → add `hermes@protean-theater-497013-j9.iam.gserviceaccount.com` as **Editor**
4. Copy the spreadsheet ID from the URL
5. Use `write` and `read` commands with that ID (service account, headless)

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────┐
│                    ONE-TIME SETUP                           │
│                                                             │
│  Browser ──► OAuth Flow ──► google_oauth_token.json         │
│  (human)     (client_secret)   (refresh token, persists)    │
│                                                             │
│  OAuth Token ──► Create Spreadsheet ──► Share with SA email │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    RUNTIME (headless)                       │
│                                                             │
│  Service Account JSON ──► gspread / google-api-python-client│
│  (no browser, no expiry)   ──► Read / Write / Append        │
│                             ──► Sheet is shared with SA     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Why Not Just Use OAuth for Everything?

You could — the OAuth refresh token works headlessly after the initial browser auth. But:

1. **Service accounts are more robust for automation** — no token refresh logic needed, no expiry risk, no user account dependency
2. **Separation of concerns** — OAuth creates (needs Drive storage), SA reads/writes (needs only Sheets API access)
3. **The service account is already set up** — key exists, APIs enabled, gspread installed
4. **If the OAuth token is revoked** (e.g. user changes password), the SA still works for read/write on already-shared sheets

## Why Not Use a Shared Drive?

Shared drives bypass the per-user storage quota, but:
- Service accounts **cannot create** shared drives (`403 userCannotCreateTeamDrives`)
- A human would need to create a shared drive first, then add the SA as a member
- This is more complex than simply sharing a single spreadsheet
- Only worth it if you need many spreadsheets organized in a shared drive

## Key Files

| File | Purpose | Required? |
|------|---------|-----------|
| `~/.hermes/profiles/oracle/.secrets/google-service-account.json` | SA key for headless read/write | Yes |
| `~/.hermes/profiles/oracle/.secrets/google_client_secret.json` | OAuth client for one-time auth | For create only |
| `~/.hermes/profiles/oracle/.secrets/google_oauth_token.json` | OAuth refresh token | For create only |

## Existing Skill

The `gspread-sheets` skill (installed from Hermes skills hub) documents the gspread API patterns. It notes: "Service account has its own Drive; share sheets with its email to access." This is correct — the skill assumes the spreadsheet already exists and is shared.

The `cloud-productivity-suite` skill has a `google_api.py` script at:
`~/.hermes/profiles/oracle/skills/productivity/cloud-productivity-suite/scripts/google-workspace/google_api.py`

This script uses OAuth tokens (not service accounts) for all operations including `sheets create`. It expects:
- `google_client_secret.json` at `~/.hermes/` (or profile home)
- `google_token.json` at `~/.hermes/` (or profile home)

It supports: `sheets create`, `sheets get`, `sheets update`, `sheets append`, `drive share`, and more. If you set up the OAuth token (Step 1 above), you can use this existing script directly:

```bash
# Create a sheet
python3 google_api.py sheets create --title "My Sheet"

# Share it with the service account
python3 google_api.py drive share FILE_ID --role writer --type user --email hermes@protean-theater-497013-j9.iam.gserviceaccount.com

# Then use gspread/service account for ongoing headless read/write
```

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `storageQuotaExceeded` | SA has 0 bytes Drive quota | Use OAuth to create, then share with SA |
| `The caller does not have permission` (Sheets create) | Same quota issue, Sheets API also blocked | Same fix |
| `404 Requested entity was not found` (Sheets get) | Sheet doesn't exist or isn't shared with SA | Share the sheet with SA email |
| `403 invalid_client` (OAuth) | Client secret mismatch | Ensure client_secret.json is from the right project |
| Token expired (OAuth) | Access token expired (normal) | Refresh token auto-refreshes; if refresh token revoked, re-run Step 1 |