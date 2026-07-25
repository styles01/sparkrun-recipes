#!/usr/bin/env python3
"""Google Sheets helper for Hermes Agent (Oracle profile).

Two-mode operation:
  create  — uses OAuth user token (needs refresh token from one-time browser auth)
  read/write — uses service account (headless, no browser, never expires)

Usage:
  python3 google_sheets_helper.py create "My Spreadsheet"
  python3 google_sheets_helper.py write SPREADSHEET_ID "A1:B2" '[["a","b"],["c","d"]]'
  python3 google_sheets_helper.py read SPREADSHEET_ID "A1:Z100"
  python3 google_sheets_helper.py append SPREADSHEET_ID "A1" '[["new","row"]]'
  python3 google_sheets_helper.py share SPREADSHEET_ID

See: research/google-sheets-setup.md for full documentation.
"""

import json
import sys
from pathlib import Path

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
    """Create a spreadsheet using OAuth user token, then share with service account."""
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

    # Share with service account for headless read/write
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
        "spreadsheetUrl": result["spreadsheetUrl"],
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
    """Write data using service account (headless)."""
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