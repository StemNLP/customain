from pathlib import Path
import base64
import email
import mailbox

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
CREDS_FILE = "credentials.json"
TOKEN_FILE = "token.json"
MBOX_FILE = "primary_export.mbox"


def get_service():
    creds = None

    if Path(TOKEN_FILE).exists():
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)

        Path(TOKEN_FILE).write_text(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def iter_primary_message_ids(service):
    page_token = None

    while True:
        response = service.users().messages().list(
            userId="me",
            q="category:primary",
            maxResults=500,
            pageToken=page_token,
        ).execute()

        for message in response.get("messages", []):
            yield message["id"]

        page_token = response.get("nextPageToken")
        if not page_token:
            break


def export_primary_to_mbox(service):
    mbox = mailbox.mbox(MBOX_FILE)
    count = 0

    for message_id in iter_primary_message_ids(service):
        raw_message = service.users().messages().get(
            userId="me",
            id=message_id,
            format="raw",
        ).execute()

        raw_bytes = base64.urlsafe_b64decode(raw_message["raw"].encode("utf-8"))
        msg = email.message_from_bytes(raw_bytes)
        mbox.add(msg)

        count += 1
        if count % 100 == 0:
            mbox.flush()
            print(f"Exported {count} messages...")

    mbox.flush()
    mbox.close()
    print(f"Done. Exported {count} messages to {MBOX_FILE}")


def main():
    service = get_service()
    export_primary_to_mbox(service)


if __name__ == "__main__":
    main()