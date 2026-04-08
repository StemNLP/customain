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
MBOX_FILE = "replied_threads.mbox"


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


def iter_replied_thread_ids(service):
    """Find thread IDs where the user sent a reply."""
    page_token = None
    seen = set()

    while True:
        response = service.users().messages().list(
            userId="me",
            q="from:me in:sent",
            maxResults=500,
            pageToken=page_token,
        ).execute()

        for msg in response.get("messages", []):
            tid = msg["threadId"]
            if tid not in seen:
                seen.add(tid)
                yield tid

        page_token = response.get("nextPageToken")
        if not page_token:
            break


def export_replied_threads(service):
    mbox = mailbox.mbox(MBOX_FILE)
    thread_count = 0
    msg_count = 0

    for thread_id in iter_replied_thread_ids(service):
        thread = service.users().threads().get(
            userId="me",
            id=thread_id,
            format="full",
        ).execute()

        # Only keep threads with at least 2 messages (a received + a reply)
        messages = thread.get("messages", [])
        if len(messages) < 2:
            continue

        for msg_resource in messages:
            raw_message = service.users().messages().get(
                userId="me",
                id=msg_resource["id"],
                format="raw",
            ).execute()
            raw_bytes = base64.urlsafe_b64decode(raw_message["raw"].encode("utf-8"))
            msg = email.message_from_bytes(raw_bytes)
            mbox.add(msg)
            msg_count += 1

        thread_count += 1
        if thread_count % 50 == 0:
            mbox.flush()
            print(f"Exported {thread_count} threads ({msg_count} messages)...")

    mbox.flush()
    mbox.close()
    print(f"Done. {thread_count} threads, {msg_count} messages -> {MBOX_FILE}")


def main():
    service = get_service()
    export_replied_threads(service)


if __name__ == "__main__":
    main()
