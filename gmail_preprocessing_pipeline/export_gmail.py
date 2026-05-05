import argparse
import base64
import email
import mailbox
from datetime import datetime
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
BASE_DIR = Path(__file__).parent.parent
CREDS_FILE = str(BASE_DIR / ".secrets" / "credentials.json")
TOKEN_FILE = str(BASE_DIR / ".secrets" / "token.json")
EXPORTS_DIR = BASE_DIR / "data" / "exports"


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


def _build_query(base_query: str | None = None, newer_than_days: int | None = None) -> str:
    parts = ["from:me in:sent"]
    if newer_than_days is not None:
        parts.append(f"newer_than:{newer_than_days}d")
    if base_query:
        parts.append(base_query)
    return " ".join(parts)


def iter_replied_thread_ids(
    service,
    base_query: str | None = None,
    newer_than_days: int | None = None,
    max_threads: int | None = None,
):
    """Find thread IDs where the user sent a reply."""
    page_token = None
    seen = set()
    page = 0
    query = _build_query(base_query, newer_than_days)

    print("Discovering threads with replies...")
    print(f"  Query: {query}")
    while True:
        page += 1
        response = service.users().messages().list(
            userId="me",
            q=query,
            maxResults=500,
            pageToken=page_token,
        ).execute()

        for msg in response.get("messages", []):
            tid = msg["threadId"]
            if tid in seen:
                continue

            seen.add(tid)
            yield tid
            if max_threads is not None and len(seen) >= max_threads:
                print(f"Reached max thread limit ({max_threads}).")
                print(f"Found {len(seen)} threads total.")
                return

        print(f"  page {page}: {len(seen)} unique threads so far")
        page_token = response.get("nextPageToken")
        if not page_token:
            break

    print(f"Found {len(seen)} threads total.")


def export_replied_threads(
    service,
    gmail_query: str | None = None,
    newer_than_days: int | None = None,
    max_threads: int | None = None,
) -> Path:
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mbox_path = EXPORTS_DIR / f"new_threads_{timestamp}.mbox"
    mbox = mailbox.mbox(str(mbox_path))
    thread_count = 0
    msg_count = 0
    skipped = 0

    print(f"\nExporting threads to {mbox_path}...")
    for thread_id in iter_replied_thread_ids(
        service,
        base_query=gmail_query,
        newer_than_days=newer_than_days,
        max_threads=max_threads,
    ):
        thread = service.users().threads().get(
            userId="me",
            id=thread_id,
            format="full",
        ).execute()

        messages = thread.get("messages", [])
        if len(messages) < 2:
            skipped += 1
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
        if thread_count % 10 == 0:
            mbox.flush()
            print(f"  {thread_count} threads ({msg_count} messages), {skipped} skipped...")

    mbox.flush()
    mbox.close()
    print(f"\nDone. {thread_count} threads, {msg_count} messages -> {mbox_path}")
    print(f"  ({skipped} single-message threads skipped)")
    return mbox_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export replied Gmail threads to an mbox file")
    parser.add_argument("--gmail-query", type=str, default=None)
    parser.add_argument("--newer-than-days", type=int, default=None)
    parser.add_argument("--max-threads", type=int, default=None)
    args = parser.parse_args()

    service = get_service()
    export_replied_threads(
        service,
        gmail_query=args.gmail_query,
        newer_than_days=args.newer_than_days,
        max_threads=args.max_threads,
    )


if __name__ == "__main__":
    main()