import json
import mailbox
from email.header import decode_header
from email.utils import parseaddr, parsedate_to_datetime
from pathlib import Path
from typing import Callable, Iterator

from .cleaner import clean_body
from .models import EmailRecord, ReplyPair

MY_EMAIL = "meghdad@calibrion.ai"

# ---------------------------------------------------------------------------
# Header transforms
# ---------------------------------------------------------------------------

def _decode_str(value: str | None) -> str | None:
    if not value:
        return None
    parts = decode_header(value)
    decoded = []
    for part, charset in parts:
        if isinstance(part, bytes):
            decoded.append(part.decode(charset or "utf-8", errors="replace"))
        else:
            decoded.append(part)
    return "".join(decoded) or None


def _addr_name(value: str | None) -> str | None:
    if not value:
        return None
    name, _ = parseaddr(_decode_str(value))
    return name or None


def _addr_email(value: str | None) -> str | None:
    if not value:
        return None
    _, addr = parseaddr(_decode_str(value))
    return addr.lower() or None


def _parse_date(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return parsedate_to_datetime(value).isoformat()
    except Exception:
        return value


def _parse_labels(value: str | None) -> list[str]:
    if not value:
        return []
    return [label.strip() for label in value.split(",") if label.strip()]


def _decode_payload(part) -> str | None:
    payload = part.get_payload(decode=True)
    if payload is None:
        return None
    charset = part.get_content_charset() or "utf-8"
    return payload.decode(charset, errors="replace").strip() or None


def _get_body(msg: mailbox.mboxMessage) -> str | None:
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                return _decode_payload(part)
    elif msg.get_content_type() == "text/plain":
        return _decode_payload(msg)
    return None


# ---------------------------------------------------------------------------
# Declarative field map: EmailRecord field name -> (header, transform)
# Add a new field here + in EmailRecord and nothing else needs to change.
# ---------------------------------------------------------------------------

FIELD_MAP: dict[str, tuple[str | None, Callable]] = {
    "message_id":   ("Message-Id",      lambda m, h: m.get(h)),
    "thread_id":    ("X-GM-THRID",      lambda m, h: m.get(h)),
    "gmail_labels": ("X-Gmail-Labels",  lambda m, h: _parse_labels(m.get(h))),
    "date":         ("Date",            lambda m, h: _parse_date(m.get(h))),
    "subject":      ("Subject",         lambda m, h: _decode_str(m.get(h))),
    "from_name":    ("From",            lambda m, h: _addr_name(m.get(h))),
    "from_email":   ("From",            lambda m, h: _addr_email(m.get(h))),
    "to_name":      ("To",              lambda m, h: _addr_name(m.get(h))),
    "to_email":     ("To",              lambda m, h: _addr_email(m.get(h))),
    "body_text":    (None,              lambda m, _: _get_body(m)),
    "in_reply_to":  ("In-Reply-To",     lambda m, h: m.get(h)),
}


def _to_record(msg: mailbox.mboxMessage) -> EmailRecord:
    kwargs = {field: fn(msg, header) for field, (header, fn) in FIELD_MAP.items()}
    return EmailRecord(**kwargs)


# ---------------------------------------------------------------------------
# Public iterators
# ---------------------------------------------------------------------------

def iter_emails(mbox_path: Path) -> Iterator[EmailRecord]:
    """Yield all emails as EmailRecords."""
    for msg in mailbox.mbox(str(mbox_path)):
        yield _to_record(msg)


def iter_reply_pairs(mbox_path: Path) -> Iterator[ReplyPair]:
    """Yield (received, reply) pairs where the reply was sent by MY_EMAIL."""
    by_message_id: dict[str, EmailRecord] = {}
    replies: list[EmailRecord] = []

    for record in iter_emails(mbox_path):
        if record.message_id:
            by_message_id[record.message_id.strip()] = record
        if record.from_email == MY_EMAIL and record.in_reply_to:
            replies.append(record)

    for reply in replies:
        received = by_message_id.get(reply.in_reply_to.strip())
        if received and received.from_email != MY_EMAIL:
            yield ReplyPair(
                subject=received.subject,
                received_body=clean_body(received.body_text),
                reply_body=clean_body(reply.body_text),
            )


# ---------------------------------------------------------------------------
# Exporters
# ---------------------------------------------------------------------------

def export_jsonl(mbox_path: Path, output_path: Path) -> int:
    """Export all emails to JSONL."""
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for record in iter_emails(mbox_path):
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def export_reply_pairs_jsonl(mbox_path: Path, output_path: Path) -> int:
    """Export email-reply pairs to JSONL."""
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for pair in iter_reply_pairs(mbox_path):
            f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count
