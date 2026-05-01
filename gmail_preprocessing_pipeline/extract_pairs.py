#!/usr/bin/env python
"""Extract email-reply pairs from an mbox file.

Usage:
    uv run python gmail_preprocessing_pipeline/extract_pairs.py
    uv run python gmail_preprocessing_pipeline/extract_pairs.py --input data/foo.mbox --output data/foo_pairs.jsonl
"""

import argparse
import json
import mailbox
from dataclasses import asdict, dataclass
from email.header import decode_header
from email.utils import parseaddr, parsedate_to_datetime
from pathlib import Path
from typing import Callable, Iterator, Optional

MY_EMAIL = "meghdad@calibrion.ai"

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class EmailRecord:
    message_id: Optional[str]
    thread_id: Optional[str]
    date: Optional[str]
    subject: Optional[str]
    from_email: Optional[str]
    to_email: Optional[str]
    body_text: Optional[str]
    in_reply_to: Optional[str]


@dataclass
class ReplyPair:
    subject: Optional[str]
    received_body: Optional[str]
    reply_body: Optional[str]

    def to_dict(self) -> dict:
        return asdict(self)

# ---------------------------------------------------------------------------
# Header parsing helpers
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


FIELD_MAP: dict[str, tuple[str | None, Callable]] = {
    "message_id":  ("Message-Id",  lambda m, h: m.get(h)),
    "thread_id":   ("X-GM-THRID",  lambda m, h: m.get(h)),
    "date":        ("Date",        lambda m, h: _parse_date(m.get(h))),
    "subject":     ("Subject",     lambda m, h: _decode_str(m.get(h))),
    "from_email":  ("From",        lambda m, h: _addr_email(m.get(h))),
    "to_email":    ("To",          lambda m, h: _addr_email(m.get(h))),
    "body_text":   (None,          lambda m, _: _get_body(m)),
    "in_reply_to": ("In-Reply-To", lambda m, h: m.get(h)),
}

# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _to_record(msg: mailbox.mboxMessage) -> EmailRecord:
    kwargs = {field: fn(msg, header) for field, (header, fn) in FIELD_MAP.items()}
    return EmailRecord(**kwargs)


def iter_reply_pairs(mbox_path: Path) -> Iterator[ReplyPair]:
    """Yield (received, reply) pairs where the reply was sent by MY_EMAIL."""
    by_message_id: dict[str, EmailRecord] = {}
    replies: list[EmailRecord] = []

    for msg in mailbox.mbox(str(mbox_path)):
        record = _to_record(msg)
        if record.message_id:
            by_message_id[record.message_id.strip()] = record
        if record.from_email == MY_EMAIL and record.in_reply_to:
            replies.append(record)

    for reply in replies:
        received = by_message_id.get(reply.in_reply_to.strip())
        if received and received.from_email != MY_EMAIL and received.body_text and reply.body_text:
            yield ReplyPair(
                subject=received.subject,
                received_body=received.body_text,
                reply_body=reply.body_text,
            )

# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def process_file(input_path: Path, output_path: Path) -> None:
    print(f"Extracting reply pairs from {input_path} ...")
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for pair in iter_reply_pairs(input_path):
            f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    print(f"Done. {count} reply pairs -> {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/_intermediate/new_threads.mbox"))
    parser.add_argument("--output", type=Path, default=Path("data/_intermediate/reply_pairs_raw.jsonl"))
    args = parser.parse_args()
    process_file(args.input, args.output)


if __name__ == "__main__":
    main()
