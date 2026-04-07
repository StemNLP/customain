from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class EmailRecord:
    message_id: Optional[str]
    thread_id: Optional[str]
    gmail_labels: list[str]
    date: Optional[str]          # ISO 8601
    subject: Optional[str]
    from_name: Optional[str]
    from_email: Optional[str]
    to_name: Optional[str]
    to_email: Optional[str]
    body_text: Optional[str]
    in_reply_to: Optional[str]   # Message-Id of parent

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReplyPair:
    subject: Optional[str]
    received_body: Optional[str]
    reply_body: Optional[str]

    def to_dict(self) -> dict:
        return asdict(self)
