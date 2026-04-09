import re
from email_reply_parser import EmailReplyParser


# Leftover noise: image placeholders, tracking codes, raw angle-bracket URLs
_NOISE = re.compile(
    r"\[image:[^\]]*\]"              # [image: foo]
    r"|#[A-Z0-9]{4,}-[A-Z0-9]{2,}"  # tracking codes: #CC65-CZC
    r"|<https?://\S+>"               # <https://...> from sig blocks
    r"|Sent via \w[\w\s]*\([^)]*\)"   # "Sent via Superhuman iOS (https://...)"
    r"|\[.{20,}?==\]"               # base64 image blobs: [g9OQ...==]
    r"|\[Image\]"                    # placeholder after blob cleanup
    ,
    re.IGNORECASE,
)

# Inline quoted blocks not caught by email-reply-parser (no > markers).
# Matches the attribution line AND everything after it to end of body.
_INLINE_QUOTE = re.compile(
    r"\n+(?:"
    r"(?:Von|From):[ \t][^\n]+\n[ \t]*(?:Sent|Date|An|To):[ \t]"  # Outlook "From:\nSent:"
    r"|Von:[ \t]"                                                    # German short form
    r"|On\s[\s\S]{0,300}?wrote\s*:"                                  # "On ... wrote:"
    r"|[^\n]{0,80}@[^\n]{0,80}\s+wrote\s*:"                          # "email wrote:"
    r"|[^\n]{0,80}(?:пише|écrit|schrieb|escribió|написав)\s*:?"      # other langs
    r")[\s\S]*$",
    re.IGNORECASE,
)

# Signals that a line is part of a contact/signature block
_SIG_SIGNALS = [
    re.compile(r"\+\d[\d\s\-()]{7,}"),            # phone: +49 170 2600122
    re.compile(r"mailto:", re.IGNORECASE),          # mailto: links
    re.compile(r"tel:", re.IGNORECASE),             # tel: links
    re.compile(r"Sent via \w+", re.IGNORECASE),     # "Sent via Superhuman"
    re.compile(r"www\.\S+\.\S+"),                   # www.company.com
    re.compile(r"HRB \d+", re.IGNORECASE),          # German company register
    re.compile(r"Geschäftsführer:", re.IGNORECASE),  # German managing director
    re.compile(r"Registered Office:", re.IGNORECASE),
    re.compile(r"Registry Court:", re.IGNORECASE),
    re.compile(r"Managing Director", re.IGNORECASE),
]

# Minimum signal lines to consider it a signature block
_SIG_THRESHOLD = 2


def _is_sig_line(line: str) -> bool:
    """Check if a single line looks like part of a contact/signature block."""
    return any(sig.search(line) for sig in _SIG_SIGNALS)


def _strip_contact_block(text: str) -> str:
    """Detect and strip trailing contact/signature blocks without `-- ` delimiters."""
    lines = text.split("\n")

    # Walk up from the bottom. A sig block is a contiguous run of sig-like lines
    # and short filler lines (blanks, names, titles) anchored at the bottom.
    # Stop when we hit a "content" line — long text with no sig signals.
    cut = len(lines)
    sig_hits = 0

    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].strip()
        if not stripped:
            # blank line — keep walking
            continue
        if _is_sig_line(stripped):
            sig_hits += 1
            cut = i
        elif len(stripped) < 50:
            # Short line (name, title, company) — tentatively include
            cut = i
        else:
            # Long content line without sig signals — stop
            break

    if sig_hits >= _SIG_THRESHOLD:
        return "\n".join(lines[:cut])

    return text


def clean_body(text: str | None) -> str | None:
    if not text:
        return None
    # Normalize line endings; ensure blank line before `-- ` for parser
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n(-- )", r"\n\n\1", text)
    # Drop quoted replies and `-- ` signature blocks
    parsed = EmailReplyParser.read(text)
    visible = [f.content for f in parsed.fragments if not f.quoted and not f.signature]
    text = "\n".join(visible)
    # Strip inline quote blocks (attribution + everything after)
    text = _INLINE_QUOTE.sub("", text)
    # Strip contact/signature blocks without `-- ` delimiter
    text = _strip_contact_block(text)
    # Remove leftover noise
    text = _NOISE.sub("", text)
    # Collapse extra blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() or None
