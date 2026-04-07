import re
from email_reply_parser import EmailReplyParser


# Leftover noise: image placeholders, tracking codes, raw angle-bracket URLs
_NOISE = re.compile(
    r"\[image:[^\]]*\]"              # [image: foo]
    r"|#[A-Z0-9]{4,}-[A-Z0-9]{2,}"  # tracking codes: #CC65-CZC
    r"|<https?://\S+>"               # <https://...> from sig blocks
    ,
    re.IGNORECASE,
)

# Inline quoted blocks not caught by email-reply-parser (no > markers).
# Matches the attribution line AND everything after it to end of body.
#   "Von: Name"          — German/Austrian Outlook attribution
#   "On <date> X wrote:" — standard, may wrap across lines
#   "email@domain wrote:"— bare attribution without "On"
#   Cyrillic/French/etc  — single-line variants
_INLINE_QUOTE = re.compile(
    r"\n+(?:"
    r"(?:Von|From):[ \t][^\n]+\n[ \t]*(?:Sent|Date|An|To):[ \t]"  # Outlook "From:\nSent:" (any lang)
    r"|Von:[ \t]"                                                    # German short form
    r"|On\s[\s\S]{0,300}?wrote\s*:"                                  # "On ... wrote:"
    r"|[^\n]{0,80}@[^\n]{0,80}\s+wrote\s*:"                          # "email wrote:"
    r"|[^\n]{0,80}(?:пише|écrit|schrieb|escribió|написав)\s*:?"      # other langs
    r")[\s\S]*$",
    re.IGNORECASE,
)


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
    # Remove leftover noise
    text = _NOISE.sub("", text)
    # Collapse extra blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() or None
