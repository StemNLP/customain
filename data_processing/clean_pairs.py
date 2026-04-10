#!/usr/bin/env python
"""Clean reply pairs using an LLM to strip signatures and contact blocks.

Reads raw reply pairs JSONL, sends each body to Claude Haiku for cleaning,
writes cleaned pairs to a new JSONL file.

Usage:
    uv run python data_processing/clean_pairs.py
    uv run python data_processing/clean_pairs.py --input data/raw.jsonl --output data/clean.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import anthropic

MODEL = "claude-haiku-4-5-20251001"
TEST_MODE = True
TEST_LIMIT = 20

SYSTEM_PROMPT = """\
You are an email body cleaner. Your job is to remove signature blocks, \
contact information, and disclaimers from email bodies while preserving \
the actual message content EXACTLY as written.

Remove:
- Email signatures (name, title, company, phone, address blocks)
- Contact blocks (mailto: links, tel: links, phone numbers in sig format)
- Legal disclaimers and confidentiality notices
- "Sent via Superhuman" or similar client tags
- Inline image placeholders like [image: ...] or [cid:...]
- Tracking codes (e.g. #CC65-CZC)
- Social media links in signature blocks
- Company registration info (HRB numbers, registered office, etc.)

Preserve EXACTLY (do not modify, rephrase, or summarize):
- The entire message body text
- Greetings and sign-offs that are part of the message (e.g. "Best, Name")
- Replace ALL URLs (http/https links) with [LINK]
- All formatting, line breaks, and whitespace in the message body

Also remove quoted replies:
- Lines starting with >
- "On [date] [person] wrote:" blocks and everything after
- "Von: ... / Sent: ..." Outlook attribution blocks and everything after
- Other language variants of reply attribution

Return ONLY the cleaned email body text. No explanation, no wrapping."""


def clean_body(client: anthropic.Anthropic, text: str) -> str:
    """Send a single email body to the LLM for cleaning."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Clean this email body:\n\n{text}"}],
    )
    return response.content[0].text.strip()


def process_file(input_path: Path, output_path: Path):
    client = anthropic.Anthropic()
    lines = input_path.read_text(encoding="utf-8").strip().splitlines()
    if TEST_MODE:
        lines = lines[:TEST_LIMIT]
        print(f"TEST MODE: limited to {TEST_LIMIT} pairs")
    total = len(lines)
    print(f"Cleaning {total} pairs from {input_path} ...")

    with output_path.open("w", encoding="utf-8") as f:
        for i, line in enumerate(lines, 1):
            pair = json.loads(line)

            if pair.get("received_body"):
                pair["received_body"] = clean_body(client, pair["received_body"])
            if pair.get("reply_body"):
                pair["reply_body"] = clean_body(client, pair["reply_body"])

            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

            if i % 10 == 0 or i == total:
                print(f"  {i}/{total} pairs cleaned")

    print(f"Done. {total} cleaned pairs -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/reply_pairs_raw.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/reply_pairs_clean.jsonl"))
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found. Run extract_pairs.py first.", file=sys.stderr)
        sys.exit(1)

    process_file(args.input, args.output)


if __name__ == "__main__":
    main()
