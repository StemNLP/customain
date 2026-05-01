#!/usr/bin/env python
"""Anonymize person names in reply pairs by replacing them with [NAME].

Uses an LLM to detect and replace all person names while preserving
everything else exactly. Designed as a pipeline step between
filter_pairs.py and format_for_sft.py.

Usage:
    uv run python data_processing/anonymize_pairs.py
    uv run python data_processing/anonymize_pairs.py --input data/reply_pairs_filtered.jsonl
"""

import argparse
import json
import random
import sys
from pathlib import Path

import anthropic

MODEL = "claude-haiku-4-5-20251001"
TEST_MODE = False
TEST_LIMIT = 20

SYSTEM_PROMPT = """\
You are a PII anonymizer. Your job is to replace all person names in the \
given text with [NAME]. This includes first names, last names, and full \
names in any position: greetings, sign-offs, mid-sentence mentions, \
subject lines, introductions, etc.

Replace ONLY person names. Do not touch anything else.

Return ONLY the anonymized text. No explanation, no wrapping."""


def anonymize_text(client: anthropic.Anthropic, text: str) -> str:
    if not text or not text.strip():
        return text
    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": f"Anonymize person names:\n\n{text}"}
        ],
    )
    return response.content[0].text.strip()


def process_file(input_path: Path, output_path: Path) -> None:
    client = anthropic.Anthropic()
    lines = input_path.read_text(encoding="utf-8").strip().splitlines()
    if TEST_MODE:
        lines = random.sample(lines, min(TEST_LIMIT, len(lines)))
        print(f"TEST MODE: randomly sampled {len(lines)} pairs")
    total = len(lines)
    print(f"Anonymizing {total} pairs from {input_path} ...")

    with output_path.open("w", encoding="utf-8") as f:
        for i, line in enumerate(lines, 1):
            pair = json.loads(line)

            if pair.get("received_body"):
                pair["received_body"] = anonymize_text(
                    client, pair["received_body"]
                )
            if pair.get("reply_body"):
                pair["reply_body"] = anonymize_text(client, pair["reply_body"])
            if pair.get("subject"):
                pair["subject"] = anonymize_text(client, pair["subject"])

            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

            if i % 10 == 0 or i == total:
                print(f"  {i}/{total} pairs anonymized")

    print(f"Done. {total} anonymized pairs -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("data/reply_pairs_filtered.jsonl")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/reply_pairs_anon.jsonl")
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(
            f"Error: {args.input} not found. Run filter_pairs.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    process_file(args.input, args.output)


if __name__ == "__main__":
    main()
