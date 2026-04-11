#!/usr/bin/env python
"""Quality filter for reply pairs. Drops low-quality entries using an LLM.

Reads cleaned reply pairs JSONL, asks the LLM to judge each pair,
writes only the pairs that pass to a new JSONL file.

Usage:
    uv run python data_processing/filter_pairs.py
    uv run python data_processing/filter_pairs.py --input data/clean.jsonl --output data/filtered.jsonl
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
You are an email quality filter for a dataset of email-reply pairs. \
Your job is to decide whether a pair should be KEPT or DROPPED.

DROP the pair if ANY of the following apply:
- Email warmup / toaster: generic, incoherent text that looks auto-generated \
  (e.g. random compliments, unrelated sentences stitched together, the reply \
  greets the sender by their own name)
- Irrelevant: spam, mass marketing, newsletters, automated notifications
- Too short: either the received or reply body has less than 2 meaningful sentences
- Incoherent: the reply clearly does not relate to the received email

KEEP the pair if it is a genuine, coherent human email exchange with \
substantive content on both sides.

Respond with EXACTLY one of these two words: KEEP or DROP
Nothing else."""


def judge_pair(client: anthropic.Anthropic, pair: dict) -> str:
    """Return 'KEEP' or 'DROP'."""
    prompt = (
        f"Subject: {pair.get('subject', '(none)')}\n\n"
        f"--- Received email ---\n{pair.get('received_body', '')}\n\n"
        f"--- Reply ---\n{pair.get('reply_body', '')}"
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=4,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip().upper()


def process_file(input_path: Path, output_path: Path):
    client = anthropic.Anthropic()
    lines = input_path.read_text(encoding="utf-8").strip().splitlines()
    if TEST_MODE:
        lines = random.sample(lines, min(TEST_LIMIT, len(lines)))
        print(f"TEST MODE: randomly sampled {len(lines)} pairs")
    total = len(lines)
    kept = 0
    dropped = 0

    print(f"Filtering {total} pairs from {input_path} ...")

    with output_path.open("w", encoding="utf-8") as f:
        for i, line in enumerate(lines, 1):
            pair = json.loads(line)
            verdict = judge_pair(client, pair)

            if verdict == "KEEP":
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
                kept += 1
            else:
                dropped += 1

            if i % 10 == 0 or i == total:
                print(f"  {i}/{total} processed (kept {kept}, dropped {dropped})")

    print(f"\nDone. {kept} kept, {dropped} dropped -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/reply_pairs_clean.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/reply_pairs_filtered.jsonl"))
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found. Run clean_pairs.py first.", file=sys.stderr)
        sys.exit(1)

    process_file(args.input, args.output)


if __name__ == "__main__":
    main()
