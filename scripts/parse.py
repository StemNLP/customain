#!/usr/bin/env python
"""Parse an mbox file into JSONL datasets."""

import argparse
from pathlib import Path

from mbox_parser.pipeline import export_jsonl, export_reply_pairs_jsonl

DEFAULT_INPUT = Path(__file__).parent.parent / "data" / "gmail2025.mbox"
DEFAULT_ALL = Path(__file__).parent.parent / "data" / "emails.jsonl"
DEFAULT_PAIRS = Path(__file__).parent.parent / "data" / "reply_pairs.jsonl"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-all", type=Path, default=DEFAULT_ALL,
                        help="All emails (default: data/emails.jsonl)")
    parser.add_argument("--output-pairs", type=Path, default=DEFAULT_PAIRS,
                        help="Reply pairs only (default: data/reply_pairs.jsonl)")
    args = parser.parse_args()

    print(f"Parsing {args.input} ...")
    count = export_jsonl(args.input, args.output_all)
    print(f"  {count} emails -> {args.output_all}")
    count = export_reply_pairs_jsonl(args.input, args.output_pairs)
    print(f"  {count} reply pairs -> {args.output_pairs}")


if __name__ == "__main__":
    main()
