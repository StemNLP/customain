#!/usr/bin/env python
"""Format reply pairs into OpenAI SFT fine-tuning format and split into train/test.

Reads filtered reply pairs JSONL and produces:
  - data/sft_train.jsonl  (training split)
  - data/sft_test.jsonl   (test split)

Each line follows the OpenAI chat fine-tuning format:
  {"messages": [{"role": "system", "content": "..."},
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}]}

Usage:
    uv run python gmail_preprocessing_pipeline/format_for_sft.py
    uv run python gmail_preprocessing_pipeline/format_for_sft.py --input data/custom.jsonl --test-ratio 0.15
"""

import argparse
import json
import random
from pathlib import Path

USER_INSTRUCTION = (
    "Write a reply to the following email. "
    "Output only the reply body — no preamble, no subject line, no explanation."
)
TEST_RATIO = 0.2
SEED = 42


def format_pair(pair: dict) -> dict:
    """Convert a reply pair to OpenAI chat fine-tuning format."""
    subject = pair.get("subject") or ""
    received = pair.get("received_body") or ""
    reply = pair.get("reply_body") or ""

    email = f"Subject: {subject}\n\n{received}" if subject else received
    user_content = f"{USER_INSTRUCTION}\n\n{email}"

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": reply},
        ]
    }


def process_file(
    input_path: Path,
    output_dir: Path,
    test_ratio: float = TEST_RATIO,
    seed: int = SEED,
) -> None:
    lines = input_path.read_text(encoding="utf-8").strip().splitlines()
    pairs = [json.loads(line) for line in lines]
    formatted = [format_pair(p) for p in pairs]

    random.seed(seed)
    random.shuffle(formatted)

    split_idx = max(1, int(len(formatted) * test_ratio))
    test_set = formatted[:split_idx]
    train_set = formatted[split_idx:]

    train_path = output_dir / "sft_train.jsonl"
    test_path = output_dir / "sft_test.jsonl"

    for path, data in [(train_path, train_set), (test_path, test_set)]:
        with path.open("w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Total: {len(formatted)} pairs")
    print(f"Train: {len(train_set)} -> {train_path}")
    print(f"Test:  {len(test_set)} -> {test_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/_intermediate/reply_pairs_anon.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--test-ratio", type=float, default=TEST_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    process_file(args.input, args.output_dir, args.test_ratio, args.seed)


if __name__ == "__main__":
    main()
