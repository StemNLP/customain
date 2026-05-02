#!/usr/bin/env python
"""Format reply pairs into OpenAI DPO fine-tuning format and split into train/test.

Reads anonymized reply pairs JSONL, generates non-preferred outputs using a
cheap baseline model, and produces:
  - data/dpo_train.jsonl  (training split)
  - data/dpo_test.jsonl   (test split)

Each line follows the OpenAI DPO fine-tuning format:
  {"input": {"messages": [...]},
   "preferred_output": [...],
   "non_preferred_output": [...]}

Usage:
    uv run python gmail_preprocessing_pipeline/format_for_dpo.py
    uv run python gmail_preprocessing_pipeline/format_for_dpo.py --input data/custom.jsonl --test-ratio 0.15
"""

import argparse
import json
import random
from pathlib import Path

from openai import OpenAI

USER_INSTRUCTION = (
    "Write a reply to the following email. "
    "Output only the reply body — no preamble, no subject line, no explanation."
)
BASELINE_MODEL = "gpt-4.1-nano-2025-04-14"
TEST_RATIO = 0.2
SEED = 42


def _build_user_content(pair: dict) -> str:
    subject = pair.get("subject") or ""
    received = pair.get("received_body") or ""
    email = f"Subject: {subject}\n\n{received}" if subject else received
    return f"{USER_INSTRUCTION}\n\n{email}"


def _generate_baseline_reply(client: OpenAI, user_content: str) -> str:
    response = client.chat.completions.create(
        model=BASELINE_MODEL,
        temperature=0.7,
        messages=[{"role": "user", "content": user_content}],
    )
    return response.choices[0].message.content.strip()


def format_dpo_example(pair: dict, client: OpenAI) -> dict:
    user_content = _build_user_content(pair)
    reply = pair.get("reply_body") or ""
    non_preferred = _generate_baseline_reply(client, user_content)

    return {
        "input": {
            "messages": [
                {"role": "user", "content": user_content},
            ],
        },
        "preferred_output": [
            {"role": "assistant", "content": reply},
        ],
        "non_preferred_output": [
            {"role": "assistant", "content": non_preferred},
        ],
    }


def process_file(
    input_path: Path,
    output_dir: Path,
    test_ratio: float = TEST_RATIO,
    seed: int = SEED,
) -> None:
    client = OpenAI()
    lines = input_path.read_text(encoding="utf-8").strip().splitlines()
    pairs = [json.loads(line) for line in lines]
    total = len(pairs)
    print(f"Generating DPO examples for {total} pairs (baseline: {BASELINE_MODEL})...")

    formatted = []
    for i, pair in enumerate(pairs, 1):
        formatted.append(format_dpo_example(pair, client))
        if i % 10 == 0 or i == total:
            print(f"  {i}/{total} pairs processed")

    random.seed(seed)
    random.shuffle(formatted)

    split_idx = max(1, int(len(formatted) * test_ratio))
    test_set = formatted[:split_idx]
    train_set = formatted[split_idx:]

    train_path = output_dir / "dpo_train.jsonl"
    test_path = output_dir / "dpo_test.jsonl"

    for path, data in [(train_path, train_set), (test_path, test_set)]:
        with path.open("w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Total: {len(formatted)} pairs")
    print(f"Train: {len(train_set)} -> {train_path}")
    print(f"Test:  {len(test_set)} -> {test_path}")


def main():
    from ._load_secrets import load_secrets
    load_secrets()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/_intermediate/reply_pairs_anon.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--test-ratio", type=float, default=TEST_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    process_file(args.input, args.output_dir, args.test_ratio, args.seed)


if __name__ == "__main__":
    main()
