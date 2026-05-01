"""Extract training data for the authorship classifier from existing SFT data.

Positives: assistant replies (the author's writing).
Negatives: incoming emails from other people (extracted from user prompts).
"""

import argparse
import json
import random
import re
from pathlib import Path


def extract_email_body(user_content: str) -> str | None:
    match = re.search(r"^Subject:[^\n]*\n\n", user_content, re.MULTILINE)
    if match:
        body = user_content[match.end() :].strip()
        return body if body else None
    return None


def extract_from_sft(sft_path: str) -> tuple[list[str], list[str]]:
    positives: list[str] = []
    negatives: list[str] = []
    with open(sft_path) as f:
        for line in f:
            record = json.loads(line)
            for msg in record["messages"]:
                if msg["role"] == "assistant":
                    positives.append(msg["content"])
                elif msg["role"] == "user":
                    body = extract_email_body(msg["content"])
                    if body:
                        negatives.append(body)
    return positives, negatives


def main() -> None:
    args = _parse_args()
    random.seed(args.seed)

    all_positives: list[str] = []
    all_negatives: list[str] = []
    for sft_file in args.sft_files:
        pos, neg = extract_from_sft(sft_file)
        all_positives.extend(pos)
        all_negatives.extend(neg)

    samples = [{"text": t, "label": 1} for t in all_positives] + [
        {"text": t, "label": 0} for t in all_negatives
    ]
    random.shuffle(samples)

    split_idx = int(len(samples) * (1 - args.val_ratio))
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, data in [("train.jsonl", train_samples), ("val.jsonl", val_samples)]:
        with open(output_dir / name, "w") as f:
            for sample in data:
                f.write(json.dumps(sample) + "\n")

    pos_count = len(all_positives)
    neg_count = len(all_negatives)
    print(f"Extracted {pos_count} positive, {neg_count} negative samples")
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")
    print(f"Saved to {output_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare authorship classifier data from SFT files"
    )
    parser.add_argument(
        "--sft-files",
        nargs="+",
        default=["data/sft_train.jsonl", "data/sft_test.jsonl"],
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/classifiers/authorship"
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main()
