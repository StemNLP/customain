"""Extract training data for the authorship classifier from existing SFT data.

Positives: assistant replies (the author's writing).
Negatives: incoming emails from other people (extracted from user prompts).
"""

import argparse
import json
import random
import re
from pathlib import Path

from gmail_preprocessing_pipeline.datasets import find_latest_dataset_dir


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


def extract_from_pairs(pairs_path: str) -> tuple[list[str], list[str]]:
    positives: list[str] = []
    negatives: list[str] = []
    with open(pairs_path) as f:
        for line in f:
            record = json.loads(line)
            reply = (record.get("reply_body") or "").strip()
            received = (record.get("received_body") or "").strip()
            if reply:
                positives.append(reply)
            if received:
                negatives.append(received)
    return positives, negatives


def write_dataset(
    positives: list[str],
    negatives: list[str],
    output_dir: str,
    val_ratio: float,
    seed: int,
) -> None:
    random.seed(seed)

    samples = [{"text": t, "label": 1} for t in positives] + [
        {"text": t, "label": 0} for t in negatives
    ]
    random.shuffle(samples)

    split_idx = int(len(samples) * (1 - val_ratio))
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for name, data in [("train.jsonl", train_samples), ("val.jsonl", val_samples)]:
        with open(output_path / name, "w") as f:
            for sample in data:
                f.write(json.dumps(sample) + "\n")

    print(f"Extracted {len(positives)} positive, {len(negatives)} negative samples")
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")
    print(f"Saved to {output_path}")


def main() -> None:
    args = _parse_args()

    all_positives: list[str] = []
    all_negatives: list[str] = []
    if args.pairs_files:
        for pair_file in args.pairs_files:
            pos, neg = extract_from_pairs(pair_file)
            all_positives.extend(pos)
            all_negatives.extend(neg)
    else:
        sft_files = args.sft_files or _default_sft_files(args.data_dir)
        for sft_file in sft_files:
            pos, neg = extract_from_sft(sft_file)
            all_positives.extend(pos)
            all_negatives.extend(neg)

    output_dir = args.output_dir or str(_default_output_dir(args.data_dir))
    write_dataset(
        all_positives,
        all_negatives,
        output_dir,
        args.val_ratio,
        args.seed,
    )


def _default_sft_files(data_dir: str) -> list[str]:
    dataset_dir = find_latest_dataset_dir(Path(data_dir))
    return [str(dataset_dir / "sft" / "train.jsonl"), str(dataset_dir / "sft" / "test.jsonl")]


def _default_output_dir(data_dir: str) -> Path:
    dataset_dir = find_latest_dataset_dir(Path(data_dir))
    return dataset_dir / "authorship"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare authorship classifier data from SFT files"
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument(
        "--sft-files",
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--pairs-files",
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--output-dir", type=str, default=None
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main()
