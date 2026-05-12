#!/usr/bin/env python
"""Run the Gmail preprocessing pipeline.

Transforms raw Gmail data into versioned datasets under data/gmail/<timestamp>/.

Usage:
    uv run python -m gmail_preprocessing_pipeline.run_pipeline
    uv run python -m gmail_preprocessing_pipeline.run_pipeline --skip 1
    uv run python -m gmail_preprocessing_pipeline.run_pipeline --targets sft authorship
"""

import argparse
from pathlib import Path

from ._load_secrets import load_secrets
from .datasets import find_latest_export, resolve_dataset_dir_from_mbox


def run_pipeline(
    data_dir: str = "data",
    targets: list[str] | None = None,
    skip_steps: list[int] | None = None,
    start_from: int = 1,
    gmail_query: str | None = None,
    newer_than_days: int | None = None,
    max_threads: int | None = None,
) -> None:
    load_secrets()
    skip = set(skip_steps or [])
    skip |= set(range(1, start_from))
    selected_targets = targets or ["sft", "dpo", "authorship"]
    if not selected_targets:
        raise ValueError("At least one dataset target is required")

    data = Path(data_dir)
    exports = data / "exports"
    exports.mkdir(parents=True, exist_ok=True)

    print("Gmail preprocessing pipeline")
    print(f"  Data directory: {data}")
    print(f"  Targets:        {', '.join(selected_targets)}")
    print(f"  Skipping steps: {sorted(skip) if skip else 'none'}")
    print()

    mbox_path = None
    if 1 not in skip:
        print("=== Step 1/4: Export Gmail threads ===")
        from .export_gmail import get_service, export_replied_threads

        service = get_service()
        mbox_path = export_replied_threads(
            service,
            gmail_query=gmail_query,
            newer_than_days=newer_than_days,
            max_threads=max_threads,
        )
    else:
        print("=== Step 1/4: Export Gmail threads [skipped] ===")

    if mbox_path is None:
        mbox_path = find_latest_export(exports)
        print(f"  Using latest export: {mbox_path}")

    dataset_dir = resolve_dataset_dir_from_mbox(data, mbox_path)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = dataset_dir / "_intermediate"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Dataset version: {dataset_dir}")
    print(f"  Intermediate:   {tmp_dir}")

    raw_pairs_path = tmp_dir / "reply_pairs_raw.jsonl"
    processed_pairs_path = tmp_dir / "reply_pairs_processed.jsonl"

    if 2 not in skip:
        print("\n=== Step 2/4: Extract reply pairs ===")
        from .extract_pairs import process_file as extract_pairs

        extract_pairs(mbox_path, raw_pairs_path)
    else:
        print("=== Step 2/4: Extract reply pairs [skipped] ===")

    if 3 not in skip:
        print("\n=== Step 3/4: Transform pairs ===")
        from .transform_pairs import process_file as transform_pairs

        transform_pairs(raw_pairs_path, processed_pairs_path)
    else:
        print("=== Step 3/4: Transform pairs [skipped] ===")

    if 4 not in skip:
        print("\n=== Step 4/4: Build selected datasets ===")
        manifest_entries: list[str] = []

        if "sft" in selected_targets:
            from .format_for_sft import process_file as build_sft

            build_sft(processed_pairs_path, dataset_dir / "sft")
            manifest_entries.extend(
                [
                    "sft/train.jsonl",
                    "sft/test.jsonl",
                    "sft/train_mock.jsonl",
                    "sft/test_mock.jsonl",
                ]
            )

        if "dpo" in selected_targets:
            from .format_for_dpo import process_file as build_dpo

            build_dpo(processed_pairs_path, dataset_dir / "dpo")
            manifest_entries.extend(
                [
                    "dpo/train.jsonl",
                    "dpo/test.jsonl",
                    "dpo/train_mock.jsonl",
                    "dpo/test_mock.jsonl",
                ]
            )

        if "authorship" in selected_targets:
            from classifiers.authorship.prepare_data import extract_from_pairs, write_dataset

            positives, negatives = extract_from_pairs(str(processed_pairs_path))
            write_dataset(
                positives,
                negatives,
                str(dataset_dir / "authorship"),
                val_ratio=0.2,
                seed=42,
            )
            manifest_entries.extend([
                "authorship/train.jsonl",
                "authorship/val.jsonl",
            ])

        from .manifest import write_manifest

        manifest_path = write_manifest(dataset_dir, manifest_entries)
        print(f"\nManifest written to {manifest_path}")
    else:
        print("=== Step 4/4: Build selected datasets [skipped] ===")

    print("\nPipeline complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=["sft", "dpo", "authorship"],
        default=["sft", "dpo", "authorship"],
    )
    parser.add_argument(
        "--skip",
        type=int,
        nargs="*",
        default=[],
        help="Step numbers to skip (e.g. --skip 1 2)",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=1,
        help="Start from this step (skips all earlier steps)",
    )
    parser.add_argument("--gmail-query", type=str, default=None)
    parser.add_argument("--newer-than-days", type=int, default=None)
    parser.add_argument("--max-threads", type=int, default=None)
    args = parser.parse_args()
    run_pipeline(
        data_dir=args.data_dir,
        targets=args.targets,
        skip_steps=args.skip,
        start_from=args.start_from,
        gmail_query=args.gmail_query,
        newer_than_days=args.newer_than_days,
        max_threads=args.max_threads,
    )


if __name__ == "__main__":
    main()