#!/usr/bin/env python
"""Run the Gmail preprocessing pipeline (steps 1-6).

Transforms raw Gmail data into anonymized SFT training files.

Usage:
    uv run python -m gmail_preprocessing_pipeline.run_pipeline
    uv run python -m gmail_preprocessing_pipeline.run_pipeline --skip 1  # already exported
    uv run python -m gmail_preprocessing_pipeline.run_pipeline --start-from 5  # re-run from anonymize
"""

import argparse
from pathlib import Path

STEPS = [
    (1, "Export Gmail threads"),
    (2, "Extract reply pairs"),
    (3, "Clean pairs (LLM)"),
    (4, "Filter pairs (LLM)"),
    (5, "Anonymize names (LLM)"),
    (6, "Format for SFT"),
]


def run_pipeline(
    data_dir: str = "data",
    skip_steps: list[int] | None = None,
    start_from: int = 1,
) -> None:
    skip = set(skip_steps or [])
    skip |= set(range(1, start_from))
    data = Path(data_dir)
    tmp = data / "_intermediate"
    tmp.mkdir(parents=True, exist_ok=True)

    print("Gmail preprocessing pipeline")
    print(f"  Data directory: {data}")
    print(f"  Intermediate:   {tmp}")
    print(f"  Skipping steps: {sorted(skip) if skip else 'none'}")
    print()

    if 1 not in skip:
        print("=== Step 1/6: Export Gmail threads ===")
        from .export_gmail import get_service, export_replied_threads

        service = get_service()
        export_replied_threads(service)
    else:
        print("=== Step 1/6: Export Gmail threads [skipped] ===")

    if 2 not in skip:
        print("\n=== Step 2/6: Extract reply pairs ===")
        from .extract_pairs import process_file

        process_file(tmp / "new_threads.mbox", tmp / "reply_pairs_raw.jsonl")
    else:
        print("=== Step 2/6: Extract reply pairs [skipped] ===")

    if 3 not in skip:
        print("\n=== Step 3/6: Clean pairs ===")
        from .clean_pairs import process_file

        process_file(tmp / "reply_pairs_raw.jsonl", tmp / "reply_pairs_clean.jsonl")
    else:
        print("=== Step 3/6: Clean pairs [skipped] ===")

    if 4 not in skip:
        print("\n=== Step 4/6: Filter pairs ===")
        from .filter_pairs import process_file

        process_file(
            tmp / "reply_pairs_clean.jsonl", tmp / "reply_pairs_filtered.jsonl"
        )
    else:
        print("=== Step 4/6: Filter pairs [skipped] ===")

    if 5 not in skip:
        print("\n=== Step 5/6: Anonymize names ===")
        from .anonymize_pairs import process_file

        process_file(
            tmp / "reply_pairs_filtered.jsonl", tmp / "reply_pairs_anon.jsonl"
        )
    else:
        print("=== Step 5/6: Anonymize names [skipped] ===")

    if 6 not in skip:
        print("\n=== Step 6/6: Format for SFT ===")
        from .format_for_sft import process_file

        process_file(tmp / "reply_pairs_anon.jsonl", data)
    else:
        print("=== Step 6/6: Format for SFT [skipped] ===")

    print("\nPipeline complete.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument(
        "--skip", type=int, nargs="*", default=[],
        help="Step numbers to skip (e.g. --skip 1 2)",
    )
    parser.add_argument(
        "--start-from", type=int, default=1,
        help="Start from this step (skips all earlier steps)",
    )
    args = parser.parse_args()
    run_pipeline(args.data_dir, args.skip, args.start_from)


if __name__ == "__main__":
    main()
