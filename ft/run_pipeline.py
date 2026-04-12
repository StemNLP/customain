#!/usr/bin/env python
"""Run the fine-tuning pipeline (steps 1-4).

Usage:
    uv run python -m ft.run_pipeline --train-file data/sft_train.jsonl --test-file data/sft_test.jsonl
    uv run python -m ft.run_pipeline --skip 1 2   # skip steps 1 and 2, only run 3 and 4
"""

import argparse
from pathlib import Path
import logging
from .logging_config import setup_logger

logger = setup_logger(log_level=logging.INFO)

TEST_TRAIN_LIMIT = 10
TEST_TEST_LIMIT = 3


def _make_subset(src: str, limit: int, suffix: str) -> str:
    """Write the first `limit` lines of `src` to a sibling file and return its path."""
    src_path = Path(src)
    dst_path = src_path.with_name(f"{src_path.stem}{suffix}{src_path.suffix}")
    lines = src_path.read_text(encoding="utf-8").splitlines()[:limit]
    dst_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"Test mode: wrote {len(lines)} lines to {dst_path}")
    return str(dst_path)


def run_pipeline(train_file: str,
                 test_file: str,
                 train_file_oai_id: str | None = None,
                 test_file_oai_id: str | None = None,
                 skip_steps: list[int] | None = None,
                 test_run: bool = False):
    """
    Run fine-tuning steps 1 through 4.

    Args:
        train_file: Local path to training JSONL.
        test_file: Local path to test JSONL.
        train_file_oai_id: OpenAI file ID for training data (if already uploaded).
        test_file_oai_id: OpenAI file ID for test data (if already uploaded).
        skip_steps: List of step numbers to skip (e.g. [1, 2]).
        test_run: If True, subset to TEST_TRAIN_LIMIT train and TEST_TEST_LIMIT test examples.
    """
    skip = set(skip_steps or [])

    if test_run:
        logger.info(f"=== TEST RUN: {TEST_TRAIN_LIMIT} train, {TEST_TEST_LIMIT} test ===")
        train_file = _make_subset(train_file, TEST_TRAIN_LIMIT, "_testrun")
        test_file = _make_subset(test_file, TEST_TEST_LIMIT, "_testrun")

    if 1 not in skip:
        logger.info("=== Step 1: Generating configs and launching FT jobs ===")
        from .step_1_run_ft_jobs import generate_configurations, run_experiments
        from .training_configs import llms, batch_sizes, learning_rate_multipliers
        from .finetuning import upload_file_for_ft

        if not train_file_oai_id:
            train_file_oai_id = upload_file_for_ft(train_file)
        if not test_file_oai_id:
            test_file_oai_id = upload_file_for_ft(test_file)

        configs = generate_configurations(
            train_file=train_file,
            test_file=test_file,
            train_file_oai_id=train_file_oai_id,
            test_file_oai_id=test_file_oai_id,
            llms=llms,
            batch_sizes=batch_sizes,
            learning_rate_multipliers=learning_rate_multipliers,
        )
        run_experiments(configs)
    else:
        logger.info("Skipping step 1")

    if 2 not in skip:
        logger.info("=== Step 2: Updating experiment status ===")
        from .step_2_update_experiments import update_experiments
        update_experiments()
    else:
        logger.info("Skipping step 2")

    if 3 not in skip:
        logger.info("=== Step 3: Running FT models on test set ===")
        from .step_3_eval_run_ft_models import eval_run_all_fted_models
        eval_run_all_fted_models(test_file=test_file)
    else:
        logger.info("Skipping step 3")

    if 4 not in skip:
        logger.info("=== Step 4: Running evaluation ===")
        from .step_4_run_evaluation import evaluate_all_ft_models
        evaluate_all_ft_models()
    else:
        logger.info("Skipping step 4")

    logger.info("Pipeline complete.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=str, required=True)
    parser.add_argument("--test-file", type=str, required=True)
    parser.add_argument("--train-file-oai-id", type=str, default=None)
    parser.add_argument("--test-file-oai-id", type=str, default=None)
    parser.add_argument("--skip", type=int, nargs="*", default=[],
                        help="Step numbers to skip (e.g. --skip 1 2)")
    parser.add_argument("--test-run", action="store_true",
                        help=f"Use only {TEST_TRAIN_LIMIT} train and {TEST_TEST_LIMIT} test examples")
    args = parser.parse_args()

    run_pipeline(
        train_file=args.train_file,
        test_file=args.test_file,
        train_file_oai_id=args.train_file_oai_id,
        test_file_oai_id=args.test_file_oai_id,
        skip_steps=args.skip,
        test_run=args.test_run,
    )


if __name__ == "__main__":
    main()
