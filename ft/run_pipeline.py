#!/usr/bin/env python
"""Run the fine-tuning pipeline (steps 1-4).

Reads training_methods from training_configs.py and auto-resolves data files:
  - supervised -> data/sft_train.jsonl, data/sft_test.jsonl
  - dpo        -> data/dpo_train.jsonl, data/dpo_test.jsonl

Usage:
    uv run python -m ft.run_pipeline
    uv run python -m ft.run_pipeline --data-dir data
    uv run python -m ft.run_pipeline --skip 1 2   # skip steps 1 and 2, only run 3 and 4
"""

import argparse
import json
import time
from pathlib import Path
import logging
from .logging_config import setup_logger

logger = setup_logger(log_level=logging.INFO)

TEST_TRAIN_LIMIT = 50
TEST_TEST_LIMIT = 25
POLL_INTERVAL_SECONDS = 300

DATA_FILE_PREFIXES = {
    "supervised": "sft",
    "dpo": "dpo",
}


def _make_subset(src: str, limit: int, suffix: str) -> str:
    """Write the first `limit` lines of `src` to a sibling file and return its path."""
    src_path = Path(src)
    dst_path = src_path.with_name(f"{src_path.stem}{suffix}{src_path.suffix}")
    lines = src_path.read_text(encoding="utf-8").splitlines()[:limit]
    dst_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"Test mode: wrote {len(lines)} lines to {dst_path}")
    return str(dst_path)


def run_pipeline(data_dir: str = "data",
                 skip_steps: list[int] | None = None,
                 test_run: bool = False):
    """
    Run fine-tuning steps 1 through 4.

    Reads training_methods from training_configs.py and auto-resolves data
    files per method (sft_*.jsonl for supervised, dpo_*.jsonl for dpo).

    Args:
        data_dir: Directory containing training/test JSONL files.
        skip_steps: List of step numbers to skip (e.g. [1, 2]).
        test_run: If True, subset to TEST_TRAIN_LIMIT train and TEST_TEST_LIMIT test examples.
    """
    skip = set(skip_steps or [])
    data_path = Path(data_dir)

    if test_run:
        logger.info(f"=== TEST RUN: {TEST_TRAIN_LIMIT} train, {TEST_TEST_LIMIT} test ===")

    if 1 not in skip:
        logger.info("=== Step 1: Generating configs and launching FT jobs ===")
        from .step_1_run_ft_jobs import generate_configurations, run_experiments
        from .training_configs import training_methods, llms, batch_sizes, learning_rate_multipliers
        from .finetuning import upload_file_for_ft

        all_configs = []
        for method in training_methods:
            prefix = DATA_FILE_PREFIXES[method]
            train_file = str(data_path / f"{prefix}_train.jsonl")
            test_file = str(data_path / f"{prefix}_test.jsonl")

            if test_run:
                train_file = _make_subset(train_file, TEST_TRAIN_LIMIT, "_testrun")
                test_file = _make_subset(test_file, TEST_TEST_LIMIT, "_testrun")

            train_oai_id = upload_file_for_ft(train_file)
            test_oai_id = upload_file_for_ft(test_file)

            configs = generate_configurations(
                train_file=train_file,
                test_file=test_file,
                train_file_oai_id=train_oai_id,
                test_file_oai_id=test_oai_id,
                llms=llms,
                batch_sizes=batch_sizes,
                learning_rate_multipliers=learning_rate_multipliers,
                training_method=method,
            )
            all_configs.extend(configs)

        experiments = run_experiments(all_configs)
        if experiments is None:
            logger.info("Pipeline aborted by user at step 1.")
            return
    else:
        logger.info("Skipping step 1")

    if 2 not in skip:
        logger.info("=== Step 2: Waiting for fine-tuning jobs to complete ===")
        from .step_2_update_experiments import update_experiments
        experiments_path = Path(__file__).parent / "_experiments.json"

        while True:
            update_experiments()
            with open(experiments_path, "r") as f:
                experiments = json.load(f)
            if experiments and all("ft_model_id" in exp for exp in experiments.values()):
                logger.info("All fine-tuning jobs completed successfully")
                break
            minutes, seconds = divmod(POLL_INTERVAL_SECONDS, 60)
            logger.info(f"Not all jobs completed, waiting {minutes}m {seconds}s...")
            time.sleep(POLL_INTERVAL_SECONDS)
    else:
        logger.info("Skipping step 2")

    if 3 not in skip:
        logger.info("=== Step 3: Running FT models on test set ===")
        from .step_3_eval_run_ft_models import eval_run_all_fted_models
        eval_test_file = str(data_path / "sft_test.jsonl")
        if test_run:
            eval_test_file = _make_subset(eval_test_file, TEST_TEST_LIMIT, "_testrun")
        eval_run_all_fted_models(test_file=eval_test_file)
    else:
        logger.info("Skipping step 3")

    if 4 not in skip:
        logger.info("=== Step 4: Running evaluation ===")
        from .step_4_run_evaluation import evaluate_all_ft_models
        from .training_configs import skip_evaluators
        evaluate_all_ft_models(skip_evaluators=skip_evaluators)
    else:
        logger.info("Skipping step 4")

    logger.info("Pipeline complete.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory containing training/test JSONL files (default: data)")
    parser.add_argument("--skip", type=int, nargs="*", default=[],
                        help="Step numbers to skip (e.g. --skip 1 2)")
    parser.add_argument("--test-run", action="store_true",
                        help=f"Use only {TEST_TRAIN_LIMIT} train and {TEST_TEST_LIMIT} test examples")
    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        skip_steps=args.skip,
        test_run=args.test_run,
    )


if __name__ == "__main__":
    main()
