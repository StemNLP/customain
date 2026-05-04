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

POLL_INTERVAL_SECONDS = 300

DATA_FILE_PREFIXES = {
    "supervised": "sft",
    "dpo": "dpo",
}


def run_pipeline(data_dir: str = "data",
                 skip_steps: list[int] | None = None,
                 test_run: bool = False):
    """
    Run fine-tuning steps 1 through 4.

    Reads training_methods from training_configs.py and auto-resolves data
    files per method (sft_*.jsonl for supervised, dpo_*.jsonl for dpo).
    With --test-run, uses pre-generated mock files from the preprocessing pipeline.

    Args:
        data_dir: Directory containing training/test JSONL files.
        skip_steps: List of step numbers to skip (e.g. [1, 2]).
        test_run: If True, use mock data files produced by the preprocessing pipeline.
    """
    skip = set(skip_steps or [])
    data_path = Path(data_dir)
    mock_suffix = "_mock" if test_run else ""

    if test_run:
        logger.info("=== TEST RUN: using mock data files ===")

    if 1 not in skip:
        logger.info("=== Step 1: Generating configs and launching FT jobs ===")
        from .step_1_run_ft_jobs import generate_configurations, run_experiments
        from .training_configs import training_methods, llms, batch_sizes, learning_rate_multipliers

        all_configs = []
        for method in training_methods:
            prefix = DATA_FILE_PREFIXES[method]
            train_file = str(data_path / f"{prefix}_train{mock_suffix}.jsonl")
            test_file = str(data_path / f"{prefix}_test{mock_suffix}.jsonl")

            configs = generate_configurations(
                train_file=train_file,
                test_file=test_file,
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
        eval_test_file = str(data_path / f"sft_test{mock_suffix}.jsonl")
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
                        help="Use mock data files (small subsets produced by preprocessing pipeline)")
    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        skip_steps=args.skip,
        test_run=args.test_run,
    )


if __name__ == "__main__":
    main()
