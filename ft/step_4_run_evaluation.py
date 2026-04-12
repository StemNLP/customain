from ft.evaluation.core import run_evaluators
from ft.evaluation.registry import get_evaluator_registry
from pathlib import Path
import json
import logging
import wandb
from .logging_config import setup_logger

logger = setup_logger(log_level=logging.INFO)


def evaluate_ft_model(model_results: list, evaluator_registry: dict) -> list[dict]:
    """
    Evaluate all responses for a single fine-tuned model.

    Args:
        model_results: List of dicts with keys: datapoint_id, user_prompt,
                       expected_response, generated_response.
        evaluator_registry: Registry of evaluator instances.

    Returns:
        List of per-datapoint result dicts.
    """
    if not model_results:
        logger.warning("Empty model results provided")
        return []

    all_results = []

    for eval_run in model_results:
        datapoint_id = eval_run["datapoint_id"]
        eval_input = {
            "expected": eval_run["expected_response"],
            "generated": eval_run["generated_response"],
        }

        try:
            result = run_evaluators(evaluator_registry, eval_input)
        except Exception as e:
            logger.error(f"Evaluator error for datapoint {datapoint_id}: {e}")
            continue

        all_results.append({"datapoint_id": datapoint_id, **result})

    return all_results


def evaluate_all_ft_models() -> None:
    """
    Evaluate all fine-tuned model results from _ft_models_eval_runs.json.
    Writes per-model evaluation results to _evaluation_results.json.
    """
    eval_run_results = Path(__file__).parent / "_ft_models_eval_runs.json"
    output_path = Path(__file__).parent / "_evaluation_results.json"

    logger.info(f"Starting evaluation of results from {eval_run_results}")
    evaluator_registry = get_evaluator_registry()

    if not evaluator_registry:
        logger.error("No evaluators found in registry. Add evaluators to ft/evaluation/evaluators/.")
        return

    logger.info(f"Loaded evaluators: {list(evaluator_registry.keys())}")

    wandb.init(project="customain", job_type="evaluation")

    with open(eval_run_results, "r") as f:
        ft_models_eval_runs = json.load(f)

    all_evaluations = {}
    per_model_avgs = {}  # {model_id: {metric_name: avg_score}}

    for ft_model_id, content in ft_models_eval_runs.items():
        logger.info(f"Evaluating model {ft_model_id} ({len(content)} datapoints)")
        results = evaluate_ft_model(content, evaluator_registry)

        # Compute averages per evaluator
        avg_metrics = {}
        if results:
            evaluator_names = [k for k in results[0] if k != "datapoint_id"]
            for name in evaluator_names:
                scores = [r[name] for r in results if name in r and isinstance(r[name], (int, float))]
                if scores:
                    avg_metrics[name] = round(sum(scores) / len(scores), 4)
            logger.info(f"  Model {ft_model_id} averages: {avg_metrics}")
            per_model_avgs[ft_model_id] = avg_metrics

        all_evaluations[ft_model_id] = {
            "per_datapoint": results,
            "averages": avg_metrics,
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_evaluations, f, indent=2, ensure_ascii=False)
    logger.info(f"Evaluation results saved to {output_path}")

    # Log a bar chart per metric: one bar per model
    if per_model_avgs:
        all_metric_names = sorted({m for avgs in per_model_avgs.values() for m in avgs})
        for metric in all_metric_names:
            rows = [[model_id, avgs.get(metric, 0.0)] for model_id, avgs in per_model_avgs.items()]
            table = wandb.Table(data=rows, columns=["model", metric])
            wandb.log({
                f"{metric}_bar": wandb.plot.bar(table, "model", metric, title=f"{metric} by model"),
            })

        # Also log a summary table with all metrics side by side
        summary_rows = [
            [model_id] + [avgs.get(m, 0.0) for m in all_metric_names]
            for model_id, avgs in per_model_avgs.items()
        ]
        summary_table = wandb.Table(data=summary_rows, columns=["model"] + all_metric_names)
        wandb.log({"metrics_summary": summary_table})

    wandb.finish()
