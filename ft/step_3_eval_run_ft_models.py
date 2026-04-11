from ft.finetuning import query_fted_model_chat_completion
from pathlib import Path
import json
import logging
from .logging_config import setup_logger

logger = setup_logger(log_level=logging.INFO)


def eval_run_fted_model(ft_model_id: str,
                        test_file: str) -> str:
    """
    Run a fine-tuned model on the test dataset and return the results.

    Args:
        ft_model_id (str): The ID of the fine-tuned model to evaluate.
        test_file (str): Path to the test dataset file.

    Returns:
        list: A list of dictionaries containing evaluation results for each example.
    """
    
    logger.info(f"Starting evaluation for model {ft_model_id} on {test_file}")


    ft_model_results = []
    
    with open(test_file, "r") as test_f:
        for i, line in enumerate(test_f):
            logger.debug(f"Processing eval example {i+1}")
            data = json.loads(line)
            user_prompt = next((msg['content'] for msg in data['messages'] if msg['role'] == 'user'), None)
            expected_response = next((msg['content'] for msg in data['messages'] if msg['role'] == 'assistant'), None)

            if not user_prompt:
                logger.warning(f"Could not find user prompt for example {i+1}. Skipping...")
                continue
            
            response = query_fted_model_chat_completion(model_id=ft_model_id,
                               user_query=user_prompt)[0]

            ft_model_results.append({
                "datapoint_id": i + 1,
                "user_prompt": user_prompt,
                "expected_response": expected_response,
                "generated_response": response,
            })

    return ft_model_results


def eval_run_all_fted_models(test_file: str) -> None:
    """
    Evaluate all fine-tuned models and baseline models using the test split.

    Args:
        test_file (str): Path to the test dataset JSONL file.
    """
    from .training_configs import baseline_models

    experiments_path = Path(__file__).parent / "_experiments.json"
    with open(experiments_path, 'r') as f:
        experiments = json.load(f)

    results = {}

    # Run baseline (non-fine-tuned) models
    for model_id in baseline_models:
        logger.info(f"Evaluating baseline model {model_id}")
        res = eval_run_fted_model(model_id, test_file)
        results[f"baseline:{model_id}"] = res

    # Run fine-tuned models
    for exp_id, exp_data in experiments.items():
        ft_model_id = exp_data.get('ft_model_id')
        if not ft_model_id:
            logger.warning(f"Experiment {exp_id} does not have a fine-tuned model ID. Skipping...")
            continue

        logger.info(f"Evaluating experiment {exp_id} with model {ft_model_id}")
        res = eval_run_fted_model(ft_model_id, test_file)
        results[ft_model_id] = res

    output_path = Path(__file__).parent / "_ft_models_eval_runs.json"
    with open(output_path, "w", encoding="utf-8") as out_f:
        json.dump(results, out_f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {output_path}")
