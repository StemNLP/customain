import json
import shortuuid
from pathlib import Path
from ft.finetuning import run_finetuning
import logging
import wandb
from .logging_config import setup_logger

logger = setup_logger(log_level=logging.INFO)

# TODO Test if all combinations are generated correctly
def generate_configurations(train_file: str,
                          test_file: str,
                          train_file_oai_id: str,
                          test_file_oai_id: str,
                          llms: list,
                          batch_sizes: list,
                          learning_rate_multipliers: list):
    """
    Generate all hyperparameter configurations for fine-tuning experiments.

    Args:
        train_file (str): Local path to the training file.
        test_file (str): Local path to the test file.
        train_file_oai_id (str): OpenAI file ID for the training file.
        test_file_oai_id (str): OpenAI file ID for the test file.
        llms (list): List of LLM model names to be used for fine-tuning.
        batch_sizes (list): List of batch sizes to be used in the experiments.
        learning_rate_multipliers (list): List of learning rate multipliers to be used in the experiments.

    Returns:
        list: A list of dictionaries, each containing a configuration for a fine-tuning experiment.
    """
    logger.info("Generating configurations for fine-tuning experiments...")
    configs = []
    
    # Loop to add the default config (None hyperparams) for each model
    for llm in llms:
        configs.append({
            "model": llm,
            "training_file": train_file,
            "training_file_oai_id": train_file_oai_id,
            "test_file": test_file,
            "test_file_oai_id": test_file_oai_id,
            "hyperparameters": None
        })
    
    # Loop to generate all hyperparam combinations/configurations 
    for llm in llms:
        for batch_size in batch_sizes:
            for lr_mult in learning_rate_multipliers:
                config = {
                    "model": llm,
                    "training_file": train_file,
                    "training_file_oai_id": train_file_oai_id,
                    "test_file": test_file,
                    "test_file_oai_id": test_file_oai_id,
                    "hyperparameters": {
                        "batch_size": batch_size,
                        "learning_rate_multiplier": lr_mult,
                        "n_epochs": 4
                    }
                }
                configs.append(config)
    return configs

def run_experiments(training_configurations):
    """
    Run fine-tuning experiments based on the provided configurations.
    Args:
        training_configurations (list): List of dictionaries containing configurations for fine-tuning.

        Returns:
            dict: A dictionary where each key is a unique identifier for an experiment,
                  and each value is a dictionary containing the experiment details.
                    Example:
                    {
                        "VJGHLBfagGsopVwsAG48ND": {
                            "model": "gpt-4.1-mini-2025-04-14",
                            "training_file_oai_id": "file-G8tstQfCKpgCcE8mzzoxrC",
                            "training_file": "file-views90-train",
                            "training_file": "file-views90-val",
                            "hyperparameters": null,
                            "ft_job_id": "ftjob-42a982ad"
                        },
                        "gWKFh54X2xGShKpsh5G934": {
                            "model": "gpt-4.1-mini-2025-04-14",
                            "training_file_oai_id": "file-views18425-train",
                            "training_file": "file-views18425-train",
                            "training_file": "file-views18425-val",
                            "hyperparameters": null,
                            "ft_job_id": "ftjob-5a2ef1e8"
                        }
                    }
    """
    experiments = {}
    
    logger.warning(f"This will run \"{len(training_configurations)}\" experiments which can become costly.")
    user_input = input(f"Do you want to continue experiments? (y/N): ")
    if user_input.lower() != 'y':
        logger.info("Aborting experiment run.")
        return None
    logger.info("Proceeding with experiments...")

    wandb.init(project="customain", job_type="fine-tuning")

    for config in training_configurations:
        # Validate required fields
        if not all(key in config for key in ["model", "training_file", "training_file_oai_id", "hyperparameters"]):
            logger.error(f"Missing required fields in config: {config}. Skipping this configuration.")
            continue
            
        method_config = None if config["hyperparameters"] is None else {
            "type": "supervised",
            "supervised": {
                "hyperparameters": config["hyperparameters"]
            }
        }
        
        try:
            response = run_finetuning(
                model=config["model"],
                training_file=config["training_file_oai_id"],
                ft_method_config=method_config
            )
            
            # Generate unique experiment ID
            experiment_id = str(shortuuid.uuid())
            
            # Store experiment config with UUID as key
            experiment_data = {
                "model": config["model"],
                "training_file": config["training_file"],
                "training_file_oai_id": config["training_file_oai_id"],
                "test_file": config["test_file"] if "test_file" in config else None,
                "test_file_oai_id": config["test_file_oai_id"] if "test_file_oai_id" in config else None,
                "hyperparameters": config["hyperparameters"],
                "ft_job_id": response.id,
            }
            experiments[experiment_id] = experiment_data

            wandb.log({
                "experiment_id": experiment_id,
                "model": config["model"],
                "ft_job_id": response.id,
                **(config["hyperparameters"] or {}),
            })

        except Exception as e:
            logger.exception(f"Error running experiment with config {config}: {str(e)}")
    
    output_path = Path(__file__).parent / "_experiments.json"
    with open(output_path, "w") as f:
        json.dump(experiments, f, indent=4)
    logger.info(f"Generated {len(experiments)} experiments.")
    logger.info(f"Experiments saved to {output_path}")

    wandb.finish()

    return experiments

