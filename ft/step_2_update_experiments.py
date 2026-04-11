import json
from pathlib import Path
import logging
from .logging_config import setup_logger
from .finetuning import client

logger = setup_logger(log_level=logging.INFO)

def update_experiments():
    """
    Update the experiment results with finetuned model IDs from OpenAI's fine-tuning jobs.
    This function reads the experiment results from a JSON file, retrieves the status of each fine-tuning job,
    and updates the JSON file with the finetuned model IDs if all jobs have succeeded.

    It's necessary step because the finetuned model IDs are not returned immediately after job creation,
    and we need to check the status of each job to ensure they have completed successfully before updating the results.
    """

    json_path = Path(__file__).parent / '_experiments.json'
    with open(json_path, 'r') as f:
        experiments = json.load(f)

    all_succeeded = True
    job_results = {}

    for exp_id, exp_data in experiments.items():
        ft_job_id = exp_data['ft_job_id']
        try:
            response = client.fine_tuning.jobs.retrieve(ft_job_id)
            if response.status != 'succeeded':
                logger.info(f"Job {ft_job_id} status: {response.status}")
                all_succeeded = False
                break
            job_results[exp_id] = response.fine_tuned_model
        except Exception as e:
            logger.exception(f"Error retrieving job {ft_job_id}: {str(e)}")
            all_succeeded = False
            break

    if all_succeeded:
        for exp_id, fine_tuned_model in job_results.items():
            experiments[exp_id]['ft_model_id'] = fine_tuned_model

        with open(json_path, 'w') as f:
            json.dump(experiments, f, indent=4)
        logger.info("Successfully updated all fine-tuned model IDs")
    else:
        logger.warning("Update aborted: Not all jobs have succeeded")

