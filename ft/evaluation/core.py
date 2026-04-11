import logging
from ..logging_config import setup_logger

logger = setup_logger(log_level=logging.INFO)


def run_evaluators(evaluator_registry: dict,
                   input: dict,
                   skip_evaluators: list[str] = []) -> dict:
    """
    Run all evaluators passed by the evaluator_registry on the provided input.

    Args:
        evaluator_registry (dict): A dictionary of evaluator instances.
        input (dict): A dictionary containing input data for the evaluators.
            Example:
            {
                "expected": "the expected text",
                "generated": "the generated text"
            }
        skip_evaluators (list[str], optional): List of evaluator names to skip. Defaults to [].
    
    Returns:
        dict: A dictionary containing the results from evaluators.
        Each key is the evaluator's name and the value is a dictionary with the results.
    """
    results = {}
    logger.debug("Running evaluators on input")
    
    for name, evaluator in evaluator_registry.items():
        if name in skip_evaluators:
            logger.info(f"Skipping evaluator: {name}")
            continue
            
        logger.info(f"Preparing evaluator: {name}")
        required_keys = evaluator.required_inputs()
        if all(k in input for k in required_keys):
            input_subset = {k: input[k] for k in required_keys}
            logger.debug(f"Running evaluator: {name} - with required keys: {input_subset.keys()}")
            try:
                results[name] = evaluator.run(**input_subset)
                logger.info(f"Evaluator {name} completed successfully.")
            except Exception as e:
                logger.error(f"Error running evaluator {name}: {str(e)}")
                raise
        else:
            logger.error(f"Missing required inputs for evaluator {name}:. Required keys: {required_keys}")
            raise ValueError(f"Missing required inputs for evaluator {name}: {required_keys}")

    return results


def run_evaluators_on_batch(evaluator_registry: dict,
                            inputs: list[dict],
                            skip_evaluators: list[str] = []) -> list[dict]:
    """
    Run all evaluators passed by the evaluator_registry on the provided batch of inputs.

    Args:
        evaluator_registry (dict): A dictionary of evaluator instances.
        inputs (list[dict]): A list of dictionaries containing input data for the evaluators.
        skip_evaluators (list[str], optional): List of evaluator names to skip. Defaults to None.
    
    Returns:
        list[dict]: A list of dictionaries containing the results from evaluators for each input.
    """
    results = []
    for i, input in enumerate(inputs):
        logger.debug(f"Running evaluators for input {i}")
        result = run_evaluators(evaluator_registry, input, skip_evaluators)
        results.append(result)

    return results

