from openai import OpenAI
import json
import logging
from pathlib import Path
from .logging_config import setup_logger

logger = setup_logger(log_level=logging.DEBUG)

SECRETS_FILE = Path(__file__).parents[1] / ".secrets" / "api_keps.json"

with open(SECRETS_FILE, "r") as f:
    credentials = json.load(f)

client = OpenAI(
    api_key=credentials.get("openai_api_key"),
)


def run_finetuning(training_file,
                   model,
                   ft_method_config=None,
                   ):

    logger.debug(f"Training file: {training_file}")
    logger.debug(f"Model: {model}")
    
    kwargs = {}
    if ft_method_config is not None:
        kwargs["method"] = ft_method_config
        logger.debug("Overriding default fine-tuning method config with custom config.")
        logger.debug(f"Fine-tuning method config: {ft_method_config}")

    response = client.fine_tuning.jobs.create(
        training_file=training_file,
        model=model,
        **kwargs
    )
    return response


def query_fted_model_chat_completion(model_id,
                     user_query,
                     system_role_content="You are a helpful assistant.",
                     temperature=0.0,
                     num_responses=1,
                     ):
    """
    Query the fine-tuned model with a user query and return the response.
    
    Args:
        model_id (str): The ID of the fine-tuned model.
        user_query (str): The user's query.
        temperature (float): Sampling temperature. Must be between 0 and 2.
                             Higher values like 0.8 will make the response more random and creative, while
                             lower values like make it more deterministic.
        num_responses (int): The number of responses to generate. Default is 1.
    
    Returns:
        list: A list of responses from the model. The size of the list is equal to num_responses.

    """
    completion = client.chat.completions.create(
        model=model_id,
        n=num_responses,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_role_content},
            {"role": "user", "content": user_query}],
            # response_format={"type": "json_object"}
            )
    responses = []
    for i in range(num_responses):
        responses.append(completion.choices[i].message.content)
    return responses


def query_fted_model_responses(model_id,
                             user_query,
                             system_role_content="You are a helpful assistant.",
                             temperature=0.0,
                             num_responses=1,
                             ):
    """
    Query the fine-tuned model with a user query using the responses API and return the response.
    
    Args:
        model_id (str): The ID of the fine-tuned model.
        user_query (str): The user's query.
        system_role_content (str): The system role content for the prompt.
        temperature (float): Sampling temperature. Must be between 0 and 2.
                             Higher values like 0.8 will make the response more random and creative, while
                             lower values like make it more deterministic.
        num_responses (int): The number of responses to generate. Default is 1.
    
    Returns:
        list: A list of responses from the model. The size of the list is equal to num_responses.
    """
    response = client.responses.create(
        model=model_id,
        temperature=temperature,
        input=[
            {"role": "system", "content": system_role_content},
            {"role": "user", "content": user_query}],
            )
    return response.output_text


