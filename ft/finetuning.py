import logging

from .logging_config import setup_logger
from .providers import get_provider

logger = setup_logger(log_level=logging.DEBUG)

_openai_provider = get_provider("openai")
client = _openai_provider.client


def upload_file_for_ft(local_path: str) -> str:
    """Backward-compatible OpenAI file upload helper."""
    return _openai_provider.upload_file(local_path)


def run_finetuning(training_file, model, ft_method_config=None):
    """Backward-compatible OpenAI fine-tuning helper."""
    logger.debug(f"Training file: {training_file}")
    logger.debug(f"Model: {model}")
    return _openai_provider.create_fine_tuning_job(
        training_file=training_file,
        model=model,
        method_config=ft_method_config,
    )


def query_fted_model_chat_completion(
    model_id,
    user_query,
    temperature=0.0,
    num_responses=1,
):
    """
    Query an OpenAI fine-tuned model with a user query and return responses.
    """
    return _openai_provider.query_model(
        model_id=model_id,
        user_query=user_query,
        temperature=temperature,
        num_responses=num_responses,
    )


def query_fted_model_responses(
    model_id,
    user_query,
    temperature=0.0,
    num_responses=1,
):
    """
    Query an OpenAI fine-tuned model with the Responses API.
    """
    response = client.responses.create(
        model=model_id,
        temperature=temperature,
        input=[{"role": "user", "content": user_query}],
    )
    return response.output_text
