from openai import OpenAI
import hashlib
import json
import logging
import os
from pathlib import Path
from .logging_config import setup_logger

logger = setup_logger(log_level=logging.DEBUG)

SECRETS_FILE = Path(__file__).parents[1] / ".secrets" / "api_keps.json"

with open(SECRETS_FILE, "r") as f:
    credentials = json.load(f)

client = OpenAI(
    api_key=credentials.get("openai_api_key"),
)

# Set wandb API key from secrets if not already in env
if credentials.get("wandb_api_key") and not os.environ.get("WANDB_API_KEY"):
    os.environ["WANDB_API_KEY"] = credentials["wandb_api_key"]


UPLOAD_CACHE_FILE = Path(__file__).parent / "_uploaded_files.json"


def _hash_file(path: str) -> str:
    """SHA-256 of file content."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_upload_cache() -> dict:
    if UPLOAD_CACHE_FILE.exists():
        return json.loads(UPLOAD_CACHE_FILE.read_text())
    return {}


def _save_upload_cache(cache: dict) -> None:
    UPLOAD_CACHE_FILE.write_text(json.dumps(cache, indent=2))


def upload_file_for_ft(local_path: str) -> str:
    """
    Upload a local JSONL file to OpenAI for fine-tuning and return the file ID.

    Caches uploads by file content hash in `_uploaded_files.json` to avoid
    re-uploading the same content on subsequent runs.
    """
    file_hash = _hash_file(local_path)
    cache = _load_upload_cache()

    if file_hash in cache:
        cached_id = cache[file_hash]["oai_id"]
        # Verify the file still exists on OpenAI's side
        try:
            client.files.retrieve(cached_id)
            logger.info(f"Reusing cached upload for {local_path} -> {cached_id}")
            return cached_id
        except Exception:
            logger.info(f"Cached file {cached_id} no longer on OpenAI, re-uploading")
            del cache[file_hash]

    logger.info(f"Uploading {local_path} to OpenAI...")
    with open(local_path, "rb") as f:
        response = client.files.create(file=f, purpose="fine-tune")
    logger.info(f"Uploaded {local_path} -> {response.id}")

    cache[file_hash] = {"oai_id": response.id, "local_path": local_path}
    _save_upload_cache(cache)
    return response.id


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
            {"role": "user", "content": user_query}],
            )
    responses = []
    for i in range(num_responses):
        responses.append(completion.choices[i].message.content)
    return responses


def query_fted_model_responses(model_id,
                             user_query,
                             temperature=0.0,
                             num_responses=1,
                             ):
    """
    Query the fine-tuned model with a user query using the responses API and return the response.

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
    response = client.responses.create(
        model=model_id,
        temperature=temperature,
        input=[
            {"role": "user", "content": user_query}],
            )
    return response.output_text


