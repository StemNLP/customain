from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

from openai import OpenAI

from .base import FineTuningProvider
from ..logging_config import setup_logger

logger = setup_logger(log_level=logging.DEBUG)

SECRETS_FILE = Path(__file__).parents[2] / ".secrets" / "api_keps.json"
UPLOAD_CACHE_FILE = Path(__file__).parents[1] / "_uploaded_files.json"


def _load_credentials() -> dict:
    if not SECRETS_FILE.exists():
        return {}
    with open(SECRETS_FILE, "r") as f:
        return json.load(f)


def _hash_file(path: str) -> str:
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


class OpenAIProvider(FineTuningProvider):
    """OpenAI FT provider.

    A custom `base_url` can be supplied in `api_keps.json` for providers that
    expose an OpenAI-compatible fine-tuning API surface.
    """

    name = "openai"

    def __init__(
        self,
        api_key_name: str = "openai_api_key",
        base_url_name: str = "openai_base_url",
        provider_name: str = "openai",
    ):
        credentials = _load_credentials()
        api_key = credentials.get(api_key_name) or os.environ.get(api_key_name.upper())
        base_url = credentials.get(base_url_name) or os.environ.get(base_url_name.upper())
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.name = provider_name

        if credentials.get("wandb_api_key") and not os.environ.get("WANDB_API_KEY"):
            os.environ["WANDB_API_KEY"] = credentials["wandb_api_key"]

    def upload_file(self, local_path: str) -> str:
        file_hash = _hash_file(local_path)
        cache_key = f"{self.name}:{file_hash}"
        cache = _load_upload_cache()

        if cache_key in cache:
            cached_id = cache[cache_key]["provider_file_id"]
            try:
                self.client.files.retrieve(cached_id)
                logger.info(f"Reusing cached upload for {local_path} -> {cached_id}")
                return cached_id
            except Exception:
                logger.info(f"Cached file {cached_id} no longer exists, re-uploading")
                del cache[cache_key]

        logger.info(f"Uploading {local_path} to {self.name}...")
        with open(local_path, "rb") as f:
            response = self.client.files.create(file=f, purpose="fine-tune")
        logger.info(f"Uploaded {local_path} -> {response.id}")

        cache[cache_key] = {"provider_file_id": response.id, "local_path": local_path}
        _save_upload_cache(cache)
        return response.id

    def create_fine_tuning_job(
        self,
        *,
        training_file: str,
        model: str,
        method_config: dict | None = None,
    ):
        kwargs = {}
        if method_config is not None:
            kwargs["method"] = method_config
        return self.client.fine_tuning.jobs.create(
            training_file=training_file,
            model=model,
            **kwargs,
        )

    def retrieve_fine_tuning_job(self, job_id: str):
        return self.client.fine_tuning.jobs.retrieve(job_id)

    def query_model(
        self,
        *,
        model_id: str,
        user_query: str,
        temperature: float = 0.0,
        num_responses: int = 1,
    ) -> list[str]:
        completion = self.client.chat.completions.create(
            model=model_id,
            n=num_responses,
            temperature=temperature,
            messages=[{"role": "user", "content": user_query}],
        )
        return [choice.message.content for choice in completion.choices]
