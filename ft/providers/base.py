from __future__ import annotations

from abc import ABC, abstractmethod


class FineTuningProvider(ABC):
    """Provider API used by the experiment pipeline."""

    name: str

    @abstractmethod
    def upload_file(self, local_path: str) -> str:
        """Upload a training/eval file and return the provider file ID."""

    @abstractmethod
    def create_fine_tuning_job(
        self,
        *,
        training_file: str,
        model: str,
        method_config: dict | None = None,
    ):
        """Create a fine-tuning job and return the provider response."""

    @abstractmethod
    def retrieve_fine_tuning_job(self, job_id: str):
        """Return the latest job object for a fine-tuning job ID."""

    @abstractmethod
    def query_model(
        self,
        *,
        model_id: str,
        user_query: str,
        temperature: float = 0.0,
        num_responses: int = 1,
    ) -> list[str]:
        """Query a base or fine-tuned model and return generated text."""
