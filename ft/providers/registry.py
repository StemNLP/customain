from __future__ import annotations

from .base import FineTuningProvider
from .openai_provider import OpenAIProvider


_OPENAI_COMPATIBLE_PROVIDER_KEYS = {
    "openai_compatible",
    "together",
    "fireworks",
}


def get_provider(provider_name: str = "openai") -> FineTuningProvider:
    """Return a provider implementation by name.

    `openai_compatible`, `together`, and `fireworks` use the OpenAI SDK with
    provider-specific key/base-url entries from `.secrets/api_keps.json`, e.g.
    `together_api_key` and `together_base_url`.
    """
    normalized = provider_name.lower()
    if normalized == "openai":
        return OpenAIProvider()
    if normalized in _OPENAI_COMPATIBLE_PROVIDER_KEYS:
        return OpenAIProvider(
            api_key_name=f"{normalized}_api_key",
            base_url_name=f"{normalized}_base_url",
            provider_name=normalized,
        )
    raise ValueError(f"Unsupported fine-tuning provider: {provider_name}")
