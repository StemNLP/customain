import json
import os
from pathlib import Path

_SECRETS_PATH = Path(__file__).parent.parent / ".secrets" / "api_keps.json"


def load_secrets() -> None:
    """Load API keys from .secrets/api_keps.json into env vars (if not already set)."""
    if not _SECRETS_PATH.exists():
        return
    keys = json.loads(_SECRETS_PATH.read_text())
    _set_if_missing("ANTHROPIC_API_KEY", keys.get("anthropic_api_key"))
    _set_if_missing("OPENAI_API_KEY", keys.get("openai_api_key"))
    _set_if_missing("WANDB_API_KEY", keys.get("wandb_api_key"))


def _set_if_missing(env_var: str, value: str | None) -> None:
    if value and not os.environ.get(env_var):
        os.environ[env_var] = value
