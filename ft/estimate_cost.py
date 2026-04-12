"""Estimate OpenAI fine-tuning cost before launching jobs.

OpenAI fine-tuning is priced per 1M training tokens:
    cost = (tokens_in_training_file * n_epochs * price_per_1M_tokens) / 1_000_000

Reference: https://openai.com/api/pricing/
"""

import json
from pathlib import Path
import tiktoken

# USD per 1M training tokens. Update as OpenAI prices change.
TRAINING_PRICE_PER_1M = {
    "gpt-4.1-2025-04-14": 25.00,
    "gpt-4.1-mini-2025-04-14": 5.00,
    "gpt-4.1-nano-2025-04-14": 1.50,
    "gpt-4o-2024-08-06": 25.00,
    "gpt-4o-mini-2024-07-18": 3.00,
    "o4-mini-2025-04-16": 100.00,
}

DEFAULT_N_EPOCHS = 3


def count_tokens_in_file(train_file: str, model: str = "gpt-4o") -> int:
    """Count tokens across all message contents in a training JSONL file."""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("o200k_base")

    total = 0
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            for msg in data.get("messages", []):
                total += len(enc.encode(msg.get("content", "")))
    return total


def estimate_cost(train_file: str, model: str, n_epochs: int = DEFAULT_N_EPOCHS) -> dict:
    """
    Estimate fine-tuning cost for a given training file and model.

    Returns:
        dict with keys: tokens, n_epochs, price_per_1M, cost_usd
    """
    tokens = count_tokens_in_file(train_file, model=model)
    price = TRAINING_PRICE_PER_1M.get(model)
    if price is None:
        return {"tokens": tokens, "n_epochs": n_epochs, "price_per_1M": None, "cost_usd": None}
    cost = (tokens * n_epochs * price) / 1_000_000
    return {"tokens": tokens, "n_epochs": n_epochs, "price_per_1M": price, "cost_usd": round(cost, 4)}


def estimate_configs_cost(configs: list) -> tuple[list[dict], float]:
    """
    Estimate cost for a list of fine-tuning configs.

    Returns:
        (per_config_estimates, total_cost_usd)
    """
    estimates = []
    total = 0.0
    for cfg in configs:
        n_epochs = (cfg.get("hyperparameters") or {}).get("n_epochs", DEFAULT_N_EPOCHS)
        est = estimate_cost(cfg["training_file"], cfg["model"], n_epochs=n_epochs)
        est["model"] = cfg["model"]
        est["hyperparameters"] = cfg.get("hyperparameters")
        estimates.append(est)
        if est["cost_usd"] is not None:
            total += est["cost_usd"]
    return estimates, round(total, 4)
