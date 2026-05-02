# Baselines — not fine-tuned, but evaluated on the test set for comparison
baseline_models = [
    "gpt-4.1-2025-04-14",
]

llms = [
    # "gpt-4.1-nano-2025-04-14", # Cheapest model $1.50 / hour or training
    # "gpt-4o-mini-2024-07-18",  # Second cheapest model $3 / hour for training
    # "gpt-4.1-mini-2025-04-14", # $5.00 / hour for training
    "gpt-4.1-2025-04-14", # Powerful, but expensive. $25.00 / hour for training
    # o4-mini-2025-04-16, # Powerful, best model in coding as of June 2025. but very expensive. $100.00 / hour for training
]


# NOTE: All of the following values must be set in order to run the fine-tuning jobs with these combinations.
# Only the llms will be used.
# If e.g. batch_sizes is empty, other hyperparameters will not be used.

# OpenAI uses a default batch size dynamically calculated, 
# which is roughly 0.2% of the training examples, capped at 256.
# This approach is generally found to be effective for larger datasets. 
# Set to: 4,8,16,32
batch_sizes = [
    # 4, 8
]

# Best when set to: 0.02, 0.05, 0.1, 0.2
learning_rate_multipliers = [
    # 0.05
]

# If True, add one extra FT job per LLM with all hyperparameters left to OpenAI's
# defaults — useful as a reference baseline alongside swept configs.
include_default_hyperparam_config = True

# Training methods to run: "supervised" (SFT) and/or "dpo" (Direct Preference Optimization).
# The FT pipeline auto-resolves data files: sft_train.jsonl for supervised, dpo_train.jsonl for dpo.
training_methods = ["supervised"]

# Evaluator names to skip during evaluation (step 4).
# e.g. ["bleu", "meteor", "semantic_similarity"]
skip_evaluators = ["bleu", "meteor", "semantic_similarity"]
