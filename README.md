# Customain

> [!WARNING]
> Customain is in transition. As OpenAI and other commercial providers discontinue or reshape hosted fine-tuning APIs, the project is planned to move soon toward open-weight model workflows and providers such as Modal, Together AI, and FireworksHQ. Expect provider support and setup instructions to change as this migration lands.

**Post-training experiment bench for managed and open-weight provider workflows.**

Customain is for running post-training experiments, evaluating the resulting models with pluggable metrics, and selecting the best model for a task. It is no longer centered on learning one person's email style; the core project is a generic experimentation pipeline for provider-hosted and open-weight post-training workflows.

```text
Generic JSONL data -> post-training sweeps across providers/models -> eval runs -> weighted model ranking
```

## What This Project Is

Customain focuses on the operational loop around post-training:

1. Define generic SFT or DPO datasets.
2. Sweep models, providers, methods, and hyperparameters.
3. Launch provider-hosted or open-weight post-training jobs.
4. Run baseline and post-trained models on the same test split.
5. Evaluate outputs with pluggable task metrics.
6. Rank models with configurable metric weights.

The pipeline is currently provider-API first, with an upcoming shift toward open-weight post-training infrastructure. If you want local full training today, use a project built for training infrastructure such as [torchtune](https://docs.pytorch.org/torchtune/stable/), [Axolotl](https://github.com/axolotl-ai-cloud/axolotl), [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), or [Unsloth](https://docs.unsloth.ai/).

## Supported Providers

| Provider | Models | Methods | Status |
| -------- | ------ | ------- | ------ |
| **OpenAI** | GPT-4.1, 4.1-mini, 4.1-nano| SFT, DPO | ✅ Available |
| **Together AI** | Llama, Mixtral, Qwen + any HF model | -- | 🔜 Planned |
<!-- | **Fireworks AI** | Llama, Mixtral + open-source models | -- | 🔜 Planned |
| **Google Vertex AI** | Gemini 2.5 Flash/Lite, 2.0 Flash | -- | 🔜 Planned |
| **Mistral AI** | Mistral Small 3.1, Medium | -- | 🔜 Planned |
| **Cohere** | Command A, R+, R, Aya Expanse | -- | 🔜 Planned | -->

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- API keys for the post-training providers you plan to use
- Optional: Weights & Biases key for experiment tracking

### Install

```bash
git clone https://github.com/user/customain.git
cd customain
uv sync
```

### Configure Secrets

Create `.secrets/api_keps.json`:

```json
{
  "openai_api_key": "sk-...",
  "wandb_api_key": "optional",
  "together_api_key": "optional",
  "together_base_url": "https://api.together.xyz/v1",
  "fireworks_api_key": "optional",
  "fireworks_base_url": "https://api.fireworks.ai/inference/v1"
}
```

Only configure providers you use.

### Configure Experiments

Edit `ft/training_configs.py`:

```python
baseline_models = [
    {"provider": "openai", "model": "gpt-4.1-2025-04-14"},
]

llms = [
    {"provider": "openai", "model": "gpt-4.1-mini-2025-04-14"},
    {"provider": "openai", "model": "gpt-4.1-2025-04-14"},
]

training_methods = ["supervised", "dpo"]

metric_weights = {
    "task_judge": 1.0,
}
```

### Run The Pipeline

```bash
uv run python -m ft.run_pipeline --data-dir data/my_experiment
```

For a small smoke test, use mock files:

```bash
uv run python -m ft.run_pipeline \
  --data-dir data/my_experiment \
  --test-run
```

Skip completed stages when iterating:

```bash
uv run python -m ft.run_pipeline \
  --data-dir data/my_experiment \
  --skip 1 2
```

The pipeline writes:

| File | Purpose |
| --- | --- |
| `ft/_experiments.json` | Provider/model/method/job metadata |
| `ft/_ft_models_eval_runs.json` | Raw generations from baseline and post-trained models |
| `ft/_evaluation_results.json` | Per-datapoint and average metric scores |
| `ft/_model_ranking.json` | Weighted ranking used for model selection |

## Evaluation

Evaluation is pluggable. Drop a new evaluator into `ft/evaluation/evaluators/`; it will be auto-discovered if it subclasses `BaseEvaluator`.

The default direction is task-oriented model selection, not similarity scoring. The main generic evaluator is:

| Evaluator | What it measures |
| --- | --- |
| `task_judge` | LLM-as-judge score for task quality, instruction following, correctness, completeness, and clarity |

Legacy/specialized evaluators remain available but are skipped by default:

| Evaluator | Use when |
| --- | --- |
| `bleu`, `meteor`, `semantic_similarity` | You explicitly want reference similarity metrics |
| `tone_judge` | You explicitly care about style/register matching |
| `authorship_classifier` | You are running an authorship/style experiment with a trained classifier |

Configure default skips and the model-selection formula in `ft/training_configs.py`:

```python
skip_evaluators = [
    "authorship_classifier",
    "bleu",
    "meteor",
    "semantic_similarity",
    "tone_judge",
]

metric_weights = {
    "task_judge": 1.0,
}
```

Evaluators can require any subset of:

- `prompt`
- `expected`
- `generated`

This keeps the evaluation layer independent from Gmail, email tone, or reference-similarity assumptions.

## Optional Gmail Dataset Builder

The old Gmail preprocessing pipeline can still build SFT/DPO data if you want to experiment on email-style tasks:

```bash
uv run python -m gmail_preprocessing_pipeline.run_pipeline --targets sft dpo
```

That path is now optional project history, not the center of Customain.

## License

This project is licensed under the [GNU Affero General Public License v3.0 (AGPLv3)](license.txt).
