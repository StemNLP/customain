```
   ┌─────────────────-┐      ┌───────────┐     ┌─────────────────┐
   │  Your writing:   │      │           │     │                 │
   │                  │      │ Fine-Tune │     │  AI that writes │
   |                  │      | on YOUR   |     │    and sounds   │
   │  emails, docs,   │─────>│  style    │────>│  just like you. │
   │  messages, notes │      │           │     │                 │
   │  ...anything     │      │           │     │                 │
   └─────────────────-┘      └───────────┘     └─────────────────┘
```

# Customain

**Fine-tune OpenAI models to sound like you.**

Customain extracts your writing style from real conversations and text content, builds a training dataset, and fine-tunes language models to mimic your tone, voice, and communication patterns. The result is an AI that writes the way *you* would — not generic, not robotic, but authentically yours.

## How It Works

```
Your emails → Extract & clean → Fine-tune → A model that writes like you
```

1. **Connect** a content source (Gmail today, more coming)
2. **Process** your text into high-quality, anonymized training pairs
3. **Fine-tune** OpenAI models on your writing style
4. **Evaluate** how well the model captures your tone — with both classical metrics and a trained authorship classifier

## Supported Sources

| Source          | Status       |
| --------------- | ------------ |
| Gmail           | ✅ Available |
| Outlook         | 🔜 Planned   |
| Slack           | 🔜 Planned   |
| Notion          | 🔜 Planned   |
| Google Docs     | 🔜 Planned   |

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- OpenAI API key
- Gmail OAuth credentials (for Gmail source)

### Installation

```bash
git clone https://github.com/user/customain.git
cd customain
uv sync
```

### Configure API Keys

Create `.secrets/api_keps.json`:

```json
{
  "openai_api_key": "sk-...",
  "wandb_api_key": "optional-for-tracking"
}
```

For Gmail, you'll also need OAuth credentials — see [Google's guide](https://developers.google.com/gmail/api/quickstart/python).

### Step 1 — Build Your Dataset

Run the full Gmail preprocessing pipeline:

```bash
uv run python -m gmail_preprocessing_pipeline.run_pipeline
```

Or skip steps you've already completed:

```bash
# Already exported Gmail — start from extract
uv run python -m gmail_preprocessing_pipeline.run_pipeline --start-from 2

# Re-run just anonymize + format
uv run python -m gmail_preprocessing_pipeline.run_pipeline --start-from 5
```

The pipeline runs 6 steps:

1. **Export** Gmail threads to mbox
2. **Extract** email-reply pairs
3. **Clean** signatures, quotes, links (LLM)
4. **Filter** low-quality pairs (LLM)
5. **Anonymize** person names → `[NAME]` (LLM)
6. **Format** into SFT train/test split
email processing pipeline
Output: `data/sft_train.jsonl` and `data/sft_test.jsonl`

### Step 2 — Fine-Tune & Evaluate

Configure which models and hyperparameters to try in `ft/training_configs.py`, then run the full pipeline:

```bash
uv run python -m ft.run_pipeline \
  --train-file data/sft_train.jsonl \
  --test-file data/sft_test.jsonl
```

Or run a quick test with a small subset first:

```bash
uv run python -m ft.run_pipeline \
  --train-file data/sft_train.jsonl \
  --test-file data/sft_test.jsonl \
  --test-run
```

You can also skip steps you've already completed:

```bash
# Skip data upload and job launch, just evaluate
uv run python -m ft.run_pipeline \
  --train-file data/sft_train.jsonl \
  --test-file data/sft_test.jsonl \
  --skip 1 2
```

The pipeline will:
1. Upload data and launch fine-tuning jobs across your configured model/hyperparameter combinations
2. Poll until all jobs complete
3. Run each fine-tuned model on the test set
4. Evaluate results and log metrics to [Weights & Biases](https://wandb.ai)

## Evaluation

Customain includes a pluggable evaluation framework. Evaluators are auto-discovered — just drop a new one into `ft/evaluation/evaluators/`.

| Evaluator                | What it measures                           |
| ------------------------ | ------------------------------------------ |
| `authorship_classifier`  | CNN-based authorship probability score     |
| `tone_judge`             | LLM-as-judge scoring tone & style fidelity |
| `bleu`                   | N-gram overlap (BLEU score)                |
| `meteor`                 | Token-level alignment (METEOR score)       |
| `semantic_similarity`    | Embedding cosine similarity                |

Configure which evaluators to skip in `ft/training_configs.py`:

```python
skip_evaluators = ["bleu", "meteor"]  # Only run tone_judge and semantic_similarity
```

### Authorship Classifier

A character-level CNN trained to distinguish the author's writing from other people's emails. Unlike LLM-as-judge evaluators, this learns style patterns directly from data.

```bash
# Prepare training data from existing SFT data
uv run python -m classifiers.authorship.prepare_data

# Train (logs to W&B under customain-classifiers)
uv run python -m classifiers.authorship.train \
  --train-data data/classifiers/authorship/train.jsonl \
  --val-data data/classifiers/authorship/val.jsonl

# The authorship_classifier evaluator auto-registers and uses the trained checkpoint
```

## Project Structure

```
customain/
├── gmail_preprocessing_pipeline/
│   ├── run_pipeline.py          # End-to-end preprocessing orchestrator
│   ├── export_gmail.py          # Export replied threads from Gmail API
│   ├── extract_pairs.py         # mbox → raw email-reply pairs (JSONL)
│   ├── clean_pairs.py           # LLM-based body cleaning
│   ├── filter_pairs.py          # LLM-based quality filtering
│   ├── anonymize_pairs.py       # LLM-based name/PII anonymization
│   └── format_for_sft.py        # Format into OpenAI SFT JSONL + train/test split
├── ft/
│   ├── run_pipeline.py          # End-to-end fine-tuning pipeline
│   ├── finetuning.py            # OpenAI fine-tuning API wrapper
│   ├── training_configs.py      # Model, hyperparameter, and evaluator configs
│   ├── step_1_run_ft_jobs.py    # Launch FT jobs
│   ├── step_2_update_experiments.py  # Poll job status
│   ├── step_3_eval_run_ft_models.py  # Run FT models on test set
│   ├── step_4_run_evaluation.py      # Run evaluators
│   └── evaluation/
│       ├── core.py              # Evaluator runner
│       ├── registry.py          # Auto-discovery registry
│       └── evaluators/          # Drop-in evaluator modules
├── classifiers/
│   └── authorship/
│       ├── prepare_data.py      # Extract classifier data from SFT files
│       ├── dataset.py           # Char-level tokenization and Dataset
│       ├── model.py             # TextCNN architecture
│       ├── train.py             # Training loop with W&B logging
│       └── predict.py           # Inference utility
├── .secrets/                    # API keys and OAuth tokens (gitignored)
├── data/                        # All data artifacts (gitignored)
│   ├── _intermediate/           # Pipeline artifacts (mbox, raw/clean/filtered pairs)
│   ├── sft_train.jsonl          # Final training data
│   ├── sft_test.jsonl           # Final test data
│   └── classifiers/authorship/  # Classifier training data
└── pyproject.toml
```

## License

This project is licensed under the [GNU Affero General Public License v3.0 (AGPLv3)](license.txt).
