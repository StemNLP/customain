# Customain

Email dataset pipeline for building instruction-tuning data from Gmail, and fine-tuning OpenAI models on it.

## Repo Structure

```
customain/
├── data_processing/
│   ├── export_gmail.py       # Step 0: Export replied threads from Gmail API
│   ├── extract_pairs.py      # Step 1: mbox -> raw email-reply pairs (JSONL)
│   ├── clean_pairs.py        # Step 2: LLM-based body cleaning (signatures, quotes, links)
│   ├── filter_pairs.py       # Step 3: LLM-based quality filter (drops warmup, spam, too-short)
│   └── format_for_sft.py     # Step 4: Format into OpenAI SFT JSONL + train/test split
├── ft/
│   ├── finetuning.py         # OpenAI fine-tuning API wrapper
│   ├── training_configs.py   # Model and hyperparameter configs
│   ├── step_1_run_ft_jobs.py # Launch FT jobs from configs
│   ├── step_2_update_experiments.py  # Poll job status, update model IDs
│   ├── step_3_eval_run_ft_models.py  # Run FT models on test set
│   ├── step_4_run_evaluation.py      # Evaluate with registered evaluators
│   ├── evaluation/
│   │   ├── core.py           # Generic evaluator runner
│   │   ├── registry.py       # Auto-discovers evaluators
│   │   └── evaluators/
│   │       ├── base.py               # Abstract base class
│   │       ├── bleu.py               # BLEU score
│   │       ├── meteor.py             # METEOR score
│   │       └── semantic_similarity.py # Cosine similarity (sentence-transformers)
│   └── logging_config.py    # Logging setup
├── .secrets/                 # API keys and OAuth tokens (gitignored)
├── data/                     # All data artifacts (gitignored)
└── pyproject.toml
```

## Data Processing Pipeline

```
Gmail API -> mbox -> raw pairs -> cleaned pairs -> filtered pairs -> SFT format
 (step 0)  (step 1)   (step 2)      (step 3)        (step 4)
```

### Step 0: Export from Gmail

```bash
uv run python data_processing/export_gmail.py
```

### Step 1: Extract reply pairs

```bash
uv run python data_processing/extract_pairs.py
```

### Step 2: Clean bodies (LLM)

```bash
uv run python data_processing/clean_pairs.py
```

### Step 3: Quality filter (LLM)

```bash
uv run python data_processing/filter_pairs.py
```

### Step 4: Format for SFT

```bash
uv run python data_processing/format_for_sft.py
```

Output: `data/sft_train.jsonl`, `data/sft_test.jsonl`

## Fine-Tuning Pipeline

After data processing, the `ft/` module handles fine-tuning and evaluation:

1. Configure models and hyperparameters in `ft/training_configs.py`
2. Launch fine-tuning jobs (step 1)
3. Poll for completion (step 2)
4. Run models on test set (step 3)
5. Evaluate with BLEU, METEOR, and semantic similarity (step 4)

Evaluators are auto-discovered from `ft/evaluation/evaluators/`. Add new evaluators by subclassing `BaseEvaluator`.

## Setup

```bash
uv sync
```

Requires Python 3.11+. API keys in `.secrets/api_keps.json`.
