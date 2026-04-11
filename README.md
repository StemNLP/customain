# Customain

Email dataset pipeline for building instruction-tuning data from Gmail exports.

## Repo Structure

```
customain/
├── data_processing/
│   ├── export_gmail.py       # Step 0: Export replied threads from Gmail API
│   ├── extract_pairs.py      # Step 1: mbox -> raw email-reply pairs (JSONL)
│   ├── clean_pairs.py        # Step 2: LLM-based body cleaning (signatures, quotes, links)
│   └── filter_pairs.py       # Step 3: LLM-based quality filter (drops warmup, spam, too-short)
├── .secrets/                  # OAuth credentials and tokens (gitignored)
├── data/                      # All data artifacts (gitignored)
└── pyproject.toml
```

## Data Processing Pipeline

Each step reads from the previous step's output and writes a new file.
Steps are run independently so you can re-run any step without repeating earlier ones.

```
Gmail API -> mbox -> raw pairs -> cleaned pairs -> filtered pairs
  (step 0)   (step 1)  (step 2)      (step 3)
```

### Step 0: Export from Gmail

Fetches threads where you replied via the Gmail API. Requires OAuth credentials in `.secrets/`.

```bash
uv run python data_processing/export_gmail.py
```

Output: `data/new_threads.mbox`

### Step 1: Extract reply pairs

Parses the mbox file, matches emails by `In-Reply-To` headers, and outputs raw email-reply pairs.
Skips pairs where either body is missing.

```bash
uv run python data_processing/extract_pairs.py
```

Output: `data/reply_pairs_raw.jsonl`

### Step 2: Clean bodies (LLM)

Sends each email body to Claude Haiku to strip signatures, contact blocks, disclaimers, quoted replies, and replaces URLs with `[LINK]`.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
uv run python data_processing/clean_pairs.py
```

Output: `data/reply_pairs_clean.jsonl`

### Step 3: Quality filter (LLM)

Drops low-quality pairs: warmup/toaster emails, spam, too-short exchanges, incoherent replies.

```bash
uv run python data_processing/filter_pairs.py
```

Output: `data/reply_pairs_filtered.jsonl`

## Output Format

Each line in the final JSONL is a flat record:

```json
{"subject": "Re: Meeting next week", "received_body": "Hi Meghdad, ...", "reply_body": "Hi Alex, ..."}
```

## Setup

```bash
uv sync
```

Requires Python 3.11+ and `ANTHROPIC_API_KEY` for steps 2-3.
