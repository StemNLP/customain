"""Dataset helpers for generic fine-tuning experiment directories."""

from __future__ import annotations

from pathlib import Path


def find_latest_dataset_dir(data_root: str | Path) -> Path:
    """Return the latest dataset directory under a data root.

    A dataset directory is any child directory that contains an `sft` or `dpo`
    folder. This keeps the FT pipeline independent from a specific data source
    such as Gmail.
    """
    root = Path(data_root)
    if (root / "sft").exists() or (root / "dpo").exists():
        return root

    candidates = [
        path for path in root.glob("*")
        if path.is_dir() and ((path / "sft").exists() or (path / "dpo").exists())
    ]

    # Backwards compatibility for existing data/gmail/<timestamp> datasets.
    gmail_root = root / "gmail"
    if gmail_root.exists():
        candidates.extend(
            path for path in gmail_root.glob("*")
            if path.is_dir() and ((path / "sft").exists() or (path / "dpo").exists())
        )

    if not candidates:
        raise FileNotFoundError(
            f"No dataset directories with sft/ or dpo/ splits found under {root}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def extract_eval_prompt_and_expected(example: dict) -> tuple[str | None, str | None]:
    """Extract prompt/reference text from common FT JSONL formats."""
    if "messages" in example:
        prompt = next(
            (msg.get("content") for msg in example["messages"] if msg.get("role") == "user"),
            None,
        )
        expected = next(
            (msg.get("content") for msg in reversed(example["messages"]) if msg.get("role") == "assistant"),
            None,
        )
        return prompt, expected

    if "input" in example:
        input_value = example["input"]
        if isinstance(input_value, dict) and "messages" in input_value:
            prompt = next(
                (msg.get("content") for msg in input_value["messages"] if msg.get("role") == "user"),
                None,
            )
        else:
            prompt = str(input_value)

        preferred = example.get("preferred_output")
        if isinstance(preferred, list):
            expected = next(
                (msg.get("content") for msg in preferred if msg.get("role") == "assistant"),
                None,
            )
        else:
            expected = str(preferred) if preferred is not None else None
        return prompt, expected

    prompt = example.get("prompt") or example.get("question") or example.get("instruction")
    expected = (
        example.get("completion")
        or example.get("response")
        or example.get("answer")
        or example.get("expected")
    )
    return prompt, expected
