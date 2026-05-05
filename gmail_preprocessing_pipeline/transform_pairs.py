#!/usr/bin/env python
"""Clean, filter, and anonymize reply pairs in a single LLM pass."""

import argparse
import json
import random
import sys
from pathlib import Path

import anthropic

MODEL = "claude-haiku-4-5-20251001"
TEST_MODE = False
TEST_LIMIT = 20

SYSTEM_PROMPT = """\
You are preparing a high-quality email reply dataset.

For each email-reply pair you must do all of the following in one pass:
1. Clean both emails by removing signatures, contact blocks, legal disclaimers,
   quoted replies, inline image placeholders, and client tags.
2. Replace every URL with [LINK].
3. Replace every person name with [NAME] in the subject and both bodies.
4. Decide whether the pair should be kept.

Drop the pair if ANY of the following apply:
- Spam, newsletter, marketing, or automated notification
- Warmup / toaster / obviously synthetic or incoherent email text
- The reply does not meaningfully respond to the received email
- Either side has fewer than 2 meaningful sentences after cleaning

Preserve the actual message text exactly aside from the required cleaning,
URL replacement, and name anonymization.

Return STRICT JSON with this shape and nothing else:
{
  "keep": true,
  "subject": "...",
  "received_body": "...",
  "reply_body": "..."
}

If the pair should be dropped, return:
{
  "keep": false
}
"""


def _parse_response(text: str) -> dict:
    content = text.strip()
    if content.startswith("```"):
        lines = content.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"Model response did not contain JSON: {text}")
    return json.loads(content[start : end + 1])


def transform_pair(client: anthropic.Anthropic, pair: dict) -> dict | None:
    prompt = json.dumps(
        {
            "subject": pair.get("subject") or "",
            "received_body": pair.get("received_body") or "",
            "reply_body": pair.get("reply_body") or "",
        },
        ensure_ascii=False,
        indent=2,
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    payload = _parse_response(response.content[0].text)
    if not payload.get("keep"):
        return None
    return {
        "subject": payload.get("subject") or "",
        "received_body": payload.get("received_body") or "",
        "reply_body": payload.get("reply_body") or "",
    }


def process_file(input_path: Path, output_path: Path) -> None:
    client = anthropic.Anthropic()
    lines = input_path.read_text(encoding="utf-8").strip().splitlines()
    if TEST_MODE:
        lines = random.sample(lines, min(TEST_LIMIT, len(lines)))
        print(f"TEST MODE: randomly sampled {len(lines)} pairs")

    total = len(lines)
    kept = 0
    dropped = 0

    print(f"Transforming {total} reply pairs from {input_path} ...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for i, line in enumerate(lines, 1):
            pair = json.loads(line)
            transformed = transform_pair(client, pair)
            if transformed is None:
                dropped += 1
            else:
                f.write(json.dumps(transformed, ensure_ascii=False) + "\n")
                kept += 1

            if i % 10 == 0 or i == total:
                print(f"  {i}/{total} processed (kept {kept}, dropped {dropped})")

    print(f"Done. {kept} kept, {dropped} dropped -> {output_path}")


def main() -> None:
    from ._load_secrets import load_secrets

    load_secrets()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/_intermediate/reply_pairs_raw.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/_intermediate/reply_pairs_processed.jsonl"),
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found. Run extract_pairs.py first.", file=sys.stderr)
        sys.exit(1)

    process_file(args.input, args.output)


if __name__ == "__main__":
    main()