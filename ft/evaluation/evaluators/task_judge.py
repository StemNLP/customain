import json
from pathlib import Path

from openai import OpenAI

from .base import BaseEvaluator

SECRETS_FILE = Path(__file__).parents[3] / ".secrets" / "api_keps.json"


def _load_client() -> OpenAI:
    credentials = {}
    if SECRETS_FILE.exists():
        with open(SECRETS_FILE, "r") as f:
            credentials = json.load(f)
    return OpenAI(api_key=credentials.get("openai_api_key"))


JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for fine-tuning experiments. You will be given a task prompt, an optional reference answer, and a model-generated answer.

Score the generated answer from 0.0 to 1.0 for overall task quality.

Consider:
- Correctness and factual consistency with the prompt
- Completeness and usefulness
- Instruction following
- Clarity and concision
- Whether the generated answer avoids unsupported claims

Use the reference answer only as guidance, not as a required exact match. Do not reward wording similarity by itself.

Respond with ONLY a JSON object in this exact format:
{"score": <float between 0.0 and 1.0>, "reasoning": "<brief explanation>"}"""

JUDGE_USER_PROMPT = """Prompt:
{prompt}

Reference answer:
{expected}

Generated answer:
{generated}"""


class TaskJudgeEvaluator(BaseEvaluator):
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self._client = None

    def name(self) -> str:
        return "task_judge"

    def required_inputs(self) -> list:
        return ["prompt", "expected", "generated"]

    def run(self, prompt: str, expected: str | None, generated: str) -> float:
        if self._client is None:
            self._client = _load_client()

        response = self._client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": JUDGE_USER_PROMPT.format(
                    prompt=prompt,
                    expected=expected or "(none provided)",
                    generated=generated,
                )},
            ],
        )
        content = response.choices[0].message.content.strip()
        parsed = json.loads(content)
        score = float(parsed["score"])
        return max(0.0, min(1.0, score))
