import json
from openai import OpenAI
from pathlib import Path
from .base import BaseEvaluator

SECRETS_FILE = Path(__file__).parents[3] / ".secrets" / "api_keps.json"

with open(SECRETS_FILE, "r") as f:
    _credentials = json.load(f)

_client = OpenAI(api_key=_credentials.get("openai_api_key"))

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of email tone and style. You will be given an expected (reference) email response and a generated email response. Rate how closely the generated response matches the tone and style of the expected response on a scale from 0.0 to 1.0.

Consider:
- Formality level (casual, professional, formal)
- Warmth and politeness
- Sentence structure and length patterns
- Use of greetings, sign-offs, and pleasantries
- Overall voice and personality

Ignore factual content, correctness, or completeness — focus only on tone and style.

Score guidelines:
- 1.0: Tone and style are virtually identical
- 0.7-0.9: Very similar tone with minor differences
- 0.4-0.6: Noticeably different tone but in the same general register
- 0.1-0.3: Very different tone (e.g. casual vs formal)
- 0.0: Completely mismatched tone

Respond with ONLY a JSON object in this exact format:
{"score": <float between 0.0 and 1.0>, "reasoning": "<brief explanation>"}"""

JUDGE_USER_PROMPT = """Expected response:
{expected}

Generated response:
{generated}"""


class ToneJudgeEvaluator(BaseEvaluator):
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model

    def name(self) -> str:
        return "tone_judge"

    def required_inputs(self) -> list:
        return ["expected", "generated"]

    def run(self, expected: str, generated: str) -> float:
        response = _client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": JUDGE_USER_PROMPT.format(
                    expected=expected, generated=generated
                )},
            ],
        )
        content = response.choices[0].message.content.strip()
        parsed = json.loads(content)
        score = float(parsed["score"])
        return max(0.0, min(1.0, score))
