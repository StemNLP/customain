from nltk.translate.meteor_score import meteor_score
from .base import BaseEvaluator


class MeteorEvaluator(BaseEvaluator):
    def name(self) -> str:
        return "meteor"

    def required_inputs(self) -> list:
        return ["expected", "generated"]

    def run(self, expected: str, generated: str) -> float:
        reference = expected.split()
        hypothesis = generated.split()
        return meteor_score([reference], hypothesis)
