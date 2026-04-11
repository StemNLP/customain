from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from .base import BaseEvaluator


class BleuEvaluator(BaseEvaluator):
    def name(self) -> str:
        return "bleu"

    def required_inputs(self) -> list:
        return ["expected", "generated"]

    def run(self, expected: str, generated: str) -> float:
        reference = expected.split()
        hypothesis = generated.split()
        smoothie = SmoothingFunction().method1
        return sentence_bleu([reference], hypothesis, smoothing_function=smoothie)
