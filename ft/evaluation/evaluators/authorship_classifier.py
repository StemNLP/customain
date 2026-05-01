from .base import BaseEvaluator
from classifiers.authorship.predict import predict


_DEFAULT_CHECKPOINT = "classifiers/checkpoints/best_authorship_cnn.pt"


class AuthorshipClassifierEvaluator(BaseEvaluator):
    def __init__(self, checkpoint_path: str = _DEFAULT_CHECKPOINT):
        self._checkpoint_path = checkpoint_path

    def name(self) -> str:
        return "authorship_classifier"

    def required_inputs(self) -> list:
        return ["generated"]

    def run(self, generated: str) -> float:
        return predict(generated, self._checkpoint_path)
