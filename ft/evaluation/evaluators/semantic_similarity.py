from sentence_transformers import SentenceTransformer, util
from .base import BaseEvaluator

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


class SemanticSimilarityEvaluator(BaseEvaluator):
    def name(self) -> str:
        return "semantic_similarity"

    def required_inputs(self) -> list:
        return ["expected", "generated"]

    def run(self, expected: str, generated: str) -> float:
        model = _get_model()
        emb_expected = model.encode(expected, convert_to_tensor=True)
        emb_generated = model.encode(generated, convert_to_tensor=True)
        score = util.cos_sim(emb_expected, emb_generated).item()
        return score
