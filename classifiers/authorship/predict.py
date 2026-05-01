import argparse

import torch

from classifiers.authorship.dataset import encode_text
from classifiers.authorship.model import TextCNN

DEFAULT_CHECKPOINT = "classifiers/checkpoints/best_authorship_cnn.pt"

_model: TextCNN | None = None
_config: dict | None = None
_device: torch.device | None = None


def load_model(checkpoint_path: str = DEFAULT_CHECKPOINT) -> None:
    global _model, _config, _device
    if torch.cuda.is_available():
        _device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        _device = torch.device("mps")
    else:
        _device = torch.device("cpu")

    checkpoint = torch.load(checkpoint_path, map_location=_device, weights_only=True)
    _config = checkpoint["config"]
    _model = TextCNN(
        vocab_size=_config["vocab_size"],
        embed_dim=_config["embed_dim"],
        num_filters=_config["num_filters"],
        kernel_sizes=_config["kernel_sizes"],
        dropout=_config["dropout"],
    ).to(_device)
    _model.load_state_dict(checkpoint["model_state_dict"])
    _model.eval()


def predict(text: str, checkpoint_path: str = DEFAULT_CHECKPOINT) -> float:
    if _model is None:
        load_model(checkpoint_path)
    encoded = encode_text(text, _config["max_len"])
    x = torch.tensor([encoded], dtype=torch.long).to(_device)
    with torch.no_grad():
        return _model(x).item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict authorship probability")
    parser.add_argument("text", type=str)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    score = predict(args.text, args.checkpoint)
    print(f"Authorship probability: {score:.4f}")
