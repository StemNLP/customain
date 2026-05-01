import json
import string

import torch
from torch.utils.data import Dataset

PAD_IDX = 0
UNK_IDX = 1
CHARS = string.printable
CHAR_TO_IDX = {c: i + 2 for i, c in enumerate(CHARS)}
VOCAB_SIZE = len(CHARS) + 2

DEFAULT_MAX_LEN = 1024


def encode_text(text: str, max_len: int = DEFAULT_MAX_LEN) -> list[int]:
    indices = [CHAR_TO_IDX.get(c, UNK_IDX) for c in text[:max_len]]
    return indices + [PAD_IDX] * (max_len - len(indices))


class AuthorshipDataset(Dataset):
    def __init__(self, path: str, max_len: int = DEFAULT_MAX_LEN):
        self.max_len = max_len
        self.samples: list[tuple[str, int]] = []
        with open(path) as f:
            for line in f:
                record = json.loads(line)
                self.samples.append((record["text"], record["label"]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        text, label = self.samples[idx]
        encoded = encode_text(text, self.max_len)
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.float),
        )
