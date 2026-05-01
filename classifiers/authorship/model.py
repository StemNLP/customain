import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 32,
        num_filters: int = 64,
        kernel_sizes: list[int] | None = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7]
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(num_filters * len(kernel_sizes), 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)              # (B, L, E)
        x = x.transpose(1, 2)              # (B, E, L)
        conv_outs = []
        for conv in self.convs:
            c = F.relu(conv(x))             # (B, F, L-k+1)
            c = c.max(dim=2).values         # (B, F)
            conv_outs.append(c)
        x = torch.cat(conv_outs, dim=1)     # (B, F*num_kernels)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return torch.sigmoid(self.fc2(x)).squeeze(1)
