import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

from classifiers.authorship.dataset import AuthorshipDataset, VOCAB_SIZE, DEFAULT_MAX_LEN
from classifiers.authorship.model import TextCNN


def train() -> None:
    args = _parse_args()

    if not args.no_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))

    train_ds = AuthorshipDataset(args.train_data, args.max_len)
    val_ds = AuthorshipDataset(args.val_data, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    device = _get_device()
    model = TextCNN(
        VOCAB_SIZE, args.embed_dim, args.num_filters, args.kernel_sizes, args.dropout
    ).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = 0.0
    patience_counter = 0

    for epoch in range(args.epochs):
        train_loss = _train_epoch(model, train_loader, criterion, optimizer, device)
        train_loss /= len(train_ds)

        metrics = _validate(model, val_loader, criterion, device, len(val_ds))
        metrics["epoch"] = epoch
        metrics["train_loss"] = train_loss

        print(
            f"Epoch {epoch:3d} | train_loss={train_loss:.4f} "
            f"val_loss={metrics['val_loss']:.4f} acc={metrics['val_accuracy']:.3f} "
            f"prec={metrics['val_precision']:.3f} rec={metrics['val_recall']:.3f} "
            f"f1={metrics['val_f1']:.3f}"
        )

        if not args.no_wandb:
            wandb.log(metrics)

        if metrics["val_f1"] > best_f1:
            best_f1 = metrics["val_f1"]
            patience_counter = 0
            _save_checkpoint(model, args, metrics, checkpoint_dir)
            print(f"  -> Saved best model (f1={best_f1:.3f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if not args.no_wandb:
        wandb.finish()


def _train_epoch(
    model: TextCNN,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss


def _validate(
    model: TextCNN,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    dataset_size: int,
) -> dict:
    model.eval()
    val_loss = 0.0
    tp = fp = fn = tn = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            val_loss += criterion(pred, y).item() * x.size(0)
            binary = (pred >= 0.5).float()
            tp += ((binary == 1) & (y == 1)).sum().item()
            fp += ((binary == 1) & (y == 0)).sum().item()
            fn += ((binary == 0) & (y == 1)).sum().item()
            tn += ((binary == 0) & (y == 0)).sum().item()

    total = max(tp + fp + fn + tn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "val_loss": val_loss / dataset_size,
        "val_accuracy": (tp + tn) / total,
        "val_precision": precision,
        "val_recall": recall,
        "val_f1": f1,
    }


def _save_checkpoint(
    model: TextCNN, args: argparse.Namespace, metrics: dict, checkpoint_dir: Path
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "vocab_size": VOCAB_SIZE,
                "embed_dim": args.embed_dim,
                "num_filters": args.num_filters,
                "kernel_sizes": args.kernel_sizes,
                "dropout": args.dropout,
                "max_len": args.max_len,
            },
            "metrics": metrics,
        },
        checkpoint_dir / "best_authorship_cnn.pt",
    )


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train authorship CNN classifier")
    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--val-data", type=str, required=True)
    parser.add_argument(
        "--checkpoint-dir", type=str, default="classifiers/checkpoints"
    )
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--num-filters", type=int, default=64)
    parser.add_argument("--kernel-sizes", type=int, nargs="+", default=[3, 5, 7])
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--wandb-project", type=str, default="customain-classifiers"
    )
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train()
