"""A small, offline PyTorch CNN baseline for image classification.

The synthetic task distinguishes noisy horizontal and vertical bars. It is not
the final course project; it verifies the entire data/training/evaluation path
before a real dataset is introduced.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

RANDOM_STATE = 42


def seed_everything(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_bar_dataset(
    samples: int = 1_200,
    image_size: int = 16,
    seed: int = RANDOM_STATE,
) -> TensorDataset:
    """Return noisy images; class 0 is vertical and class 1 is horizontal."""
    generator = torch.Generator().manual_seed(seed)
    images = 0.15 * torch.randn(samples, 1, image_size, image_size, generator=generator)
    labels = torch.randint(0, 2, (samples,), generator=generator)
    positions = torch.randint(3, image_size - 3, (samples,), generator=generator)
    for index, (label, position) in enumerate(zip(labels, positions, strict=True)):
        location = int(position.item())
        if int(label.item()) == 0:
            images[index, 0, :, location - 1 : location + 1] += 1.0
        else:
            images[index, 0, location - 1 : location + 1, :] += 1.0
    return TensorDataset(images.clamp(0.0, 1.0), labels)


def make_loaders(
    batch_size: int = 64,
    seed: int = RANDOM_STATE,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    dataset = make_bar_dataset(seed=seed)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)
    train_end = int(0.70 * len(dataset))
    val_end = int(0.85 * len(dataset))

    def subset(start: int, end: int) -> TensorDataset:
        selected = indices[start:end]
        images, labels = dataset.tensors
        return TensorDataset(images[selected], labels[selected])

    train_loader = DataLoader(
        subset(0, train_end), batch_size=batch_size, shuffle=True, generator=generator
    )
    val_loader = DataLoader(subset(train_end, val_end), batch_size=batch_size)
    test_loader = DataLoader(subset(val_end, len(dataset)), batch_size=batch_size)
    return train_loader, val_loader, test_loader


class TinyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(16, 2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        return self.classifier(features.flatten(1))


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, float]:
    model.eval()
    loss_function = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    count = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += float(loss_function(logits, labels).item()) * len(labels)
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        count += len(labels)
    return total_loss / count, correct / count


def train(
    epochs: int = 8,
    learning_rate: float = 1e-3,
    batch_size: int = 64,
) -> tuple[TinyCNN, list[dict[str, float]], float]:
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = make_loaders(batch_size=batch_size)
    model = TinyCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_function(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(labels)
            seen += len(labels)
        val_loss, val_accuracy = evaluate(model, val_loader, device)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": running_loss / seen,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )

    _, test_accuracy = evaluate(model, test_loader, device)
    return model, history, test_accuracy


def run_experiment(output_dir: Path, epochs: int = 8) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model, history, test_accuracy = train(epochs=epochs)
    torch.save(model.state_dict(), output_dir / "tiny_cnn_state.pt")

    epoch_values = [int(row["epoch"]) for row in history]
    plt.figure(figsize=(7, 4))
    plt.plot(epoch_values, [row["train_loss"] for row in history], label="train")
    plt.plot(epoch_values, [row["val_loss"] for row in history], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Tiny CNN learning curves")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "learning_curves.png", dpi=160)
    plt.close()

    result: dict[str, object] = {
        "task": "synthetic horizontal-vs-vertical bar classification",
        "epochs": epochs,
        "random_state": RANDOM_STATE,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "test_accuracy": test_accuracy,
        "history": history,
        "note": "The test split was not used during training or model selection.",
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/cnn"))
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    result = run_experiment(args.output, epochs=args.epochs)
    print(f"device={result['device']}, test_accuracy={result['test_accuracy']:.3f}")
    print(f"Saved outputs to {args.output.resolve()}")


if __name__ == "__main__":
    main()
