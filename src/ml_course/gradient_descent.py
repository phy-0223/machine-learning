"""Linear regression implemented from scratch with NumPy gradient descent."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RANDOM_STATE = 42


@dataclass
class LinearModel:
    weights: np.ndarray
    bias: float
    loss_history: list[float]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights + self.bias


def make_regression_data(
    samples: int = 400, noise: float = 0.1, seed: int = RANDOM_STATE
) -> tuple[np.ndarray, np.ndarray]:
    """Generate data with known parameters w=[3, -2], b=0.5."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(samples, 2))
    y = X @ np.array([3.0, -2.0]) + 0.5 + rng.normal(scale=noise, size=samples)
    return X, y


def mse_and_gradients(
    X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float
) -> tuple[float, np.ndarray, float]:
    """Return mean squared error and its analytical gradients."""
    residual = X @ weights + bias - y
    loss = float(np.mean(residual**2))
    grad_weights = (2.0 / len(X)) * X.T @ residual
    grad_bias = float(2.0 * residual.mean())
    return loss, grad_weights, grad_bias


def finite_difference_gradient(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    bias: float,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, float]:
    """Compute a central-difference gradient for checking backprop algebra."""
    numeric_weights = np.zeros_like(weights, dtype=float)
    for index in range(len(weights)):
        plus = weights.copy()
        minus = weights.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        loss_plus = mse_and_gradients(X, y, plus, bias)[0]
        loss_minus = mse_and_gradients(X, y, minus, bias)[0]
        numeric_weights[index] = (loss_plus - loss_minus) / (2 * epsilon)

    loss_plus = mse_and_gradients(X, y, weights, bias + epsilon)[0]
    loss_minus = mse_and_gradients(X, y, weights, bias - epsilon)[0]
    numeric_bias = (loss_plus - loss_minus) / (2 * epsilon)
    return numeric_weights, float(numeric_bias)


def train_linear_regression(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.1,
    epochs: int = 300,
) -> LinearModel:
    weights = np.zeros(X.shape[1], dtype=float)
    bias = 0.0
    history: list[float] = []
    for _ in range(epochs):
        loss, grad_weights, grad_bias = mse_and_gradients(X, y, weights, bias)
        history.append(loss)
        weights -= learning_rate * grad_weights
        bias -= learning_rate * grad_bias
    final_loss = mse_and_gradients(X, y, weights, bias)[0]
    history.append(final_loss)
    return LinearModel(weights=weights, bias=bias, loss_history=history)


def run_experiment(output_dir: Path) -> LinearModel:
    output_dir.mkdir(parents=True, exist_ok=True)
    X, y = make_regression_data()
    model = train_linear_regression(X, y)

    plt.figure(figsize=(7, 4))
    plt.plot(model.loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Mean squared error")
    plt.title("Gradient descent convergence")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=160)
    plt.close()

    summary = {
        "learned_weights": model.weights.tolist(),
        "learned_bias": model.bias,
        "true_weights": [3.0, -2.0],
        "true_bias": 0.5,
        "initial_loss": model.loss_history[0],
        "final_loss": model.loss_history[-1],
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/gradient"))
    args = parser.parse_args()
    model = run_experiment(args.output)
    print(f"weights={model.weights}, bias={model.bias:.4f}")
    print(f"loss: {model.loss_history[0]:.4f} -> {model.loss_history[-1]:.6f}")
    print(f"Saved outputs to {args.output.resolve()}")


if __name__ == "__main__":
    main()
