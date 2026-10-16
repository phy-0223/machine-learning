"""A two-layer NumPy network with explicit forward and backward passes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RANDOM_STATE = 42


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass
class TwoLayerNetwork:
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray
    loss_history: list[float]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        hidden = np.tanh(X @ self.W1 + self.b1)
        return sigmoid(hidden @ self.W2 + self.b2)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)


def xor_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([[0.0], [1.0], [1.0], [0.0]])
    return X, y


def train_xor(
    hidden_units: int = 8,
    learning_rate: float = 0.5,
    epochs: int = 5_000,
    seed: int = RANDOM_STATE,
) -> TwoLayerNetwork:
    X, y = xor_data()
    rng = np.random.default_rng(seed)
    W1 = rng.normal(scale=np.sqrt(1 / X.shape[1]), size=(X.shape[1], hidden_units))
    b1 = np.zeros((1, hidden_units))
    W2 = rng.normal(scale=np.sqrt(1 / hidden_units), size=(hidden_units, 1))
    b2 = np.zeros((1, 1))
    history: list[float] = []

    for _ in range(epochs):
        z1 = X @ W1 + b1
        a1 = np.tanh(z1)
        probabilities = sigmoid(a1 @ W2 + b2)
        probabilities_safe = np.clip(probabilities, 1e-8, 1 - 1e-8)
        loss = -np.mean(
            y * np.log(probabilities_safe) + (1 - y) * np.log(1 - probabilities_safe)
        )
        history.append(float(loss))

        # For sigmoid + binary cross-entropy, dL/dz2 simplifies to (p-y)/n.
        dz2 = (probabilities - y) / len(X)
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0, keepdims=True)
        dz1 = (dz2 @ W2.T) * (1 - a1**2)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0, keepdims=True)

        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

    return TwoLayerNetwork(W1, b1, W2, b2, history)


def main() -> None:
    X, y = xor_data()
    network = train_xor()
    probabilities = network.predict_proba(X)
    predictions = network.predict(X)
    print("XOR probabilities:", np.round(probabilities.ravel(), 4))
    print("XOR predictions:  ", predictions.ravel())
    print("Targets:          ", y.astype(int).ravel())
    print(f"loss: {network.loss_history[0]:.4f} -> {network.loss_history[-1]:.6f}")


if __name__ == "__main__":
    main()

