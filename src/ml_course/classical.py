"""Leakage-safe comparison of KNN, decision tree, and random forest.

The bundled breast-cancer dataset keeps this exercise reproducible and offline.
Model selection uses only the training split; the untouched test split is used
once, after every candidate has been evaluated by stratified cross-validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42


def build_models() -> dict[str, Pipeline]:
    """Return three deliberately small, interpretable baseline pipelines."""
    return {
        "knn": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=7)),
            ]
        ),
        "decision_tree": Pipeline(
            [
                (
                    "model",
                    DecisionTreeClassifier(
                        max_depth=5,
                        min_samples_leaf=4,
                        random_state=RANDOM_STATE,
                    ),
                )
            ]
        ),
        "random_forest": Pipeline(
            [
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        min_samples_leaf=2,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                )
            ]
        ),
    }


def run_experiment(output_dir: Path) -> pd.DataFrame:
    """Evaluate candidates by CV, then evaluate each once on the test split."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    y = dataset.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    rows: list[dict[str, Any]] = []
    fitted_models: dict[str, Pipeline] = {}
    for name, model in build_models().items():
        cv_result = cross_validate(
            model,
            X_train,
            y_train,
            scoring=scoring,
            cv=cv,
            n_jobs=None,
        )
        model.fit(X_train, y_train)
        fitted_models[name] = model
        prediction = model.predict(X_test)
        probability = model.predict_proba(X_test)[:, 1]

        row: dict[str, Any] = {"model": name}
        for metric in scoring:
            values = cv_result[f"test_{metric}"]
            row[f"cv_{metric}_mean"] = float(values.mean())
            row[f"cv_{metric}_std"] = float(values.std())
        row.update(
            {
                "test_accuracy": float(accuracy_score(y_test, prediction)),
                "test_precision": float(precision_score(y_test, prediction)),
                "test_recall": float(recall_score(y_test, prediction)),
                "test_f1": float(f1_score(y_test, prediction)),
                "test_roc_auc": float(roc_auc_score(y_test, probability)),
            }
        )
        rows.append(row)

        ConfusionMatrixDisplay.from_predictions(
            y_test,
            prediction,
            display_labels=dataset.target_names,
            cmap="Blues",
            colorbar=False,
        )
        plt.title(f"Confusion matrix: {name}")
        plt.tight_layout()
        plt.savefig(output_dir / f"confusion_matrix_{name}.png", dpi=160)
        plt.close()

    results = pd.DataFrame(rows).sort_values("cv_f1_mean", ascending=False)
    results.to_csv(output_dir / "model_comparison.csv", index=False)

    best_name = str(results.iloc[0]["model"])
    metadata = {
        "dataset": "sklearn.datasets.load_breast_cancer",
        "random_state": RANDOM_STATE,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "selection_metric": "mean 5-fold stratified CV F1 on training data",
        "selected_model": best_name,
        "test_set_policy": "untouched during model selection; evaluated once",
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/classical"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_experiment(args.output)
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved reproducible outputs to {args.output.resolve()}")


if __name__ == "__main__":
    main()
