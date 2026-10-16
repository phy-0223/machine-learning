import json

import pytest

pytest.importorskip("sklearn")

from ml_course.classical import run_experiment


def test_classical_workflow_writes_reproducible_outputs(tmp_path) -> None:
    results = run_experiment(tmp_path)
    assert set(results["model"]) == {"knn", "decision_tree", "random_forest"}
    assert results["test_f1"].between(0.8, 1.0).all()
    metadata = json.loads((tmp_path / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["test_set_policy"] == "untouched during model selection; evaluated once"
    assert (tmp_path / "model_comparison.csv").exists()

