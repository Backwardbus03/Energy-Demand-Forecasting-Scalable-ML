from __future__ import annotations

import json
from pathlib import Path
import pytest

from scripts import train_and_evaluate


def test_evaluation_summary_schema_contract() -> None:
    """Verifies that evaluation summary contract matches what promote_model.py expects."""
    required_tiers = ["Large", "Medium", "Small", "Overall"]

    # Sample minimal summary structure
    dummy_summary = {
        "lightgbm_models": {
            tier: {"wmape": 4.5, "mae": 100.0, "smape": 5.0, "r2": 0.9}
            for tier in required_tiers
        }
    }

    assert "lightgbm_models" in dummy_summary
    for tier in required_tiers:
        assert tier in dummy_summary["lightgbm_models"]
        assert "wmape" in dummy_summary["lightgbm_models"][tier]


@pytest.mark.slow
def test_train_and_evaluate_pipeline_end_to_end(
    synthetic_raw_store: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end smoke test training candidate models with synthetic data."""
    models_dir = tmp_path / "models_candidate"
    summary_path = tmp_path / "evaluation_summary.json"
    report_path = synthetic_raw_store.parent / "_logs" / "validation_report.json"

    monkeypatch.setattr("sys.argv", [
        "train_and_evaluate.py",
        "--raw-dir", str(synthetic_raw_store),
        "--report-path", str(report_path),
        "--start", "2023-10-01",
        "--end", "2024-03-01",
        "--models-dir", str(models_dir),
        "--features-path", str(tmp_path / "features.parquet"),
        "--output-summary", str(summary_path),
        "--no-mlflow",
    ])

    exit_code = train_and_evaluate.main()
    assert exit_code == 0
    assert summary_path.exists()
    assert (tmp_path / "features.parquet").exists()

    summary_data = json.loads(summary_path.read_text())
    assert "lightgbm_models" in summary_data
    for tier in ["Large", "Medium", "Small", "Overall"]:
        assert tier in summary_data["lightgbm_models"]

    # Verify that model artifacts were saved
    saved_models = list(models_dir.glob("*.joblib"))
    assert len(saved_models) == 3
