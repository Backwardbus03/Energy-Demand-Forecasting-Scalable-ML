from __future__ import annotations

import json
from pathlib import Path
import pytest

from scripts import promote_model


def _create_summary(tiers_wmape: dict[str, float]) -> dict:
    return {
        "lightgbm_models": {
            tier: {"wmape": wmape, "mae": 100.0, "smape": 5.0, "r2": 0.9}
            for tier, wmape in tiers_wmape.items()
        }
    }


def test_promote_model_missing_candidate_files_exits_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    candidate_dir = tmp_path / "models_cand"
    candidate_summary = tmp_path / "cand_summary.json"
    live_dir = tmp_path / "models_live"
    live_summary = tmp_path / "live_summary.json"

    monkeypatch.setattr("sys.argv", [
        "promote_model.py",
        "--candidate-models-dir", str(candidate_dir),
        "--candidate-summary", str(candidate_summary),
        "--live-models-dir", str(live_dir),
        "--live-summary", str(live_summary),
        "--no-registry",
    ])

    exit_code = promote_model.main()
    assert exit_code == 2


def test_promote_model_first_deploy_promotes_unconditionally(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    candidate_dir = tmp_path / "models_cand"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    (candidate_dir / "model_tier_large.joblib").write_text("model_data_large")

    candidate_summary = tmp_path / "cand_summary.json"
    candidate_summary.write_text(json.dumps(_create_summary({
        "Large": 4.5, "Medium": 5.0, "Small": 6.0, "Overall": 5.1
    })))

    live_dir = tmp_path / "models_live"
    live_summary = tmp_path / "live_summary.json"

    monkeypatch.setattr("sys.argv", [
        "promote_model.py",
        "--candidate-models-dir", str(candidate_dir),
        "--candidate-summary", str(candidate_summary),
        "--live-models-dir", str(live_dir),
        "--live-summary", str(live_summary),
        "--no-registry",
    ])

    exit_code = promote_model.main()
    assert exit_code == 0
    assert (live_dir / "model_tier_large.joblib").exists()
    assert live_summary.exists()


def test_promote_model_promotes_when_candidate_better(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    candidate_dir = tmp_path / "models_cand"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    (candidate_dir / "model_tier_large.joblib").write_text("new_model")

    live_dir = tmp_path / "models_live"
    live_dir.mkdir(parents=True, exist_ok=True)
    (live_dir / "model_tier_large.joblib").write_text("old_model")

    candidate_summary = tmp_path / "cand_summary.json"
    candidate_summary.write_text(json.dumps(_create_summary({
        "Large": 4.0, "Medium": 4.5, "Small": 5.5, "Overall": 4.6
    })))

    live_summary = tmp_path / "live_summary.json"
    live_summary.write_text(json.dumps(_create_summary({
        "Large": 5.0, "Medium": 5.0, "Small": 6.0, "Overall": 5.3
    })))

    monkeypatch.setattr("sys.argv", [
        "promote_model.py",
        "--candidate-models-dir", str(candidate_dir),
        "--candidate-summary", str(candidate_summary),
        "--live-models-dir", str(live_dir),
        "--live-summary", str(live_summary),
        "--no-registry",
    ])

    exit_code = promote_model.main()
    assert exit_code == 0
    assert (live_dir / "model_tier_large.joblib").read_text() == "new_model"


def test_promote_model_rejects_when_regression_exceeds_threshold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    candidate_dir = tmp_path / "models_cand"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    (candidate_dir / "model_tier_large.joblib").write_text("candidate_bad")

    live_dir = tmp_path / "models_live"
    live_dir.mkdir(parents=True, exist_ok=True)
    (live_dir / "model_tier_large.joblib").write_text("live_good")

    # Large tier regressed by 2.5 pp (exceeds default 2.0 pp tolerance)
    candidate_summary = tmp_path / "cand_summary.json"
    candidate_summary.write_text(json.dumps(_create_summary({
        "Large": 7.5, "Medium": 5.0, "Small": 6.0, "Overall": 5.3
    })))

    live_summary = tmp_path / "live_summary.json"
    live_summary.write_text(json.dumps(_create_summary({
        "Large": 5.0, "Medium": 5.0, "Small": 6.0, "Overall": 5.3
    })))

    monkeypatch.setattr("sys.argv", [
        "promote_model.py",
        "--candidate-models-dir", str(candidate_dir),
        "--candidate-summary", str(candidate_summary),
        "--live-models-dir", str(live_dir),
        "--live-summary", str(live_summary),
        "--no-registry",
    ])

    exit_code = promote_model.main()
    assert exit_code == 1
    # Verify live model was left untouched
    assert (live_dir / "model_tier_large.joblib").read_text() == "live_good"

