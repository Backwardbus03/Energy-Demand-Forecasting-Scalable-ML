from __future__ import annotations

from pathlib import Path
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "daily_retraining.yml"


def test_workflow_file_exists_and_is_valid_yaml() -> None:
    assert WORKFLOW_PATH.exists(), f"Workflow not found at {WORKFLOW_PATH}"
    content = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    assert isinstance(content, dict)
    assert "jobs" in content


def test_workflow_contains_pytest_test_gate() -> None:
    content = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    jobs = content["jobs"]
    assert "test" in jobs, "Workflow must contain a 'test' gate job"
    test_job = jobs["test"]

    # Verify pytest execution in test job
    runs = [step.get("run", "") for step in test_job.get("steps", []) if "run" in step]
    assert any("pytest" in r for r in runs), "Test job must execute pytest"


def test_retraining_job_depends_on_test_gate() -> None:
    content = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    jobs = content["jobs"]
    assert "retrain_and_monitor" in jobs
    retrain_job = jobs["retrain_and_monitor"]

    needs = retrain_job.get("needs")
    assert needs == "test" or (isinstance(needs, list) and "test" in needs), (
        "retrain_and_monitor job must declare 'needs: test'"
    )


def test_retraining_job_has_write_permission() -> None:
    content = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    retrain_job = content["jobs"]["retrain_and_monitor"]
    permissions = retrain_job.get("permissions", {})
    assert permissions.get("contents") == "write", (
        "retrain_and_monitor must have 'permissions: contents: write' to commit promoted models"
    )


def test_step_ordering_training_before_drift_monitoring() -> None:
    content = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    steps = content["jobs"]["retrain_and_monitor"]["steps"]

    train_step_idx = -1
    drift_step_idx = -1
    promotion_step_idx = -1

    for idx, step in enumerate(steps):
        run_cmd = step.get("run", "")
        if "train_and_evaluate.py" in run_cmd:
            train_step_idx = idx
        elif "run_drift_monitoring.py" in run_cmd:
            drift_step_idx = idx
        elif "promote_model.py" in run_cmd:
            promotion_step_idx = idx

    assert train_step_idx != -1, "train_and_evaluate.py step not found"
    assert drift_step_idx != -1, "run_drift_monitoring.py step not found"
    assert promotion_step_idx != -1, "promote_model.py step not found"

    assert train_step_idx < drift_step_idx, (
        "Candidate training must run BEFORE drift monitoring so fresh features are analyzed"
    )
    assert drift_step_idx < promotion_step_idx, (
        "Drift monitoring must run before model promotion gate"
    )

