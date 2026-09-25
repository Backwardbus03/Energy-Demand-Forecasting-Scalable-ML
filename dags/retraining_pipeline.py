"""Phase 8/13 - Airflow DAG: automated ingestion + retraining pipeline.

Runs on the project's own `.venv` (Python 3.13) via subprocess, invoked from
a separate `.venv-airflow` (Python 3.12) that only hosts the Airflow
scheduler/webserver. No project code runs inside Airflow's interpreter.

Pipeline:
    fetch (incremental) -> validate (gate) -> train candidate -> promote gate
        -> reload live serving models

Exit codes from the underlying scripts are used to gate progression:
    fetch_raw_data.py       0 = ok, 1 = failed / no data
    validate_raw_data.py    0 = passed, 1 = structural errors, 2 = no data
    train_and_evaluate.py   0 = ok
    promote_model.py        0 = promoted, 1 = rejected (regression), 2 = error

A rejected promotion is NOT a pipeline failure: it means the guardrail
worked and the previous production model was correctly kept in place.
"""

from __future__ import annotations

import pendulum
from airflow.decorators import dag, task
from airflow.exceptions import AirflowSkipException
from airflow.operators.bash import BashOperator

PROJECT_ROOT = "/Users/aryankulkarni/Desktop/BE/honours/Mini Project"
PROJECT_PYTHON = f'"{PROJECT_ROOT}/.venv/bin/python"'


def run(script_and_args: str) -> str:
    """Build a shell command that runs a project script with the project venv."""
    return f'cd "{PROJECT_ROOT}" && {PROJECT_PYTHON} {script_and_args}'


default_args = {
    "owner": "energy-forecasting",
    "retries": 1,
    "retry_delay": pendulum.duration(minutes=5),
}


@dag(
    dag_id="energy_demand_retraining_pipeline",
    description="Incremental EIA ingestion, validation, retraining, and gated promotion.",
    schedule="0 6 * * *",  # daily at 06:00
    start_date=pendulum.datetime(2026, 1, 1, tz="America/New_York"),
    catchup=False,
    default_args=default_args,
    tags=["energy", "retraining", "mlops"],
)
def energy_demand_retraining_pipeline():

    fetch = BashOperator(
        task_id="fetch_incremental_data",
        bash_command=run("scripts/fetch_raw_data.py --incremental"),
    )

    validate = BashOperator(
        task_id="validate_raw_data",
        bash_command=run("scripts/validate_raw_data.py"),
    )

    train_candidate = BashOperator(
        task_id="train_candidate_model",
        bash_command=run(
            "scripts/train_and_evaluate.py "
            '--models-dir "{{ params.project_root }}/models_candidate" '
            '--output-summary "{{ params.project_root }}/data/evaluation_summary_candidate.json"'
        ),
        params={"project_root": PROJECT_ROOT},
    )

    @task(task_id="promote_if_not_regressed")
    def promote_if_not_regressed() -> str:
        import subprocess

        result = subprocess.run(
            [
                f"{PROJECT_ROOT}/.venv/bin/python",
                f"{PROJECT_ROOT}/scripts/promote_model.py",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        print(result.stderr)

        if result.returncode == 0:
            return "promoted"
        if result.returncode == 1:
            # Guardrail correctly rejected a regressed candidate; not a
            # pipeline failure, but downstream reload should be skipped.
            raise AirflowSkipException("Candidate rejected by promotion gate (regression detected).")
        raise RuntimeError(f"promote_model.py errored (exit {result.returncode})")

    reload_live_models = BashOperator(
        task_id="reload_live_models",
        bash_command=(
            'curl -sf -X POST http://127.0.0.1:8000/admin/reload-models '
            '|| echo "serving app not reachable; models will load fresh on next server start"'
        ),
    )

    promotion = promote_if_not_regressed()

    fetch >> validate >> train_candidate >> promotion >> reload_live_models


energy_demand_retraining_pipeline()
