from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ._types import err as _fail


def _config_path(project_path: str) -> Path:
    return Path(project_path).expanduser().resolve() / ".mlops" / "config.json"


def _load_config(project_path: str) -> dict[str, Any]:
    path = _config_path(project_path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_config(project_path: str, payload: dict[str, Any]) -> None:
    path = _config_path(project_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _mlflow_module():
    try:
        import mlflow
    except ImportError:
        return None, _fail("mlflow not installed.")
    return mlflow, None


def mlflow_check_available(project_path: str = ".") -> dict[str, Any]:
    mlflow, error = _mlflow_module()
    if error:
        return error

    try:
        tracking_uri = mlflow.get_tracking_uri()
        config = _load_config(project_path)
        return {
            "success": True,
            "available": True,
            "tracking_uri": tracking_uri,
            "configured_tracking_uri": config.get("mlflow_tracking_uri"),
        }
    except Exception as exc:  # noqa: BLE001
        return _fail(str(exc))


def set_mlflow_tracking_uri(tracking_uri: str, project_path: str = ".") -> dict[str, Any]:
    mlflow, error = _mlflow_module()
    if error:
        return error

    if not tracking_uri.strip():
        return _fail("tracking_uri must not be empty")

    try:
        mlflow.set_tracking_uri(tracking_uri)
        config = _load_config(project_path)
        config["mlflow_tracking_uri"] = tracking_uri
        _save_config(project_path, config)
        return {
            "success": True,
            "tracking_uri": tracking_uri,
            "config_path": str(_config_path(project_path)),
        }
    except Exception as exc:  # noqa: BLE001
        return _fail(str(exc))


def list_mlflow_experiments() -> dict[str, Any]:
    mlflow, error = _mlflow_module()
    if error:
        return error

    try:
        experiments = mlflow.search_experiments()
        return {
            "success": True,
            "count": len(experiments),
            "experiments": [
                {
                    "experiment_id": str(exp.experiment_id),
                    "name": exp.name,
                    "artifact_location": exp.artifact_location,
                }
                for exp in experiments
            ],
        }
    except Exception as exc:  # noqa: BLE001
        return _fail(str(exc))


def get_mlflow_runs(experiment_id: str, max_results: int = 100) -> dict[str, Any]:
    mlflow, error = _mlflow_module()
    if error:
        return error

    try:
        runs_df = mlflow.search_runs(
            experiment_ids=[experiment_id], max_results=max_results
        )
        rows = runs_df.to_dict(orient="records")
        return {
            "success": True,
            "experiment_id": experiment_id,
            "count": len(rows),
            "runs": rows,
        }
    except Exception as exc:  # noqa: BLE001
        return _fail(str(exc))


def log_artifact_to_mlflow(local_path: str, run_id: str | None = None) -> dict[str, Any]:
    mlflow, error = _mlflow_module()
    if error:
        return error

    artifact = Path(local_path).expanduser().resolve()
    if not artifact.exists() or not artifact.is_file():
        return _fail(f"artifact file not found: {artifact}")

    try:
        if run_id:
            with mlflow.start_run(run_id=run_id):
                mlflow.log_artifact(str(artifact))
        else:
            mlflow.log_artifact(str(artifact))
        return {
            "success": True,
            "path": str(artifact),
            "run_id": run_id,
        }
    except Exception as exc:  # noqa: BLE001
        return _fail(str(exc))


def download_mlflow_artifact(
    run_id: str,
    artifact_path: str,
    destination_path: str,
) -> dict[str, Any]:
    mlflow, error = _mlflow_module()
    if error:
        return error

    destination = Path(destination_path).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    try:
        downloaded = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=artifact_path,
            dst_path=str(destination),
        )
        return {
            "success": True,
            "run_id": run_id,
            "artifact_path": artifact_path,
            "downloaded_path": str(downloaded),
        }
    except Exception as exc:  # noqa: BLE001
        return _fail(str(exc))


def register_model_in_mlflow(model_path: str, model_name: str) -> dict[str, Any]:
    mlflow, error = _mlflow_module()
    if error:
        return error

    target = Path(model_path).expanduser().resolve()
    if not target.exists() or not target.is_file():
        return _fail(f"model file not found: {target}")

    try:
        result = mlflow.register_model(model_uri=str(target), name=model_name)
        return {
            "success": True,
            "model_name": model_name,
            "model_uri": str(target),
            "version": result.version,
        }
    except Exception as exc:  # noqa: BLE001
        return _fail(str(exc))


def get_mlflow_model_versions(model_name: str) -> dict[str, Any]:
    mlflow, error = _mlflow_module()
    if error:
        return error

    try:
        client = mlflow.MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        return {
            "success": True,
            "model_name": model_name,
            "count": len(versions),
            "versions": [
                {
                    "name": mv.name,
                    "version": str(mv.version),
                    "current_stage": mv.current_stage,
                    "run_id": getattr(mv, "run_id", None),
                    "source": getattr(mv, "source", None),
                }
                for mv in versions
            ],
        }
    except Exception as exc:  # noqa: BLE001
        return _fail(str(exc))
