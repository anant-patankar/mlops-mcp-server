from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ._types import err as _fail


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tracker_root(project_path: str) -> Path:
    root = Path(project_path).expanduser().resolve()
    return root / ".mlops" / "experiments"


def _registry_path(project_path: str) -> Path:
    return _tracker_root(project_path) / "registry.json"


def _run_dir(project_path: str, run_id: str) -> Path:
    return _tracker_root(project_path) / run_id


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f"{path.suffix}.tmp")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp.replace(path)


def _update_registry(project_path: str, record: dict[str, Any]) -> None:
    reg_path = _registry_path(project_path)
    registry = _load_json(reg_path, {"runs": []})
    runs = [r for r in registry.get("runs", []) if r.get(
        "run_id") != record["run_id"]]
    runs.append(record)
    registry["runs"] = runs
    _save_json(reg_path, registry)


def _get_registry(project_path: str) -> dict[str, Any]:
    return _load_json(_registry_path(project_path), {"runs": []})


def _sync_mlflow(fn) -> bool:
    try:
        fn()
        return True
    except Exception:
        return False


def init_experiment_tracker(project_path: str = ".") -> dict[str, Any]:
    try:
        root = _tracker_root(project_path)
        root.mkdir(parents=True, exist_ok=True)
        reg_path = root / "registry.json"
        if not reg_path.exists():
            _save_json(reg_path, {"runs": []})

        return {"success": True, "tracker_path": str(root)}
    except OSError as exc:
        return _fail(str(exc))


def create_run(project_path: str = ".") -> dict[str, Any]:
    initialized = init_experiment_tracker(project_path)
    if not initialized.get("success"):
        return initialized

    run_id = str(uuid.uuid4())
    run_path = _run_dir(project_path, run_id)
    started_at = _now_iso()

    try:
        run_path.mkdir(parents=True, exist_ok=False)
        _save_json(run_path / "params.json", {})
        _save_json(run_path / "metrics.json", [])
        _save_json(run_path / "artifacts.json", [])
        _save_json(run_path / "status.json",
                   {"status": "running", "started_at": started_at})

        _update_registry(
            project_path,
            {
                "run_id": run_id,
                "status": "running",
                "started_at": started_at,
            },
        )

        return {
            "success": True,
            "project_path": str(Path(project_path).resolve()),
            "run_id": run_id,
            "run_path": str(run_path),
            "status": "running",
            "started_at": started_at,
        }
    except OSError as exc:
        return _fail(str(exc))


def log_params(project_path: str, run_id: str, params: dict[str, Any]) -> dict[str, Any]:
    run_path = _run_dir(project_path, run_id)
    if not run_path.exists() or not run_path.is_dir():
        return _fail(f"run not found: {run_id}")

    try:
        existing = _load_json(run_path / "params.json", {})
        existing.update(params)
        _save_json(run_path / "params.json", existing)

        mlflow_logged = _sync_mlflow(lambda: __import__('mlflow').log_params(params))

        return {
            "success": True,
            "run_id": run_id,
            "params_count": len(existing),
            "mlflow_logged": mlflow_logged,
        }
    except (OSError, ValueError) as exc:
        return _fail(str(exc))


def log_metrics(
    project_path: str,
    run_id: str,
    metrics: dict[str, float],
    step: int,
) -> dict[str, Any]:
    run_path = _run_dir(project_path, run_id)
    if not run_path.exists() or not run_path.is_dir():
        return _fail(f"run not found: {run_id}")

    try:
        history = _load_json(run_path / "metrics.json", [])
        entry = {"step": step, "timestamp": _now_iso(), **metrics}
        history.append(entry)
        _save_json(run_path / "metrics.json", history)

        mlflow_logged = _sync_mlflow(
            lambda: [__import__('mlflow').log_metric(k, float(v), step=step) for k, v in metrics.items()]
        )

        return {
            "success": True,
            "run_id": run_id,
            "step": step,
            "metric_keys": sorted(metrics.keys()),
            "metric_entries": len(history),
            "mlflow_logged": mlflow_logged,
        }
    except (OSError, ValueError) as exc:
        return _fail(str(exc))


def log_artifact(project_path: str, run_id: str, artifact_path: str) -> dict[str, Any]:
    run_path = _run_dir(project_path, run_id)
    source = Path(artifact_path).expanduser().resolve()
    if not run_path.exists() or not run_path.is_dir():
        return _fail(f"run not found: {run_id}")
    if not source.exists() or not source.is_file():
        return _fail(f"artifact file not found: {source}")

    try:
        artifacts_dir = run_path / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        destination = artifacts_dir / source.name
        shutil.copy2(source, destination)

        manifest = _load_json(run_path / "artifacts.json", [])
        entry = {
            "source": str(source),
            "path": str(destination.relative_to(run_path)),
            "size": destination.stat().st_size,
            "logged_at": _now_iso(),
        }
        manifest.append(entry)
        _save_json(run_path / "artifacts.json", manifest)

        mlflow_logged = _sync_mlflow(lambda: __import__('mlflow').log_artifact(str(source)))

        return {
            "success": True,
            "run_id": run_id,
            "artifact": entry,
            "artifact_count": len(manifest),
            "mlflow_logged": mlflow_logged,
        }
    except (OSError, ValueError) as exc:
        return _fail(str(exc))


def finish_run(project_path: str, run_id: str, status: str = "success") -> dict[str, Any]:
    if status not in {"success", "failed"}:
        return _fail("status must be one of: success, failed")

    run_path = _run_dir(project_path, run_id)
    if not run_path.exists() or not run_path.is_dir():
        return _fail(f"run not found: {run_id}")

    try:
        status_file = run_path / "status.json"
        status_data = _load_json(status_file, {})
        started_at = status_data.get("started_at")
        ended_at = _now_iso()

        duration_seconds = None
        if started_at:
            try:
                start_dt = datetime.fromisoformat(started_at)
                end_dt = datetime.fromisoformat(ended_at)
                duration_seconds = max(
                    0.0, (end_dt - start_dt).total_seconds())
            except ValueError:
                duration_seconds = None

        _save_json(status_file, {
            "status": status,
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_seconds": duration_seconds,
        })

        registry = _get_registry(project_path)
        runs = registry.get("runs", [])
        for item in runs:
            if item.get("run_id") == run_id:
                item["status"] = status
                item["ended_at"] = ended_at
                item["duration_seconds"] = duration_seconds
        _save_json(_registry_path(project_path), registry)

        return {
            "success": True,
            "run_id": run_id,
            "status": status,
            "ended_at": ended_at,
            "duration_seconds": duration_seconds,
        }
    except OSError as exc:
        return _fail(str(exc))


def get_run(project_path: str, run_id: str) -> dict[str, Any]:
    run_path = _run_dir(project_path, run_id)
    if not run_path.exists() or not run_path.is_dir():
        return _fail(f"run not found: {run_id}")

    try:
        params = _load_json(run_path / "params.json", {})
        metrics = _load_json(run_path / "metrics.json", [])
        artifacts = _load_json(run_path / "artifacts.json", [])
        status = _load_json(run_path / "status.json", {})
        return {
            "success": True,
            "run_id": run_id,
            "run_path": str(run_path),
            "params": params,
            "metrics": metrics,
            "artifacts": artifacts,
            "status": status,
        }
    except (OSError, ValueError) as exc:
        return _fail(str(exc))


def list_runs(project_path: str = ".") -> dict[str, Any]:
    root = _tracker_root(project_path)
    if not root.exists():
        return {
            "success": True,
            "project_path": str(Path(project_path).resolve()),
            "run_count": 0,
            "runs": [],
        }

    try:
        registry = _get_registry(project_path)
        rows: list[dict[str, Any]] = []
        for record in registry.get("runs", []):
            run_id = record.get("run_id")
            if not run_id:
                continue
            run_state = get_run(project_path, run_id)
            if not run_state.get("success"):
                continue
            metrics = run_state["metrics"]
            latest_metrics = metrics[-1] if metrics else {}
            rows.append(
                {
                    "run_id": run_id,
                    "status": run_state["status"].get("status", record.get("status", "unknown")),
                    "duration_seconds": run_state["status"].get("duration_seconds"),
                    "started_at": run_state["status"].get("started_at", record.get("started_at")),
                    "updated_at": run_state["status"].get("ended_at")
                    or run_state["status"].get("started_at"),
                    "latest_metrics": latest_metrics,
                }
            )

        rows.sort(key=lambda row: row.get("updated_at") or "", reverse=True)
        return {
            "success": True,
            "project_path": str(Path(project_path).resolve()),
            "run_count": len(rows),
            "runs": rows,
        }
    except OSError as exc:
        return _fail(str(exc))


def compare_runs(project_path: str, run_ids: list[str]) -> dict[str, Any]:
    if len(run_ids) < 2:
        return _fail("run_ids must contain at least two runs")

    snapshots: dict[str, dict[str, Any]] = {}
    for run_id in run_ids:
        row = get_run(project_path, run_id)
        if not row.get("success"):
            return row
        latest_metrics = row["metrics"][-1] if row["metrics"] else {}
        snapshots[run_id] = {
            "params": row["params"],
            "latest_metrics": latest_metrics,
            "status": row["status"],
        }

    diff: dict[str, Any]
    base_id = run_ids[0]
    try:
        from deepdiff import DeepDiff

        diff = {
            rid: DeepDiff(snapshots[base_id],
                          snapshots[rid], ignore_order=True).to_dict()
            for rid in run_ids[1:]
        }
    except ImportError:
        def _flat(d, prefix=""):
            out = {}
            for k, v in d.items():
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    out.update(_flat(v, key))
                else:
                    out[key] = v
            return out

        base_flat = _flat(snapshots[base_id])
        diff = {}
        for rid in run_ids[1:]:
            rid_flat = _flat(snapshots[rid])
            all_keys = base_flat.keys() | rid_flat.keys()
            diff[rid] = {
                k: {"old": base_flat.get(k), "new": rid_flat.get(k)}
                for k in all_keys
                if base_flat.get(k) != rid_flat.get(k)
            }

    return {
        "success": True,
        "run_ids": run_ids,
        "snapshots": snapshots,
        "diff": diff,
    }


def get_best_run(
    project_path: str,
    metric_name: str,
    direction: str = "max",
) -> dict[str, Any]:
    if direction not in {"max", "min"}:
        return _fail("direction must be one of: max, min")

    rows = list_runs(project_path)
    if not rows.get("success"):
        return rows

    candidates: list[dict[str, Any]] = []
    for row in rows.get("runs", []):
        value = row.get("latest_metrics", {}).get(metric_name)
        if isinstance(value, (int, float)):
            candidates.append(
                {"run_id": row["run_id"], "metric_value": float(value), "row": row})

    if not candidates:
        return _fail(f"metric '{metric_name}' not found in any run")

    best = max(candidates, key=lambda x: x["metric_value"]) if direction == "max" else min(
        candidates, key=lambda x: x["metric_value"]
    )

    return {
        "success": True,
        "metric_name": metric_name,
        "direction": direction,
        "best_run_id": best["run_id"],
        "metric_value": best["metric_value"],
        "run": best["row"],
    }


def delete_run(project_path: str, run_id: str, dry_run: bool = False) -> dict[str, Any]:
    run_path = _run_dir(project_path, run_id)
    if not run_path.exists() or not run_path.is_dir():
        return _fail(f"run not found: {run_id}")

    try:
        if not dry_run:
            shutil.rmtree(run_path)

            registry = _get_registry(project_path)
            registry["runs"] = [r for r in registry.get(
                "runs", []) if r.get("run_id") != run_id]
            _save_json(_registry_path(project_path), registry)

        return {
            "success": True,
            "run_id": run_id,
            "dry_run": dry_run,
            "deleted": not dry_run,
        }
    except OSError as exc:
        return _fail(str(exc))


def export_runs_csv(project_path: str, output_path: str) -> dict[str, Any]:
    runs = list_runs(project_path)
    if not runs.get("success"):
        return runs

    try:
        import pandas as pd
    except ImportError:
        return _fail("pandas not installed.")

    rows: list[dict[str, Any]] = []
    for summary in runs.get("runs", []):
        run_id = summary["run_id"]
        full = get_run(project_path, run_id)
        if not full.get("success"):
            continue

        params = full["params"]
        latest_metrics = summary.get("latest_metrics", {})
        row = {
            "run_id": run_id,
            "status": summary.get("status"),
            "started_at": summary.get("started_at"),
            "duration_seconds": summary.get("duration_seconds"),
        }
        for key, value in params.items():
            row[f"param_{key}"] = value
        for key, value in latest_metrics.items():
            if key in {"timestamp", "step"}:
                continue
            row[f"metric_{key}"] = value
        rows.append(row)

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(destination, index=False)
    return {
        "success": True,
        "output_path": str(destination),
        "row_count": int(frame.shape[0]),
    }

