from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ._types import err as _fail
from .experiments import get_run


_MODEL_EXTENSIONS = {
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".joblib",
    ".pkl",
    ".safetensors",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _registry_dir(project_path: str) -> Path:
    return Path(project_path).expanduser().resolve() / ".mlops" / "model_registry"


def _registry_path(project_path: str) -> Path:
    return _registry_dir(project_path) / "models.json"


def _load_registry(project_path: str) -> dict[str, Any]:
    path = _registry_path(project_path)
    if not path.exists():
        return {"models": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_registry(project_path: str, registry: dict[str, Any]) -> None:
    path = _registry_path(project_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    tmp.replace(path)


def _detect_framework(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".pt", ".pth", ".ckpt"}:
        return "pytorch"
    if ext == ".onnx":
        return "onnx"
    if ext in {".joblib", ".pkl"}:
        return "sklearn"
    if ext == ".safetensors":
        return "safetensors"
    if ext in {".pb", ".savedmodel"}:
        return "tensorflow"
    return "unknown"


def _lookup_version(
    registry: dict[str, Any], model_name: str, version: int
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    versions = registry.get("models", {}).get(model_name, [])
    for entry in versions:
        if int(entry.get("version", -1)) == int(version):
            return entry, versions
    return None, versions


def init_model_registry(project_path: str = ".") -> dict[str, Any]:
    try:
        root = _registry_dir(project_path)
        root.mkdir(parents=True, exist_ok=True)
        reg = _registry_path(project_path)
        if not reg.exists():
            _save_registry(project_path, {"models": {}})
        return {"success": True, "registry_path": str(reg), "registry_dir": str(root)}
    except OSError as exc:
        return _fail(str(exc))


def register_model(
    project_path: str,
    model_name: str,
    model_path: str,
    run_id: str | None = None,
    version: int | None = None,
    stage: str = "dev",
    tags: dict[str, Any] | None = None,
) -> dict[str, Any]:
    initialized = init_model_registry(project_path)
    if not initialized.get("success"):
        return initialized

    model_file = Path(model_path).expanduser().resolve()
    if not model_file.exists() or not model_file.is_file():
        return _fail(f"model file not found: {model_file}")

    try:
        registry = _load_registry(project_path)
        model_versions = registry.setdefault(
            "models", {}).setdefault(model_name, [])
        if version is None:
            next_version = (
                max((int(v.get("version", 0))
                    for v in model_versions), default=0) + 1
            )
        else:
            next_version = int(version)
            if any(int(v.get("version", -1)) == next_version for v in model_versions):
                return _fail(f"version already exists for model '{model_name}': {next_version}")

        entry = {
            "name": model_name,
            "version": next_version,
            "stage": stage,
            "run_id": run_id,
            "path": str(model_file),
            "size": model_file.stat().st_size,
            "framework": _detect_framework(model_file),
            "registered_at": _now_iso(),
            "tags": tags or {},
            "stage_history": [{"stage": stage, "timestamp": _now_iso()}],
        }
        model_versions.append(entry)
        model_versions.sort(key=lambda item: int(item.get("version", 0)))
        _save_registry(project_path, registry)

        mlflow_registered = False
        try:
            import mlflow

            mlflow.register_model(model_uri=str(model_file), name=model_name)
            mlflow_registered = True
        except (ImportError, Exception):
            mlflow_registered = False

        return {
            "success": True,
            "model_name": model_name,
            "version": next_version,
            "stage": stage,
            "path": str(model_file),
            "framework": entry["framework"],
            "mlflow_registered": mlflow_registered,
        }
    except (OSError, ValueError) as exc:
        return _fail(str(exc))


def list_models(project_path: str = ".") -> dict[str, Any]:
    try:
        registry = _load_registry(project_path)
        entries: list[dict[str, Any]] = []
        for name, versions in registry.get("models", {}).items():
            if not versions:
                continue
            sorted_versions = sorted(
                versions, key=lambda x: int(x.get("version", 0)))
            entries.append(
                {
                    "model_name": name,
                    "versions": [int(v["version"]) for v in sorted_versions],
                    "current_stage": sorted_versions[-1].get("stage", "unknown"),
                }
            )
        entries.sort(key=lambda m: m["model_name"])
        return {"success": True, "project_path": str(Path(project_path).resolve()), "models": entries, "model_count": len(entries)}
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _fail(str(exc))


def get_model_versions(project_path: str, model_name: str) -> dict[str, Any]:
    try:
        registry = _load_registry(project_path)
        versions = registry.get("models", {}).get(model_name)
        if versions is None:
            return _fail(f"model not found: {model_name}")

        enriched: list[dict[str, Any]] = []
        for entry in sorted(versions, key=lambda item: int(item.get("version", 0))):
            model_file = Path(entry.get("path", ""))
            exists = model_file.exists()
            size = model_file.stat().st_size if exists else None
            item = dict(entry)
            item["file_exists"] = exists
            item["size"] = size if size is not None else entry.get("size")
            enriched.append(item)

        return {"success": True, "model_name": model_name, "versions": enriched}
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _fail(str(exc))


def get_model_info(project_path: str, model_name: str, version: int) -> dict[str, Any]:
    try:
        registry = _load_registry(project_path)
        entry, _ = _lookup_version(registry, model_name, version)
        if entry is None:
            return _fail(f"model/version not found: {model_name} v{version}")

        model_file = Path(entry.get("path", ""))
        framework = _detect_framework(model_file) if model_file.exists(
        ) else entry.get("framework", "unknown")
        payload = dict(entry)
        payload["framework"] = framework
        payload["file_exists"] = model_file.exists()
        if model_file.exists():
            payload["size"] = model_file.stat().st_size

        return {"success": True, "model": payload}
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _fail(str(exc))


def promote_model(
    project_path: str,
    model_name: str,
    version: int,
    new_stage: str,
) -> dict[str, Any]:
    try:
        registry = _load_registry(project_path)
        entry, _ = _lookup_version(registry, model_name, version)
        if entry is None:
            return _fail(f"model/version not found: {model_name} v{version}")

        old_stage = entry.get("stage", "unknown")
        entry["stage"] = new_stage
        entry.setdefault("stage_history", []).append(
            {"stage": new_stage, "from": old_stage, "timestamp": _now_iso()}
        )
        _save_registry(project_path, registry)
        return {
            "success": True,
            "model_name": model_name,
            "version": int(version),
            "old_stage": old_stage,
            "new_stage": new_stage,
        }
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _fail(str(exc))


def tag_model(
    project_path: str,
    model_name: str,
    version: int,
    tags: dict[str, Any],
) -> dict[str, Any]:
    try:
        registry = _load_registry(project_path)
        entry, _ = _lookup_version(registry, model_name, version)
        if entry is None:
            return _fail(f"model/version not found: {model_name} v{version}")

        current = entry.setdefault("tags", {})
        current.update(tags)
        _save_registry(project_path, registry)
        return {
            "success": True,
            "model_name": model_name,
            "version": int(version),
            "tags": current,
        }
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _fail(str(exc))


def compare_model_versions(
    project_path: str,
    model_name: str,
    version_a: int,
    version_b: int,
) -> dict[str, Any]:
    info_a = get_model_info(project_path, model_name, version_a)
    if not info_a.get("success"):
        return info_a
    info_b = get_model_info(project_path, model_name, version_b)
    if not info_b.get("success"):
        return info_b

    model_a = info_a["model"]
    model_b = info_b["model"]

    run_metrics_a: dict[str, Any] | None = None
    run_metrics_b: dict[str, Any] | None = None
    if model_a.get("run_id"):
        run_a = get_run(project_path, str(model_a["run_id"]))
        if run_a.get("success") and run_a.get("metrics"):
            run_metrics_a = run_a["metrics"][-1]
    if model_b.get("run_id"):
        run_b = get_run(project_path, str(model_b["run_id"]))
        if run_b.get("success") and run_b.get("metrics"):
            run_metrics_b = run_b["metrics"][-1]

    diff = {
        "size": {"a": model_a.get("size"), "b": model_b.get("size")},
        "stage": {"a": model_a.get("stage"), "b": model_b.get("stage")},
        "framework": {"a": model_a.get("framework"), "b": model_b.get("framework")},
        "tags": {"a": model_a.get("tags", {}), "b": model_b.get("tags", {})},
        "latest_run_metrics": {"a": run_metrics_a, "b": run_metrics_b},
    }
    return {
        "success": True,
        "model_name": model_name,
        "version_a": int(version_a),
        "version_b": int(version_b),
        "diff": diff,
    }


def deprecate_model(
    project_path: str,
    model_name: str,
    version: int,
    reason: str,
) -> dict[str, Any]:
    promoted = promote_model(project_path, model_name,
                             version, new_stage="deprecated")
    if not promoted.get("success"):
        return promoted

    try:
        registry = _load_registry(project_path)
        entry, _ = _lookup_version(registry, model_name, version)
        if entry is None:
            return _fail(f"model/version not found: {model_name} v{version}")
        entry["deprecation_reason"] = reason
        _save_registry(project_path, registry)
        return {
            "success": True,
            "model_name": model_name,
            "version": int(version),
            "stage": "deprecated",
            "reason": reason,
        }
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _fail(str(exc))


def delete_model_version(
    project_path: str,
    model_name: str,
    version: int,
    delete_file: bool = True,
) -> dict[str, Any]:
    try:
        registry = _load_registry(project_path)
        models = registry.get("models", {})
        versions = models.get(model_name)
        if versions is None:
            return _fail(f"model not found: {model_name}")

        idx = next(
            (i for i, v in enumerate(versions) if int(v.get("version", -1)) == int(version)),
            None,
        )
        entry = versions[idx] if idx is not None else None
        if idx is None or entry is None:
            return _fail(f"model/version not found: {model_name} v{version}")

        model_file = Path(entry.get("path", ""))
        if delete_file and model_file.exists() and model_file.is_file():
            model_file.unlink()

        versions.pop(idx)
        if not versions:
            models.pop(model_name, None)
        _save_registry(project_path, registry)
        return {
            "success": True,
            "model_name": model_name,
            "version": int(version),
            "deleted_file": bool(delete_file),
        }
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _fail(str(exc))


def get_model_lineage(project_path: str, model_name: str, version: int) -> dict[str, Any]:
    info = get_model_info(project_path, model_name, version)
    if not info.get("success"):
        return info

    model = info["model"]
    run_id = model.get("run_id")
    run_payload = None
    datasets: list[str] = []
    if run_id:
        run = get_run(project_path, str(run_id))
        if run.get("success"):
            run_payload = run
            params = run.get("params", {})
            for key, value in params.items():
                if "data" in key.lower() or "dataset" in key.lower():
                    datasets.append(f"{key}={value}")

    return {
        "success": True,
        "model_name": model_name,
        "version": int(version),
        "run_id": run_id,
        "model": model,
        "run": run_payload,
        "dataset_hints": datasets,
    }


def create_model_card(
    project_path: str,
    model_name: str,
    version: int,
    output_path: str | None = None,
) -> dict[str, Any]:
    info = get_model_info(project_path, model_name, version)
    if not info.get("success"):
        return info
    lineage = get_model_lineage(project_path, model_name, version)
    if not lineage.get("success"):
        return lineage

    model = info["model"]
    out = (
        Path(output_path).expanduser().resolve()
        if output_path
        else Path(project_path).expanduser().resolve() / "MODEL_CARD.md"
    )

    payload = {
        "model_name": model_name,
        "version": int(version),
        "stage": model.get("stage"),
        "framework": model.get("framework"),
        "path": model.get("path"),
        "run_id": model.get("run_id"),
        "tags": model.get("tags", {}),
        "dataset_hints": lineage.get("dataset_hints", []),
    }

    try:
        try:
            from jinja2 import Template

            template = Template("""# MODEL_CARD

- Name: {{ model_name }}
- Version: {{ version }}
- Stage: {{ stage }}
- Framework: {{ framework }}
- Path: {{ path }}
- Run ID: {{ run_id }}

## Tags
{% for key, value in tags.items() %}- {{ key }}: {{ value }}
{% endfor %}
## Dataset Hints
{% for item in dataset_hints %}- {{ item }}
{% endfor %}""")
            content = template.render(**payload)
        except ImportError:
            lines = [
                "# MODEL_CARD",
                "",
                f"- Name: {payload['model_name']}",
                f"- Version: {payload['version']}",
                f"- Stage: {payload['stage']}",
                f"- Framework: {payload['framework']}",
                f"- Path: {payload['path']}",
                f"- Run ID: {payload['run_id']}",
                "",
                "## Tags",
            ]
            lines += [f"- {k}: {v}" for k, v in payload["tags"].items()]
            lines.append("")
            lines.append("## Dataset Hints")
            lines += [f"- {item}" for item in payload["dataset_hints"]]
            content = "\n".join(lines) + "\n"

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(content, encoding="utf-8")
        return {
            "success": True,
            "model_name": model_name,
            "version": int(version),
            "output_path": str(out),
        }
    except OSError as exc:
        return _fail(str(exc))


def model_list(project_path: str = ".") -> dict[str, Any]:
    listed = list_models(project_path)
    if listed.get("success") and listed.get("model_count", 0) > 0:
        return {
            "success": True,
            "project_path": listed["project_path"],
            "model_count": listed["model_count"],
            "models": listed["models"],
            "source": "registry",
        }

    try:
        root = Path(project_path).expanduser().resolve()
        search_roots = [root / "models", root]

        seen: set[Path] = set()
        discovered: list[dict[str, Any]] = []
        for search_root in search_roots:
            if not search_root.exists() or not search_root.is_dir():
                continue
            for item in search_root.rglob("*"):
                if not item.is_file() or item.suffix.lower() not in _MODEL_EXTENSIONS:
                    continue
                if item in seen:
                    continue
                seen.add(item)
                discovered.append(
                    {
                        "path": str(item.relative_to(root)),
                        "size": item.stat().st_size,
                        "extension": item.suffix.lower(),
                    }
                )

        discovered.sort(key=lambda m: m["path"])
        return {
            "success": True,
            "project_path": str(root),
            "model_count": len(discovered),
            "models": discovered,
            "source": "filesystem",
        }
    except OSError as exc:
        return _fail(str(exc))
