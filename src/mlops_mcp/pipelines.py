from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import yaml

from ._types import err as _fail


def _get_stages(payload: Any) -> dict[str, Any]:
    return payload.get("stages", {}) if isinstance(payload, dict) else {}


def _read_pipeline(path: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not path.exists() or not path.is_file():
        return None, _fail(f"pipeline file not found: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return payload, None
    except (OSError, yaml.YAMLError) as exc:
        return None, _fail(str(exc))


def _write_pipeline(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(
            payload, sort_keys=False), encoding="utf-8")
        return {"success": True, "pipeline_path": str(path)}
    except OSError as exc:
        return _fail(str(exc))


def _build_stage_graph(stages: dict[str, Any]) -> dict[str, set[str]]:
    producers: dict[str, str] = {}
    for stage_name, stage in stages.items():
        for out in stage.get("outs", []) or []:
            producers[str(out)] = stage_name

    graph: dict[str, set[str]] = {name: set() for name in stages}
    for stage_name, stage in stages.items():
        for dep in stage.get("deps", []) or []:
            producer = producers.get(str(dep))
            if producer and producer != stage_name:
                graph[producer].add(stage_name)
    return graph


def _has_cycle(graph: dict[str, set[str]]) -> bool:
    # Kahn's algorithm — if topological sort can't visit all nodes, there's a cycle.
    indegree = {node: 0 for node in graph}
    for src in graph:
        for dest in graph[src]:
            indegree[dest] += 1

    queue = deque([node for node, deg in indegree.items() if deg == 0])
    visited = 0
    while queue:
        node = queue.popleft()
        visited += 1
        for nxt in graph.get(node, set()):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    return visited != len(indegree)


def create_pipeline(pipeline_path: str, stages: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not stages:
        return _fail("stages must not be empty")
    payload = {"stages": stages}
    return _write_pipeline(Path(pipeline_path).expanduser().resolve(), payload)


def get_pipeline(pipeline_path: str) -> dict[str, Any]:
    payload, error = _read_pipeline(Path(pipeline_path).expanduser().resolve())
    if error:
        return error
    stages = _get_stages(payload)
    return {
        "success": True,
        "pipeline_path": str(Path(pipeline_path).expanduser().resolve()),
        "stage_count": len(stages),
        "pipeline": payload,
    }


def validate_pipeline(pipeline_path: str, project_root: str = ".") -> dict[str, Any]:
    path = Path(pipeline_path).expanduser().resolve()
    payload, error = _read_pipeline(path)
    if error:
        return error

    stages = _get_stages(payload)
    root = Path(project_root).expanduser().resolve()
    errors: list[dict[str, Any]] = []

    for stage_name, stage in stages.items():
        if not isinstance(stage, dict):
            errors.append(
                {"stage": stage_name, "error": "stage definition must be a mapping"})
            continue
        if not stage.get("cmd"):
            errors.append({"stage": stage_name, "error": "missing cmd"})

        for dep in stage.get("deps", []) or []:
            dep_path = root / str(dep)
            if not dep_path.exists():
                errors.append(
                    {
                        "stage": stage_name,
                        "error": "missing dependency",
                        "path": str(dep),
                    }
                )

    graph = _build_stage_graph(stages)
    if _has_cycle(graph):
        errors.append({"stage": None, "error": "circular dependency detected"})

    return {
        "success": True,
        "pipeline_path": str(path),
        "valid": len(errors) == 0,
        "errors": errors,
    }


def add_pipeline_stage(
    pipeline_path: str,
    stage_name: str,
    stage_definition: dict[str, Any],
) -> dict[str, Any]:
    path = Path(pipeline_path).expanduser().resolve()
    payload, error = _read_pipeline(path)
    if error:
        return error
    if payload is None:
        return _fail("failed to load pipeline")

    stages = payload.setdefault("stages", {})
    if stage_name in stages:
        return _fail(f"stage already exists: {stage_name}")
    stages[stage_name] = stage_definition
    return _write_pipeline(path, payload)


def remove_pipeline_stage(pipeline_path: str, stage_name: str) -> dict[str, Any]:
    path = Path(pipeline_path).expanduser().resolve()
    payload, error = _read_pipeline(path)
    if error:
        return error
    if payload is None:
        return _fail("failed to load pipeline")

    stages = payload.setdefault("stages", {})
    if stage_name not in stages:
        return _fail(f"stage not found: {stage_name}")

    dependents: list[str] = []
    removed_outs = set(stages[stage_name].get("outs", []) or [])
    for name, stage in stages.items():
        if name == stage_name:
            continue
        deps = set(stage.get("deps", []) or [])
        if removed_outs & deps:
            dependents.append(name)

    del stages[stage_name]
    written = _write_pipeline(path, payload)
    if not written.get("success"):
        return written
    written.update({"removed_stage": stage_name, "dependent_stages": sorted(dependents)})
    return written


def get_pipeline_status(pipeline_path: str, project_root: str = ".") -> dict[str, Any]:
    path = Path(pipeline_path).expanduser().resolve()
    payload, error = _read_pipeline(path)
    if error:
        return error

    stages = _get_stages(payload)
    root = Path(project_root).expanduser().resolve()
    stage_results: list[dict[str, Any]] = []

    for stage_name, stage in stages.items():
        deps = [root / str(dep) for dep in (stage.get("deps", []) or [])]
        outs = [root / str(out) for out in (stage.get("outs", []) or [])]

        missing_outs = [str(out.relative_to(root))
                        for out in outs if not out.exists()]
        if missing_outs:
            stage_results.append(
                {
                    "stage": stage_name,
                    "status": "stale",
                    "reason": "missing_outputs",
                    "missing_outputs": missing_outs,
                }
            )
            continue

        dep_times = [dep.stat().st_mtime for dep in deps if dep.exists()]
        out_times = [out.stat().st_mtime for out in outs if out.exists()]
        if dep_times and out_times and max(dep_times) > min(out_times):
            stage_results.append({"stage": stage_name, "status": "stale",
                        "reason": "deps_newer_than_outs"})
        else:
            stage_results.append({"stage": stage_name, "status": "up-to-date"})

    return {
        "success": True,
        "pipeline_path": str(path),
        "stage_count": len(stage_results),
        "stages": stage_results,
    }


def list_pipelines(project_path: str = ".") -> dict[str, Any]:
    root = Path(project_path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return _fail(f"project path not found: {root}")

    try:
        found = [
            str(item.relative_to(root))
            for item in root.rglob("*.pipeline.yaml")
            if item.is_file()
        ]
        found.sort()
        return {
            "success": True,
            "project_path": str(root),
            "count": len(found),
            "pipelines": found,
        }
    except OSError as exc:
        return _fail(str(exc))


def visualize_pipeline(pipeline_path: str) -> dict[str, Any]:
    path = Path(pipeline_path).expanduser().resolve()
    payload, error = _read_pipeline(path)
    if error:
        return error

    stages = _get_stages(payload)
    graph = _build_stage_graph(stages)

    lines = ["graph TD"]
    for stage_name in stages:
        lines.append(f"    {stage_name}[{stage_name}]")
    for src, targets in graph.items():
        for dest in sorted(targets):
            lines.append(f"    {src} --> {dest}")

    mermaid = "\n".join(lines)
    return {
        "success": True,
        "pipeline_path": str(path),
        "mermaid": mermaid,
    }
