from __future__ import annotations

import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ._types import err as _fail


def _lineage_path(project_path: str) -> Path:
    return Path(project_path).expanduser().resolve() / ".mlops" / "lineage.json"


def _load_lineage(project_path: str) -> dict[str, Any]:
    path = _lineage_path(project_path)
    if not path.exists():
        return {"records": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_lineage(project_path: str, payload: dict[str, Any]) -> None:
    path = _lineage_path(project_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


_TYPE_MAP: dict[str, str] = {
    ".csv": "dataset", ".parquet": "dataset", ".jsonl": "dataset",
    ".tsv": "dataset", ".feather": "dataset",
    ".pt": "model", ".pth": "model", ".ckpt": "model",
    ".onnx": "model", ".joblib": "model", ".pkl": "model",
    ".safetensors": "model",
    ".yaml": "config", ".yml": "config", ".json": "config", ".toml": "config",
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".svg": "image",
    ".txt": "text", ".md": "text", ".log": "text",
}


def _artifact_type(path: str) -> str:
    return _TYPE_MAP.get(Path(path).suffix.lower(), "artifact")


def record_lineage(
    project_path: str,
    output: str,
    inputs: list[str],
    run_id: str,
) -> dict[str, Any]:
    if not output.strip():
        return _fail("output must not be empty")
    if not inputs:
        return _fail("inputs must not be empty")
    if not run_id.strip():
        return _fail("run_id must not be empty")

    try:
        payload = _load_lineage(project_path)
        records = payload.setdefault("records", [])
        entry = {
            "output": output,
            "inputs": inputs,
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        records.append(entry)
        _save_lineage(project_path, payload)
        return {
            "success": True,
            "lineage_path": str(_lineage_path(project_path)),
            "record": entry,
            "record_count": len(records),
        }
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _fail(str(exc))


def get_artifact_provenance(project_path: str, artifact: str) -> dict[str, Any]:
    try:
        payload = _load_lineage(project_path)
        records = payload.get("records", [])

        by_output: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            by_output.setdefault(record.get("output", ""), []).append(record)

        queue = deque([artifact])
        visited: set[str] = set()
        provenance_edges: list[dict[str, Any]] = []
        source_artifacts: set[str] = set()

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            candidates = by_output.get(current, [])
            if not candidates:
                source_artifacts.add(current)
                continue

            for rec in candidates:
                edge = {
                    "output": rec.get("output"),
                    "inputs": rec.get("inputs", []),
                    "run_id": rec.get("run_id"),
                    "timestamp": rec.get("timestamp"),
                }
                provenance_edges.append(edge)
                for parent in rec.get("inputs", []):
                    queue.append(parent)

        return {
            "success": True,
            "artifact": artifact,
            "record_count": len(provenance_edges),
            "provenance": provenance_edges,
            "source_artifacts": sorted(source_artifacts),
        }
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _fail(str(exc))


def list_lineage_artifacts(project_path: str) -> dict[str, Any]:
    try:
        payload = _load_lineage(project_path)
        records = payload.get("records", [])

        latest: dict[str, str] = {}
        all_artifacts: set[str] = set()
        for record in records:
            output = record.get("output")
            if output:
                all_artifacts.add(output)
                latest[output] = record.get("timestamp", latest.get(output, ""))

            for item in record.get("inputs", []):
                all_artifacts.add(item)
                latest.setdefault(item, record.get("timestamp", ""))

        listed = [
            {
                "artifact": artifact,
                "type": _artifact_type(artifact),
                "last_updated": latest.get(artifact),
            }
            for artifact in sorted(all_artifacts)
        ]
        return {
            "success": True,
            "lineage_path": str(_lineage_path(project_path)),
            "count": len(listed),
            "artifacts": listed,
        }
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _fail(str(exc))


def visualize_lineage(project_path: str, artifact: str) -> dict[str, Any]:
    provenance = get_artifact_provenance(project_path, artifact)
    if not provenance.get("success"):
        return provenance

    edges = provenance.get("provenance", [])
    lines = ["graph TD"]
    nodes: list[str] = []
    for edge in edges:
        output = str(edge.get("output"))
        if output not in nodes:
            nodes.append(output)
        for parent in edge.get("inputs", []):
            parent_node = str(parent)
            if parent_node not in nodes:
                nodes.append(parent_node)
            lines.append(f"    {parent_node} --> {output}")

    if not edges:
        lines.append(f"    {artifact}[{artifact}]")
    else:
        for node in sorted(nodes):
            lines.insert(1, f"    {node}[{node}]")

    return {
        "success": True,
        "artifact": artifact,
        "mermaid": "\n".join(lines),
    }


def check_lineage_integrity(project_path: str) -> dict[str, Any]:
    try:
        root = Path(project_path).expanduser().resolve()
        payload = _load_lineage(project_path)
        records = payload.get("records", [])

        broken_edges = []
        for record in records:
            output = record.get("output")
            for parent in record.get("inputs", []):
                candidate = root / str(parent)
                if not candidate.exists():
                    broken_edges.append(
                        {
                            "output": output,
                            "missing_input": parent,
                            "run_id": record.get("run_id"),
                        }
                    )

        return {
            "success": True,
            "lineage_path": str(_lineage_path(project_path)),
            "is_valid": len(broken_edges) == 0,
            "broken_edge_count": len(broken_edges),
            "broken_edges": broken_edges,
        }
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _fail(str(exc))
