from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

from ._types import err as _fail


def _run_dvc(command: list[str], cwd: str = ".") -> dict[str, Any]:
    check = dvc_check_available()
    if not check.get("success"):
        return check

    try:
        completed = subprocess.run(
            command,
            cwd=Path(cwd).expanduser().resolve(),
            capture_output=True,
            text=True,
            check=False,
        )
        return {
            "success": completed.returncode == 0,
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except OSError as exc:
        return _fail(str(exc))


def dvc_check_available() -> dict[str, Any]:
    dvc_binary = shutil.which("dvc")
    if not dvc_binary:
        return _fail("dvc CLI not found. Install DVC and ensure it is on PATH.")

    try:
        completed = subprocess.run(
            [dvc_binary, "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            return _fail(completed.stderr.strip() or "failed to check dvc version")

        return {
            "success": True,
            "available": True,
            "binary": dvc_binary,
            "version": completed.stdout.strip(),
        }
    except OSError as exc:
        return _fail(str(exc))


def _wrap_result(result: dict[str, Any], repo_path: str, **extra: Any) -> dict[str, Any]:
    return {
        "success": True,
        "repo_path": str(Path(repo_path).expanduser().resolve()),
        "stdout": result.get("stdout", ""),
        **extra,
    }


def dvc_init(repo_path: str = ".") -> dict[str, Any]:
    result = _run_dvc(["dvc", "init"], cwd=repo_path)
    if not result.get("success"):
        return result
    return _wrap_result(result, repo_path)


def dvc_add(path: str, repo_path: str = ".") -> dict[str, Any]:
    result = _run_dvc(["dvc", "add", path], cwd=repo_path)
    if not result.get("success"):
        return result
    return _wrap_result(result, repo_path, tracked_path=str(Path(path).expanduser().resolve()))


def dvc_push(repo_path: str = ".") -> dict[str, Any]:
    result = _run_dvc(["dvc", "push"], cwd=repo_path)
    if not result.get("success"):
        return result
    return _wrap_result(result, repo_path)


def dvc_pull(repo_path: str = ".") -> dict[str, Any]:
    result = _run_dvc(["dvc", "pull"], cwd=repo_path)
    if not result.get("success"):
        return result
    return _wrap_result(result, repo_path)


def dvc_status(repo_path: str = ".") -> dict[str, Any]:
    result = _run_dvc(["dvc", "status"], cwd=repo_path)
    if not result.get("success"):
        return result

    stdout = result.get("stdout", "")
    changed: list[str] = []
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        changed.append(line)

    return {
        "success": True,
        "repo_path": str(Path(repo_path).expanduser().resolve()),
        "changed": changed,
        "count": len(changed),
    }


def dvc_repro(repo_path: str = ".") -> dict[str, Any]:
    result = _run_dvc(["dvc", "repro"], cwd=repo_path)
    if not result.get("success"):
        return result
    return _wrap_result(result, repo_path)


def create_dvc_pipeline(
    repo_path: str,
    stages: dict[str, dict[str, Any]],
    output_file: str = "dvc.yaml",
) -> dict[str, Any]:
    if not stages:
        return _fail("stages must not be empty")

    root = Path(repo_path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return _fail(f"repository path not found: {root}")

    payload = {"stages": stages}
    output = root / output_file
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(yaml.safe_dump(
            payload, sort_keys=False), encoding="utf-8")
        return {
            "success": True,
            "repo_path": str(root),
            "pipeline_path": str(output),
            "stage_count": len(stages),
        }
    except OSError as exc:
        return _fail(str(exc))
