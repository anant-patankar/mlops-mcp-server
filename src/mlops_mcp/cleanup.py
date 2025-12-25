import json
import shutil
from pathlib import Path
from typing import Any

from ._types import err as _fail


_REMOVABLE_DIRS = {"__pycache__", ".ipynb_checkpoints", "wandb", "lightning_logs"}
_REMOVABLE_NAMES = {".DS_Store"}
_REMOVABLE_SUFFIXES = {".pyc", ".tmp", ".log"}


def cleanup_project(project_path: str, dry_run: bool = False) -> dict[str, Any]:
    root = Path(project_path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return _fail(f"project directory not found: {root}")

    removed_files: list[str] = []
    removed_dirs: list[str] = []
    empty_dirs_removed: list[str] = []

    try:
        for item in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            rel = str(item.relative_to(root))
            if item.is_dir() and item.name in _REMOVABLE_DIRS:
                removed_dirs.append(rel)
                if not dry_run:
                    shutil.rmtree(item, ignore_errors=False)
                continue

            if item.is_file() and (
                item.name in _REMOVABLE_NAMES or item.suffix.lower() in _REMOVABLE_SUFFIXES
            ):
                removed_files.append(rel)
                if not dry_run:
                    item.unlink(missing_ok=True)

        for directory in sorted(
            [d for d in root.rglob("*") if d.is_dir()],
            key=lambda p: len(p.parts),
            reverse=True,
        ):
            if directory == root:
                continue
            try:
                if any(directory.iterdir()):
                    continue
            except OSError:
                continue

            rel = str(directory.relative_to(root))
            empty_dirs_removed.append(rel)
            if not dry_run:
                directory.rmdir()

        return {
            "success": True,
            "project_path": str(root),
            "dry_run": dry_run,
            "removed_files": sorted(removed_files),
            "removed_directories": sorted(removed_dirs),
            "removed_empty_directories": sorted(empty_dirs_removed),
        }
    except OSError as exc:
        return _fail(str(exc))


def cleanup_old_checkpoints(
    checkpoints_path: str,
    keep: int = 3,
    dry_run: bool = False,
) -> dict[str, Any]:
    root = Path(checkpoints_path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return _fail(f"checkpoint directory not found: {root}")
    if keep < 0:
        return _fail("keep must be >= 0")

    checkpoint_exts = {".pt", ".pth", ".ckpt", ".onnx", ".safetensors"}
    files = [p for p in root.rglob(
        "*") if p.is_file() and p.suffix.lower() in checkpoint_exts]

    try:
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        kept = [str(p.relative_to(root)) for p in files[:keep]]
        to_delete = files[keep:]
        removed = [str(p.relative_to(root)) for p in to_delete]

        if not dry_run:
            for path in to_delete:
                path.unlink(missing_ok=True)

        return {
            "success": True,
            "checkpoints_path": str(root),
            "dry_run": dry_run,
            "kept": kept,
            "removed": removed,
            "removed_count": len(removed),
        }
    except OSError as exc:
        return _fail(str(exc))


def cleanup_failed_runs(
    runs_path: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    root = Path(runs_path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return _fail(f"runs directory not found: {root}")

    removed: list[str] = []
    kept: list[str] = []

    try:
        for run_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
            rel = str(run_dir.relative_to(root))
            metrics_file = run_dir / "metrics.json"
            status_file = run_dir / "status.json"

            has_metrics = metrics_file.is_file()
            status_ok = False
            if status_file.is_file():
                try:
                    status_data = json.loads(
                        status_file.read_text(encoding="utf-8"))
                    status_ok = status_data.get("status") == "success"
                except (json.JSONDecodeError, OSError):
                    status_ok = False

            if has_metrics and status_ok:
                kept.append(rel)
                continue

            removed.append(rel)
            if not dry_run:
                shutil.rmtree(run_dir, ignore_errors=False)

        return {
            "success": True,
            "runs_path": str(root),
            "dry_run": dry_run,
            "kept": kept,
            "removed": removed,
            "removed_count": len(removed),
        }
    except OSError as exc:
        return _fail(str(exc))


def cleanup_empty_logs(
    logs_path: str,
    min_size_bytes: int = 100,
    dry_run: bool = False,
) -> dict[str, Any]:
    root = Path(logs_path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return _fail(f"logs directory not found: {root}")
    if min_size_bytes < 0:
        return _fail("min_size_bytes must be >= 0")

    removed: list[str] = []
    kept: list[str] = []

    try:
        for item in root.rglob("*.log"):
            rel = str(item.relative_to(root))
            size = item.stat().st_size
            if size < min_size_bytes:
                removed.append(rel)
                if not dry_run:
                    item.unlink(missing_ok=True)
            else:
                kept.append(rel)

        return {
            "success": True,
            "logs_path": str(root),
            "dry_run": dry_run,
            "min_size_bytes": min_size_bytes,
            "removed": sorted(removed),
            "kept": sorted(kept),
            "removed_count": len(removed),
        }
    except OSError as exc:
        return _fail(str(exc))
