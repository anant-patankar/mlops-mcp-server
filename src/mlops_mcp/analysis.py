from __future__ import annotations

import hashlib
import mimetypes
from collections import defaultdict
from datetime import datetime
from difflib import unified_diff
from pathlib import Path
from typing import Any

from ._types import err as err


def _hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def get_file_info(path: str) -> dict[str, Any]:
    try:
        target = Path(path)
        if not target.exists():
            return err(f"file not found: {target}")
        if not target.is_file():
            return err(f"path is not a file: {target}")

        stat = target.stat()
        mime_type, _ = mimetypes.guess_type(target.name)
        info: dict[str, Any] = {
            "success": True,
            "path": str(target.resolve()),
            "size": stat.st_size,
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "permissions": oct(stat.st_mode & 0o777),
            "mime_type": mime_type or "application/octet-stream",
        }

        if stat.st_size <= 100 * 1024 * 1024:
            info["md5"] = _hash_file(target)
        else:
            info["md5"] = None
            info["md5_skipped"] = True

        return info
    except OSError as exc:
        return err(str(exc))


def classify_file(path: str) -> dict[str, Any]:
    try:
        target = Path(path)
        if not target.exists():
            return err(f"file not found: {target}")
        if not target.is_file():
            return err(f"path is not a file: {target}")

        ext = target.suffix.lower()
        if ext in {".csv", ".parquet", ".jsonl", ".tsv", ".feather"}:
            category = "dataset"
        elif ext in {".pt", ".pth", ".onnx", ".joblib", ".pkl", ".safetensors", ".ckpt"}:
            category = "model"
        elif ext == ".ipynb":
            category = "notebook"
        elif ext in {".yaml", ".yml", ".json", ".toml", ".ini"}:
            category = "config"
        elif ext in {".py", ".sh", ".bash"}:
            category = "script"
        elif ext in {".md", ".rst", ".txt"}:
            category = "doc"
        elif ext in {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}:
            category = "image"
        elif ext in {".log", ".out"}:
            category = "log"
        else:
            category = "other"

        return {
            "success": True,
            "path": str(target.resolve()),
            "extension": ext,
            "category": category,
        }
    except OSError as exc:
        return err(str(exc))


def analysis_directory(path: str) -> dict[str, Any]:
    try:
        root = Path(path)
        if not root.exists():
            return {"success": False, "error": f"directory not found: {path}"}

        if not root.is_dir():
            return {"success": False, "error": f"path is not a directory: {path}"}

        file_count = 0
        dir_count = 0
        total_bytes = 0
        by_extension: dict[str, int] = {}

        for item in root.rglob("*"):
            if item.is_dir():
                dir_count += 1
                continue

            file_count += 1
            total_bytes += item.stat().st_size
            ext = item.suffix.lower() or "<no_ext>"
            by_extension[ext] = by_extension.get(ext, 0) + 1

        return {
            "success": True,
            "path": str(root),
            "file_count": file_count,
            "directory_count": dir_count,
            "total_bytes": total_bytes,
            "by_extension": dict(sorted(by_extension.items())),
        }
    except OSError as exc:
        return err(str(exc))


def find_duplicate_files(
    dir_path: str,
    min_size: int = 1024,
    max_size: int = 100 * 1024 * 1024,
) -> dict[str, Any]:
    try:
        root_dir = Path(dir_path)
        if not root_dir.exists() or not root_dir.is_dir():
            return err(f"directory not found: {root_dir}")

        buckets: dict[str, list[str]] = defaultdict(list)
        for item in root_dir.rglob("*"):
            if not item.is_file():
                continue
            size = item.stat().st_size
            if size < min_size or size > max_size:
                continue
            file_hash = _hash_file(item)
            buckets[file_hash].append(str(item.relative_to(root_dir)))

        duplicates = [files for files in buckets.values() if len(files) > 1]
        duplicate_groups = sorted(duplicates, key=len, reverse=True)
        return {
            "success": True,
            "path": str(root_dir.resolve()),
            "group_count": len(duplicate_groups),
            "duplicate_groups": duplicate_groups,
        }
    except OSError as exc:
        return err(str(exc))


def storage_report(dir_path: str, top_k: int = 10):
    try:
        dir_path = Path(dir_path).resolve()

        if not dir_path.exists():
            return err(f"Directory does not exists: {dir_path}")

        if not dir_path.is_dir():
            return err(f"Provided path is not directory: {dir_path}")

        analysis_report = analysis_directory(path = dir_path)
        if not analysis_report["success"]:
            return analysis_report

        files: list[tuple[str, int]] = []
        for p in dir_path.rglob("*"):
            if p.is_file():
                files.append((p.relative_to(dir_path), p.stat().st_size))

        largest = sorted(files, key=lambda entry: entry[1], reverse=True)[:top_k]

        analysis_report["top_k"] = top_k
        analysis_report["largest_files"] = [{"path": str(entry[0]), "size": entry[1]} for entry in largest]

        duplicates = find_duplicate_files(dir_path = dir_path)
        if duplicates["success"]:
            saved = 0
            for group in duplicates["duplicate_groups"]:
                if not group:
                    continue

                first = dir_path / group[0] # first file to be saved

                try:
                    saved += first.stat().st_size * (len(group) - 1)
                except OSError:
                    continue

            analysis_report["duplicate_groups"] = duplicates["group_count"]
            analysis_report["duplicate_savings_bytes"] = saved

        return analysis_report

    except OSError as exc:
        return err(str(exc))


def batch_classify(paths: list[str]) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for p in paths:
        row = classify_file(p)
        if row.get("success"):
            results.append(row)
        else:
            failures.append(
                {"path": p, "error": row.get("error", "unknown")})
    return {
        "success": True,
        "count": len(results),
        "classifications": results,
        "failures": failures,
    }


def get_dataset_stats(data_path: str) -> dict[str, Any]:
    data_path_resolved = Path(data_path)
    if not data_path_resolved.exists() or not data_path_resolved.is_file():
        return err(f"dataset not found: {data_path_resolved}")

    ext = data_path_resolved.suffix.lower()
    if ext == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError:
            return err("pyarrow not installed.")

        try:
            table = pq.read_table(data_path_resolved)
            return {
                "success": True,
                "path": str(data_path_resolved.resolve()),
                "rows": table.num_rows,
                "columns": table.num_columns,
                "column_names": table.column_names,
                "dtypes": {field.name: str(field.type) for field in table.schema},
            }
        except OSError as exc:
            return err(str(exc))

    try:
        import pandas as pd
    except ImportError:
        return err("pandas not installed.")

    if ext not in {".csv", ".json", ".jsonl", ".tsv", ".feather"}:
        return err(f"unsupported dataset extension: {ext}")

    try:
        if ext == ".csv":
            dataframe = pd.read_csv(data_path_resolved)
        elif ext == ".tsv":
            dataframe = pd.read_csv(data_path_resolved, sep="\t")
        elif ext in {".json", ".jsonl"}:
            dataframe = pd.read_json(data_path_resolved, lines=(ext == ".jsonl"))
        else:
            dataframe = pd.read_feather(data_path_resolved)

        return {
            "success": True,
            "path": str(data_path_resolved.resolve()),
            "rows": int(dataframe.shape[0]),
            "columns": int(dataframe.shape[1]),
            "dtypes": {col: str(dtype) for col, dtype in dataframe.dtypes.items()},
            "null_counts": {col: int(v) for col, v in dataframe.isna().sum().items()},
            "memory_bytes": int(dataframe.memory_usage(deep=True).sum()),
        }
    except (ValueError, OSError) as exc:
        return err(str(exc))


def detect_model_framework(model_path: str) -> dict[str, Any]:
    target_path = Path(model_path)
    if not target_path.exists() or not target_path.is_file():
        return err(f"model file not found: {target_path}")

    ext = target_path.suffix.lower()
    if ext in {".pt", ".pth", ".ckpt"}:
        framework = "pytorch"
    elif ext in {".pb", ".savedmodel"}:
        framework = "tensorflow"
    elif ext == ".onnx":
        framework = "onnx"
    elif ext in {".joblib", ".pkl"}:
        framework = "sklearn"
    elif ext == ".safetensors":
        framework = "safetensors"
    else:
        framework = "unknown"

    return {
        "success": True,
        "path": str(target_path.resolve()),
        "framework": framework,
        "extension": ext,
    }


def compare_files(path_a: str, path_b: str) -> dict[str, Any]:
    try:
        file_a = Path(path_a)
        file_b = Path(path_b)
        if not file_a.exists() or not file_a.is_file():
            return err(f"file not found: {file_a}")
        if not file_b.exists() or not file_b.is_file():
            return err(f"file not found: {file_b}")

        text_a = file_a.read_text(encoding="utf-8").splitlines()
        text_b = file_b.read_text(encoding="utf-8").splitlines()
        diff_lines = list(
            unified_diff(
                text_a, text_b,
                fromfile=str(file_a), tofile=str(file_b),
                lineterm="",
            )
        )
        added   = sum(1 for l in diff_lines if l.startswith("+") and not l.startswith("+++"))
        removed = sum(1 for l in diff_lines if l.startswith("-") and not l.startswith("---"))
        return {
            "success":    True,
            "path_a":     str(file_a.resolve()),
            "path_b":     str(file_b.resolve()),
            "identical":  not bool(diff_lines),
            "diff_lines": diff_lines,
            "added":      added,
            "removed":    removed,
        }
    except (OSError, UnicodeDecodeError) as exc:
        return err(str(exc))
    

def compare_directories(path_a: str, path_b: str) -> dict[str, Any]:
    try:
        dir_a = Path(path_a)
        dir_b = Path(path_b)
        if not dir_a.exists() or not dir_a.is_dir():
            return err(f"directory not found: {dir_a}")
        if not dir_b.exists() or not dir_b.is_dir():
            return err(f"directory not found: {dir_b}")

        files_a = {
            str(item.relative_to(dir_a)): item
            for item in dir_a.rglob("*")
            if item.is_file()
        }
        files_b = {
            str(item.relative_to(dir_b)): item
            for item in dir_b.rglob("*")
            if item.is_file()
        }

        set_a = set(files_a)
        set_b = set(files_b)

        only_in_a = sorted(set_a - set_b)
        only_in_b = sorted(set_b - set_a)
        shared = sorted(set_a & set_b)

        different_content: list[str] = []
        for rel in shared:
            if _hash_file(files_a[rel]) != _hash_file(files_b[rel]):
                different_content.append(rel)

        return {
            "success": True,
            "path_a": str(dir_a.resolve()), "path_b": str(dir_b.resolve()),
            "only_in_a": only_in_a, "only_in_b": only_in_b,
            "different_content": sorted(different_content),
        }
    except OSError as exc:
        return err(str(exc))

def get_notebook_summary(path: str) -> dict[str, Any]:
    try:
        import nbformat
    except ImportError:
        return err("nbformat not installed")

    target = Path(path)
    if not target.exists() or not target.is_file():
        return err(f"notebook file not found: {target}")

    try:
        notebook = nbformat.read(target, as_version=4)
    except OSError as exc:
        return err(str(exc))

    code_cells = 0
    markdown_cells = 0
    last_exec = None
    for cell in notebook.cells:
        if cell.cell_type == "code":
            code_cells += 1
            if cell.get("execution_count") is not None:
                last_exec = cell["execution_count"]
        elif cell.cell_type == "markdown":
            markdown_cells += 1

    return {
        "success": True,
        "path": str(target.resolve()),
        "cell_count": len(notebook.cells),
        "code_cells": code_cells,
        "markdown_cells": markdown_cells,
        "last_execution_count": last_exec,
    }
