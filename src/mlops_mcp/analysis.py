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
                "dtypes": [str(field.type) for field in table.schema],
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
