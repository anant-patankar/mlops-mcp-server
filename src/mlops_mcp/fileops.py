from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import re
import shutil

from ._types import err


_AUDIT_TRAIL: list[dict[str, Any]] = []


class FileManager:
    def __init__(self, base_path: str = ".") -> None:
        self.base = Path(base_path).expanduser().resolve()

    def resolve(self, path: str) -> Path:
        target = Path(path).expanduser()
        if target.is_absolute():
            return target.resolve()
        return (self.base / target).resolve()

def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()

def _log_op(operation: str, **payload: Any) -> None:
    _AUDIT_TRAIL.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operation": operation,
        **payload
    })
    if len(_AUDIT_TRAIL) > 1000:
        _AUDIT_TRAIL.pop(0)

def create_file(
    path: str,
    content: str = "",
    encoding: str = "utf-8",
    overwrite: bool = False,
    base_path: str = ".",
) -> dict[str, Any]:
    try:
        target = FileManager(base_path).resolve(path)
        if target.exists() and not overwrite:
            return err(f"won't overwrite existing file: {target}")

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding=encoding)

        _log_op("create_file", path=str(target),
                       bytes_written=target.stat().st_size)
        return {
            "success": True,
            "path": str(target),
            "bytes_written": target.stat().st_size,
        }
    except OSError as exc:
        return err(str(exc))


def read_file(path: str, encoding: str = "utf-8", base_path: str = ".") -> dict[str, Any]:
    try:
        target = FileManager(base_path).resolve(path)
        if not target.exists():
            return err(f"file not found: {target}")

        if not target.is_file():
            return err(f"not a regular file: {target}")

        content = target.read_text(encoding=encoding)
        return {
            "success": True,
            "path": str(target),
            "content": content,
            "size": target.stat().st_size,
        }
    except (OSError, UnicodeDecodeError) as exc:
        return err(str(exc))


def write_file(
    path: str,
    content: str,
    encoding: str = "utf-8",
    base_path: str = ".",
) -> dict[str, Any]:
    try:
        target = FileManager(base_path).resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        temp_path = target.with_suffix(f"{target.suffix}.tmp")
        temp_path.write_text(content, encoding=encoding)
        temp_path.replace(target)

        _log_op("write_file", path=str(target),
                       bytes_written=target.stat().st_size)

        return {
            "success": True,
            "path": str(target),
            "bytes_written": target.stat().st_size,
        }
    except OSError as exc:
        return err(str(exc))


def delete_file(path: str, dry_run: bool = False, base_path: str = ".") -> dict[str, Any]:
    try:
        target = FileManager(base_path).resolve(path)
        if not target.exists():
            return err(f"file not found: {target}")
        if not target.is_file():
            return err(f"can't delete non-file path: {target}")

        if dry_run:
            return {"success": True, "path": str(target), "dry_run": True, "deleted": False}

        target.unlink()
        _log_op("delete_file", path=str(target))
        return {"success": True, "path": str(target), "dry_run": False, "deleted": True}
    except OSError as exc:
        return err(str(exc))


def move_file(
    source_path: str,
    destination_path: str,
    dry_run: bool = False,
    base_path: str = ".",
) -> dict[str, Any]:
    try:
        fm = FileManager(base_path)
        source = fm.resolve(source_path)
        destination = fm.resolve(destination_path)

        if not source.exists() or not source.is_file():
            return err(f"source file missing: {source}")

        if dry_run:
            return {
                "success": True,
                "dry_run": True,
                "source": str(source),
                "destination": str(destination),
            }

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        _log_op("move_file", source=str(
            source), destination=str(destination))
        return {
            "success": True,
            "dry_run": False,
            "source": str(source),
            "destination": str(destination),
        }
    except OSError as exc:
        return err(str(exc))


def copy_file(
    source_path: str,
    destination_path: str,
    dry_run: bool = False,
    base_path: str = ".",
) -> dict[str, Any]:
    try:
        fm = FileManager(base_path)
        source = fm.resolve(source_path)
        destination = fm.resolve(destination_path)

        if not source.exists() or not source.is_file():
            return err(f"source file missing: {source}")

        if dry_run:
            return {
                "success": True,
                "dry_run": True,
                "source": str(source),
                "destination": str(destination),
            }

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        _log_op("copy_file", source=str(
            source), destination=str(destination))
        return {
            "success": True,
            "dry_run": False,
            "source": str(source),
            "destination": str(destination),
            "bytes_copied": destination.stat().st_size,
        }
    except OSError as exc:
        return err(str(exc))


def create_directory(path: str, base_path: str = ".") -> dict[str, Any]:
    try:
        target = FileManager(base_path).resolve(path)
        target.mkdir(parents=True, exist_ok=True)
        _log_op("create_directory", path=str(target))
        return {"success": True, "path": str(target)}
    except OSError as exc:
        return err(str(exc))


def rename_file(
    path: str,
    new_name: str,
    dry_run: bool = False,
    base_path: str = ".",
) -> dict[str, Any]:
    try:
        target = FileManager(base_path).resolve(path)
        if not target.exists() or not target.is_file():
            return err(f"file not found: {target}")

        renamed = target.with_name(new_name)
        if dry_run:
            return {
                "success": True,
                "dry_run": True,
                "source": str(target),
                "destination": str(renamed),
            }

        target.rename(renamed)
        _log_op("rename_file", source=str(
            target), destination=str(renamed))
        return {
            "success": True,
            "dry_run": False,
            "source": str(target),
            "destination": str(renamed),
        }
    except OSError as exc:
        return err(str(exc))


def list_files(
    path: str,
    pattern: str = "*",
    recursive: bool = False,
    base_path: str = ".",
) -> dict[str, Any]:
    try:
        root = FileManager(base_path).resolve(path)
        if not root.exists():
            return err(f"directory not found: {root}")

        if not root.is_dir():
            return err(f"not a directory: {root}")

        if recursive:
            entries = [str(item.relative_to(root))
                       for item in root.rglob(pattern)]
        else:
            entries = [item.name for item in root.glob(pattern)]

        return {
            "success": True,
            "path": str(root),
            "pattern": pattern,
            "recursive": recursive,
            "count": len(entries),
            "files": sorted(entries),
        }
    except OSError as exc:
        return err(str(exc))


def search_files(
    path: str,
    name_pattern: str = "*",
    extension: str | None = None,
    min_size: int | None = None,
    max_size: int | None = None,
    base_path: str = ".",
) -> dict[str, Any]:
    try:
        root = FileManager(base_path).resolve(path)
        if not root.exists():
            return err(f"directory not found: {root}")

        if not root.is_dir():
            return err(f"not a directory: {root}")

        ext_filter = extension.lower() if extension else None
        matches: list[dict[str, Any]] = []
        for fp in root.rglob(name_pattern):
            if not fp.is_file():
                continue

            size = fp.stat().st_size
            if ext_filter and fp.suffix.lower() != ext_filter:
                continue
            if min_size is not None and size < min_size:
                continue
            if max_size is not None and size > max_size:
                continue
            matches.append({"path": str(fp.relative_to(root)), "size": size})

        matches.sort(key=lambda row: row["path"])
        return {
            "success": True,
            "path": str(root),
            "name_pattern": name_pattern,
            "extension": extension,
            "count": len(matches),
            "matches": matches,
        }
    except OSError as exc:
        return err(str(exc))


def search_file_content(
    path: str,
    query: str,
    file_pattern: str = "*",
    is_regex: bool = False,
    case_sensitive: bool = False,
    max_matches: int = 200,
    base_path: str = ".",
) -> dict[str, Any]:
    try:
        root = FileManager(base_path).resolve(path)
        if not root.exists() or not root.is_dir():
            return err(f"directory not found: {root}")

        flags = 0 if case_sensitive else re.IGNORECASE
        regex = re.compile(query, flags=flags) if is_regex else None

        results: list[dict[str, Any]] = []
        for fp in root.rglob(file_pattern):
            if not fp.is_file():
                continue

            try:
                text = fp.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue

            for line_no, line in enumerate(text.splitlines(), start=1):
                matched = bool(regex.search(line)) if regex else (
                    query in line if case_sensitive else query.lower() in line.lower()
                )
                if not matched:
                    continue

                results.append(
                    {
                        "path": str(fp.relative_to(root)),
                        "line": line_no,
                        "text": line,
                    }
                )
                if len(results) >= max_matches:
                    return {
                        "success": True,
                        "path": str(root),
                        "query": query,
                        "is_regex": is_regex,
                        "count": len(results),
                        "matches": results,
                        "truncated": True,
                    }

        return {
            "success": True,
            "path": str(root),
            "query": query,
            "is_regex": is_regex,
            "count": len(results),
            "matches": results,
            "truncated": False,
        }
    except OSError as exc:
        return err(str(exc))


def get_disk_usage(path: str = ".", base_path: str = ".") -> dict[str, Any]:
    try:
        target = FileManager(base_path).resolve(path)
        if not target.exists():
            return err(f"path not found: {target}")

        usage = shutil.disk_usage(target)
        return {
            "success": True,
            "path": str(target),
            "total": usage.total,
            "used": usage.used,
            "free": usage.free,
        }
    except OSError as exc:
        return err(str(exc))


def batch_delete(paths: list[str], dry_run: bool = False, base_path: str = ".") -> dict[str, Any]:
    fm = FileManager(base_path)
    deleted: list[str] = []
    skipped: list[dict[str, str]] = []

    for p in paths:
        try:
            target = fm.resolve(p)
            if not target.exists() or not target.is_file():
                skipped.append(
                    {"path": str(target), "reason": "not_found_or_not_file"})
                continue

            if dry_run:
                deleted.append(str(target))
                continue

            target.unlink()
            deleted.append(str(target))
        except OSError as exc:
            skipped.append({"path": p, "reason": str(exc)})

    if not dry_run and deleted:
        _log_op("batch_delete", deleted_count=len(deleted))

    return {
        "success": True,
        "dry_run": dry_run,
        "deleted_count": len(deleted),
        "deleted": deleted,
        "skipped": skipped,
    }


def get_operation_history(limit: int = 50) -> dict[str, Any]:
    if limit < 0:
        return err("limit must be >= 0")

    history = _AUDIT_TRAIL[-limit:] if limit else []
    return {"success": True, "count": len(history), "history": history}


def file_read(path: str, encoding: str = "utf-8") -> dict[str, Any]:
    return read_file(path=path, encoding=encoding)


def file_write(path: str, content: str, encoding: str = "utf-8") -> dict[str, Any]:
    return write_file(path=path, content=content, encoding=encoding)


def file_list(path: str, recursive: bool = False) -> dict[str, Any]:
    result = list_files(path=path, recursive=recursive)
    if result.get("success"):
        result["entries"] = result.get("files", [])
    return result


def file_search(path: str, pattern: str = "*") -> dict[str, Any]:
    result = search_files(path=path, name_pattern=pattern)
    if result.get("success"):
        result["matches"] = [item["path"]
                             for item in result.get("matches", [])]
        result["pattern"] = pattern
    return result
