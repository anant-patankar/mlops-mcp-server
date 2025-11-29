from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import tarfile
import zipfile

from ._types import err as _fail


def _is_within_directory(root: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _archive_members(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return [item for item in source.rglob("*") if item.is_file()]


def create_archive(
    source_path: str,
    archive_path: str,
    archive_format: str = "zip",
    compression_level: int = 6,
) -> dict[str, Any]:
    source = Path(source_path).expanduser().resolve()
    target = Path(archive_path).expanduser().resolve()

    if not source.exists():
        return _fail(f"source path not found: {source}")

    if archive_format not in {"zip", "tar", "tar.gz"}:
        return _fail("archive_format must be one of: zip, tar, tar.gz")

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        members = _archive_members(source)

        if archive_format == "zip":
            with zipfile.ZipFile(
                target,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=compression_level,
            ) as handle:
                for item in members:
                    arcname = item.relative_to(
                        source.parent if source.is_file() else source)
                    handle.write(item, arcname=str(arcname))
        else:
            mode = "w:gz" if archive_format == "tar.gz" else "w"
            with tarfile.open(target, mode=mode) as handle:
                base = source.parent if source.is_file() else source
                for item in members:
                    arcname = item.relative_to(base)
                    handle.add(item, arcname=str(arcname))

        return {
            "success": True,
            "source_path": str(source),
            "archive_path": str(target),
            "archive_format": archive_format,
            "file_count": len(members),
            "size": target.stat().st_size,
        }
    except OSError as exc:
        return _fail(str(exc))


def list_archive_contents(archive_path: str) -> dict[str, Any]:
    target = Path(archive_path).expanduser().resolve()
    if not target.exists() or not target.is_file():
        return _fail(f"archive file not found: {target}")

    try:
        if zipfile.is_zipfile(target):
            with zipfile.ZipFile(target, "r") as handle:
                names = handle.namelist()
            archive_type = "zip"
        elif tarfile.is_tarfile(target):
            with tarfile.open(target, "r:*") as handle:
                names = handle.getnames()
            archive_type = "tar"
        else:
            return _fail(f"unsupported archive type: {target}")

        return {
            "success": True,
            "archive_path": str(target),
            "archive_type": archive_type,
            "file_count": len(names),
            "entries": names,
        }
    except OSError as exc:
        return _fail(str(exc))


def extract_archive(archive_path: str, destination_path: str) -> dict[str, Any]:
    archive = Path(archive_path).expanduser().resolve()
    destination = Path(destination_path).expanduser().resolve()

    if not archive.exists() or not archive.is_file():
        return _fail(f"archive file not found: {archive}")

    try:
        destination.mkdir(parents=True, exist_ok=True)

        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive, "r") as handle:
                for name in handle.namelist():
                    out_path = destination / name
                    if not _is_within_directory(destination, out_path):
                        return _fail(f"unsafe archive member path: {name}")
                handle.extractall(destination)
                extracted = handle.namelist()

        elif tarfile.is_tarfile(archive):
            with tarfile.open(archive, "r:*") as handle:
                members = handle.getmembers()
                for member in members:
                    out_path = destination / member.name
                    if not _is_within_directory(destination, out_path):
                        return _fail(f"unsafe archive member path: {member.name}")
                handle.extractall(destination, filter="data")
                extracted = [member.name for member in members]
        else:
            return _fail(f"unsupported archive type: {archive}")

        return {
            "success": True,
            "archive_path": str(archive),
            "destination_path": str(destination),
            "extracted_count": len(extracted),
            "entries": extracted,
        }
    except OSError as exc:
        return _fail(str(exc))


def archive_experiment(
    experiment_path: str,
    output_directory: str | None = None,
    compression_level: int = 6,
) -> dict[str, Any]:
    source = Path(experiment_path).expanduser().resolve()
    if not source.exists() or not source.is_dir():
        return _fail(f"experiment directory not found: {source}")

    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
    archive_root = (
        Path(output_directory).expanduser().resolve()
        if output_directory
        else source.parent / "archives"
    )
    archive_root.mkdir(parents=True, exist_ok=True)

    archive_file = archive_root / f"{source.name}_{stamp}.zip"
    manifest_file = archive_root / f"{source.name}_{stamp}_manifest.json"

    created = create_archive(
        source_path=str(source),
        archive_path=str(archive_file),
        archive_format="zip",
        compression_level=compression_level,
    )
    if not created.get("success"):
        return created

    try:
        payload = {
            "experiment_path": str(source),
            "archive_path": str(archive_file),
            "created_at": stamp,
            "file_count": created["file_count"],
            "archive_size": created["size"],
        }
        manifest_file.write_text(json.dumps(
            payload, indent=2), encoding="utf-8")
        payload["success"] = True
        payload["manifest_path"] = str(manifest_file)
        return payload
    except OSError as exc:
        return _fail(str(exc))
