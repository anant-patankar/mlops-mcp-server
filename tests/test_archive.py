import io
import json
import tarfile
import zipfile
from pathlib import Path

import pytest

from mlops_mcp.archive import (
    _archive_members,
    _is_within_directory,
    archive_experiment,
    create_archive,
    extract_archive,
    list_archive_contents,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_tree(root: Path) -> None:
    (root / "a.txt").write_text("alpha")
    (root / "sub").mkdir()
    (root / "sub" / "b.txt").write_text("beta")


# ---------------------------------------------------------------------------
# _is_within_directory
# ---------------------------------------------------------------------------

def test_is_within_directory_safe(tmp_path):
    assert _is_within_directory(tmp_path, tmp_path / "sub" / "file.txt") is True


def test_is_within_directory_traversal(tmp_path):
    assert _is_within_directory(tmp_path, tmp_path / ".." / "escape.txt") is False


def test_is_within_directory_same_path(tmp_path):
    assert _is_within_directory(tmp_path, tmp_path) is True


# ---------------------------------------------------------------------------
# _archive_members
# ---------------------------------------------------------------------------

def test_archive_members_file_returns_self(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("hi")
    assert _archive_members(f) == [f]


def test_archive_members_directory_recursive(tmp_path):
    make_tree(tmp_path)
    members = _archive_members(tmp_path)
    names = {p.name for p in members}
    assert names == {"a.txt", "b.txt"}


def test_archive_members_skips_subdirs(tmp_path):
    make_tree(tmp_path)
    members = _archive_members(tmp_path)
    assert all(p.is_file() for p in members)


# ---------------------------------------------------------------------------
# create_archive
# ---------------------------------------------------------------------------

def test_create_archive_zip(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    make_tree(src)
    out = tmp_path / "out.zip"
    result = create_archive(str(src), str(out), archive_format="zip")
    assert result["success"] is True
    assert result["file_count"] == 2
    assert Path(result["archive_path"]).exists()


def test_create_archive_tar(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    make_tree(src)
    out = tmp_path / "out.tar"
    result = create_archive(str(src), str(out), archive_format="tar")
    assert result["success"] is True
    assert tarfile.is_tarfile(str(out))


def test_create_archive_tar_gz(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    make_tree(src)
    out = tmp_path / "out.tar.gz"
    result = create_archive(str(src), str(out), archive_format="tar.gz")
    assert result["success"] is True
    assert result["archive_format"] == "tar.gz"


def test_create_archive_single_file(tmp_path):
    f = tmp_path / "model.pkl"
    f.write_bytes(b"\x00" * 100)
    out = tmp_path / "model.zip"
    result = create_archive(str(f), str(out))
    assert result["success"] is True
    assert result["file_count"] == 1


def test_create_archive_creates_parent_dirs(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "f.txt").write_text("x")
    out = tmp_path / "nested" / "deep" / "out.zip"
    result = create_archive(str(src), str(out))
    assert result["success"] is True


def test_create_archive_bad_format(tmp_path):
    src = tmp_path / "x.txt"
    src.write_text("x")
    result = create_archive(str(src), str(tmp_path / "out.7z"), archive_format="7z")
    assert result["success"] is False
    assert "archive_format" in result["error"]


def test_create_archive_missing_source(tmp_path):
    result = create_archive(str(tmp_path / "ghost"), str(tmp_path / "out.zip"))
    assert result["success"] is False


def test_create_archive_returns_size(tmp_path):
    src = tmp_path / "x.txt"
    src.write_text("hello world")
    out = tmp_path / "out.zip"
    result = create_archive(str(src), str(out))
    assert result["size"] > 0


# ---------------------------------------------------------------------------
# list_archive_contents
# ---------------------------------------------------------------------------

def test_list_archive_contents_zip(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    make_tree(src)
    out = tmp_path / "out.zip"
    create_archive(str(src), str(out))
    result = list_archive_contents(str(out))
    assert result["success"] is True
    assert result["archive_type"] == "zip"
    assert result["file_count"] == 2
    assert len(result["entries"]) == 2


def test_list_archive_contents_tar(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    make_tree(src)
    out = tmp_path / "out.tar"
    create_archive(str(src), str(out), archive_format="tar")
    result = list_archive_contents(str(out))
    assert result["success"] is True
    assert result["archive_type"] == "tar"


def test_list_archive_contents_missing_file(tmp_path):
    result = list_archive_contents(str(tmp_path / "ghost.zip"))
    assert result["success"] is False


def test_list_archive_contents_unsupported_format(tmp_path):
    f = tmp_path / "data.7z"
    f.write_bytes(b"not an archive")
    result = list_archive_contents(str(f))
    assert result["success"] is False
    assert "unsupported" in result["error"]


# ---------------------------------------------------------------------------
# extract_archive
# ---------------------------------------------------------------------------

def test_extract_archive_zip(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    make_tree(src)
    out = tmp_path / "out.zip"
    create_archive(str(src), str(out))
    dest = tmp_path / "extracted"
    result = extract_archive(str(out), str(dest))
    assert result["success"] is True
    assert result["extracted_count"] == 2
    assert (dest / "a.txt").exists()
    assert (dest / "sub" / "b.txt").exists()


def test_extract_archive_tar(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    make_tree(src)
    out = tmp_path / "out.tar"
    create_archive(str(src), str(out), archive_format="tar")
    dest = tmp_path / "extracted"
    result = extract_archive(str(out), str(dest))
    assert result["success"] is True
    assert result["extracted_count"] == 2


def test_extract_archive_creates_destination(tmp_path):
    src = tmp_path / "f.txt"
    src.write_text("x")
    out = tmp_path / "out.zip"
    create_archive(str(src), str(out))
    dest = tmp_path / "new" / "dir"
    result = extract_archive(str(out), str(dest))
    assert result["success"] is True
    assert dest.exists()


def test_extract_archive_path_traversal_zip(tmp_path):
    # craft a zip with a path traversal member
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("../escape.txt", "pwned")
    dest = tmp_path / "dest"
    dest.mkdir()
    result = extract_archive(str(evil), str(dest))
    assert result["success"] is False
    assert "unsafe" in result["error"]


def test_extract_archive_missing_file(tmp_path):
    result = extract_archive(str(tmp_path / "ghost.zip"), str(tmp_path / "dest"))
    assert result["success"] is False


def test_extract_archive_unsupported_format(tmp_path):
    f = tmp_path / "data.7z"
    f.write_bytes(b"not an archive")
    result = extract_archive(str(f), str(tmp_path / "dest"))
    assert result["success"] is False


# ---------------------------------------------------------------------------
# archive_experiment
# ---------------------------------------------------------------------------

def test_archive_experiment_creates_zip_and_manifest(tmp_path):
    exp = tmp_path / "run01"
    exp.mkdir()
    make_tree(exp)
    result = archive_experiment(str(exp))
    assert result["success"] is True
    assert Path(result["archive_path"]).exists()
    assert Path(result["manifest_path"]).exists()


def test_archive_experiment_manifest_content(tmp_path):
    exp = tmp_path / "run01"
    exp.mkdir()
    (exp / "metrics.json").write_text('{"acc": 0.9}')
    result = archive_experiment(str(exp))
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    assert manifest["experiment_path"] == str(exp.resolve())
    assert "created_at" in manifest
    assert manifest["file_count"] == 1


def test_archive_experiment_custom_output_dir(tmp_path):
    exp = tmp_path / "run01"
    exp.mkdir()
    (exp / "f.txt").write_text("x")
    out_dir = tmp_path / "my_archives"
    result = archive_experiment(str(exp), output_directory=str(out_dir))
    assert result["success"] is True
    assert Path(result["archive_path"]).parent == out_dir.resolve()


def test_archive_experiment_default_output_dir(tmp_path):
    exp = tmp_path / "run01"
    exp.mkdir()
    (exp / "f.txt").write_text("x")
    result = archive_experiment(str(exp))
    assert Path(result["archive_path"]).parent.name == "archives"


def test_archive_experiment_missing_source(tmp_path):
    result = archive_experiment(str(tmp_path / "ghost"))
    assert result["success"] is False


def test_archive_experiment_source_is_file_not_dir(tmp_path):
    f = tmp_path / "model.pkl"
    f.write_bytes(b"\x00")
    result = archive_experiment(str(f))
    assert result["success"] is False


def test_archive_experiment_file_count_in_return(tmp_path):
    exp = tmp_path / "run01"
    exp.mkdir()
    make_tree(exp)
    result = archive_experiment(str(exp))
    assert result["file_count"] == 2


# TODO: test archive_experiment manifest created_at matches filename stamp exactly
@pytest.mark.skip(reason="timestamp coupling between filename and manifest not yet validated")
def test_archive_experiment_stamp_consistency():
    pass
