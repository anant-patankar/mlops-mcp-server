import pytest
from pathlib import Path

from mlops_mcp.fileops import (
    create_file,
    read_file,
    write_file,
    delete_file,
    move_file,
    copy_file,
    create_directory,
    rename_file,
    list_files,
    search_files,
    search_file_content,
    get_disk_usage,
    batch_delete,
    get_operation_history,
)


# ── create_file ──────────────────────────────────────────────────

class TestCreateFile:
    def test_basic(self, tmp_path):
        r = create_file(str(tmp_path / "new.txt"), content="hello", base_path=str(tmp_path))
        assert r["success"] is True
        assert (tmp_path / "new.txt").read_text() == "hello"

    def test_no_overwrite_by_default(self, tmp_path):
        f = tmp_path / "existing.txt"
        f.write_text("original")
        r = create_file(str(f), content="new", base_path=str(tmp_path))
        assert r["success"] is False
        assert f.read_text() == "original"  # untouched

    def test_overwrite_flag(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("old")
        r = create_file(str(f), content="new", overwrite=True, base_path=str(tmp_path))
        assert r["success"] is True
        assert f.read_text() == "new"

    def test_creates_parent_dirs(self, tmp_path):
        deep = str(tmp_path / "a" / "b" / "c" / "file.txt")
        r = create_file(deep, content="nested", base_path=str(tmp_path))
        assert r["success"] is True


# ── write_file (atomic) ──────────────────────────────────────────

class TestWriteFile:
    def test_atomic_no_tmp_left_behind(self, tmp_path):
        target = tmp_path / "output.txt"
        write_file(str(target), "hello world", base_path=str(tmp_path))
        # the .tmp file must be gone after successful write
        tmp_file = target.with_suffix(".txt.tmp")
        assert not tmp_file.exists()
        assert target.read_text() == "hello world"

    def test_overwrites_existing(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("v1")
        write_file(str(f), "v2", base_path=str(tmp_path))
        assert f.read_text() == "v2"


# ── read_file ────────────────────────────────────────────────────

class TestReadFile:
    def test_basic(self, tmp_path):
        f = tmp_path / "sample.txt"
        f.write_text("read me")
        r = read_file(str(f), base_path=str(tmp_path))
        assert r["success"] is True
        assert r["content"] == "read me"

    def test_missing(self, tmp_path):
        r = read_file(str(tmp_path / "ghost.txt"), base_path=str(tmp_path))
        assert r["success"] is False

    def test_directory_rejected(self, tmp_path):
        r = read_file(str(tmp_path), base_path=str(tmp_path))
        assert r["success"] is False


# ── delete_file ──────────────────────────────────────────────────

class TestDeleteFile:
    def test_deletes(self, tmp_path):
        f = tmp_path / "bye.txt"
        f.write_text("gone")
        r = delete_file(str(f), base_path=str(tmp_path))
        assert r["success"] is True
        assert not f.exists()

    def test_dry_run_leaves_file_intact(self, tmp_path):
        f = tmp_path / "keep.txt"
        f.write_text("stay")
        r = delete_file(str(f), dry_run=True, base_path=str(tmp_path))
        assert r["success"] is True
        assert r["deleted"] is False
        assert f.exists()

    def test_missing_file(self, tmp_path):
        r = delete_file(str(tmp_path / "nope.txt"), base_path=str(tmp_path))
        assert r["success"] is False


# ── move_file ────────────────────────────────────────────────────

class TestMoveFile:
    def test_moves(self, tmp_path):
        src = tmp_path / "src.txt"
        src.write_text("moving")
        dst = tmp_path / "dst.txt"
        r = move_file(str(src), str(dst), base_path=str(tmp_path))
        assert r["success"] is True
        assert not src.exists()
        assert dst.read_text() == "moving"

    def test_dry_run(self, tmp_path):
        src = tmp_path / "src.txt"
        src.write_text("stay")
        dst = tmp_path / "dst.txt"
        r = move_file(str(src), str(dst), dry_run=True, base_path=str(tmp_path))
        assert r["dry_run"] is True
        assert src.exists()
        assert not dst.exists()

    def test_missing_source(self, tmp_path):
        r = move_file(str(tmp_path / "ghost.txt"), str(tmp_path / "dst.txt"), base_path=str(tmp_path))
        assert r["success"] is False


# ── copy_file ────────────────────────────────────────────────────

class TestCopyFile:
    def test_copies(self, tmp_path):
        src = tmp_path / "original.txt"
        src.write_text("copy me")
        dst = tmp_path / "copy.txt"
        r = copy_file(str(src), str(dst), base_path=str(tmp_path))
        assert r["success"] is True
        assert src.exists()       # source survives
        assert dst.read_text() == "copy me"

    def test_bytes_copied_in_response(self, tmp_path):
        src = tmp_path / "src.txt"
        src.write_bytes(b"x" * 500)
        dst = tmp_path / "dst.txt"
        r = copy_file(str(src), str(dst), base_path=str(tmp_path))
        assert r["bytes_copied"] == 500


# ── list_files ───────────────────────────────────────────────────

class TestListFiles:
    def test_flat(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        r = list_files(str(tmp_path), base_path=str(tmp_path))
        assert r["success"] is True
        assert r["count"] == 2

    def test_recursive(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "root.txt").write_text("r")
        (sub / "nested.txt").write_text("n")
        r = list_files(str(tmp_path), recursive=True, base_path=str(tmp_path))
        # rglob("*") includes directories in the listing — sub/ counts as an entry
        assert r["count"] == 3
        assert "root.txt" in r["files"]
        assert "sub/nested.txt" in r["files"] or "sub" in r["files"]

    def test_missing_dir(self, tmp_path):
        r = list_files(str(tmp_path / "nope"), base_path=str(tmp_path))
        assert r["success"] is False


# ── search_file_content ──────────────────────────────────────────

class TestSearchFileContent:
    def test_finds_string(self, tmp_path):
        f = tmp_path / "log.txt"
        f.write_text("line1\nERROR something failed\nline3\n")
        r = search_file_content(str(tmp_path), "ERROR", base_path=str(tmp_path))
        assert r["count"] == 1
        assert r["matches"][0]["line"] == 2

    def test_case_insensitive_by_default(self, tmp_path):
        f = tmp_path / "notes.txt"
        f.write_text("Training loss: 0.02\ntraining accuracy: 98%\n")
        r = search_file_content(str(tmp_path), "training", base_path=str(tmp_path))
        assert r["count"] == 2

    def test_regex_mode(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("epoch=1\nepoch=10\nbatch=5\n")
        r = search_file_content(str(tmp_path), r"epoch=\d+", is_regex=True, base_path=str(tmp_path))
        assert r["count"] == 2

    def test_max_matches_truncation(self, tmp_path):
        f = tmp_path / "big.txt"
        f.write_text("\n".join(["match"] * 50))
        r = search_file_content(str(tmp_path), "match", max_matches=10, base_path=str(tmp_path))
        assert r["count"] == 10
        assert r["truncated"] is True


# ── batch_delete ─────────────────────────────────────────────────

class TestBatchDelete:
    def test_deletes_all(self, tmp_path):
        files = []
        for i in range(3):
            f = tmp_path / f"f{i}.txt"
            f.write_text("x")
            files.append(str(f))
        r = batch_delete(files, base_path=str(tmp_path))
        assert r["deleted_count"] == 3
        assert r["skipped"] == []

    def test_skips_missing(self, tmp_path):
        real = tmp_path / "real.txt"
        real.write_text("x")
        r = batch_delete([str(real), "/ghost/path.txt"], base_path=str(tmp_path))
        assert r["deleted_count"] == 1
        assert len(r["skipped"]) == 1

    def test_dry_run(self, tmp_path):
        f = tmp_path / "keep.txt"
        f.write_text("x")
        r = batch_delete([str(f)], dry_run=True, base_path=str(tmp_path))
        assert r["dry_run"] is True
        assert f.exists()


# ── get_operation_history ────────────────────────────────────────

class TestOperationHistory:
    def test_limit_respected(self, tmp_path):
        # generate some operations
        for i in range(5):
            f = tmp_path / f"op{i}.txt"
            create_file(str(f), content="x", base_path=str(tmp_path))

        r = get_operation_history(limit=3)
        assert r["success"] is True
        assert r["count"] <= 3

    def test_zero_limit(self):
        r = get_operation_history(limit=0)
        assert r["success"] is True
        assert r["count"] == 0

    def test_negative_limit_is_error(self):
        r = get_operation_history(limit=-1)
        assert r["success"] is False