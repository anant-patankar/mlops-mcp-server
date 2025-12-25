import json
import os
import time

import pytest

from mlops_mcp.cleanup import (
    cleanup_empty_logs,
    cleanup_failed_runs,
    cleanup_old_checkpoints,
    cleanup_project,
)


# ── cleanup_project ──────────────────────────────────────────────

class TestCleanupProject:
    def test_removes_pycache_dirs(self, tmp_path):
        cache = tmp_path / "__pycache__"
        cache.mkdir()
        (cache / "mod.cpython-312.pyc").write_bytes(b"")
        r = cleanup_project(str(tmp_path))
        assert r["success"] is True
        assert not cache.exists()
        assert any("__pycache__" in d for d in r["removed_directories"])

    def test_removes_pyc_files(self, tmp_path):
        f = tmp_path / "module.pyc"
        f.write_bytes(b"")
        r = cleanup_project(str(tmp_path))
        assert not f.exists()
        assert any("module.pyc" in p for p in r["removed_files"])

    def test_removes_ds_store(self, tmp_path):
        ds = tmp_path / ".DS_Store"
        ds.write_bytes(b"")
        r = cleanup_project(str(tmp_path))
        assert not ds.exists()
        assert any(".DS_Store" in p for p in r["removed_files"])

    def test_dry_run_does_not_delete(self, tmp_path):
        cache = tmp_path / "__pycache__"
        cache.mkdir()
        f = tmp_path / "stale.pyc"
        f.write_bytes(b"")
        r = cleanup_project(str(tmp_path), dry_run=True)
        assert r["success"] is True
        assert r["dry_run"] is True
        assert cache.exists()
        assert f.exists()
        assert len(r["removed_directories"]) > 0
        assert len(r["removed_files"]) > 0

    def test_empty_dir_removed_after_cleanup(self, tmp_path):
        empty = tmp_path / "empty_subdir"
        empty.mkdir()
        r = cleanup_project(str(tmp_path))
        assert not empty.exists()
        assert any("empty_subdir" in d for d in r["removed_empty_directories"])

    def test_missing_project_path_returns_error(self, tmp_path):
        r = cleanup_project(str(tmp_path / "ghost"))
        assert r["success"] is False

    def test_response_has_required_keys(self, tmp_path):
        r = cleanup_project(str(tmp_path))
        assert "removed_files" in r
        assert "removed_directories" in r
        assert "removed_empty_directories" in r

    def test_non_removable_files_left_intact(self, tmp_path):
        keep = tmp_path / "model.pkl"
        keep.write_bytes(b"\x00" * 10)
        cleanup_project(str(tmp_path))
        assert keep.exists()


# ── cleanup_old_checkpoints ──────────────────────────────────────

class TestCleanupOldCheckpoints:
    def _make_ckpts(self, root, names):
        # write files with distinct mtimes so sort order is deterministic
        paths = []
        for i, name in enumerate(names):
            p = root / name
            p.write_bytes(b"\x00")
            os.utime(p, (1000 + i, 1000 + i))
            paths.append(p)
        return paths

    def test_keeps_n_most_recent_removes_rest(self, tmp_path):
        self._make_ckpts(tmp_path, ["a.ckpt", "b.ckpt", "c.ckpt", "d.ckpt"])
        r = cleanup_old_checkpoints(str(tmp_path), keep=2)
        assert r["success"] is True
        assert len(r["kept"]) == 2
        assert len(r["removed"]) == 2
        # newest two (c, d) kept; oldest two (a, b) removed
        assert not (tmp_path / "a.ckpt").exists()
        assert not (tmp_path / "b.ckpt").exists()
        assert (tmp_path / "c.ckpt").exists()
        assert (tmp_path / "d.ckpt").exists()

    def test_keep_zero_removes_all(self, tmp_path):
        self._make_ckpts(tmp_path, ["x.pt", "y.pt"])
        r = cleanup_old_checkpoints(str(tmp_path), keep=0)
        assert r["removed_count"] == 2
        assert not (tmp_path / "x.pt").exists()

    def test_keep_larger_than_total_keeps_all(self, tmp_path):
        self._make_ckpts(tmp_path, ["a.ckpt", "b.ckpt"])
        r = cleanup_old_checkpoints(str(tmp_path), keep=10)
        assert r["removed_count"] == 0
        assert len(r["kept"]) == 2

    def test_dry_run_does_not_delete(self, tmp_path):
        self._make_ckpts(tmp_path, ["a.ckpt", "b.ckpt", "c.ckpt"])
        r = cleanup_old_checkpoints(str(tmp_path), keep=1, dry_run=True)
        assert r["dry_run"] is True
        assert r["removed_count"] == 2
        # files must still exist
        assert (tmp_path / "a.ckpt").exists()
        assert (tmp_path / "b.ckpt").exists()

    def test_negative_keep_returns_error(self, tmp_path):
        r = cleanup_old_checkpoints(str(tmp_path), keep=-1)
        assert r["success"] is False
        assert ">= 0" in r["error"]

    def test_missing_directory_returns_error(self, tmp_path):
        r = cleanup_old_checkpoints(str(tmp_path / "ghost"))
        assert r["success"] is False

    def test_non_checkpoint_files_ignored(self, tmp_path):
        (tmp_path / "notes.txt").write_text("hi")
        self._make_ckpts(tmp_path, ["a.ckpt"])
        r = cleanup_old_checkpoints(str(tmp_path), keep=0)
        # only the .ckpt removed; .txt untouched
        assert r["removed_count"] == 1
        assert (tmp_path / "notes.txt").exists()

    def test_removed_count_matches_removed_list(self, tmp_path):
        self._make_ckpts(tmp_path, ["a.pt", "b.pt", "c.pt"])
        r = cleanup_old_checkpoints(str(tmp_path), keep=1)
        assert r["removed_count"] == len(r["removed"])


# ── cleanup_failed_runs ──────────────────────────────────────────

class TestCleanupFailedRuns:
    def _make_run(self, root, name, status=None, with_metrics=False):
        run = root / name
        run.mkdir()
        if status is not None:
            (run / "status.json").write_text(
                json.dumps({"status": status}), encoding="utf-8"
            )
        if with_metrics:
            (run / "metrics.json").write_text(
                json.dumps({"acc": 0.9}), encoding="utf-8"
            )
        return run

    def test_removes_run_with_no_status_file(self, tmp_path):
        self._make_run(tmp_path, "run_a")
        r = cleanup_failed_runs(str(tmp_path))
        assert r["success"] is True
        assert "run_a" in r["removed"]
        assert not (tmp_path / "run_a").exists()

    def test_removes_run_with_failed_status(self, tmp_path):
        self._make_run(tmp_path, "run_b", status="failed", with_metrics=True)
        r = cleanup_failed_runs(str(tmp_path))
        assert "run_b" in r["removed"]

    def test_keeps_successful_run_with_metrics(self, tmp_path):
        self._make_run(tmp_path, "run_ok", status="success", with_metrics=True)
        r = cleanup_failed_runs(str(tmp_path))
        assert "run_ok" in r["kept"]
        assert (tmp_path / "run_ok").exists()

    def test_dry_run_does_not_delete(self, tmp_path):
        self._make_run(tmp_path, "run_bad", status="failed")
        r = cleanup_failed_runs(str(tmp_path), dry_run=True)
        assert r["dry_run"] is True
        assert "run_bad" in r["removed"]
        assert (tmp_path / "run_bad").exists()

    def test_missing_runs_path_returns_error(self, tmp_path):
        r = cleanup_failed_runs(str(tmp_path / "ghost"))
        assert r["success"] is False

    def test_removed_count_matches_removed_list(self, tmp_path):
        self._make_run(tmp_path, "r1")
        self._make_run(tmp_path, "r2", status="success", with_metrics=True)
        self._make_run(tmp_path, "r3")
        r = cleanup_failed_runs(str(tmp_path))
        assert r["removed_count"] == len(r["removed"])

    def test_success_without_metrics_is_removed(self, tmp_path):
        # status=success but no metrics.json — should still be removed
        self._make_run(tmp_path, "run_no_metrics", status="success")
        r = cleanup_failed_runs(str(tmp_path))
        assert "run_no_metrics" in r["removed"]


# ── cleanup_empty_logs ───────────────────────────────────────────

class TestCleanupEmptyLogs:
    def test_removes_log_below_min_size(self, tmp_path):
        small = tmp_path / "small.log"
        small.write_text("hi")
        r = cleanup_empty_logs(str(tmp_path), min_size_bytes=100)
        assert r["success"] is True
        assert not small.exists()
        assert "small.log" in r["removed"]

    def test_keeps_log_at_or_above_min_size(self, tmp_path):
        big = tmp_path / "big.log"
        big.write_bytes(b"x" * 200)
        r = cleanup_empty_logs(str(tmp_path), min_size_bytes=100)
        assert big.exists()
        assert "big.log" in r["kept"]

    def test_dry_run_does_not_delete(self, tmp_path):
        small = tmp_path / "tiny.log"
        small.write_text("")
        r = cleanup_empty_logs(str(tmp_path), min_size_bytes=100, dry_run=True)
        assert r["dry_run"] is True
        assert "tiny.log" in r["removed"]
        assert small.exists()

    def test_negative_min_size_returns_error(self, tmp_path):
        r = cleanup_empty_logs(str(tmp_path), min_size_bytes=-1)
        assert r["success"] is False
        assert ">= 0" in r["error"]

    def test_missing_logs_path_returns_error(self, tmp_path):
        r = cleanup_empty_logs(str(tmp_path / "ghost"))
        assert r["success"] is False

    def test_non_log_files_ignored(self, tmp_path):
        txt = tmp_path / "notes.txt"
        txt.write_text("small")
        r = cleanup_empty_logs(str(tmp_path), min_size_bytes=100)
        assert txt.exists()
        assert r["removed"] == []

    def test_removed_count_matches_removed_list(self, tmp_path):
        for name in ("a.log", "b.log", "c.log"):
            (tmp_path / name).write_text("x")
        (tmp_path / "big.log").write_bytes(b"x" * 500)
        r = cleanup_empty_logs(str(tmp_path), min_size_bytes=100)
        assert r["removed_count"] == len(r["removed"])

    def test_min_size_boundary_exact_match_is_kept(self, tmp_path):
        exact = tmp_path / "exact.log"
        exact.write_bytes(b"x" * 100)
        r = cleanup_empty_logs(str(tmp_path), min_size_bytes=100)
        assert exact.exists()
        assert "exact.log" in r["kept"]
