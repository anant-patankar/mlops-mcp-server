from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mlops_mcp.git_ops import (
    create_gitignore,
    detect_uncommitted_changes,
    git_add,
    git_commit,
    git_init,
    git_log,
    git_status,
)

_NOT_A_REPO = (None, {"success": False, "error": "not a git repo"}, None)


def _make_repo(tmp_path):
    repo = MagicMock()
    repo.working_tree_dir = str(tmp_path)
    repo.git_dir = str(tmp_path / ".git")
    repo.head.is_detached = False
    repo.active_branch.name = "main"
    repo.untracked_files = []
    repo.index.diff.return_value = []
    repo.iter_commits.return_value = []
    return repo


# ── git_init ─────────────────────────────────────────────────────

class TestGitInit:
    def test_success_returns_repo_root_and_git_dir(self, tmp_path):
        mock_repo = _make_repo(tmp_path)
        mock_git = MagicMock()
        mock_git.Repo.init.return_value = mock_repo
        with patch("mlops_mcp.git_ops._git_module", return_value=(mock_git, None)):
            r = git_init(str(tmp_path))
        assert r["success"] is True
        assert "repo_root" in r
        assert "git_dir" in r

    def test_gitpython_not_installed_returns_error(self, tmp_path):
        with patch("mlops_mcp.git_ops._git_module", return_value=(None, {"success": False, "error": "gitpython not installed."})):
            r = git_init(str(tmp_path))
        assert r["success"] is False

    def test_create_ignore_false_skips_gitignore(self, tmp_path):
        mock_repo = _make_repo(tmp_path)
        mock_git = MagicMock()
        mock_git.Repo.init.return_value = mock_repo
        with patch("mlops_mcp.git_ops._git_module", return_value=(mock_git, None)), \
             patch("mlops_mcp.git_ops.create_gitignore") as mock_ignore:
            git_init(str(tmp_path), create_ignore=False)
        mock_ignore.assert_not_called()


# ── git_status ───────────────────────────────────────────────────

class TestGitStatus:
    def test_success_returns_branch_and_file_lists(self, tmp_path):
        repo = _make_repo(tmp_path)
        with patch("mlops_mcp.git_ops._get_repo", return_value=(repo, None, None)):
            r = git_status(str(tmp_path))
        assert r["success"] is True
        assert r["branch"] == "main"
        assert "staged" in r
        assert "unstaged" in r
        assert "untracked" in r

    def test_not_a_git_repo_returns_error(self, tmp_path):
        with patch("mlops_mcp.git_ops._get_repo", return_value=_NOT_A_REPO):
            r = git_status(str(tmp_path))
        assert r["success"] is False

    def test_staged_and_unstaged_are_sorted_and_deduped(self, tmp_path):
        repo = _make_repo(tmp_path)
        item_b = MagicMock(); item_b.a_path = "b.py"
        item_a = MagicMock(); item_a.a_path = "a.py"
        item_dup = MagicMock(); item_dup.a_path = "a.py"
        # staged returns b then a (unsorted), unstaged returns a duplicate
        repo.index.diff.side_effect = lambda ref: (
            [item_b, item_a] if ref == "HEAD" else [item_dup]
        )
        with patch("mlops_mcp.git_ops._get_repo", return_value=(repo, None, None)):
            r = git_status(str(tmp_path))
        assert r["staged"] == ["a.py", "b.py"]
        assert r["unstaged"] == ["a.py"]  # deduped


# ── git_add ──────────────────────────────────────────────────────

class TestGitAdd:
    def test_success_with_explicit_files(self, tmp_path):
        repo = _make_repo(tmp_path)
        with patch("mlops_mcp.git_ops._get_repo", return_value=(repo, None, None)):
            r = git_add(str(tmp_path), files=["data.csv", "model.py"])
        assert r["success"] is True
        assert r["added"] == ["data.csv", "model.py"]

    def test_no_files_defaults_to_dot(self, tmp_path):
        repo = _make_repo(tmp_path)
        with patch("mlops_mcp.git_ops._get_repo", return_value=(repo, None, None)):
            r = git_add(str(tmp_path))
        assert r["added"] == ["."]

    def test_not_a_git_repo_returns_error(self, tmp_path):
        with patch("mlops_mcp.git_ops._get_repo", return_value=_NOT_A_REPO):
            r = git_add(str(tmp_path))
        assert r["success"] is False


# ── git_commit ───────────────────────────────────────────────────

class TestGitCommit:
    def test_success_returns_hash_and_message(self, tmp_path):
        repo = _make_repo(tmp_path)
        fake_commit = MagicMock()
        fake_commit.hexsha = "abc1234def5678"
        fake_commit.message = "add features\n"
        repo.index.commit.return_value = fake_commit
        with patch("mlops_mcp.git_ops._get_repo", return_value=(repo, None, None)):
            r = git_commit(str(tmp_path), message="add features")
        assert r["success"] is True
        assert r["commit"] == "abc1234def5678"
        assert r["message"] == "add features"

    def test_empty_message_returns_error(self, tmp_path):
        repo = _make_repo(tmp_path)
        with patch("mlops_mcp.git_ops._get_repo", return_value=(repo, None, None)):
            r = git_commit(str(tmp_path), message="")
        assert r["success"] is False

    def test_whitespace_only_message_returns_error(self, tmp_path):
        repo = _make_repo(tmp_path)
        with patch("mlops_mcp.git_ops._get_repo", return_value=(repo, None, None)):
            r = git_commit(str(tmp_path), message="   ")
        assert r["success"] is False

    def test_not_a_git_repo_returns_error(self, tmp_path):
        with patch("mlops_mcp.git_ops._get_repo", return_value=_NOT_A_REPO):
            r = git_commit(str(tmp_path), message="update")
        assert r["success"] is False


# ── git_log ──────────────────────────────────────────────────────

class TestGitLog:
    def _fake_commit(self, sha, msg, ts="2025-01-01T00:00:00+00:00", author="dev"):
        c = MagicMock()
        c.hexsha = sha
        c.message = msg + "\n"
        c.committed_datetime.isoformat.return_value = ts
        c.author.__str__ = lambda self: author
        return c

    def test_success_returns_commits_with_correct_fields(self, tmp_path):
        repo = _make_repo(tmp_path)
        repo.iter_commits.return_value = [
            self._fake_commit("abc1234def56789", "first"),
        ]
        with patch("mlops_mcp.git_ops._get_repo", return_value=(repo, None, None)):
            r = git_log(str(tmp_path), limit=5)
        assert r["success"] is True
        assert r["count"] == 1
        entry = r["commits"][0]
        assert entry["hash"] == "abc1234def56789"
        assert entry["short_hash"] == "abc1234"
        assert entry["message"] == "first"

    def test_limit_zero_returns_error(self, tmp_path):
        repo = _make_repo(tmp_path)
        with patch("mlops_mcp.git_ops._get_repo", return_value=(repo, None, None)):
            r = git_log(str(tmp_path), limit=0)
        assert r["success"] is False

    def test_negative_limit_returns_error(self, tmp_path):
        repo = _make_repo(tmp_path)
        with patch("mlops_mcp.git_ops._get_repo", return_value=(repo, None, None)):
            r = git_log(str(tmp_path), limit=-1)
        assert r["success"] is False

    def test_not_a_git_repo_returns_error(self, tmp_path):
        with patch("mlops_mcp.git_ops._get_repo", return_value=_NOT_A_REPO):
            r = git_log(str(tmp_path))
        assert r["success"] is False


# ── create_gitignore ─────────────────────────────────────────────

class TestCreateGitignore:
    def test_creates_gitignore_with_ml_patterns(self, tmp_path):
        r = create_gitignore(str(tmp_path))
        assert r["success"] is True
        assert r["created"] is True
        content = (tmp_path / ".gitignore").read_text()
        assert "*.pt" in content
        assert "mlruns/" in content
        assert "__pycache__/" in content

    def test_existing_file_not_overwritten_by_default(self, tmp_path):
        (tmp_path / ".gitignore").write_text("custom\n")
        r = create_gitignore(str(tmp_path))
        assert r["created"] is False
        assert r["reason"] == "already_exists"
        assert (tmp_path / ".gitignore").read_text() == "custom\n"

    def test_overwrite_true_replaces_existing_file(self, tmp_path):
        (tmp_path / ".gitignore").write_text("custom\n")
        r = create_gitignore(str(tmp_path), overwrite=True)
        assert r["created"] is True
        assert "*.pt" in (tmp_path / ".gitignore").read_text()

    def test_missing_directory_returns_error(self, tmp_path):
        r = create_gitignore(str(tmp_path / "ghost"))
        assert r["success"] is False


# ── detect_uncommitted_changes ───────────────────────────────────

class TestDetectUncommittedChanges:
    def _status(self, tmp_path, staged=None, unstaged=None, untracked=None):
        return {
            "success": True,
            "repo_root": str(tmp_path),
            "branch": "main",
            "staged": staged or [],
            "unstaged": unstaged or [],
            "untracked": untracked or [],
        }

    def test_no_changes_returns_has_changes_false(self, tmp_path):
        with patch("mlops_mcp.git_ops.git_status", return_value=self._status(tmp_path)):
            r = detect_uncommitted_changes(str(tmp_path))
        assert r["success"] is True
        assert r["has_changes"] is False
        assert r["dirty_files"] == []

    def test_staged_files_returns_has_changes_true(self, tmp_path):
        with patch("mlops_mcp.git_ops.git_status", return_value=self._status(tmp_path, staged=["model.py"])):
            r = detect_uncommitted_changes(str(tmp_path))
        assert r["has_changes"] is True
        assert "model.py" in r["dirty_files"]

    def test_untracked_files_returns_has_changes_true(self, tmp_path):
        with patch("mlops_mcp.git_ops.git_status", return_value=self._status(tmp_path, untracked=["new.csv"])):
            r = detect_uncommitted_changes(str(tmp_path))
        assert r["has_changes"] is True

    def test_git_status_failure_propagates_error(self, tmp_path):
        with patch("mlops_mcp.git_ops.git_status", return_value={"success": False, "error": "not a repo"}):
            r = detect_uncommitted_changes(str(tmp_path))
        assert r["success"] is False
