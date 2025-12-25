from __future__ import annotations

from pathlib import Path
from typing import Any

from ._types import err as _fail


def _git_module():
    try:
        import git
    except ImportError:
        return None, _fail("gitpython not installed.")
    return git, None


def _get_repo(repo_path: str):
    git, error = _git_module()
    if error:
        return None, error, None
    try:
        repo = git.Repo(Path(repo_path).expanduser(),
                        search_parent_directories=True)
        return repo, None, git
    except git.InvalidGitRepositoryError:
        return None, _fail(f"not a git repository: {repo_path}"), git
    except OSError as exc:
        return None, _fail(str(exc)), git


def git_init(repo_path: str = ".", create_ignore: bool = True) -> dict[str, Any]:
    git, error = _git_module()
    if error:
        return error

    target = Path(repo_path).expanduser().resolve()
    try:
        target.mkdir(parents=True, exist_ok=True)
        repo = git.Repo.init(target)
        ignore_result = None
        if create_ignore:
            ignore_result = create_gitignore(str(target), overwrite=False)
        return {
            "success": True,
            "repo_root": str(repo.working_tree_dir),
            "git_dir": str(Path(repo.git_dir)),
            "gitignore_created": bool(ignore_result and ignore_result.get("created")),
        }
    except OSError as exc:
        return _fail(str(exc))


def git_status(repo_path: str = ".") -> dict[str, Any]:
    repo, error, _ = _get_repo(repo_path)
    if error:
        return error

    try:
        staged = [item.a_path for item in repo.index.diff("HEAD")]
        unstaged = [item.a_path for item in repo.index.diff(None)]
        untracked = sorted(repo.untracked_files)

        branch = None
        if not repo.head.is_detached:
            branch = repo.active_branch.name

        return {
            "success": True,
            "repo_root": str(repo.working_tree_dir),
            "branch": branch,
            "staged": sorted(set(staged)),
            "unstaged": sorted(set(unstaged)),
            "untracked": untracked,
        }
    except OSError as exc:
        return _fail(str(exc))


def git_add(repo_path: str = ".", files: list[str] | None = None) -> dict[str, Any]:
    repo, error, _ = _get_repo(repo_path)
    if error:
        return error

    targets = files or ["."]
    try:
        repo.index.add(targets)
        return {
            "success": True,
            "repo_root": str(repo.working_tree_dir),
            "added": targets,
        }
    except OSError as exc:
        return _fail(str(exc))


def git_commit(repo_path: str = ".", message: str = "update") -> dict[str, Any]:
    repo, error, _ = _get_repo(repo_path)
    if error:
        return error
    if not message.strip():
        return _fail("commit message must not be empty")

    try:
        commit = repo.index.commit(message)
        return {
            "success": True,
            "repo_root": str(repo.working_tree_dir),
            "commit": commit.hexsha,
            "message": commit.message.strip(),
        }
    except OSError as exc:
        return _fail(str(exc))


def git_log(repo_path: str = ".", limit: int = 10) -> dict[str, Any]:
    repo, error, _ = _get_repo(repo_path)
    if error:
        return error
    if limit <= 0:
        return _fail("limit must be > 0")

    try:
        commits = [
            {
                "hash": c.hexsha,
                "short_hash": c.hexsha[:7],
                "message": c.message.strip(),
                "timestamp": c.committed_datetime.isoformat(),
                "author": str(c.author),
            }
            for c in repo.iter_commits(max_count=limit)
        ]
        return {
            "success": True,
            "repo_root": str(repo.working_tree_dir),
            "count": len(commits),
            "commits": commits,
        }
    except OSError as exc:
        return _fail(str(exc))


def create_gitignore(repo_path: str = ".", overwrite: bool = False) -> dict[str, Any]:
    root = Path(repo_path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return _fail(f"directory not found: {root}")

    target = root / ".gitignore"
    if target.exists() and not overwrite:
        return {
            "success": True,
            "path": str(target),
            "created": False,
            "reason": "already_exists",
        }

    content = "\n".join(
        (
            "# Python",
            "__pycache__/",
            "*.pyc",
            ".venv/",
            ".env",
            "",
            "# ML artifacts",
            "*.pt",
            "*.ckpt",
            "*.onnx",
            "mlruns/",
            "wandb/",
            "data/raw/",
            "",
            "# OS/editor",
            ".DS_Store",
        )
    ) + "\n"

    try:
        target.write_text(content, encoding="utf-8")
        return {"success": True, "path": str(target), "created": True}
    except OSError as exc:
        return _fail(str(exc))


def detect_uncommitted_changes(repo_path: str = ".") -> dict[str, Any]:
    status = git_status(repo_path)
    if not status.get("success"):
        return status

    dirty_files = sorted(
        set(status.get("staged", []))
        | set(status.get("unstaged", []))
        | set(status.get("untracked", []))
    )
    return {
        "success": True,
        "repo_root": status["repo_root"],
        "has_changes": len(dirty_files) > 0,
        "dirty_files": dirty_files,
    }
