from __future__ import annotations

import ast
import importlib.metadata as importlib_metadata
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import yaml

from ._types import err as _fail

_STDLIB_MODULES = {
    "abc", "argparse", "ast", "asyncio",
    "collections", "contextlib", "csv",
    "dataclasses", "datetime", "functools",
    "glob", "hashlib", "itertools", "json",
    "logging", "math", "os", "pathlib",
    "random", "re", "shutil", "string",
    "subprocess", "sys", "tempfile", "time",
    "typing", "statistics", "unittest", "uuid",
}


def _parse_requirements(path: Path) -> dict[str, str | None]:
    parsed: dict[str, str | None] = {}
    if not path.exists() or not path.is_file():
        return parsed

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" in line:
            name, version = line.split("==", 1)
            parsed[name.strip().lower()] = version.strip()
        else:
            parsed[line.lower()] = None
    return parsed


def scan_imports(project_path: str = ".") -> dict[str, Any]:
    root = Path(project_path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return _fail(f"project folder not found: {root}")

    imports = set()
    syntax_errors = []

    for fp in root.rglob("*.py"):
        try:
            tree = ast.parse(fp.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            syntax_errors.append(
                {"file": str(fp.relative_to(root)), "error": str(exc)})
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split(".")[0])

    third_party = sorted(imports - _STDLIB_MODULES)
    return {
        "success": True,
        "project_path": str(root),
        "imports": third_party,
        "count": len(third_party),
        "syntax_errors": syntax_errors,
    }


def generate_requirements(
    project_path: str = ".",
    output_path: str | None = None,
    pin_versions: bool = False,
) -> dict[str, Any]:
    scanned = scan_imports(project_path)
    if not scanned.get("success"):
        return scanned

    root = Path(project_path).expanduser().resolve()
    destination = (
        Path(output_path).expanduser().resolve()
        if output_path
        else root / "requirements.txt"
    )

    distributions = importlib_metadata.packages_distributions()
    lines: list[str] = []

    for module in scanned["imports"]:
        dist_names = distributions.get(module, [module])
        dist_name = dist_names[0]
        if pin_versions:
            try:
                version = importlib_metadata.version(dist_name)
                lines.append(f"{dist_name}=={version}")
            except importlib_metadata.PackageNotFoundError:
                lines.append(dist_name)
        else:
            lines.append(dist_name)

    unique_lines = sorted(set(lines), key=str.lower)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("\n".join(unique_lines) +
                               ("\n" if unique_lines else ""), encoding="utf-8")
        return {
            "success": True,
            "project_path": str(root),
            "output_path": str(destination),
            "count": len(unique_lines),
            "requirements": unique_lines,
        }
    except OSError as exc:
        return _fail(str(exc))


def compare_requirements(base_path: str, target_path: str) -> dict[str, Any]:
    base = Path(base_path).expanduser().resolve()
    target = Path(target_path).expanduser().resolve()
    if not base.exists() or not base.is_file():
        return _fail(f"could not find requirements file: {base}")
    if not target.exists() or not target.is_file():
        return _fail(f"could not find requirements file: {target}")

    base_reqs = _parse_requirements(base)
    target_reqs = _parse_requirements(target)

    added = sorted(set(target_reqs) - set(base_reqs))
    removed = sorted(set(base_reqs) - set(target_reqs))
    version_changed = []
    for pkg in sorted(set(base_reqs) & set(target_reqs)):
        if base_reqs[pkg] != target_reqs[pkg]:
            version_changed.append(
                {
                    "package": pkg,
                    "from": base_reqs[pkg],
                    "to": target_reqs[pkg],
                }
            )

    return {
        "success": True,
        "base_path": str(base),
        "target_path": str(target),
        "added": added,
        "removed": removed,
        "version_changed": version_changed,
    }


def check_dependency_conflicts(project_path: str = ".") -> dict[str, Any]:
    root = Path(project_path).expanduser().resolve()
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        conflicts = []
        if completed.returncode != 0:
            for line in stdout.splitlines():
                line = line.strip()
                if line:
                    conflicts.append(line)

        return {
            "success": True,
            "project_path": str(root),
            "has_conflicts": completed.returncode != 0,
            "conflicts": conflicts,
            "stdout": stdout,
            "stderr": stderr,
            "returncode": completed.returncode,
        }
    except OSError as exc:
        return _fail(str(exc))


def create_conda_env_file(
    requirements_path: str,
    output_path: str,
    python_version: str = "3.13",
    env_name: str = "mlops-env",
) -> dict[str, Any]:
    requirements = Path(requirements_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not requirements.exists() or not requirements.is_file():
        return _fail(f"requirements file is missing: {requirements}")

    req_lines = [
        line.strip()
        for line in requirements.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    payload = {
        "name": env_name,
        "channels": ["conda-forge", "defaults"],
        "dependencies": [f"python={python_version}", "pip", {"pip": req_lines}],
    }

    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(yaml.safe_dump(
            payload, sort_keys=False), encoding="utf-8")
        return {
            "success": True,
            "output_path": str(output),
            "package_count": len(req_lines),
            "env_name": env_name,
        }
    except OSError as exc:
        return _fail(str(exc))


def create_env_template(output_path: str, variables: list[str]) -> dict[str, Any]:
    if not variables:
        return _fail("variables must not be empty")

    target = Path(output_path).expanduser().resolve()
    normalized = []
    for var in variables:
        candidate = var.strip()
        if not candidate:
            continue
        if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", candidate):
            return _fail(f"invalid environment variable name: {candidate}")
        normalized.append(candidate)

    if not normalized:
        return _fail("variables must contain at least one valid name")

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        content = "\n".join(f"{name}=" for name in normalized) + "\n"
        target.write_text(content, encoding="utf-8")
        return {
            "success": True,
            "output_path": str(target),
            "variables": normalized,
            "count": len(normalized),
        }
    except OSError as exc:
        return _fail(str(exc))


def check_env_vars(template_path: str, env_file_path: str | None = None) -> dict[str, Any]:
    template = Path(template_path).expanduser().resolve()
    if not template.exists() or not template.is_file():
        return _fail(f"template file is missing: {template}")

    required: list[str] = []
    for raw in template.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key = line.split("=", 1)[0].strip()
        if key:
            required.append(key)

    provided: dict[str, str] = dict(os.environ)
    env_file = Path(env_file_path).expanduser(
    ).resolve() if env_file_path else None
    if env_file and env_file.exists() and env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            provided.setdefault(key.strip(), value.strip())

    missing = [name for name in required if not provided.get(name)]
    return {
        "success": True,
        "template_path": str(template),
        "required_count": len(required),
        "missing": sorted(missing),
        "is_valid": len(missing) == 0,
    }


def get_python_version(project_path: str = ".") -> dict[str, Any]:
    root = Path(project_path).expanduser().resolve()
    pyproject = root / "pyproject.toml"

    requires_python = None
    if pyproject.exists() and pyproject.is_file():
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            requires_python = (
                data.get("project", {}).get("requires-python")
                if isinstance(data, dict)
                else None
            )
        except (OSError, tomllib.TOMLDecodeError):
            requires_python = None

    current_version = sys.version.split()[0]
    compatible = None
    if requires_python and requires_python.startswith(">="):  # TODO: handle ~= and compound specifiers (PEP 440)
        minimum = requires_python[2:].strip()
        try:
            compatible = tuple(map(int, current_version.split("."))) >= tuple(
                map(int, minimum.split("."))
            )
        except ValueError:
            compatible = None

    return {
        "success": True,
        "python_version": current_version,
        "requires_python": requires_python,
        "compatible": compatible,
        "pyproject_path": str(pyproject),
    }
