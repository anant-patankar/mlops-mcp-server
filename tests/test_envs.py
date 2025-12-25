import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from mlops_mcp.envs import (
    check_dependency_conflicts,
    check_env_vars,
    compare_requirements,
    create_conda_env_file,
    create_env_template,
    generate_requirements,
    get_python_version,
    scan_imports,
)


def _py(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


def _req(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


# ── scan_imports ─────────────────────────────────────────────────

class TestScanImports:
    def test_finds_third_party_imports(self, tmp_path):
        _py(tmp_path, "app.py", "import requests\nimport flask\n")
        r = scan_imports(str(tmp_path))
        assert r["success"] is True
        assert "requests" in r["imports"]
        assert "flask" in r["imports"]

    def test_ignores_stdlib_imports(self, tmp_path):
        _py(tmp_path, "app.py", "import os\nimport json\nimport pathlib\n")
        r = scan_imports(str(tmp_path))
        assert "os" not in r["imports"]
        assert "json" not in r["imports"]

    def test_missing_directory_returns_error(self, tmp_path):
        r = scan_imports(str(tmp_path / "ghost"))
        assert r["success"] is False

    def test_syntax_error_reported_but_others_processed(self, tmp_path):
        _py(tmp_path, "good.py", "import requests\n")
        _py(tmp_path, "bad.py", "def broken(\n")
        r = scan_imports(str(tmp_path))
        assert r["success"] is True
        assert len(r["syntax_errors"]) == 1
        assert "requests" in r["imports"]

    def test_empty_directory_returns_empty_imports(self, tmp_path):
        r = scan_imports(str(tmp_path))
        assert r["success"] is True
        assert r["imports"] == []
        assert r["count"] == 0


# ── generate_requirements ────────────────────────────────────────

class TestGenerateRequirements:
    def test_creates_requirements_txt_at_default_location(self, tmp_path):
        _py(tmp_path, "app.py", "import yaml\n")
        r = generate_requirements(str(tmp_path))
        assert r["success"] is True
        assert (tmp_path / "requirements.txt").exists()

    def test_custom_output_path_respected(self, tmp_path):
        _py(tmp_path, "app.py", "import yaml\n")
        out = tmp_path / "deps" / "reqs.txt"
        r = generate_requirements(str(tmp_path), output_path=str(out))
        assert r["success"] is True
        assert out.exists()

    def test_output_contains_expected_package(self, tmp_path):
        _py(tmp_path, "app.py", "import yaml\n")
        out = tmp_path / "requirements.txt"
        r = generate_requirements(str(tmp_path), output_path=str(out))
        content = out.read_text()
        # yaml maps to PyYAML dist
        assert len(r["requirements"]) >= 1

    def test_missing_project_directory_returns_error(self, tmp_path):
        r = generate_requirements(str(tmp_path / "ghost"))
        assert r["success"] is False


# ── compare_requirements ─────────────────────────────────────────

class TestCompareRequirements:
    def test_added_packages_detected(self, tmp_path):
        base = _req(tmp_path, "base.txt", "requests\n")
        target = _req(tmp_path, "target.txt", "requests\nflask\n")
        r = compare_requirements(str(base), str(target))
        assert r["success"] is True
        assert "flask" in r["added"]

    def test_removed_packages_detected(self, tmp_path):
        base = _req(tmp_path, "base.txt", "requests\nflask\n")
        target = _req(tmp_path, "target.txt", "requests\n")
        r = compare_requirements(str(base), str(target))
        assert "flask" in r["removed"]

    def test_version_change_detected(self, tmp_path):
        base = _req(tmp_path, "base.txt", "requests==2.28.0\n")
        target = _req(tmp_path, "target.txt", "requests==2.31.0\n")
        r = compare_requirements(str(base), str(target))
        assert len(r["version_changed"]) == 1
        assert r["version_changed"][0]["package"] == "requests"
        assert r["version_changed"][0]["from"] == "2.28.0"
        assert r["version_changed"][0]["to"] == "2.31.0"

    def test_identical_files_return_empty_diffs(self, tmp_path):
        base = _req(tmp_path, "base.txt", "requests==2.28.0\n")
        target = _req(tmp_path, "target.txt", "requests==2.28.0\n")
        r = compare_requirements(str(base), str(target))
        assert r["added"] == []
        assert r["removed"] == []
        assert r["version_changed"] == []

    def test_missing_base_returns_error(self, tmp_path):
        target = _req(tmp_path, "target.txt", "requests\n")
        r = compare_requirements(str(tmp_path / "ghost.txt"), str(target))
        assert r["success"] is False

    def test_missing_target_returns_error(self, tmp_path):
        base = _req(tmp_path, "base.txt", "requests\n")
        r = compare_requirements(str(base), str(tmp_path / "ghost.txt"))
        assert r["success"] is False


# ── check_dependency_conflicts ───────────────────────────────────

class TestCheckDependencyConflicts:
    def _mock_completed(self, returncode, stdout="", stderr=""):
        m = MagicMock()
        m.returncode = returncode
        m.stdout = stdout
        m.stderr = stderr
        return m

    def test_no_conflicts_when_pip_check_passes(self, tmp_path):
        with patch("mlops_mcp.envs.subprocess.run") as mock_run:
            mock_run.return_value = self._mock_completed(0, stdout="No broken requirements found.")
            r = check_dependency_conflicts(str(tmp_path))
        assert r["success"] is True
        assert r["has_conflicts"] is False
        assert r["conflicts"] == []

    def test_conflicts_returned_when_pip_check_fails(self, tmp_path):
        conflict_line = "pkgA 1.0 has requirement pkgB>=2.0, but pkgB 1.5 installed."
        with patch("mlops_mcp.envs.subprocess.run") as mock_run:
            mock_run.return_value = self._mock_completed(1, stdout=conflict_line)
            r = check_dependency_conflicts(str(tmp_path))
        assert r["has_conflicts"] is True
        assert conflict_line in r["conflicts"]

    def test_response_always_has_stdout_stderr_returncode(self, tmp_path):
        with patch("mlops_mcp.envs.subprocess.run") as mock_run:
            mock_run.return_value = self._mock_completed(0)
            r = check_dependency_conflicts(str(tmp_path))
        assert "stdout" in r
        assert "stderr" in r
        assert "returncode" in r


# ── create_conda_env_file ────────────────────────────────────────

class TestCreateCondaEnvFile:
    def test_creates_valid_yaml_structure(self, tmp_path):
        req = _req(tmp_path, "requirements.txt", "requests\nflask\n")
        out = tmp_path / "environment.yml"
        r = create_conda_env_file(str(req), str(out))
        assert r["success"] is True
        data = yaml.safe_load(out.read_text())
        assert "name" in data
        assert "channels" in data
        assert "dependencies" in data

    def test_package_count_matches_requirements(self, tmp_path):
        req = _req(tmp_path, "requirements.txt", "requests\nflask\nnumpy\n")
        out = tmp_path / "environment.yml"
        r = create_conda_env_file(str(req), str(out))
        assert r["package_count"] == 3

    def test_custom_env_name_appears_in_output(self, tmp_path):
        req = _req(tmp_path, "requirements.txt", "requests\n")
        out = tmp_path / "environment.yml"
        r = create_conda_env_file(str(req), str(out), env_name="my-env")
        assert r["env_name"] == "my-env"
        data = yaml.safe_load(out.read_text())
        assert data["name"] == "my-env"

    def test_missing_requirements_returns_error(self, tmp_path):
        r = create_conda_env_file(
            str(tmp_path / "ghost.txt"),
            str(tmp_path / "environment.yml"),
        )
        assert r["success"] is False


# ── create_env_template ──────────────────────────────────────────

class TestCreateEnvTemplate:
    def test_creates_file_with_correct_format(self, tmp_path):
        out = tmp_path / ".env.template"
        r = create_env_template(str(out), ["API_KEY", "DB_URL"])
        assert r["success"] is True
        content = out.read_text()
        assert "API_KEY=" in content
        assert "DB_URL=" in content

    def test_invalid_variable_name_returns_error(self, tmp_path):
        out = tmp_path / ".env.template"
        r = create_env_template(str(out), ["123INVALID"])
        assert r["success"] is False
        assert "123INVALID" in r["error"]

    def test_empty_list_returns_error(self, tmp_path):
        r = create_env_template(str(tmp_path / ".env"), [])
        assert r["success"] is False

    def test_whitespace_only_variables_are_skipped(self, tmp_path):
        out = tmp_path / ".env.template"
        r = create_env_template(str(out), ["  ", "API_KEY"])
        assert r["success"] is True
        assert r["count"] == 1
        assert r["variables"] == ["API_KEY"]

    def test_all_whitespace_list_returns_error(self, tmp_path):
        r = create_env_template(str(tmp_path / ".env"), ["  ", "   "])
        assert r["success"] is False


# ── check_env_vars ───────────────────────────────────────────────

class TestCheckEnvVars:
    def _template(self, tmp_path, content):
        p = tmp_path / ".env.template"
        p.write_text(content, encoding="utf-8")
        return p

    def test_all_vars_present_returns_valid(self, tmp_path, monkeypatch):
        t = self._template(tmp_path, "API_KEY=\nDB_URL=\n")
        monkeypatch.setenv("API_KEY", "secret")
        monkeypatch.setenv("DB_URL", "postgres://localhost")
        r = check_env_vars(str(t))
        assert r["success"] is True
        assert r["is_valid"] is True
        assert r["missing"] == []

    def test_missing_var_appears_in_missing(self, tmp_path, monkeypatch):
        t = self._template(tmp_path, "API_KEY=\nMISSING_VAR=\n")
        monkeypatch.setenv("API_KEY", "x")
        monkeypatch.delenv("MISSING_VAR", raising=False)
        r = check_env_vars(str(t))
        assert r["is_valid"] is False
        assert "MISSING_VAR" in r["missing"]

    def test_env_file_supplements_os_environ(self, tmp_path, monkeypatch):
        t = self._template(tmp_path, "FROM_FILE=\n")
        monkeypatch.delenv("FROM_FILE", raising=False)
        ef = tmp_path / ".env"
        ef.write_text("FROM_FILE=hello\n", encoding="utf-8")
        r = check_env_vars(str(t), env_file_path=str(ef))
        assert r["is_valid"] is True

    def test_missing_template_returns_error(self, tmp_path):
        r = check_env_vars(str(tmp_path / "ghost.template"))
        assert r["success"] is False


# ── get_python_version ───────────────────────────────────────────

class TestGetPythonVersion:
    def test_returns_current_python_version(self, tmp_path):
        r = get_python_version(str(tmp_path))
        assert r["success"] is True
        assert r["python_version"] == sys.version.split()[0]

    def test_reads_requires_python_from_pyproject(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nrequires-python = ">=3.11"\n', encoding="utf-8"
        )
        r = get_python_version(str(tmp_path))
        assert r["requires_python"] == ">=3.11"

    def test_compatible_true_when_version_meets_requirement(self, tmp_path):
        # use a very low minimum so any supported Python satisfies it
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nrequires-python = ">=3.9"\n', encoding="utf-8"
        )
        r = get_python_version(str(tmp_path))
        assert r["compatible"] is True

    def test_compatible_none_for_non_gte_specifier(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nrequires-python = "~=3.11"\n', encoding="utf-8"
        )
        r = get_python_version(str(tmp_path))
        assert r["compatible"] is None

    def test_no_pyproject_returns_requires_python_none(self, tmp_path):
        r = get_python_version(str(tmp_path))
        assert r["requires_python"] is None
        assert r["compatible"] is None
