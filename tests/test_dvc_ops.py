from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from mlops_mcp.dvc_ops import (
    create_dvc_pipeline,
    dvc_add,
    dvc_check_available,
    dvc_init,
    dvc_pull,
    dvc_push,
    dvc_repro,
    dvc_status,
)


def _completed(returncode=0, stdout="", stderr=""):
    m = MagicMock()
    m.returncode = returncode
    m.stdout = stdout
    m.stderr = stderr
    return m


def _ok():
    return {"success": True, "binary": "/usr/bin/dvc"}


# ── dvc_check_available ──────────────────────────────────────────

class TestDvcCheckAvailable:
    def test_dvc_found_returns_success_with_version(self):
        with patch("mlops_mcp.dvc_ops.shutil.which", return_value="/usr/bin/dvc"), \
             patch("mlops_mcp.dvc_ops.subprocess.run", return_value=_completed(0, stdout="3.0.0")):
            r = dvc_check_available()
        assert r["success"] is True
        assert r["version"] == "3.0.0"
        assert r["binary"] == "/usr/bin/dvc"

    def test_dvc_not_on_path_returns_error(self):
        with patch("mlops_mcp.dvc_ops.shutil.which", return_value=None):
            r = dvc_check_available()
        assert r["success"] is False

    def test_version_check_fails_returns_error(self):
        with patch("mlops_mcp.dvc_ops.shutil.which", return_value="/usr/bin/dvc"), \
             patch("mlops_mcp.dvc_ops.subprocess.run", return_value=_completed(1, stderr="bad")):
            r = dvc_check_available()
        assert r["success"] is False
        assert "bad" in r["error"]


# ── dvc_init ─────────────────────────────────────────────────────

class TestDvcInit:
    def test_success_returns_repo_path_and_stdout(self, tmp_path):
        with patch("mlops_mcp.dvc_ops.dvc_check_available", return_value=_ok()), \
             patch("mlops_mcp.dvc_ops.subprocess.run", return_value=_completed(0, stdout="Initialized")):
            r = dvc_init(str(tmp_path))
        assert r["success"] is True
        assert "repo_path" in r
        assert r["stdout"] == "Initialized"

    def test_dvc_not_available_returns_error(self, tmp_path):
        with patch("mlops_mcp.dvc_ops.dvc_check_available", return_value={"success": False, "error": "no dvc"}):
            r = dvc_init(str(tmp_path))
        assert r["success"] is False

    def test_subprocess_failure_returns_error(self, tmp_path):
        with patch("mlops_mcp.dvc_ops.dvc_check_available", return_value=_ok()), \
             patch("mlops_mcp.dvc_ops.subprocess.run", return_value=_completed(1, stderr="init failed")):
            r = dvc_init(str(tmp_path))
        assert r["success"] is False


# ── dvc_add ──────────────────────────────────────────────────────

class TestDvcAdd:
    def test_success_returns_resolved_tracked_path(self, tmp_path):
        data = tmp_path / "data.csv"
        data.write_text("x\n1\n")
        with patch("mlops_mcp.dvc_ops.dvc_check_available", return_value=_ok()), \
             patch("mlops_mcp.dvc_ops.subprocess.run", return_value=_completed(0)):
            r = dvc_add(str(data), str(tmp_path))
        assert r["success"] is True
        # tracked_path must be an absolute resolved path
        assert Path(r["tracked_path"]).is_absolute()
        assert r["tracked_path"] == str(data.resolve())

    def test_dvc_not_available_returns_error(self, tmp_path):
        with patch("mlops_mcp.dvc_ops.dvc_check_available", return_value={"success": False, "error": "no dvc"}):
            r = dvc_add("data.csv", str(tmp_path))
        assert r["success"] is False


# ── dvc_push ─────────────────────────────────────────────────────

class TestDvcPush:
    def test_success_returns_repo_path(self, tmp_path):
        with patch("mlops_mcp.dvc_ops.dvc_check_available", return_value=_ok()), \
             patch("mlops_mcp.dvc_ops.subprocess.run", return_value=_completed(0)):
            r = dvc_push(str(tmp_path))
        assert r["success"] is True
        assert "repo_path" in r

    def test_subprocess_failure_returns_error(self, tmp_path):
        with patch("mlops_mcp.dvc_ops.dvc_check_available", return_value=_ok()), \
             patch("mlops_mcp.dvc_ops.subprocess.run", return_value=_completed(1)):
            r = dvc_push(str(tmp_path))
        assert r["success"] is False


# ── dvc_pull ─────────────────────────────────────────────────────

class TestDvcPull:
    def test_success_returns_repo_path(self, tmp_path):
        with patch("mlops_mcp.dvc_ops.dvc_check_available", return_value=_ok()), \
             patch("mlops_mcp.dvc_ops.subprocess.run", return_value=_completed(0)):
            r = dvc_pull(str(tmp_path))
        assert r["success"] is True
        assert "repo_path" in r

    def test_subprocess_failure_returns_error(self, tmp_path):
        with patch("mlops_mcp.dvc_ops.dvc_check_available", return_value=_ok()), \
             patch("mlops_mcp.dvc_ops.subprocess.run", return_value=_completed(1)):
            r = dvc_pull(str(tmp_path))
        assert r["success"] is False


# ── dvc_status ───────────────────────────────────────────────────

class TestDvcStatus:
    def test_output_returns_changed_list_and_count(self, tmp_path):
        stdout = "data.csv.dvc: changed\nmodel.pkl.dvc: changed"
        with patch("mlops_mcp.dvc_ops.dvc_check_available", return_value=_ok()), \
             patch("mlops_mcp.dvc_ops.subprocess.run", return_value=_completed(0, stdout=stdout)):
            r = dvc_status(str(tmp_path))
        assert r["success"] is True
        assert r["count"] == 2
        assert len(r["changed"]) == 2

    def test_empty_output_returns_empty_changed(self, tmp_path):
        with patch("mlops_mcp.dvc_ops.dvc_check_available", return_value=_ok()), \
             patch("mlops_mcp.dvc_ops.subprocess.run", return_value=_completed(0, stdout="")):
            r = dvc_status(str(tmp_path))
        assert r["changed"] == []
        assert r["count"] == 0

    def test_dvc_not_available_returns_error(self, tmp_path):
        with patch("mlops_mcp.dvc_ops.dvc_check_available", return_value={"success": False, "error": "no dvc"}):
            r = dvc_status(str(tmp_path))
        assert r["success"] is False


# ── dvc_repro ─────────────────────────────────────────────────────

class TestDvcRepro:
    def test_success_returns_repo_path_and_stdout(self, tmp_path):
        with patch("mlops_mcp.dvc_ops.dvc_check_available", return_value=_ok()), \
             patch("mlops_mcp.dvc_ops.subprocess.run", return_value=_completed(0, stdout="Reproduced")):
            r = dvc_repro(str(tmp_path))
        assert r["success"] is True
        assert r["stdout"] == "Reproduced"
        assert "repo_path" in r

    def test_subprocess_failure_returns_error(self, tmp_path):
        with patch("mlops_mcp.dvc_ops.dvc_check_available", return_value=_ok()), \
             patch("mlops_mcp.dvc_ops.subprocess.run", return_value=_completed(1, stderr="stage failed")):
            r = dvc_repro(str(tmp_path))
        assert r["success"] is False


# ── create_dvc_pipeline ───────────────────────────────────────────

class TestCreateDvcPipeline:
    def _stages(self):
        return {
            "preprocess": {
                "cmd": "python preprocess.py",
                "deps": ["data/raw"],
                "outs": ["data/processed"],
            }
        }

    def test_creates_dvc_yaml_with_correct_structure(self, tmp_path):
        r = create_dvc_pipeline(str(tmp_path), self._stages())
        assert r["success"] is True
        out = tmp_path / "dvc.yaml"
        assert out.exists()
        data = yaml.safe_load(out.read_text())
        assert "stages" in data
        assert "preprocess" in data["stages"]

    def test_empty_stages_returns_error(self, tmp_path):
        r = create_dvc_pipeline(str(tmp_path), {})
        assert r["success"] is False

    def test_missing_repo_path_returns_error(self, tmp_path):
        r = create_dvc_pipeline(str(tmp_path / "ghost"), self._stages())
        assert r["success"] is False

    def test_custom_output_file_name_respected(self, tmp_path):
        r = create_dvc_pipeline(str(tmp_path), self._stages(), output_file="pipeline.yaml")
        assert r["success"] is True
        assert (tmp_path / "pipeline.yaml").exists()
        assert r["pipeline_path"].endswith("pipeline.yaml")
