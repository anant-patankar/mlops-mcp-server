import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import mlops_mcp.mlflow_ops as mlflow_ops_module
from mlops_mcp.mlflow_ops import (
    download_mlflow_artifact,
    get_mlflow_model_versions,
    get_mlflow_runs,
    list_mlflow_experiments,
    log_artifact_to_mlflow,
    mlflow_check_available,
    register_model_in_mlflow,
    set_mlflow_tracking_uri,
)

_NOT_INSTALLED = (None, {"success": False, "error": "mlflow not installed."})


@pytest.fixture
def mock_mlflow():
    m = MagicMock()
    return m


# ── mlflow_check_available ───────────────────────────────────────

class TestMlflowCheckAvailable:
    def test_available_returns_tracking_uri(self, mock_mlflow, tmp_path):
        mock_mlflow.get_tracking_uri.return_value = "http://localhost:5000"
        with patch.object(mlflow_ops_module, "_mlflow_module", return_value=(mock_mlflow, None)):
            r = mlflow_check_available(str(tmp_path))
        assert r["success"] is True
        assert r["available"] is True
        assert r["tracking_uri"] == "http://localhost:5000"

    def test_configured_uri_from_persisted_config(self, mock_mlflow, tmp_path):
        mock_mlflow.get_tracking_uri.return_value = "http://localhost:5000"
        config_dir = tmp_path / ".mlops"
        config_dir.mkdir()
        (config_dir / "config.json").write_text(
            json.dumps({"mlflow_tracking_uri": "http://remote:5000"})
        )
        with patch.object(mlflow_ops_module, "_mlflow_module", return_value=(mock_mlflow, None)):
            r = mlflow_check_available(str(tmp_path))
        assert r["configured_tracking_uri"] == "http://remote:5000"

    def test_not_installed_returns_error(self):
        with patch.object(mlflow_ops_module, "_mlflow_module", return_value=_NOT_INSTALLED):
            r = mlflow_check_available()
        assert r["success"] is False
        assert "mlflow not installed" in r["error"]


# ── set_mlflow_tracking_uri ──────────────────────────────────────

class TestSetMlflowTrackingUri:
    def test_sets_uri_and_persists_config(self, mock_mlflow, tmp_path):
        with patch.object(mlflow_ops_module, "_mlflow_module", return_value=(mock_mlflow, None)):
            r = set_mlflow_tracking_uri("http://new:5000", str(tmp_path))
        assert r["success"] is True
        assert r["tracking_uri"] == "http://new:5000"
        mock_mlflow.set_tracking_uri.assert_called_once_with("http://new:5000")
        config = json.loads((tmp_path / ".mlops" / "config.json").read_text())
        assert config["mlflow_tracking_uri"] == "http://new:5000"

    def test_empty_string_returns_error(self, mock_mlflow):
        with patch.object(mlflow_ops_module, "_mlflow_module", return_value=(mock_mlflow, None)):
            r = set_mlflow_tracking_uri("   ")
        assert r["success"] is False
        assert "empty" in r["error"]

    def test_not_installed_returns_error(self):
        with patch.object(mlflow_ops_module, "_mlflow_module", return_value=_NOT_INSTALLED):
            r = set_mlflow_tracking_uri("http://x:5000")
        assert r["success"] is False


# ── list_mlflow_experiments ──────────────────────────────────────

class TestListMlflowExperiments:
    def test_returns_experiment_list(self, mock_mlflow):
        exp = MagicMock()
        exp.experiment_id = "1"
        exp.name = "my-exp"
        exp.artifact_location = "s3://bucket/1"
        mock_mlflow.search_experiments.return_value = [exp]
        with patch.object(mlflow_ops_module, "_mlflow_module", return_value=(mock_mlflow, None)):
            r = list_mlflow_experiments()
        assert r["success"] is True
        assert r["count"] == 1
        assert r["experiments"][0] == {
            "experiment_id": "1",
            "name": "my-exp",
            "artifact_location": "s3://bucket/1",
        }

    def test_empty_returns_empty_list(self, mock_mlflow):
        mock_mlflow.search_experiments.return_value = []
        with patch.object(mlflow_ops_module, "_mlflow_module", return_value=(mock_mlflow, None)):
            r = list_mlflow_experiments()
        assert r["success"] is True
        assert r["count"] == 0
        assert r["experiments"] == []

    def test_not_installed_returns_error(self):
        with patch.object(mlflow_ops_module, "_mlflow_module", return_value=_NOT_INSTALLED):
            r = list_mlflow_experiments()
        assert r["success"] is False


# ── get_mlflow_runs ──────────────────────────────────────────────

class TestGetMlflowRuns:
    def test_returns_runs_as_list(self, mock_mlflow):
        fake_df = MagicMock()
        fake_df.to_dict.return_value = [{"run_id": "abc", "status": "FINISHED"}]
        mock_mlflow.search_runs.return_value = fake_df
        with patch.object(mlflow_ops_module, "_mlflow_module", return_value=(mock_mlflow, None)):
            r = get_mlflow_runs("1")
        assert r["success"] is True
        assert r["count"] == 1
        assert r["runs"][0]["run_id"] == "abc"
        fake_df.to_dict.assert_called_once_with(orient="records")

    def test_passes_max_results(self, mock_mlflow):
        fake_df = MagicMock()
        fake_df.to_dict.return_value = []
        mock_mlflow.search_runs.return_value = fake_df
        with patch.object(mlflow_ops_module, "_mlflow_module", return_value=(mock_mlflow, None)):
            get_mlflow_runs("1", max_results=10)
        mock_mlflow.search_runs.assert_called_once_with(
            experiment_ids=["1"], max_results=10
        )

    def test_not_installed_returns_error(self):
        with patch.object(mlflow_ops_module, "_mlflow_module", return_value=_NOT_INSTALLED):
            r = get_mlflow_runs("1")
        assert r["success"] is False


# ── log_artifact_to_mlflow ───────────────────────────────────────

class TestLogArtifactToMlflow:
    def test_logs_with_run_id(self, mock_mlflow, tmp_path):
        artifact = tmp_path / "model.pkl"
        artifact.write_bytes(b"\x00")
        with patch.object(mlflow_ops_module, "_mlflow_module", return_value=(mock_mlflow, None)):
            r = log_artifact_to_mlflow(str(artifact), run_id="run-123")
        assert r["success"] is True
        assert r["run_id"] == "run-123"
        mock_mlflow.start_run.assert_called_once_with(run_id="run-123")

    def test_logs_without_run_id(self, mock_mlflow, tmp_path):
        artifact = tmp_path / "model.pkl"
        artifact.write_bytes(b"\x00")
        with patch.object(mlflow_ops_module, "_mlflow_module", return_value=(mock_mlflow, None)):
            r = log_artifact_to_mlflow(str(artifact))
        assert r["success"] is True
        assert r["run_id"] is None
        mock_mlflow.log_artifact.assert_called_once_with(str(artifact.resolve()))

    def test_missing_file_returns_error(self, mock_mlflow, tmp_path):
        with patch.object(mlflow_ops_module, "_mlflow_module", return_value=(mock_mlflow, None)):
            r = log_artifact_to_mlflow(str(tmp_path / "ghost.pkl"))
        assert r["success"] is False
        assert "not found" in r["error"]

    def test_not_installed_returns_error(self):
        with patch.object(mlflow_ops_module, "_mlflow_module", return_value=_NOT_INSTALLED):
            r = log_artifact_to_mlflow("/any/path.pkl")
        assert r["success"] is False


# ── download_mlflow_artifact ─────────────────────────────────────

class TestDownloadMlflowArtifact:
    def test_returns_downloaded_path(self, mock_mlflow, tmp_path):
        dest = tmp_path / "downloads"
        mock_mlflow.artifacts.download_artifacts.return_value = str(dest / "model.pkl")
        with patch.object(mlflow_ops_module, "_mlflow_module", return_value=(mock_mlflow, None)):
            r = download_mlflow_artifact("run-123", "model.pkl", str(dest))
        assert r["success"] is True
        assert r["run_id"] == "run-123"
        assert r["artifact_path"] == "model.pkl"
        assert "model.pkl" in r["downloaded_path"]

    def test_creates_destination_dir(self, mock_mlflow, tmp_path):
        dest = tmp_path / "new" / "dir"
        mock_mlflow.artifacts.download_artifacts.return_value = str(dest / "f.pkl")
        with patch.object(mlflow_ops_module, "_mlflow_module", return_value=(mock_mlflow, None)):
            download_mlflow_artifact("run-123", "f.pkl", str(dest))
        assert dest.exists()

    def test_not_installed_returns_error(self):
        with patch.object(mlflow_ops_module, "_mlflow_module", return_value=_NOT_INSTALLED):
            r = download_mlflow_artifact("run-123", "f.pkl", "/tmp/dest")
        assert r["success"] is False


# ── register_model_in_mlflow ─────────────────────────────────────

class TestRegisterModelInMlflow:
    def test_returns_version(self, mock_mlflow, tmp_path):
        model = tmp_path / "model.pkl"
        model.write_bytes(b"\x00")
        mock_result = MagicMock()
        mock_result.version = "3"
        mock_mlflow.register_model.return_value = mock_result
        with patch.object(mlflow_ops_module, "_mlflow_module", return_value=(mock_mlflow, None)):
            r = register_model_in_mlflow(str(model), "my-model")
        assert r["success"] is True
        assert r["version"] == "3"
        assert r["model_name"] == "my-model"

    def test_missing_model_file_returns_error(self, mock_mlflow, tmp_path):
        with patch.object(mlflow_ops_module, "_mlflow_module", return_value=(mock_mlflow, None)):
            r = register_model_in_mlflow(str(tmp_path / "ghost.pkl"), "m")
        assert r["success"] is False
        assert "not found" in r["error"]

    def test_not_installed_returns_error(self):
        with patch.object(mlflow_ops_module, "_mlflow_module", return_value=_NOT_INSTALLED):
            r = register_model_in_mlflow("/any/model.pkl", "m")
        assert r["success"] is False


# ── get_mlflow_model_versions ────────────────────────────────────

class TestGetMlflowModelVersions:
    def test_returns_version_list(self, mock_mlflow):
        mv = MagicMock()
        mv.name = "my-model"
        mv.version = "2"
        mv.current_stage = "Staging"
        mv.run_id = "run-abc"
        mv.source = "s3://bucket/model"
        client = MagicMock()
        client.search_model_versions.return_value = [mv]
        mock_mlflow.MlflowClient.return_value = client
        with patch.object(mlflow_ops_module, "_mlflow_module", return_value=(mock_mlflow, None)):
            r = get_mlflow_model_versions("my-model")
        assert r["success"] is True
        assert r["count"] == 1
        v = r["versions"][0]
        assert v["name"] == "my-model"
        assert v["version"] == "2"
        assert v["current_stage"] == "Staging"
        assert v["run_id"] == "run-abc"

    def test_empty_model_returns_empty_list(self, mock_mlflow):
        client = MagicMock()
        client.search_model_versions.return_value = []
        mock_mlflow.MlflowClient.return_value = client
        with patch.object(mlflow_ops_module, "_mlflow_module", return_value=(mock_mlflow, None)):
            r = get_mlflow_model_versions("unknown")
        assert r["success"] is True
        assert r["count"] == 0
        assert r["versions"] == []

    def test_not_installed_returns_error(self):
        with patch.object(mlflow_ops_module, "_mlflow_module", return_value=_NOT_INSTALLED):
            r = get_mlflow_model_versions("m")
        assert r["success"] is False
