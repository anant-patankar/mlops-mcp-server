import json
from pathlib import Path

import pytest

from mlops_mcp.experiments import (
    compare_runs,
    create_run,
    delete_run,
    export_runs_csv,
    finish_run,
    get_best_run,
    get_run,
    init_experiment_tracker,
    list_runs,
    log_artifact,
    log_metrics,
    log_params,
)


# ── helpers ──────────────────────────────────────────────────────
# _tracker_root is not exported, so we replicate the path here to
# inspect files on disk without going through the public API.

def _tracker(tmp_path: Path) -> Path:
    return tmp_path / ".mlops" / "experiments"


def _run_dir(tmp_path: Path, run_id: str) -> Path:
    return _tracker(tmp_path) / run_id


# ── init_experiment_tracker ──────────────────────────────────────

class TestInitExperimentTracker:
    def test_creates_registry_on_first_call(self, tmp_path):
        r = init_experiment_tracker(str(tmp_path))
        assert r["success"] is True
        # registry.json must exist and contain an empty run list
        reg = json.loads((_tracker(tmp_path) / "registry.json").read_text())
        assert reg == {"runs": []}

    def test_idempotent_second_call(self, tmp_path):
        # calling twice must not wipe existing registry data
        init_experiment_tracker(str(tmp_path))
        run = create_run(str(tmp_path))
        init_experiment_tracker(str(tmp_path))  # second call
        r = list_runs(str(tmp_path))
        # the run created between the two inits must still be there
        assert r["run_count"] == 1

    def test_returns_tracker_path(self, tmp_path):
        r = init_experiment_tracker(str(tmp_path))
        assert "tracker_path" in r
        assert Path(r["tracker_path"]).exists()


# ── create_run ───────────────────────────────────────────────────

class TestCreateRun:
    def test_returns_run_id(self, tmp_path):
        r = create_run(str(tmp_path))
        assert r["success"] is True
        assert "run_id" in r
        assert len(r["run_id"]) == 36  # UUID4 length

    def test_creates_four_json_files(self, tmp_path):
        r = create_run(str(tmp_path))
        run_path = _run_dir(tmp_path, r["run_id"])
        for fname in ("params.json", "metrics.json", "artifacts.json", "status.json"):
            assert (run_path / fname).exists(), f"{fname} missing"

    def test_initial_status_is_running(self, tmp_path):
        r = create_run(str(tmp_path))
        status = json.loads(
            (_run_dir(tmp_path, r["run_id"]) / "status.json").read_text()
        )
        assert status["status"] == "running"
        assert "started_at" in status

    def test_run_appears_in_registry(self, tmp_path):
        r = create_run(str(tmp_path))
        reg = json.loads((_tracker(tmp_path) / "registry.json").read_text())
        run_ids = [entry["run_id"] for entry in reg["runs"]]
        assert r["run_id"] in run_ids


# ── log_params ───────────────────────────────────────────────────

class TestLogParams:
    def test_persists_to_disk(self, tmp_path):
        run_id = create_run(str(tmp_path))["run_id"]
        log_params(str(tmp_path), run_id, {"lr": 0.001, "batch": 32})
        on_disk = json.loads(
            (_run_dir(tmp_path, run_id) / "params.json").read_text()
        )
        assert on_disk["lr"] == 0.001
        assert on_disk["batch"] == 32

    def test_merges_on_second_call(self, tmp_path):
        # second call must ADD keys, not overwrite the file
        run_id = create_run(str(tmp_path))["run_id"]
        log_params(str(tmp_path), run_id, {"lr": 0.001})
        log_params(str(tmp_path), run_id, {"optimizer": "adam"})
        r = get_run(str(tmp_path), run_id)
        assert r["params"]["lr"] == 0.001
        assert r["params"]["optimizer"] == "adam"

    def test_missing_run_returns_error(self, tmp_path):
        r = log_params(str(tmp_path), "nonexistent-id", {"lr": 0.001})
        assert r["success"] is False


# ── log_metrics ──────────────────────────────────────────────────

class TestLogMetrics:
    def test_appends_entry_with_step(self, tmp_path):
        run_id = create_run(str(tmp_path))["run_id"]
        log_metrics(str(tmp_path), run_id, {"val_loss": 0.42}, step=1)
        log_metrics(str(tmp_path), run_id, {"val_loss": 0.35}, step=2)
        r = get_run(str(tmp_path), run_id)
        assert len(r["metrics"]) == 2
        assert r["metrics"][0]["step"] == 1
        assert r["metrics"][1]["val_loss"] == 0.35

    def test_each_entry_has_timestamp(self, tmp_path):
        run_id = create_run(str(tmp_path))["run_id"]
        log_metrics(str(tmp_path), run_id, {"acc": 0.9}, step=1)
        r = get_run(str(tmp_path), run_id)
        assert "timestamp" in r["metrics"][0]


# ── log_artifact ─────────────────────────────────────────────────

class TestLogArtifact:
    def test_copies_file_into_run_dir(self, tmp_path):
        run_id = create_run(str(tmp_path))["run_id"]
        model_file = tmp_path / "model.pt"
        model_file.write_bytes(b"weights")
        r = log_artifact(str(tmp_path), run_id, str(model_file))
        assert r["success"] is True
        dest = _run_dir(tmp_path, run_id) / "artifacts" / "model.pt"
        assert dest.exists()
        assert dest.read_bytes() == b"weights"

    def test_missing_artifact_file_returns_error(self, tmp_path):
        run_id = create_run(str(tmp_path))["run_id"]
        r = log_artifact(str(tmp_path), run_id, str(tmp_path / "ghost.pt"))
        assert r["success"] is False


# ── finish_run ───────────────────────────────────────────────────

class TestFinishRun:
    def test_writes_ended_at_and_duration(self, tmp_path):
        run_id = create_run(str(tmp_path))["run_id"]
        r = finish_run(str(tmp_path), run_id, status="success")
        assert r["success"] is True
        assert r["duration_seconds"] is not None
        assert r["duration_seconds"] >= 0.0

    def test_invalid_status_returns_error(self, tmp_path):
        run_id = create_run(str(tmp_path))["run_id"]
        r = finish_run(str(tmp_path), run_id, status="unknown")
        assert r["success"] is False

    def test_status_updated_in_registry(self, tmp_path):
        run_id = create_run(str(tmp_path))["run_id"]
        finish_run(str(tmp_path), run_id, status="failed")
        reg = json.loads((_tracker(tmp_path) / "registry.json").read_text())
        entry = next(e for e in reg["runs"] if e["run_id"] == run_id)
        assert entry["status"] == "failed"


# ── delete_run ───────────────────────────────────────────────────

class TestDeleteRun:
    def test_dry_run_leaves_directory(self, tmp_path):
        run_id = create_run(str(tmp_path))["run_id"]
        r = delete_run(str(tmp_path), run_id, dry_run=True)
        assert r["success"] is True
        assert _run_dir(tmp_path, run_id).exists()

    def test_real_delete_removes_directory(self, tmp_path):
        run_id = create_run(str(tmp_path))["run_id"]
        r = delete_run(str(tmp_path), run_id, dry_run=False)
        assert not _run_dir(tmp_path, run_id).exists()

    def test_deleted_run_removed_from_registry(self, tmp_path):
        run_id = create_run(str(tmp_path))["run_id"]
        delete_run(str(tmp_path), run_id)
        reg = json.loads((_tracker(tmp_path) / "registry.json").read_text())
        assert all(e["run_id"] != run_id for e in reg["runs"])

    def test_missing_run_returns_error(self, tmp_path):
        r = delete_run(str(tmp_path), "nonexistent-id")
        assert r["success"] is False


# ── list_runs ────────────────────────────────────────────────────

class TestListRuns:
    def test_empty_when_no_tracker(self, tmp_path):
        r = list_runs(str(tmp_path))
        assert r["success"] is True
        assert r["run_count"] == 0

    def test_counts_all_runs(self, tmp_path):
        create_run(str(tmp_path))
        create_run(str(tmp_path))
        r = list_runs(str(tmp_path))
        assert r["run_count"] == 2

    def test_latest_metrics_included(self, tmp_path):
        run_id = create_run(str(tmp_path))["run_id"]
        log_metrics(str(tmp_path), run_id, {"val_acc": 0.91}, step=1)
        r = list_runs(str(tmp_path))
        assert r["runs"][0]["latest_metrics"]["val_acc"] == 0.91


# ── compare_runs ─────────────────────────────────────────────────

class TestCompareRuns:
    def test_requires_at_least_two_runs(self, tmp_path):
        run_id = create_run(str(tmp_path))["run_id"]
        r = compare_runs(str(tmp_path), [run_id])
        assert r["success"] is False

    def test_diff_identifies_param_changes(self, tmp_path):
        # uses the fallback diff (no deepdiff installed in base env)
        run_a = create_run(str(tmp_path))["run_id"]
        run_b = create_run(str(tmp_path))["run_id"]
        log_params(str(tmp_path), run_a, {"lr": 0.001})
        log_params(str(tmp_path), run_b, {"lr": 0.01})
        r = compare_runs(str(tmp_path), [run_a, run_b])
        assert r["success"] is True
        assert "snapshots" in r
        assert run_b in r["diff"]


# ── get_best_run ─────────────────────────────────────────────────

class TestGetBestRun:
    def test_finds_max(self, tmp_path):
        run_a = create_run(str(tmp_path))["run_id"]
        run_b = create_run(str(tmp_path))["run_id"]
        log_metrics(str(tmp_path), run_a, {"val_acc": 0.80}, step=1)
        log_metrics(str(tmp_path), run_b, {"val_acc": 0.95}, step=1)
        r = get_best_run(str(tmp_path), "val_acc", direction="max")
        assert r["success"] is True
        assert r["best_run_id"] == run_b

    def test_finds_min(self, tmp_path):
        run_a = create_run(str(tmp_path))["run_id"]
        run_b = create_run(str(tmp_path))["run_id"]
        log_metrics(str(tmp_path), run_a, {"val_loss": 0.30}, step=1)
        log_metrics(str(tmp_path), run_b, {"val_loss": 0.12}, step=1)
        r = get_best_run(str(tmp_path), "val_loss", direction="min")
        assert r["best_run_id"] == run_b

    def test_missing_metric_returns_error(self, tmp_path):
        create_run(str(tmp_path))
        r = get_best_run(str(tmp_path), "nonexistent_metric")
        assert r["success"] is False

    def test_invalid_direction_returns_error(self, tmp_path):
        r = get_best_run(str(tmp_path), "val_acc", direction="sideways")
        assert r["success"] is False


# ── export_runs_csv ──────────────────────────────────────────────

class TestExportRunsCsv:
    def test_creates_csv_file(self, tmp_path):
        pytest.importorskip("pandas")  # skip if pandas not installed
        run_id = create_run(str(tmp_path))["run_id"]
        log_params(str(tmp_path), run_id, {"lr": 0.001})
        log_metrics(str(tmp_path), run_id, {"val_acc": 0.9}, step=1)
        out = tmp_path / "runs.csv"
        r = export_runs_csv(str(tmp_path), str(out))
        assert r["success"] is True
        assert out.exists()
        assert r["row_count"] == 1
