import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from mlops_mcp.models import (
    compare_model_versions,
    delete_model_version,
    deprecate_model,
    get_model_info,
    get_model_versions,
    init_model_registry,
    list_models,
    model_list,
    promote_model,
    register_model,
    tag_model,
)


def _model_file(tmp_path, name="model.pt"):
    p = tmp_path / name
    p.write_bytes(b"\x00" * 16)
    return p


def _register(tmp_path, name="classifier", fname="model.pt", stage="dev", version=None, run_id=None):
    mf = _model_file(tmp_path, fname)
    return register_model(
        str(tmp_path), name, str(mf),
        stage=stage, version=version, run_id=run_id,
    )


# ── init_model_registry ───────────────────────────────────────────

class TestInitModelRegistry:
    def test_creates_registry_file_on_first_call(self, tmp_path):
        r = init_model_registry(str(tmp_path))
        assert r["success"] is True
        assert Path(r["registry_path"]).exists()

    def test_idempotent_on_second_call(self, tmp_path):
        init_model_registry(str(tmp_path))
        r = init_model_registry(str(tmp_path))
        assert r["success"] is True


# ── register_model ────────────────────────────────────────────────

class TestRegisterModel:
    def test_registers_model_returns_version_stage_framework(self, tmp_path):
        r = _register(tmp_path)
        assert r["success"] is True
        assert r["version"] == 1
        assert r["stage"] == "dev"
        assert r["framework"] == "pytorch"

    def test_auto_increments_version(self, tmp_path):
        _register(tmp_path, fname="model.pt")
        r = _register(tmp_path, fname="model.pt")
        assert r["version"] == 2

    def test_explicit_version_is_respected(self, tmp_path):
        r = _register(tmp_path, version=5)
        assert r["version"] == 5

    def test_duplicate_version_returns_error(self, tmp_path):
        _register(tmp_path, version=3)
        r = _register(tmp_path, version=3)
        assert r["success"] is False

    def test_missing_model_file_returns_error(self, tmp_path):
        r = register_model(str(tmp_path), "x", str(tmp_path / "ghost.pt"))
        assert r["success"] is False

    def test_mlflow_registered_false_when_not_installed(self, tmp_path):
        mf = _model_file(tmp_path)
        with patch.dict(sys.modules, {"mlflow": None}):
            r = register_model(str(tmp_path), "clf", str(mf))
        assert r["mlflow_registered"] is False


# ── list_models ───────────────────────────────────────────────────

class TestListModels:
    def test_empty_registry_returns_empty_list(self, tmp_path):
        r = list_models(str(tmp_path))
        assert r["success"] is True
        assert r["models"] == []
        assert r["model_count"] == 0

    def test_registered_model_appears_in_list(self, tmp_path):
        _register(tmp_path, name="clf")
        r = list_models(str(tmp_path))
        names = [m["model_name"] for m in r["models"]]
        assert "clf" in names

    def test_models_sorted_alphabetically(self, tmp_path):
        _register(tmp_path, name="zoo", fname="zoo.pt")
        _register(tmp_path, name="alpha", fname="alpha.pt")
        r = list_models(str(tmp_path))
        names = [m["model_name"] for m in r["models"]]
        assert names == sorted(names)


# ── get_model_versions ────────────────────────────────────────────

class TestGetModelVersions:
    def test_returns_all_versions(self, tmp_path):
        _register(tmp_path, fname="model.pt", version=1)
        _register(tmp_path, fname="model.pt", version=2)
        r = get_model_versions(str(tmp_path), "classifier")
        assert r["success"] is True
        assert len(r["versions"]) == 2

    def test_file_exists_reflects_presence(self, tmp_path):
        mf = _model_file(tmp_path)
        register_model(str(tmp_path), "clf", str(mf))
        mf.unlink()  # delete after registration
        r = get_model_versions(str(tmp_path), "clf")
        assert r["versions"][0]["file_exists"] is False

    def test_unknown_model_returns_error(self, tmp_path):
        r = get_model_versions(str(tmp_path), "ghost")
        assert r["success"] is False


# ── get_model_info ────────────────────────────────────────────────

class TestGetModelInfo:
    def test_returns_correct_model_entry(self, tmp_path):
        _register(tmp_path, name="clf", stage="staging")
        r = get_model_info(str(tmp_path), "clf", 1)
        assert r["success"] is True
        assert r["model"]["stage"] == "staging"
        assert r["model"]["version"] == 1

    def test_unknown_version_returns_error(self, tmp_path):
        _register(tmp_path, name="clf")
        r = get_model_info(str(tmp_path), "clf", 99)
        assert r["success"] is False


# ── promote_model ─────────────────────────────────────────────────

class TestPromoteModel:
    def test_stage_changes_correctly(self, tmp_path):
        _register(tmp_path, name="clf", stage="dev")
        r = promote_model(str(tmp_path), "clf", 1, "production")
        assert r["success"] is True
        assert r["new_stage"] == "production"
        assert r["old_stage"] == "dev"

    def test_stage_history_updated(self, tmp_path):
        _register(tmp_path, name="clf", stage="dev")
        promote_model(str(tmp_path), "clf", 1, "staging")
        info = get_model_info(str(tmp_path), "clf", 1)
        history = info["model"]["stage_history"]
        stages = [h["stage"] for h in history]
        assert "staging" in stages

    def test_unknown_model_returns_error(self, tmp_path):
        r = promote_model(str(tmp_path), "ghost", 1, "production")
        assert r["success"] is False


# ── tag_model ─────────────────────────────────────────────────────

class TestTagModel:
    def test_tags_are_merged_not_replaced(self, tmp_path):
        _register(tmp_path, name="clf")
        tag_model(str(tmp_path), "clf", 1, {"owner": "alice"})
        r = tag_model(str(tmp_path), "clf", 1, {"env": "prod"})
        assert r["tags"]["owner"] == "alice"
        assert r["tags"]["env"] == "prod"

    def test_unknown_model_returns_error(self, tmp_path):
        r = tag_model(str(tmp_path), "ghost", 1, {"k": "v"})
        assert r["success"] is False


# ── compare_model_versions ────────────────────────────────────────

class TestCompareModelVersions:
    def test_diff_contains_expected_keys(self, tmp_path):
        _register(tmp_path, fname="model.pt", version=1, stage="dev")
        _register(tmp_path, fname="model.pt", version=2, stage="staging")
        r = compare_model_versions(str(tmp_path), "classifier", 1, 2)
        assert r["success"] is True
        assert "size" in r["diff"]
        assert "stage" in r["diff"]
        assert "framework" in r["diff"]

    def test_unknown_version_returns_error(self, tmp_path):
        _register(tmp_path, fname="model.pt", version=1)
        r = compare_model_versions(str(tmp_path), "classifier", 1, 99)
        assert r["success"] is False


# ── deprecate_model ───────────────────────────────────────────────

class TestDeprecateModel:
    def test_stage_becomes_deprecated(self, tmp_path):
        _register(tmp_path, name="clf")
        r = deprecate_model(str(tmp_path), "clf", 1, "too old")
        assert r["success"] is True
        assert r["stage"] == "deprecated"

    def test_reason_stored_in_entry(self, tmp_path):
        _register(tmp_path, name="clf")
        deprecate_model(str(tmp_path), "clf", 1, "accuracy dropped")
        info = get_model_info(str(tmp_path), "clf", 1)
        assert info["model"]["deprecation_reason"] == "accuracy dropped"


# ── delete_model_version ──────────────────────────────────────────

class TestDeleteModelVersion:
    def test_removes_version_from_registry(self, tmp_path):
        _register(tmp_path, fname="model.pt", version=1)
        _register(tmp_path, fname="model.pt", version=2)
        delete_model_version(str(tmp_path), "classifier", 1)
        r = get_model_versions(str(tmp_path), "classifier")
        assert all(v["version"] != 1 for v in r["versions"])

    def test_delete_file_true_removes_file(self, tmp_path):
        mf = _model_file(tmp_path)
        register_model(str(tmp_path), "clf", str(mf))
        delete_model_version(str(tmp_path), "clf", 1, delete_file=True)
        assert not mf.exists()

    def test_delete_file_false_keeps_file(self, tmp_path):
        mf = _model_file(tmp_path)
        register_model(str(tmp_path), "clf", str(mf))
        delete_model_version(str(tmp_path), "clf", 1, delete_file=False)
        assert mf.exists()

    def test_last_version_removal_cleans_up_model_entry(self, tmp_path):
        _register(tmp_path, name="clf")
        delete_model_version(str(tmp_path), "clf", 1)
        r = list_models(str(tmp_path))
        assert all(m["model_name"] != "clf" for m in r["models"])

    def test_unknown_version_returns_error(self, tmp_path):
        _register(tmp_path, name="clf")
        r = delete_model_version(str(tmp_path), "clf", 99)
        assert r["success"] is False


# ── model_list ────────────────────────────────────────────────────

class TestModelList:
    def test_returns_source_registry_when_registry_has_models(self, tmp_path):
        _register(tmp_path, name="clf")
        r = model_list(str(tmp_path))
        assert r["success"] is True
        assert r["source"] == "registry"

    def test_falls_back_to_filesystem_when_registry_empty(self, tmp_path):
        (tmp_path / "model.pt").write_bytes(b"\x00" * 16)
        r = model_list(str(tmp_path))
        assert r["source"] == "filesystem"

    def test_filesystem_scan_finds_pt_files(self, tmp_path):
        (tmp_path / "a.pt").write_bytes(b"\x00")
        (tmp_path / "b.onnx").write_bytes(b"\x00")
        r = model_list(str(tmp_path))
        exts = {m["extension"] for m in r["models"]}
        assert ".pt" in exts
        assert ".onnx" in exts
