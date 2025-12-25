import io
import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from mlops_mcp.datasets import (
    check_data_freshness,
    detect_data_drift,
    find_dataset_files,
    generate_dataset_card,
    merge_datasets,
    profile_dataset,
    split_dataset,
    validate_dataset_schema,
)


def _csv(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


# ── profile_dataset ──────────────────────────────────────────────

class TestProfileDataset:
    pd = pytest.importorskip("pandas")

    def test_row_and_column_counts(self, tmp_path):
        f = _csv(tmp_path, "data.csv", "a,b\n1,2\n3,4\n5,6\n")
        r = profile_dataset(str(f))
        assert r["success"] is True
        assert r["rows"] == 3
        assert r["columns"] == 2

    def test_null_counts(self, tmp_path):
        f = _csv(tmp_path, "data.csv", "a,b\n1,\n3,4\n")
        r = profile_dataset(str(f))
        assert r["null_counts"]["b"] == 1

    def test_numeric_stats_present(self, tmp_path):
        f = _csv(tmp_path, "data.csv", "x\n1\n2\n3\n")
        r = profile_dataset(str(f))
        assert "x" in r["numeric_stats"]

    def test_missing_file_returns_error(self, tmp_path):
        r = profile_dataset(str(tmp_path / "ghost.csv"))
        assert r["success"] is False

    def test_unsupported_extension_returns_error(self, tmp_path):
        f = tmp_path / "data.xlsx"
        f.write_bytes(b"fake")
        r = profile_dataset(str(f))
        assert r["success"] is False


# ── validate_dataset_schema ──────────────────────────────────────

class TestValidateDatasetSchema:
    pd = pytest.importorskip("pandas")

    def _schema(self, tmp_path, content, name="schema.json"):
        p = tmp_path / name
        if name.endswith(".json"):
            p.write_text(json.dumps(content), encoding="utf-8")
        else:
            p.write_text(yaml.dump(content), encoding="utf-8")
        return p

    def test_valid_schema_passes(self, tmp_path):
        f = _csv(tmp_path, "data.csv", "a,b\n1,2\n")
        s = self._schema(tmp_path, {"columns": {"a": "int64", "b": "int64"}})
        r = validate_dataset_schema(str(f), str(s))
        assert r["success"] is True
        assert r["valid"] is True
        assert r["errors"] == []

    def test_missing_column_reported(self, tmp_path):
        f = _csv(tmp_path, "data.csv", "a\n1\n")
        s = self._schema(tmp_path, {"columns": {"a": "int64", "b": "int64"}})
        r = validate_dataset_schema(str(f), str(s))
        assert r["valid"] is False
        cols = [e["column"] for e in r["errors"]]
        assert "b" in cols

    def test_dtype_mismatch_reported(self, tmp_path):
        f = _csv(tmp_path, "data.csv", "a\nhello\nworld\n")
        s = self._schema(tmp_path, {"columns": {"a": "int64"}})
        r = validate_dataset_schema(str(f), str(s))
        assert r["valid"] is False
        assert r["errors"][0]["error"] == "dtype_mismatch"

    def test_non_dict_schema_returns_error(self, tmp_path):
        f = _csv(tmp_path, "data.csv", "a\n1\n")
        s = tmp_path / "schema.json"
        s.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
        r = validate_dataset_schema(str(f), str(s))
        assert r["success"] is False

    def test_missing_dataset_returns_error(self, tmp_path):
        s = self._schema(tmp_path, {"columns": {}})
        r = validate_dataset_schema(str(tmp_path / "ghost.csv"), str(s))
        assert r["success"] is False

    def test_missing_schema_file_returns_error(self, tmp_path):
        f = _csv(tmp_path, "data.csv", "a\n1\n")
        r = validate_dataset_schema(str(f), str(tmp_path / "ghost.json"))
        assert r["success"] is False

    def test_yaml_schema_works(self, tmp_path):
        f = _csv(tmp_path, "data.csv", "a\n1\n2\n")
        s = self._schema(tmp_path, {"columns": {"a": "int64"}}, name="schema.yaml")
        r = validate_dataset_schema(str(f), str(s))
        assert r["success"] is True


# ── detect_data_drift ────────────────────────────────────────────

class TestDetectDataDrift:
    pd = pytest.importorskip("pandas")

    def test_numeric_identical_no_drift(self, tmp_path):
        pytest.importorskip("scipy")
        f1 = _csv(tmp_path, "ref.csv", "x\n1\n2\n3\n4\n5\n")
        f2 = _csv(tmp_path, "cand.csv", "x\n1\n2\n3\n4\n5\n")
        r = detect_data_drift(str(f1), str(f2))
        assert r["success"] is True
        assert r["drift_report"]["x"]["drift_detected"] is False

    def test_numeric_very_different_drift_detected(self, tmp_path):
        pytest.importorskip("scipy")
        ref = "\n".join(["x"] + ["1"] * 50) + "\n"
        cand = "\n".join(["x"] + ["1000"] * 50) + "\n"
        f1 = _csv(tmp_path, "ref.csv", ref)
        f2 = _csv(tmp_path, "cand.csv", cand)
        r = detect_data_drift(str(f1), str(f2))
        assert r["drift_report"]["x"]["drift_detected"] is True

    def test_categorical_identical_no_drift(self, tmp_path):
        pytest.importorskip("scipy")
        content = "cat\na\nb\na\nb\n"
        f1 = _csv(tmp_path, "ref.csv", content)
        f2 = _csv(tmp_path, "cand.csv", content)
        r = detect_data_drift(str(f1), str(f2))
        assert r["drift_report"]["cat"]["drift_detected"] is False

    def test_missing_reference_returns_error(self, tmp_path):
        f2 = _csv(tmp_path, "cand.csv", "x\n1\n")
        r = detect_data_drift(str(tmp_path / "ghost.csv"), str(f2))
        assert r["success"] is False

    def test_scipy_not_installed_returns_error(self, tmp_path):
        f1 = _csv(tmp_path, "ref.csv", "x\n1\n2\n")
        f2 = _csv(tmp_path, "cand.csv", "x\n1\n2\n")
        with patch.dict("sys.modules", {"scipy": None, "scipy.stats": None}):
            r = detect_data_drift(str(f1), str(f2))
        assert r["success"] is False


# ── split_dataset ────────────────────────────────────────────────

class TestSplitDataset:
    pd = pytest.importorskip("pandas")

    def _big_csv(self, tmp_path, rows=100):
        lines = ["a,b"] + [f"{i},{i*2}" for i in range(rows)]
        return _csv(tmp_path, "data.csv", "\n".join(lines) + "\n")

    def test_basic_split_correct_total(self, tmp_path):
        pytest.importorskip("sklearn")
        f = self._big_csv(tmp_path)
        r = split_dataset(str(f), str(tmp_path / "out"))
        assert r["success"] is True
        total = r["counts"]["train"] + r["counts"]["val"] + r["counts"]["test"]
        assert total == 100

    def test_ratios_not_summing_to_one_returns_error(self, tmp_path):
        f = self._big_csv(tmp_path)
        r = split_dataset(str(f), str(tmp_path / "out"), train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)
        assert r["success"] is False

    def test_stratify_column_not_found_returns_error(self, tmp_path):
        pytest.importorskip("sklearn")
        f = self._big_csv(tmp_path)
        r = split_dataset(str(f), str(tmp_path / "out"), stratify_column="nonexistent")
        assert r["success"] is False

    def test_missing_dataset_returns_error(self, tmp_path):
        r = split_dataset(str(tmp_path / "ghost.csv"), str(tmp_path / "out"))
        assert r["success"] is False

    def test_sklearn_not_installed_returns_error(self, tmp_path):
        f = self._big_csv(tmp_path)
        with patch.dict("sys.modules", {"sklearn": None, "sklearn.model_selection": None}):
            r = split_dataset(str(f), str(tmp_path / "out"))
        assert r["success"] is False


# ── merge_datasets ───────────────────────────────────────────────

class TestMergeDatasets:
    pd = pytest.importorskip("pandas")

    def test_merges_two_csvs(self, tmp_path):
        f1 = _csv(tmp_path, "a.csv", "x\n1\n2\n")
        f2 = _csv(tmp_path, "b.csv", "x\n3\n4\n")
        out = tmp_path / "merged.csv"
        r = merge_datasets([str(f1), str(f2)], str(out))
        assert r["success"] is True
        assert r["rows"] == 4

    def test_deduplicate_removes_dupes(self, tmp_path):
        f1 = _csv(tmp_path, "a.csv", "x\n1\n2\n")
        f2 = _csv(tmp_path, "b.csv", "x\n2\n3\n")
        out = tmp_path / "merged.csv"
        r = merge_datasets([str(f1), str(f2)], str(out), deduplicate=True)
        assert r["rows"] == 3

    def test_empty_paths_returns_error(self, tmp_path):
        r = merge_datasets([], str(tmp_path / "out.csv"))
        assert r["success"] is False

    def test_missing_dataset_returns_error(self, tmp_path):
        r = merge_datasets([str(tmp_path / "ghost.csv")], str(tmp_path / "out.csv"))
        assert r["success"] is False


# ── check_data_freshness ─────────────────────────────────────────

class TestCheckDataFreshness:
    def test_fresh_file_not_stale(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("x\n1\n")
        r = check_data_freshness(str(f), stale_after_hours=24)
        assert r["success"] is True
        assert r["is_stale"] is False

    def test_old_file_is_stale(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("x\n1\n")
        # set mtime to 2 days ago
        old_time = time.time() - 48 * 3600
        os.utime(f, (old_time, old_time))
        r = check_data_freshness(str(f), stale_after_hours=24)
        assert r["is_stale"] is True

    def test_negative_stale_hours_returns_error(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("x\n1\n")
        r = check_data_freshness(str(f), stale_after_hours=-1)
        assert r["success"] is False

    def test_missing_file_returns_error(self, tmp_path):
        r = check_data_freshness(str(tmp_path / "ghost.csv"))
        assert r["success"] is False

    def test_response_has_age_and_modified(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("x\n1\n")
        r = check_data_freshness(str(f))
        assert "age_seconds" in r
        assert "last_modified" in r


# ── generate_dataset_card ────────────────────────────────────────

class TestGenerateDatasetCard:
    pd = pytest.importorskip("pandas")

    def test_creates_default_dataset_card(self, tmp_path):
        f = _csv(tmp_path, "mydata.csv", "a,b\n1,2\n3,4\n")
        r = generate_dataset_card(str(f))
        assert r["success"] is True
        assert (tmp_path / "DATASET_CARD.md").exists()

    def test_custom_output_path_respected(self, tmp_path):
        f = _csv(tmp_path, "mydata.csv", "a,b\n1,2\n")
        out = tmp_path / "cards" / "card.md"
        r = generate_dataset_card(str(f), output_path=str(out))
        assert r["success"] is True
        assert out.exists()

    def test_output_contains_dataset_name(self, tmp_path):
        f = _csv(tmp_path, "trainset.csv", "a\n1\n2\n")
        r = generate_dataset_card(str(f))
        content = (tmp_path / "DATASET_CARD.md").read_text()
        assert "trainset.csv" in content

    def test_missing_dataset_returns_error(self, tmp_path):
        r = generate_dataset_card(str(tmp_path / "ghost.csv"))
        assert r["success"] is False

    def test_fallback_path_without_jinja2(self, tmp_path):
        f = _csv(tmp_path, "data.csv", "a\n1\n2\n")
        with patch.dict("sys.modules", {"jinja2": None}):
            r = generate_dataset_card(str(f))
        assert r["success"] is True
        content = (tmp_path / "DATASET_CARD.md").read_text()
        assert "DATASET_CARD" in content


# ── find_dataset_files ───────────────────────────────────────────

class TestFindDatasetFiles:
    pd = pytest.importorskip("pandas")

    def test_finds_csv_files(self, tmp_path):
        _csv(tmp_path, "train.csv", "a\n1\n")
        r = find_dataset_files(str(tmp_path))
        assert r["success"] is True
        paths = [d["path"] for d in r["datasets"]]
        assert any("train.csv" in p for p in paths)

    def test_ignores_non_dataset_files(self, tmp_path):
        _csv(tmp_path, "data.csv", "a\n1\n")
        (tmp_path / "readme.md").write_text("# hi")
        (tmp_path / "model.pkl").write_bytes(b"\x00")
        r = find_dataset_files(str(tmp_path))
        paths = [d["path"] for d in r["datasets"]]
        assert all(not p.endswith(".md") and not p.endswith(".pkl") for p in paths)

    def test_results_sorted(self, tmp_path):
        _csv(tmp_path, "z.csv", "a\n1\n")
        _csv(tmp_path, "a.csv", "a\n1\n")
        r = find_dataset_files(str(tmp_path))
        paths = [d["path"] for d in r["datasets"]]
        assert paths == sorted(paths)

    def test_missing_directory_returns_error(self, tmp_path):
        r = find_dataset_files(str(tmp_path / "ghost"))
        assert r["success"] is False

    def test_empty_directory_returns_empty_list(self, tmp_path):
        r = find_dataset_files(str(tmp_path))
        assert r["success"] is True
        assert r["count"] == 0
        assert r["datasets"] == []
