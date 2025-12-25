import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from mlops_mcp.docgen import (
    generate_api_docs,
    generate_dataset_card,
    generate_experiment_report,
    generate_model_card,
    generate_pipeline_docs,
    generate_project_readme,
)
from mlops_mcp.experiments import create_run, finish_run


# ── generate_model_card ──────────────────────────────────────────

class TestGenerateModelCard:
    def test_creates_output_file(self, tmp_path):
        out = tmp_path / "MODEL_CARD.md"
        r = generate_model_card(
            model_name="bert",
            task_type="classification",
            dataset="imdb",
            metrics={"f1": 0.92},
            limitations=["English only"],
            usage_example="model.predict(x)",
            output_path=str(out),
        )
        assert r["success"] is True
        assert out.exists()

    def test_output_contains_model_name_and_task(self, tmp_path):
        out = tmp_path / "card.md"
        generate_model_card(
            model_name="resnet50",
            task_type="image-classification",
            dataset="imagenet",
            metrics={},
            limitations=[],
            usage_example="",
            output_path=str(out),
        )
        content = out.read_text()
        assert "resnet50" in content
        assert "image-classification" in content

    def test_empty_metrics_renders_na(self, tmp_path):
        out = tmp_path / "card.md"
        generate_model_card("m", "t", "d", {}, [], "", output_path=str(out))
        assert "- N/A" in out.read_text()

    def test_empty_limitations_renders_none_provided(self, tmp_path):
        out = tmp_path / "card.md"
        generate_model_card("m", "t", "d", {"acc": 1.0}, [], "", output_path=str(out))
        assert "- None provided" in out.read_text()

    def test_custom_output_path_respected(self, tmp_path):
        out = tmp_path / "subdir" / "special.md"
        r = generate_model_card("m", "t", "d", {}, [], "", output_path=str(out))
        assert r["success"] is True
        assert out.exists()

    def test_missing_parent_dirs_created(self, tmp_path):
        out = tmp_path / "a" / "b" / "c" / "card.md"
        r = generate_model_card("m", "t", "d", {}, [], "", output_path=str(out))
        assert r["success"] is True
        assert out.exists()

    def test_fallback_without_jinja2(self, tmp_path):
        out = tmp_path / "card.md"
        with patch.dict(sys.modules, {"jinja2": None}):
            r = generate_model_card(
                "gpt2", "text-gen", "pile", {}, [], "", output_path=str(out)
            )
        assert r["success"] is True
        assert "gpt2" in out.read_text()


# ── generate_dataset_card ────────────────────────────────────────

class TestGenerateDatasetCard:
    pd = pytest.importorskip("pandas")

    def _make_csv(self, tmp_path, name="data.csv", content="a,b\n1,2\n3,4\n"):
        p = tmp_path / name
        p.write_text(content, encoding="utf-8")
        return p

    def test_creates_output_file(self, tmp_path):
        f = self._make_csv(tmp_path)
        out = tmp_path / "DATASET_CARD.md"
        r = generate_dataset_card(str(f), output_path=str(out))
        assert r["success"] is True
        assert out.exists()

    def test_output_contains_row_and_column_count(self, tmp_path):
        f = self._make_csv(tmp_path)
        out = tmp_path / "card.md"
        generate_dataset_card(str(f), output_path=str(out))
        content = out.read_text()
        assert "2" in content  # 2 rows

    def test_missing_dataset_returns_error(self, tmp_path):
        r = generate_dataset_card(
            str(tmp_path / "ghost.csv"),
            output_path=str(tmp_path / "card.md"),
        )
        assert r["success"] is False

    def test_description_and_license_appear_in_output(self, tmp_path):
        f = self._make_csv(tmp_path)
        out = tmp_path / "card.md"
        generate_dataset_card(
            str(f),
            output_path=str(out),
            description="Training split",
            license_name="MIT",
        )
        content = out.read_text()
        assert "Training split" in content
        assert "MIT" in content


# ── generate_experiment_report ───────────────────────────────────

class TestGenerateExperimentReport:
    def test_creates_report_with_no_runs(self, tmp_path):
        # init tracker so list_runs succeeds with empty run list
        from mlops_mcp.experiments import init_experiment_tracker
        init_experiment_tracker(str(tmp_path))
        out = tmp_path / "report.md"
        r = generate_experiment_report(str(tmp_path), output_path=str(out))
        assert r["success"] is True
        assert out.exists()
        assert "No runs" in out.read_text()

    def test_creates_report_with_two_runs(self, tmp_path):
        r1 = create_run(str(tmp_path))
        finish_run(str(tmp_path), r1["run_id"], status="success")
        r2 = create_run(str(tmp_path))
        finish_run(str(tmp_path), r2["run_id"], status="success")
        out = tmp_path / "report.md"
        r = generate_experiment_report(str(tmp_path), output_path=str(out))
        assert r["success"] is True
        content = out.read_text()
        assert r1["run_id"] in content
        assert r2["run_id"] in content

    def test_report_with_no_tracker_shows_zero_runs(self, tmp_path):
        # list_runs succeeds with empty list even for uninitialised paths
        out = tmp_path / "report.md"
        r = generate_experiment_report(str(tmp_path / "ghost"), output_path=str(out))
        assert r["success"] is True
        assert r["run_count"] == 0


# ── generate_pipeline_docs ────────────────────────────────────────

class TestGeneratePipelineDocs:
    def _make_pipeline(self, tmp_path, name="pipeline.yaml"):
        pipeline = {
            "stages": {
                "preprocess": {
                    "cmd": "python preprocess.py",
                    "deps": ["data/raw"],
                    "outs": ["data/processed"],
                },
                "train": {
                    "cmd": "python train.py",
                    "deps": ["data/processed"],
                    "outs": ["models/model.pkl"],
                },
            }
        }
        p = tmp_path / name
        p.write_text(yaml.safe_dump(pipeline), encoding="utf-8")
        return p

    def test_creates_pipeline_md_with_stage_names(self, tmp_path):
        p = self._make_pipeline(tmp_path)
        out = tmp_path / "PIPELINE.md"
        r = generate_pipeline_docs(str(p), output_path=str(out))
        assert r["success"] is True
        content = out.read_text()
        assert "preprocess" in content
        assert "train" in content

    def test_mermaid_block_in_output(self, tmp_path):
        p = self._make_pipeline(tmp_path)
        out = tmp_path / "PIPELINE.md"
        generate_pipeline_docs(str(p), output_path=str(out))
        assert "```mermaid" in out.read_text()

    def test_missing_pipeline_file_returns_error(self, tmp_path):
        r = generate_pipeline_docs(
            str(tmp_path / "ghost.yaml"),
            output_path=str(tmp_path / "PIPELINE.md"),
        )
        assert r["success"] is False


# ── generate_project_readme ───────────────────────────────────────

class TestGenerateProjectReadme:
    def test_creates_readme_for_real_directory(self, tmp_path):
        (tmp_path / "main.py").write_text("import os\n")
        (tmp_path / "utils.py").write_text("import sys\n")
        out = tmp_path / "README.md"
        r = generate_project_readme(str(tmp_path), output_path=str(out))
        assert r["success"] is True
        assert out.exists()

    def test_output_contains_file_count(self, tmp_path):
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.py").write_text("")
        out = tmp_path / "README.md"
        generate_project_readme(str(tmp_path), output_path=str(out))
        content = out.read_text()
        assert "Files" in content
        assert "2" in content

    def test_missing_project_path_returns_error(self, tmp_path):
        r = generate_project_readme(
            str(tmp_path / "ghost"),
            output_path=str(tmp_path / "README.md"),
        )
        assert r["success"] is False


# ── generate_api_docs ─────────────────────────────────────────────

class TestGenerateApiDocs:
    def _make_source(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "tools.py").write_text(
            "def predict(x, y):\n    pass\n\ndef train(data):\n    return None\n"
        )
        return src

    def test_finds_functions_in_python_files(self, tmp_path):
        src = self._make_source(tmp_path)
        out = tmp_path / "API.md"
        r = generate_api_docs(str(src), output_path=str(out))
        assert r["success"] is True
        content = out.read_text()
        assert "predict" in content
        assert "train" in content

    def test_function_count_matches_top_level_functions(self, tmp_path):
        src = self._make_source(tmp_path)
        out = tmp_path / "API.md"
        r = generate_api_docs(str(src), output_path=str(out))
        assert r["function_count"] == 2

    def test_missing_source_path_returns_error(self, tmp_path):
        r = generate_api_docs(
            str(tmp_path / "ghost"),
            output_path=str(tmp_path / "API.md"),
        )
        assert r["success"] is False

    def test_ignores_files_with_syntax_errors(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "good.py").write_text("def ok(): pass\n")
        (src / "bad.py").write_text("def broken(\n")  # syntax error
        out = tmp_path / "API.md"
        r = generate_api_docs(str(src), output_path=str(out))
        assert r["success"] is True
        assert r["function_count"] == 1
