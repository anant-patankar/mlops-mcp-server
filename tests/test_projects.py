from pathlib import Path

import pytest

from mlops_mcp.projects import (
    add_project_component,
    create_ml_project,
    list_project_templates,
    validate_project_structure,
)


# ── list_project_templates ───────────────────────────────────────

class TestListProjectTemplates:
    def test_returns_seven_templates(self):
        r = list_project_templates()
        assert r["success"] is True
        assert r["count"] == 7

    def test_all_known_templates_present(self):
        r = list_project_templates()
        names = set(r["templates"])
        assert names == {
            "basic_ml", "deep_learning", "nlp", "computer_vision",
            "time_series", "mlops_pipeline", "llm_finetuning",
        }

    def test_each_entry_has_required_fields(self):
        r = list_project_templates()
        for name, meta in r["templates"].items():
            assert "description" in meta, f"{name} missing description"
            assert "directory_count" in meta, f"{name} missing directory_count"
            assert "file_count" in meta, f"{name} missing file_count"

    def test_count_matches_templates_dict(self):
        r = list_project_templates()
        assert r["count"] == len(r["templates"])

    def test_basic_ml_directory_count(self):
        r = list_project_templates()
        assert r["templates"]["basic_ml"]["directory_count"] == 8


# ── create_ml_project ────────────────────────────────────────────

class TestCreateMlProject:
    def test_creates_expected_dirs_basic_ml(self, tmp_path):
        r = create_ml_project(str(tmp_path / "proj"))
        assert r["success"] is True
        proj = tmp_path / "proj"
        for d in ("data/raw", "data/processed", "models", "notebooks",
                  "src", "configs", "reports", "tests"):
            assert (proj / d).is_dir(), f"{d} not created"

    def test_creates_expected_files_basic_ml(self, tmp_path):
        proj = tmp_path / "proj"
        r = create_ml_project(str(proj))
        assert (proj / "README.md").exists()
        assert (proj / "configs" / "train.yaml").exists()
        assert (proj / "src" / "__init__.py").exists()

    def test_readme_contains_project_name(self, tmp_path):
        proj = tmp_path / "mymodel"
        create_ml_project(str(proj))
        content = (proj / "README.md").read_text()
        assert "mymodel" in content

    def test_custom_project_name_in_readme(self, tmp_path):
        proj = tmp_path / "proj"
        create_ml_project(str(proj), project_name="OverrideName")
        content = (proj / "README.md").read_text()
        assert "OverrideName" in content

    def test_project_path_created_if_missing(self, tmp_path):
        target = tmp_path / "new" / "nested" / "proj"
        r = create_ml_project(str(target))
        assert r["success"] is True
        assert target.is_dir()

    def test_unknown_template_returns_error(self, tmp_path):
        r = create_ml_project(str(tmp_path / "proj"), template="fantasy_ml")
        assert r["success"] is False
        assert "fantasy_ml" in r["error"]

    def test_idempotent_second_call(self, tmp_path):
        proj = tmp_path / "proj"
        create_ml_project(str(proj))
        r2 = create_ml_project(str(proj))
        assert r2["success"] is True
        # README already existed — should not appear in created_files second time
        assert "README.md" not in r2["created_files"]

    def test_created_directories_sorted(self, tmp_path):
        r = create_ml_project(str(tmp_path / "proj"))
        assert r["created_directories"] == sorted(r["created_directories"])

    def test_created_files_sorted(self, tmp_path):
        r = create_ml_project(str(tmp_path / "proj"))
        assert r["created_files"] == sorted(r["created_files"])

    def test_deep_learning_template_smoke(self, tmp_path):
        r = create_ml_project(str(tmp_path / "proj"), template="deep_learning")
        assert r["success"] is True
        assert (tmp_path / "proj" / "models" / "checkpoints").is_dir()


# ── add_project_component ────────────────────────────────────────

class TestAddProjectComponent:
    def test_adds_monitoring_dirs(self, tmp_path):
        create_ml_project(str(tmp_path))
        r = add_project_component(str(tmp_path), "monitoring")
        assert r["success"] is True
        assert (tmp_path / "monitoring").is_dir()
        assert (tmp_path / "monitoring" / "dashboards").is_dir()

    def test_adds_monitoring_files(self, tmp_path):
        create_ml_project(str(tmp_path))
        add_project_component(str(tmp_path), "monitoring")
        assert (tmp_path / "monitoring" / "README.md").exists()

    def test_adds_api_component(self, tmp_path):
        create_ml_project(str(tmp_path))
        r = add_project_component(str(tmp_path), "api")
        assert r["success"] is True
        assert (tmp_path / "api" / "routes").is_dir()
        assert (tmp_path / "api" / "app.py").exists()

    def test_unknown_component_returns_error(self, tmp_path):
        create_ml_project(str(tmp_path))
        r = add_project_component(str(tmp_path), "blockchain")
        assert r["success"] is False
        assert "blockchain" in r["error"]
        # valid options should be listed in the error
        assert "api" in r["error"] or "monitoring" in r["error"]

    def test_missing_project_dir_returns_error(self, tmp_path):
        r = add_project_component(str(tmp_path / "ghost"), "monitoring")
        assert r["success"] is False

    def test_idempotent_dirs_second_call(self, tmp_path):
        create_ml_project(str(tmp_path))
        add_project_component(str(tmp_path), "monitoring")
        r2 = add_project_component(str(tmp_path), "monitoring")
        assert r2["success"] is True

    def test_idempotent_skips_existing_files(self, tmp_path):
        create_ml_project(str(tmp_path))
        add_project_component(str(tmp_path), "monitoring")
        r2 = add_project_component(str(tmp_path), "monitoring")
        # file already existed — should not appear in created_files
        assert "monitoring/README.md" not in r2["created_files"]

    def test_response_includes_component_name(self, tmp_path):
        create_ml_project(str(tmp_path))
        r = add_project_component(str(tmp_path), "api")
        assert r["component"] == "api"


# ── validate_project_structure ───────────────────────────────────

class TestValidateProjectStructure:
    def test_valid_project_returns_valid_true(self, tmp_path):
        create_ml_project(str(tmp_path))
        r = validate_project_structure(str(tmp_path), "basic_ml")
        assert r["success"] is True
        assert r["valid"] is True
        assert r["missing_directories"] == []
        assert r["missing_files"] == []

    def test_missing_dir_appears_in_missing_directories(self, tmp_path):
        create_ml_project(str(tmp_path))
        import shutil
        shutil.rmtree(tmp_path / "models")
        r = validate_project_structure(str(tmp_path), "basic_ml")
        assert r["valid"] is False
        assert "models" in r["missing_directories"]

    def test_missing_file_appears_in_missing_files(self, tmp_path):
        create_ml_project(str(tmp_path))
        (tmp_path / "README.md").unlink()
        r = validate_project_structure(str(tmp_path), "basic_ml")
        assert r["valid"] is False
        assert "README.md" in r["missing_files"]

    def test_unknown_template_returns_error(self, tmp_path):
        r = validate_project_structure(str(tmp_path), "fantasy_ml")
        assert r["success"] is False
        assert "fantasy_ml" in r["error"]

    def test_missing_project_path_returns_error(self, tmp_path):
        r = validate_project_structure(str(tmp_path / "ghost"), "basic_ml")
        assert r["success"] is False

    def test_partially_created_project_lists_all_missing(self, tmp_path):
        # only create the project root, none of the template dirs
        (tmp_path / "myproj").mkdir()
        r = validate_project_structure(str(tmp_path / "myproj"), "basic_ml")
        assert r["valid"] is False
        assert len(r["missing_directories"]) > 0
        assert len(r["missing_files"]) > 0
