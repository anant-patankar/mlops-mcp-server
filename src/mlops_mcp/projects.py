from pathlib import Path
from typing import Any

from ._types import err as _fail


_TEMPLATES: dict[str, dict[str, Any]] = {
    "basic_ml": {
        "description": "Minimal ML project structure.",
        "dirs": [
            "data/raw",
            "data/processed",
            "models",
            "notebooks",
            "src",
            "configs",
            "reports",
            "tests",
        ],
        "files": {
            "README.md": "# {project_name}\n",
            "configs/train.yaml": "seed: 42\n",
            "src/__init__.py": "",
        },
    },
    "deep_learning": {
        "description": "Deep learning-focused project structure.",
        "dirs": [
            "data/raw",
            "data/processed",
            "models/checkpoints",
            "models/exports",
            "notebooks",
            "src/data",
            "src/models",
            "src/training",
            "configs",
            "logs",
            "tests",
        ],
        "files": {
            "README.md": "# {project_name}\n",
            "configs/model.yaml": "architecture: baseline\n",
            "src/__init__.py": "",
        },
    },
    "nlp": {
        "description": "NLP-oriented project layout.",
        "dirs": [
            "data/raw",
            "data/tokenized",
            "models",
            "notebooks",
            "src/preprocessing",
            "src/training",
            "configs",
            "reports",
            "tests",
        ],
        "files": {
            "README.md": "# {project_name}\n",
            "configs/tokenizer.yaml": "name: basic\n",
            "src/__init__.py": "",
        },
    },
    "computer_vision": {
        "description": "Computer vision project skeleton.",
        "dirs": [
            "data/images",
            "data/labels",
            "models",
            "notebooks",
            "src/datasets",
            "src/models",
            "src/training",
            "configs",
            "tests",
        ],
        "files": {
            "README.md": "# {project_name}\n",
            "configs/augmentations.yaml": "flip: true\n",
            "src/__init__.py": "",
        },
    },
    "time_series": {
        "description": "Time-series forecasting project structure.",
        "dirs": [
            "data/raw",
            "data/features",
            "models",
            "notebooks",
            "src/features",
            "src/training",
            "configs",
            "reports",
            "tests",
        ],
        "files": {
            "README.md": "# {project_name}\n",
            "configs/features.yaml": "window: 24\n",
            "src/__init__.py": "",
        },
    },
    "mlops_pipeline": {
        "description": "MLOps pipeline-oriented layout.",
        "dirs": [
            "data/raw",
            "data/processed",
            "pipelines",
            "models",
            "src/pipelines",
            "src/training",
            "configs",
            "deploy",
            "tests",
        ],
        "files": {
            "README.md": "# {project_name}\n",
            "pipelines/pipeline.yaml": "stages: []\n",
            "src/__init__.py": "",
        },
    },
    "llm_finetuning": {
        "description": "LLM fine-tuning project structure.",
        "dirs": [
            "data/raw",
            "data/formatted",
            "models/checkpoints",
            "models/merged",
            "src/data",
            "src/training",
            "src/evaluation",
            "configs",
            "tests",
        ],
        "files": {
            "README.md": "# {project_name}\n",
            "configs/finetune.yaml": 'base_model: ""\n',
            "src/__init__.py": "",
        },
    },
}

# TODO: finding MLOPs used in Data Engineering and RL and add particular templates


_COMPONENTS: dict[str, dict[str, Any]] = {
    "monitoring": {
        "dirs": ["monitoring", "monitoring/dashboards"],
        "files": {"monitoring/README.md": "# Monitoring\n"},
    },
    "api": {
        "dirs": ["api", "api/routes"],
        "files": {
            "api/app.py": "def create_app():\n    return None\n",
            "api/README.md": "# API\n",
        },
    },
    "tests": {
        "dirs": ["tests"],
        "files": {"tests/test_smoke.py": "def test_smoke():\n    assert True\n"},
    },
}


def list_project_templates() -> dict[str, Any]:
    templates = {
        name: {
            "description": meta["description"],
            "directory_count": len(meta["dirs"]),
            "file_count": len(meta["files"]),
        }
        for name, meta in _TEMPLATES.items()
    }
    return {"success": True, "count": len(templates), "templates": templates}


def create_ml_project(
    project_path: str,
    template: str = "basic_ml",
    project_name: str | None = None,
) -> dict[str, Any]:
    target = Path(project_path).expanduser().resolve()
    if template not in _TEMPLATES:
        return _fail(f"unknown template '{template}'")

    meta = _TEMPLATES[template]
    name = project_name or target.name

    try:
        target.mkdir(parents=True, exist_ok=True)
        created_dirs: list[str] = []
        created_files: list[str] = []

        for relative in meta["dirs"]:
            path = target / relative
            path.mkdir(parents=True, exist_ok=True)
            created_dirs.append(relative)

        for relative, content in meta["files"].items():
            path = target / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_text(content.format(
                    project_name=name), encoding="utf-8")
                created_files.append(relative)

        return {
            "success": True,
            "project_path": str(target),
            "template": template,
            "created_directories": sorted(created_dirs),
            "created_files": sorted(created_files),
        }
    except OSError as exc:
        return _fail(str(exc))


def add_project_component(project_path: str, component: str) -> dict[str, Any]:
    if component not in _COMPONENTS:
        valid = ", ".join(sorted(_COMPONENTS))
        return _fail(f"unknown component '{component}'. Use one of: {valid}")

    target = Path(project_path).expanduser().resolve()
    if not target.exists() or not target.is_dir():
        return _fail(f"project directory not found: {target}")

    meta = _COMPONENTS[component]
    try:
        created_dirs: list[str] = []
        created_files: list[str] = []

        for relative in meta["dirs"]:
            path = target / relative
            path.mkdir(parents=True, exist_ok=True)
            created_dirs.append(relative)

        for relative, content in meta["files"].items():
            path = target / relative
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
                created_files.append(relative)

        return {
            "success": True,
            "project_path": str(target),
            "component": component,
            "created_directories": sorted(created_dirs),
            "created_files": sorted(created_files),
        }
    except OSError as exc:
        return _fail(str(exc))


def validate_project_structure(project_path: str, template: str) -> dict[str, Any]:
    if template not in _TEMPLATES:
        return _fail(f"unknown template '{template}'")

    target = Path(project_path).expanduser().resolve()
    if not target.exists() or not target.is_dir():
        return _fail(f"project directory not found: {target}")

    meta = _TEMPLATES[template]
    missing_dirs = [relative for relative in meta["dirs"]
                    if not (target / relative).is_dir()]
    missing_files = [relative for relative in meta["files"]
                     if not (target / relative).is_file()]

    return {
        "success": True,
        "project_path": str(target),
        "template": template,
        "valid": not missing_dirs and not missing_files,
        "missing_directories": sorted(missing_dirs),
        "missing_files": sorted(missing_files),
    }
