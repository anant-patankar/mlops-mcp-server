import ast
from pathlib import Path
from typing import Any

from .envs import scan_imports
from ._types import err as _fail
from .analysis import analysis_directory
from .datasets import profile_dataset
from .experiments import compare_runs, list_runs
from .pipelines import get_pipeline, visualize_pipeline


def _render(template_text: str, context: dict[str, Any]) -> str:
    try:
        from jinja2 import Template

        return Template(template_text).render(**context)
    except ImportError:
        rendered = template_text
        for key, value in context.items():
            rendered = rendered.replace("{{ " + key + " }}", str(value))
        return rendered


def _write_output(path: Path, content: str) -> dict[str, Any]:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return {"success": True, "output_path": str(path), "bytes_written": len(content.encode("utf-8"))}
    except OSError as exc:
        return _fail(str(exc))


def generate_model_card(
    model_name: str,
    task_type: str,
    dataset: str,
    metrics: dict[str, Any],
    limitations: list[str],
    usage_example: str,
    output_path: str = "MODEL_CARD.md",
) -> dict[str, Any]:
    template = """
# MODEL_CARD

- Model: {{ model_name }}
- Task: {{ task_type }}
- Dataset: {{ dataset }}

## Metrics
{{ metrics_section }}

## Limitations
{{ limitations_section }}

## Usage
{{ usage_example }}
""".strip()

    metrics_section = "\n".join(f"- {k}: {v}" for k, v in metrics.items()) if metrics else "- N/A"
    limitations_section = "\n".join(f"- {item}" for item in limitations) if limitations else "- None provided"
    content = _render(
        template,
        {
            "model_name": model_name,
            "task_type": task_type,
            "dataset": dataset,
            "metrics_section": metrics_section,
            "limitations_section": limitations_section,
            "usage_example": usage_example,
        },
    ) + "\n"

    written = _write_output(Path(output_path).expanduser().resolve(), content)
    if not written.get("success"):
        return written
    return {"success": True, "model_name": model_name, **written}


def generate_dataset_card(
    dataset_path: str,
    output_path: str = "DATASET_CARD.md",
    description: str = "",
    license_name: str = "",
    intended_use: str = "",
) -> dict[str, Any]:
    prof = profile_dataset(dataset_path)
    if not prof.get("success"):
        return prof

    template = """
# DATASET_CARD

- Dataset Path: {{ dataset_path }}
- Rows: {{ rows }}
- Columns: {{ columns }}

## Description
{{ description }}

## License
{{ license_name }}

## Intended Use
{{ intended_use }}

## Schema
{{ schema_section }}

## Null Counts
{{ nulls_section }}
""".strip()

    schema_section = "\n".join(f"- {k}: {v}" for k, v in prof.get("dtypes", {}).items())
    nulls_section = "\n".join(f"- {k}: {v}" for k, v in prof.get("null_counts", {}).items())

    content = _render(
        template,
        {
            "dataset_path": prof.get("path"),
            "rows": prof.get("rows"),
            "columns": prof.get("columns"),
            "description": description or "N/A",
            "license_name": license_name or "N/A",
            "intended_use": intended_use or "N/A",
            "schema_section": schema_section or "- N/A",
            "nulls_section": nulls_section or "- N/A",
        },
    ) + "\n"

    written = _write_output(Path(output_path).expanduser().resolve(), content)
    if not written.get("success"):
        return written
    return {"success": True, **written}


def generate_experiment_report(
    project_path: str,
    output_path: str = "EXPERIMENT_REPORT.md",
) -> dict[str, Any]:
    runs = list_runs(project_path)
    if not runs.get("success"):
        return runs

    run_rows = runs.get("runs", [])
    run_lines = []
    for run in run_rows:
        run_lines.append(
            f"- {run.get('run_id')}: status={run.get('status')}, duration={run.get('duration_seconds')}"
        )

    compare_section = "No comparison available"
    mermaid = ""
    if len(run_rows) >= 2:
        run_ids = [run_rows[0]["run_id"], run_rows[1]["run_id"]]
        compared = compare_runs(project_path, run_ids)
        if compared.get("success"):
            compare_section = str(compared.get("diff", {}))
            mermaid = f"\n```mermaid\ngraph TD\n    run1[{run_ids[0]}] --> compare\n    run2[{run_ids[1]}] --> compare\n```\n"

    content = (
        "# EXPERIMENT_REPORT\n\n"
        f"- Project: {Path(project_path).expanduser().resolve()}\n"
        f"- Run Count: {len(run_rows)}\n\n"
        "## Runs\n"
        + ("\n".join(run_lines) if run_lines else "- No runs")
        + "\n\n## Comparison\n"
        + compare_section
        + "\n"
        + mermaid
    )

    written = _write_output(Path(output_path).expanduser().resolve(), content)
    if not written.get("success"):
        return written
    return {"success": True, **written, "run_count": len(run_rows)}


def generate_pipeline_docs(
    pipeline_path: str,
    output_path: str = "PIPELINE.md",
) -> dict[str, Any]:
    pipeline = get_pipeline(pipeline_path)
    if not pipeline.get("success"):
        return pipeline
    viz = visualize_pipeline(pipeline_path)
    if not viz.get("success"):
        return viz

    stages = pipeline.get("pipeline", {}).get("stages", {})
    stage_lines = []
    for name, spec in stages.items():
        stage_lines.append(
            f"### {name}\n"
            f"- cmd: {spec.get('cmd')}\n"
            f"- deps: {spec.get('deps', [])}\n"
            f"- outs: {spec.get('outs', [])}\n"
        )

    content = (
        "# PIPELINE\n\n"
        f"- Pipeline File: {Path(pipeline_path).expanduser().resolve()}\n"
        f"- Stage Count: {pipeline.get('stage_count')}\n\n"
        "## Stages\n\n"
        + "\n".join(stage_lines)
        + "\n## Graph\n\n```mermaid\n"
        + viz.get("mermaid", "graph TD")
        + "\n```\n"
    )

    written = _write_output(Path(output_path).expanduser().resolve(), content)
    if not written.get("success"):
        return written
    return {"success": True, **written}


def generate_project_readme(project_path: str, output_path: str = "README.md") -> dict[str, Any]:
    from .envs import scan_imports  # noqa: PLC0415

    summary = analysis_directory(project_path)
    if not summary.get("success"):
        return summary
    imports = scan_imports(project_path)
    if not imports.get("success"):
        return imports

    content = (
        "# Project Overview\n\n"
        f"- Project Path: {summary.get('path')}\n"
        f"- Files: {summary.get('file_count')}\n"
        f"- Directories: {summary.get('directory_count')}\n"
        f"- Total Size (bytes): {summary.get('total_size_bytes')}\n\n"
        "## Key Extensions\n"
        + "\n".join([f"- {ext}: {count}" for ext,
                    count in summary.get("by_extension", {}).items()])
        + "\n\n## Third-party Imports\n"
        + ("\n".join([f"- {imp}" for imp in imports.get("imports", [])]) or "- None")
        + "\n"
    )

    written = _write_output(Path(output_path).expanduser().resolve(), content)
    if not written.get("success"):
        return written
    return {"success": True, **written}


def generate_api_docs(source_path: str, output_path: str = "API.md") -> dict[str, Any]:
    root = Path(source_path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return _fail(f"source path not found: {root}")

    entries: list[dict[str, Any]] = []
    for py_file in root.rglob("*.py"):
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue

        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                args = [arg.arg for arg in node.args.args]
                annotation = (
                    ast.unparse(node.returns)
                    if node.returns is not None and hasattr(ast, "unparse")
                    else None
                )
                doc = ast.get_docstring(node)
                entries.append(
                    {
                        "file": str(py_file.relative_to(root)),
                        "function": node.name,
                        "args": args,
                        "returns": annotation,
                        "doc": doc,
                    }
                )

    lines = ["# API Documentation", ""]
    for entry in sorted(entries, key=lambda item: (item["file"], item["function"])):
        lines.append(f"## {entry['function']}")
        lines.append(f"- File: {entry['file']}")
        lines.append(f"- Args: {entry['args']}")
        lines.append(f"- Returns: {entry['returns']}")
        lines.append(f"- Doc: {entry['doc'] or 'N/A'}")
        lines.append("")

    content = "\n".join(lines) + "\n"
    written = _write_output(Path(output_path).expanduser().resolve(), content)
    if not written.get("success"):
        return written
    return {"success": True, **written, "function_count": len(entries)}
