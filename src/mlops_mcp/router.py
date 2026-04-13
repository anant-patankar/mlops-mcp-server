from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fastmcp import FastMCP

from ._types import err as _fail

from .analysis import (
    analysis_directory,
    batch_classify,
    classify_file,
    compare_directories,
    compare_files,
    detect_model_framework,
    find_duplicate_files,
    get_dataset_stats,
    get_file_info,
    get_notebook_summary,
    storage_report,
)
from .archive import archive_experiment, create_archive, extract_archive, list_archive_contents
from .cleanup import (
    cleanup_empty_logs,
    cleanup_failed_runs,
    cleanup_old_checkpoints,
    cleanup_project,
)
from .datasets import (
    check_data_freshness,
    detect_data_drift,
    find_dataset_files,
    generate_dataset_card as datasets_generate_dataset_card,
    merge_datasets,
    profile_dataset,
    split_dataset,
    validate_dataset_schema,
)
from .docgen import (
    generate_api_docs,
    generate_dataset_card as docgen_generate_dataset_card,
    generate_experiment_report,
    generate_model_card,
    generate_pipeline_docs,
    generate_project_readme,
)
from .dvc_ops import (
    create_dvc_pipeline,
    dvc_add,
    dvc_check_available,
    dvc_init,
    dvc_pull,
    dvc_push,
    dvc_repro,
    dvc_status,
)
from .envs import (
    check_dependency_conflicts,
    check_env_vars,
    compare_requirements,
    create_conda_env_file,
    create_env_template,
    generate_requirements,
    get_python_version,
    scan_imports,
)
from .experiments import (
    compare_runs,
    create_run,
    delete_run,
    experiment_list_runs,
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
from .fileops import (
    batch_delete,
    copy_file,
    create_directory,
    create_file,
    delete_file,
    file_list,
    file_read,
    file_search,
    file_write,
    get_disk_usage,
    get_operation_history,
    list_files,
    move_file,
    read_file,
    rename_file,
    search_file_content,
    search_files,
    write_file,
)
from .git_ops import (
    create_gitignore,
    detect_uncommitted_changes,
    git_add,
    git_commit,
    git_init,
    git_log,
    git_status,
)
from .lineage import (
    check_lineage_integrity,
    get_artifact_provenance,
    list_lineage_artifacts,
    record_lineage,
    visualize_lineage,
)
from .mlflow_ops import (
    download_mlflow_artifact,
    get_mlflow_model_versions,
    get_mlflow_runs,
    list_mlflow_experiments,
    log_artifact_to_mlflow,
    mlflow_check_available,
    register_model_in_mlflow,
    set_mlflow_tracking_uri,
)
from .models import (
    compare_model_versions,
    create_model_card,
    delete_model_version,
    deprecate_model,
    get_model_info,
    get_model_lineage,
    get_model_versions,
    init_model_registry,
    list_models,
    model_list,
    promote_model,
    register_model,
    tag_model,
)
from .pipelines import (
    add_pipeline_stage,
    create_pipeline,
    get_pipeline,
    get_pipeline_status,
    list_pipelines,
    remove_pipeline_stage,
    validate_pipeline,
    visualize_pipeline,
)
from .projects import (
    add_project_component,
    create_ml_project,
    list_project_templates,
    validate_project_structure,
)


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    anti_confusion_tags: tuple[str, ...]
    handler: Callable[..., dict[str, Any]]


_ACTIVE_MODULES: set[str] = set()
_ACTIVE_MODULE_TOOLS: dict[str, set[str]] = {}
_RUNTIME_REGISTERED_TOOLS: set[str] = set()
_APP: FastMCP | None = None
_TIER_ONE_TOOLS = {
    "mlops_discover",
    "mlops_activate",
    "mlops_session_status",
    "file_read",
    "file_write",
    "file_list",
    "file_search",
    "analysis_directory",
    "git_status",
    "experiment_list_runs",
    "model_list",
    "storage_report",
}

_MODULE_META: dict[str, dict[str, Any]] = {
    "fileops": {"tool_count": 14, "description": "Core file operations for ML projects."},
    "analysis": {"tool_count": 11, "description": "File analysis and ML-aware classification."},
    "projects": {"tool_count": 4, "description": "Project scaffolding and structure checks."},
    "archive": {"tool_count": 4, "description": "Archive creation, inspection, and extraction."},
    "cleanup": {"tool_count": 4, "description": "Project cleanup and stale artifact removal."},
    "experiments": {"tool_count": 12, "description": "Experiment tracking and run comparison."},
    "datasets": {"tool_count": 8, "description": "Dataset profiling, validation, and utilities."},
    "models": {"tool_count": 12, "description": "Model metadata, conversion, and validation helpers."},
    "git": {"tool_count": 7, "description": "Git-aware workflow operations for ML repos."},
    "dvc": {"tool_count": 8, "description": "Data version control workflow operations."},
    "mlflow": {"tool_count": 8, "description": "MLflow tracking and registry wrappers."},
    "pipelines": {"tool_count": 8, "description": "Pipeline orchestration and execution helpers."},
    "envs": {"tool_count": 8, "description": "Environment inspection and reproducibility tools."},
    "lineage": {"tool_count": 5, "description": "Data and model lineage tracking tools."},
    "docgen": {"tool_count": 6, "description": "Documentation and report generation helpers."},
}

TOOL_REGISTRY: dict[str, list[ToolDefinition]] = {
    "fileops": [
        ToolDefinition("create_file", "Creates a file with content. Use for: new file creation. Do NOT use for: partial file edits.",
                       ("file-create",), create_file),
        ToolDefinition(
            "read_file", "Reads a file. Use for: loading file contents. Do NOT use for: directory listing.", ("file-read",), read_file),
        ToolDefinition(
            "write_file", "Overwrites file content. Use for: replacing whole files. Do NOT use for: append-only logs.", ("file-write",), write_file),
        ToolDefinition("delete_file", "Deletes a file with optional dry run. Use for: file cleanup. Do NOT use for: removing directories.",
                       ("file-delete",), delete_file),
        ToolDefinition(
            "move_file", "Moves file paths. Use for: relocations. Do NOT use for: copying while keeping source.", ("file-move",), move_file),
        ToolDefinition(
            "copy_file", "Copies files. Use for: duplicating artifacts. Do NOT use for: renaming same file.", ("file-copy",), copy_file),
        ToolDefinition("create_directory", "Creates directories recursively. Use for: folder scaffolding. Do NOT use for: file writes.",
                       ("dir-create",), create_directory),
        ToolDefinition("rename_file", "Renames file paths. Use for: name normalization. Do NOT use for: content edits.",
                       ("file-rename",), rename_file),
        ToolDefinition(
            "list_files", "Lists files in a directory. Use for: browsing trees. Do NOT use for: full-text search.", ("file-list",), list_files),
        ToolDefinition("search_files", "Finds files by patterns. Use for: matching names. Do NOT use for: searching inside file text.",
                       ("file-search",), search_files),
        ToolDefinition("search_file_content", "Searches text in files. Use for: locating strings. Do NOT use for: binary data scan.",
                       ("content-search",), search_file_content),
        ToolDefinition("get_disk_usage", "Reports disk usage under a path. Use for: storage checks. Do NOT use for: duplicate analysis.",
                       ("disk-usage",), get_disk_usage),
        ToolDefinition("batch_delete", "Deletes multiple files with optional dry run. Use for: cleanup batches. Do NOT use for: whole directory wipes.",
                       ("batch-delete",), batch_delete),
        ToolDefinition("get_operation_history", "Returns file operation history. Use for: audit trace. Do NOT use for: git history.",
                       ("audit",), get_operation_history),
    ],
    "analysis": [
        ToolDefinition("get_file_info", "Returns file metadata and checksum. Use for: file inspection. Do NOT use for: directory summaries.",
                       ("file-info",), get_file_info),
        ToolDefinition("classify_file", "Classifies file by ML-aware type. Use for: routing workflows. Do NOT use for: content validation.",
                       ("file-classify",), classify_file),
        ToolDefinition("analysis_directory", "Summarizes directory composition. Use for: project scans. Do NOT use for: exact lineage tracing.",
                       ("dir-summary",), analysis_directory),
        ToolDefinition("find_duplicate_files", "Finds duplicate files by hash. Use for: storage cleanup prep. Do NOT use for: near-duplicate detection.",
                       ("dedupe",), find_duplicate_files),
        ToolDefinition("storage_report", "Reports largest files and storage use. Use for: disk planning. Do NOT use for: git diffing.",
                       ("storage",), storage_report),
        ToolDefinition("batch_classify", "Classifies many file paths in one call. Use for: bulk audits. Do NOT use for: mutation operations.",
                       ("bulk-classify",), batch_classify),
        ToolDefinition("get_dataset_stats", "Profiles one dataset file. Use for: training readiness checks. Do NOT use for: schema contract checks.",
                       ("dataset-stats",), get_dataset_stats),
        ToolDefinition("detect_model_framework", "Detects model framework from file. Use for: runner selection. Do NOT use for: model quality assessment.",
                       ("framework-detect",), detect_model_framework),
        ToolDefinition("get_notebook_summary", "Summarizes notebook structure. Use for: notebook audits. Do NOT use for: executing notebook code.",
                       ("notebook-summary",), get_notebook_summary),
        ToolDefinition("compare_files", "Diffs two text files. Use for: config comparisons. Do NOT use for: binary model files.",
                       ("file-diff",), compare_files),
        ToolDefinition("compare_directories", "Compares two directory trees by content. Use for: output comparisons. Do NOT use for: lineage graphing.",
                       ("dir-diff",), compare_directories),
    ],
    "projects": [
        ToolDefinition(
            name="create_ml_project",
            description=(
                "Scaffolds a new project from a named template. Use for: project"
                " bootstrap. Do NOT use for: adding components to existing projects."
            ),
            anti_confusion_tags=("scaffold", "template", "project-setup"),
            handler=create_ml_project,
        ),
        ToolDefinition(
            name="list_project_templates",
            description=(
                "Lists available project templates and metadata. Use for: choosing"
                " a template before scaffolding. Do NOT use for: validating an"
                " existing project structure."
            ),
            anti_confusion_tags=("template-discovery", "project-scaffold"),
            handler=list_project_templates,
        ),
        ToolDefinition(
            name="add_project_component",
            description=(
                "Adds a preset component into an existing project. Use for: adding"
                " api/monitoring/tests folders. Do NOT use for: creating a full"
                " project from scratch."
            ),
            anti_confusion_tags=("project-extension", "component"),
            handler=add_project_component,
        ),
        ToolDefinition(
            name="validate_project_structure",
            description=(
                "Checks a project against an expected template layout. Use for:"
                " CI preflight structure checks. Do NOT use for: linting code"
                " or running tests."
            ),
            anti_confusion_tags=("validation", "project-layout"),
            handler=validate_project_structure,
        ),
    ],
    "archive": [
        ToolDefinition(
            name="create_archive",
            description=(
                "Creates a zip or tar archive from files/directories. Use for:"
                " packaging model artifacts. Do NOT use for: inspecting archive"
                " content without extraction."
            ),
            anti_confusion_tags=("packaging", "artifacts"),
            handler=create_archive,
        ),
        ToolDefinition(
            name="extract_archive",
            description=(
                "Extracts archive contents to a destination with path safety checks."
                " Use for: unpacking shared artifacts. Do NOT use for: listing"
                " archive contents only."
            ),
            anti_confusion_tags=("unpack", "archive-safety"),
            handler=extract_archive,
        ),
        ToolDefinition(
            name="list_archive_contents",
            description=(
                "Lists files in an archive without extracting. Use for: quick"
                " inspection before unpacking. Do NOT use for: copying files out"
                " of archive."
            ),
            anti_confusion_tags=("archive-inspection", "read-only"),
            handler=list_archive_contents,
        ),
        ToolDefinition(
            name="archive_experiment",
            description=(
                "Creates timestamped archive + manifest from experiment directory."
                " Use for: run-end snapshots. Do NOT use for: partial file copy"
                " operations."
            ),
            anti_confusion_tags=("experiment-snapshot", "manifest"),
            handler=archive_experiment,
        ),
    ],
    "cleanup": [
        ToolDefinition(
            name="cleanup_project",
            description=(
                "Removes common temporary/cache artifacts from project tree. Use for:"
                " housekeeping before commit. Do NOT use for: deleting arbitrary"
                " user-selected files."
            ),
            anti_confusion_tags=("cleanup", "project-hygiene"),
            handler=cleanup_project,
        ),
        ToolDefinition(
            name="cleanup_old_checkpoints",
            description=(
                "Prunes old checkpoints while keeping newest N. Use for: controlling"
                " model artifact growth. Do NOT use for: selecting best model by"
                " metrics."
            ),
            anti_confusion_tags=("checkpoint-prune", "storage-control"),
            handler=cleanup_old_checkpoints,
        ),
        ToolDefinition(
            name="cleanup_failed_runs",
            description=(
                "Removes run directories missing success markers. Use for: deleting"
                " failed/incomplete runs. Do NOT use for: active in-progress runs."
            ),
            anti_confusion_tags=("run-cleanup", "failed-runs"),
            handler=cleanup_failed_runs,
        ),
        ToolDefinition(
            name="cleanup_empty_logs",
            description=(
                "Deletes empty or near-empty log files. Use for: removing noise"
                " logs after failed runs. Do NOT use for: general log retention"
                " policy management."
            ),
            anti_confusion_tags=("log-cleanup", "garbage-logs"),
            handler=cleanup_empty_logs,
        ),
    ],
    "experiments": [
        ToolDefinition("init_experiment_tracker", "Initializes file-based experiment tracking. Use for: first-time setup. Do NOT use for: creating a run directly.",
                       ("exp-init",), init_experiment_tracker),
        ToolDefinition(
            "create_run", "Creates a new experiment run. Use for: run start. Do NOT use for: updating existing run metadata.", ("run-create",), create_run),
        ToolDefinition(
            "log_params", "Logs run parameters. Use for: hyperparameter capture. Do NOT use for: metric timeline updates.", ("params",), log_params),
        ToolDefinition(
            "log_metrics", "Logs run metrics by step. Use for: epoch metrics. Do NOT use for: static run params.", ("metrics",), log_metrics),
        ToolDefinition(
            "log_artifact", "Copies artifact into run storage. Use for: model/plot attachment. Do NOT use for: mlflow-only logging.", ("artifact",), log_artifact),
        ToolDefinition(
            "finish_run", "Marks run success or failure. Use for: run completion. Do NOT use for: deleting runs.", ("run-finish",), finish_run),
        ToolDefinition(
            "get_run", "Returns full run metadata. Use for: run inspection. Do NOT use for: global run inventory.", ("run-get",), get_run),
        ToolDefinition(
            "list_runs", "Lists run summaries. Use for: overview and selection. Do NOT use for: deep param diff alone.", ("run-list",), list_runs),
        ToolDefinition("compare_runs", "Compares multiple runs. Use for: run delta analysis. Do NOT use for: single-run read.",
                       ("run-compare",), compare_runs),
        ToolDefinition(
            "get_best_run", "Finds best run for a metric. Use for: winner selection. Do NOT use for: multi-objective ranking.", ("best-run",), get_best_run),
        ToolDefinition(
            "delete_run", "Deletes a run with optional dry run. Use for: cleanup. Do NOT use for: archiving runs.", ("run-delete",), delete_run),
        ToolDefinition("export_runs_csv", "Exports run summaries to CSV. Use for: reporting. Do NOT use for: in-memory comparisons only.",
                       ("run-export",), export_runs_csv),
    ],
    "datasets": [
        ToolDefinition("profile_dataset", "Profiles dataset stats and schema. Use for: pre-training checks. Do NOT use for: model file inspection.",
                       ("dataset-profile",), profile_dataset),
        ToolDefinition("validate_dataset_schema", "Validates dataset against schema file. Use for: contract enforcement. Do NOT use for: drift analysis.",
                       ("dataset-validate",), validate_dataset_schema),
        ToolDefinition("detect_data_drift", "Compares column distributions between reference and candidate datasets using KS-test for numeric columns and frequency distance for categorical. Use for: retraining decisions after new data arrives. Do NOT use for: schema contract validation.",
                       ("dataset-drift",), detect_data_drift),
        ToolDefinition("split_dataset", "Splits dataset into train/val/test. Use for: data prep. Do NOT use for: dataset merge.",
                       ("dataset-split",), split_dataset),
        ToolDefinition("merge_datasets", "Merges multiple datasets to one file. Use for: consolidation. Do NOT use for: splitting.",
                       ("dataset-merge",), merge_datasets),
        ToolDefinition("check_data_freshness", "Checks dataset age against threshold. Use for: stale-data checks. Do NOT use for: drift stats.",
                       ("dataset-freshness",), check_data_freshness),
        ToolDefinition("generate_dataset_card", "Generates dataset markdown card. Use for: dataset documentation. Do NOT use for: model cards.",
                       ("dataset-card",), datasets_generate_dataset_card),
        ToolDefinition("find_dataset_files", "Finds dataset files in a tree. Use for: dataset inventory. Do NOT use for: full fileops search.",
                       ("dataset-find",), find_dataset_files),
    ],
    "models": [
        ToolDefinition("init_model_registry", "Initializes model registry files. Use for: registry setup. Do NOT use for: model listing from filesystem only.",
                       ("model-init",), init_model_registry),
        ToolDefinition("register_model", "Registers model artifact in registry. Use for: tracking versions. Do NOT use for: stage promotion only.",
                       ("model-register",), register_model),
        ToolDefinition("list_models", "Lists registry model names/versions. Use for: model inventory. Do NOT use for: detailed single version info.",
                       ("model-list",), list_models),
        ToolDefinition("get_model_versions", "Returns all versions for a model. Use for: version history. Do NOT use for: cross-model comparison.",
                       ("model-versions",), get_model_versions),
        ToolDefinition("get_model_info", "Returns detailed model version metadata. Use for: pre-deploy checks. Do NOT use for: run listing.",
                       ("model-info",), get_model_info),
        ToolDefinition("promote_model", "Moves model version to a new stage. Use for: release workflow. Do NOT use for: deprecate with reason.",
                       ("model-promote",), promote_model),
        ToolDefinition(
            "tag_model", "Adds tags to model version. Use for: metadata enrichment. Do NOT use for: stage transition.", ("model-tag",), tag_model),
        ToolDefinition("compare_model_versions", "Compares two versions of same model. Use for: promotion decisions. Do NOT use for: deleting versions.",
                       ("model-compare",), compare_model_versions),
        ToolDefinition("deprecate_model", "Marks model version deprecated with reason. Use for: retirement workflow. Do NOT use for: hard delete.",
                       ("model-deprecate",), deprecate_model),
        ToolDefinition("delete_model_version", "Deletes model version and optional file. Use for: irreversible cleanup. Do NOT use for: soft retirement.",
                       ("model-delete",), delete_model_version),
        ToolDefinition("get_model_lineage", "Returns model-to-run lineage data. Use for: provenance checks. Do NOT use for: run metric ranking.",
                       ("model-lineage",), get_model_lineage),
        ToolDefinition("create_model_card", "Generates model markdown card. Use for: documentation. Do NOT use for: registry mutation.",
                       ("model-card",), create_model_card),
    ],
    "git": [
        ToolDefinition(
            "git_init", "Initializes git repo. Use for: VCS setup. Do NOT use for: commit history reads.", ("git-init",), git_init),
        ToolDefinition(
            "git_status", "Returns repo changed state. Use for: pre-commit checks. Do NOT use for: commit creation.", ("git-status",), git_status),
        ToolDefinition(
            "git_add", "Stages files for commit. Use for: preparing commits. Do NOT use for: committing directly.", ("git-add",), git_add),
        ToolDefinition(
            "git_commit", "Creates a commit with message. Use for: recording changes. Do NOT use for: staging files.", ("git-commit",), git_commit),
        ToolDefinition(
            "git_log", "Lists recent commits with hashes and messages.", ("git-log",), git_log),
        ToolDefinition("create_gitignore", "Writes ML-aware .gitignore. Use for: repo bootstrap. Do NOT use for: dependency lockfiles.",
                       ("gitignore",), create_gitignore),
        ToolDefinition("detect_uncommitted_changes", "Flags dirty repository state. Use for: guardrails before runs. Do NOT use for: branch operations.",
                       ("git-dirty",), detect_uncommitted_changes),
    ],
    "dvc": [
        ToolDefinition(
            "dvc_init", "Initializes DVC in repo. Use for: DVC setup. Do NOT use for: data push/pull.", ("dvc-init",), dvc_init),
        ToolDefinition(
            "dvc_add", "Tracks data with DVC. Use for: dataset versioning. Do NOT use for: git staging.", ("dvc-add",), dvc_add),
        ToolDefinition(
            "dvc_push", "Pushes DVC data to remote. Use for: remote sync upload. Do NOT use for: local staging.", ("dvc-push",), dvc_push),
        ToolDefinition(
            "dvc_pull", "Pulls DVC data from remote. Use for: reproducing data state. Do NOT use for: dependency scans.", ("dvc-pull",), dvc_pull),
        ToolDefinition(
            "dvc_status", "Reports changed DVC stages/artifacts. Use for: stale checks. Do NOT use for: executing pipeline.", ("dvc-status",), dvc_status),
        ToolDefinition(
            "dvc_repro", "Runs DVC repro. Use for: stale stage recompute. Do NOT use for: pipeline file creation.", ("dvc-repro",), dvc_repro),
        ToolDefinition("create_dvc_pipeline", "Writes dvc.yaml from stage config. Use for: pipeline definition. Do NOT use for: pipeline execution.",
                       ("dvc-pipeline",), create_dvc_pipeline),
        ToolDefinition("dvc_check_available", "Checks DVC CLI availability/version. Use for: preflight checks. Do NOT use for: data operations.",
                       ("dvc-check",), dvc_check_available),
    ],
    "mlflow": [
        ToolDefinition("set_mlflow_tracking_uri", "Sets MLflow tracking URI. Use for: target config. Do NOT use for: listing runs.",
                       ("mlflow-uri",), set_mlflow_tracking_uri),
        ToolDefinition("list_mlflow_experiments", "Lists MLflow experiments. Use for: inventory. Do NOT use for: run-level metrics analysis.",
                       ("mlflow-experiments",), list_mlflow_experiments),
        ToolDefinition("get_mlflow_runs", "Fetches runs for one experiment. Use for: run comparisons. Do NOT use for: model registry versions.",
                       ("mlflow-runs",), get_mlflow_runs),
        ToolDefinition("log_artifact_to_mlflow", "Logs artifact file to MLflow. Use for: artifact tracking. Do NOT use for: file copy to local run dir.",
                       ("mlflow-artifact",), log_artifact_to_mlflow),
        ToolDefinition("download_mlflow_artifact", "Downloads MLflow artifact locally. Use for: retrieval for eval. Do NOT use for: local-only files.",
                       ("mlflow-download",), download_mlflow_artifact),
        ToolDefinition("register_model_in_mlflow", "Registers model in MLflow registry. Use for: MLflow release tracking. Do NOT use for: file-based registry only.",
                       ("mlflow-register",), register_model_in_mlflow),
        ToolDefinition("get_mlflow_model_versions", "Lists MLflow model versions. Use for: stage visibility. Do NOT use for: local model files scan.",
                       ("mlflow-versions",), get_mlflow_model_versions),
        ToolDefinition("mlflow_check_available", "Checks MLflow availability and URI. Use for: preflight. Do NOT use for: artifact operations.",
                       ("mlflow-check",), mlflow_check_available),
    ],
    "pipelines": [
        ToolDefinition("create_pipeline", "Creates pipeline yaml. Use for: pipeline bootstrap. Do NOT use for: adding one stage only.",
                       ("pipeline-create",), create_pipeline),
        ToolDefinition("get_pipeline", "Reads pipeline yaml. Use for: inspection. Do NOT use for: validation side effects.",
                       ("pipeline-get",), get_pipeline),
        ToolDefinition("validate_pipeline", "Validates stage deps and cycles. Use for: CI checks. Do NOT use for: running pipeline.",
                       ("pipeline-validate",), validate_pipeline),
        ToolDefinition("add_pipeline_stage", "Adds stage to existing pipeline. Use for: iterative updates. Do NOT use for: deleting stage.",
                       ("pipeline-add-stage",), add_pipeline_stage),
        ToolDefinition("remove_pipeline_stage", "Removes one stage from pipeline. Use for: deprecating steps. Do NOT use for: adding stage.",
                       ("pipeline-remove-stage",), remove_pipeline_stage),
        ToolDefinition("get_pipeline_status", "Checks stale/up-to-date per stage. Use for: run planning. Do NOT use for: dependency installation.",
                       ("pipeline-status",), get_pipeline_status),
        ToolDefinition("list_pipelines", "Lists pipeline files in project. Use for: inventory. Do NOT use for: stage-level metadata.",
                       ("pipeline-list",), list_pipelines),
        ToolDefinition("visualize_pipeline", "Generates Mermaid for pipeline DAG. Use for: visualization. Do NOT use for: pipeline mutation.",
                       ("pipeline-visualize",), visualize_pipeline),
    ],
    "envs": [
        ToolDefinition(
            "scan_imports", "Scans project imports statically. Use for: dependency discovery. Do NOT use for: runtime package conflicts.", ("env-scan",), scan_imports),
        ToolDefinition("generate_requirements", "Generates requirements from imports. Use for: bootstrap dependency files. Do NOT use for: diffing two files.",
                       ("env-requirements",), generate_requirements),
        ToolDefinition("compare_requirements", "Diffs two requirements files. Use for: env snapshot comparison. Do NOT use for: install operations.",
                       ("env-compare",), compare_requirements),
        ToolDefinition("check_dependency_conflicts", "Runs pip check for conflicts. Use for: pre-deploy sanity. Do NOT use for: dependency lock generation.",
                       ("env-conflicts",), check_dependency_conflicts),
        ToolDefinition("create_conda_env_file", "Creates environment.yml from requirements. Use for: conda migration. Do NOT use for: pip install.",
                       ("env-conda",), create_conda_env_file),
        ToolDefinition("create_env_template", "Creates .env template from variable names. Use for: secret docs. Do NOT use for: loading envs.",
                       ("env-template",), create_env_template),
        ToolDefinition("check_env_vars", "Checks required env vars presence. Use for: runtime preflight. Do NOT use for: generating template.",
                       ("env-check",), check_env_vars),
        ToolDefinition("get_python_version", "Reports python version and compatibility. Use for: runtime checks. Do NOT use for: installing python.",
                       ("env-python",), get_python_version),
    ],
    "lineage": [
        ToolDefinition("record_lineage", "Records artifact lineage edge. Use for: provenance capture. Do NOT use for: visualization directly.",
                       ("lineage-record",), record_lineage),
        ToolDefinition("get_artifact_provenance", "Traces artifact ancestry. Use for: root-cause tracing. Do NOT use for: file deletion checks.",
                       ("lineage-trace",), get_artifact_provenance),
        ToolDefinition("list_lineage_artifacts", "Lists known lineage artifacts. Use for: audit inventory. Do NOT use for: deep traversal.",
                       ("lineage-list",), list_lineage_artifacts),
        ToolDefinition("visualize_lineage", "Generates a Mermaid diagram tracing artifact ancestry. Use for: provenance visualization in Claude. Do NOT use for: integrity validation — use check_lineage_integrity for that.",
                       ("lineage-visualize",), visualize_lineage),
        ToolDefinition("check_lineage_integrity", "Checks lineage inputs exist. Use for: broken edge detection. Do NOT use for: creating records.",
                       ("lineage-integrity",), check_lineage_integrity),
    ],
    "docgen": [
        ToolDefinition("generate_model_card", "Generates model card markdown. Use for: model documentation. Do NOT use for: registry updates.",
                       ("doc-model",), generate_model_card),
        ToolDefinition("generate_dataset_card", "Generates dataset card markdown. Use for: data documentation. Do NOT use for: dataset mutation.",
                       ("doc-dataset",), docgen_generate_dataset_card),
        ToolDefinition("generate_experiment_report", "Generates experiment report markdown. Use for: run summaries. Do NOT use for: run deletion.",
                       ("doc-experiment",), generate_experiment_report),
        ToolDefinition("generate_pipeline_docs", "Generates pipeline docs with Mermaid. Use for: pipeline documentation. Do NOT use for: pipeline execution.",
                       ("doc-pipeline",), generate_pipeline_docs),
        ToolDefinition("generate_project_readme", "Generates project README draft. Use for: bootstrap docs. Do NOT use for: dependency installs.",
                       ("doc-readme",), generate_project_readme),
        ToolDefinition("generate_api_docs", "Generates API markdown from source AST. Use for: API reference. Do NOT use for: code generation.",
                       ("doc-api",), generate_api_docs),
    ],
}


def _ok(**payload: Any) -> dict[str, Any]:
    return {"success": True, **payload}


def list_modules() -> dict[str, dict[str, Any]]:
    return {
        name: {
            "description": meta["description"],
            "tool_count": meta["tool_count"],
        }
        for name, meta in _MODULE_META.items()
    }


def _try_register_tool(tool: ToolDefinition) -> bool:
    if _APP is None:
        return False
    if tool.name in _TIER_ONE_TOOLS:
        return False
    if tool.name in _RUNTIME_REGISTERED_TOOLS:
        return False

    _APP.tool(name=tool.name)(tool.handler)
    _RUNTIME_REGISTERED_TOOLS.add(tool.name)
    return True


def activate_module(module_name: str) -> dict[str, Any]:
    if module_name not in _MODULE_META:
        valid_modules = sorted(_MODULE_META)
        return _fail(
            f"unknown module '{module_name}'. Use one of: {', '.join(valid_modules)}"
        )

    tools = TOOL_REGISTRY.get(module_name, [])
    activated_tools: list[str] = []
    for tool in tools:
        if _try_register_tool(tool):
            activated_tools.append(tool.name)

    _ACTIVE_MODULES.add(module_name)
    _ACTIVE_MODULE_TOOLS[module_name] = {tool.name for tool in tools}

    return _ok(
        module=module_name,
        tool_count=_MODULE_META[module_name]["tool_count"],
        activated_tools=sorted(activated_tools),
        active_modules=sorted(_ACTIVE_MODULES),
    )


def deactivate_module(module_name: str) -> dict[str, Any]:
    if module_name not in _MODULE_META:
        return _fail(f"unknown module '{module_name}'")

    if module_name not in _ACTIVE_MODULES:
        return _ok(module=module_name, deactivated=False, active_modules=sorted(_ACTIVE_MODULES))

    removed_tools = sorted(_ACTIVE_MODULE_TOOLS.pop(module_name, set()))
    _ACTIVE_MODULES.remove(module_name)

    # FastMCP does not currently expose stable runtime unregistration in this setup.
    for tool_name in removed_tools:
        _RUNTIME_REGISTERED_TOOLS.discard(tool_name)

    return _ok(
        module=module_name,
        deactivated=True,
        removed_tools=removed_tools,
        active_modules=sorted(_ACTIVE_MODULES),
    )


def mlops_discover() -> dict[str, Any]:
    modules = list_modules()
    return _ok(modules=modules, active_modules=sorted(_ACTIVE_MODULES))


def mlops_activate(module: str) -> dict[str, Any]:
    return activate_module(module)


def mlops_session_status() -> dict[str, Any]:
    return _ok(
        status="bootstrapped",
        active_modules=sorted(_ACTIVE_MODULES),
        active_module_count=len(_ACTIVE_MODULES),
        runtime_registered_tool_count=len(_RUNTIME_REGISTERED_TOOLS),
    )


def reset_session_state() -> None:
    _ACTIVE_MODULES.clear()
    _ACTIVE_MODULE_TOOLS.clear()
    _RUNTIME_REGISTERED_TOOLS.clear()


def register_tier_one_tools(app: FastMCP) -> None:
    global _APP
    _APP = app

    app.tool(name="mlops_discover")(mlops_discover)
    app.tool(name="mlops_activate")(mlops_activate)
    app.tool(name="mlops_session_status")(mlops_session_status)
    app.tool(name="file_read")(file_read)
    app.tool(name="file_write")(file_write)
    app.tool(name="file_list")(file_list)
    app.tool(name="file_search")(file_search)
    app.tool(name="analysis_directory")(analysis_directory)
    app.tool(name="git_status")(git_status)
    app.tool(name="experiment_list_runs")(experiment_list_runs)
    app.tool(name="model_list")(model_list)
    app.tool(name="storage_report")(storage_report)

    for module_tools in TOOL_REGISTRY.values():
        for tool in module_tools:
            if tool.name not in _TIER_ONE_TOOLS:
                app.tool(name=tool.name)(tool.handler)
                _RUNTIME_REGISTERED_TOOLS.add(tool.name)
