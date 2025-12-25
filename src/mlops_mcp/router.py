from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fastmcp import FastMCP

from ._types import err

from .fileops import (
    batch_delete, copy_file,
    create_directory, create_file,
    delete_file, file_list,
    file_read, file_search,
    file_write, get_disk_usage,
    get_operation_history, list_files,
    move_file, read_file,
    rename_file, search_file_content,
    search_files, write_file,
    )

from .analysis import (
    analysis_directory, batch_classify, classify_file, compare_directories,
    compare_files,
    detect_model_framework,
    find_duplicate_files,
    get_dataset_stats,
    get_file_info,
    get_notebook_summary,
    storage_report,
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
    "file_read", "file_write", "file_list", "file_search",
    "analysis_directory", "storage_report",
    }

_MODULE_META: dict[str, dict[str, Any]] = {
    "fileops": {"tool_count": 14, "description": "Core file operations for ML projects."},
    "analysis": {"tool_count": 11, "description": "File analysis and ML-aware classification."},
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
        return err(
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
        return err(f"unknown module '{module_name}'")

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

    app.tool(name="file_read")(file_read)
    app.tool(name="file_write")(file_write)
    app.tool(name="file_list")(file_list)
    app.tool(name="file_search")(file_search)
    app.tool(name="analysis_directory")(analysis_directory)
    app.tool(name="storage_report")(storage_report)
