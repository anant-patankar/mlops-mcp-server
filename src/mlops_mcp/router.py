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
    }

_MODULE_META: dict[str, dict[str, Any]] = {
    "fileops": {"tool_count": 14, "description": "Core file operations for ML projects."},
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


def register_tier_one_tools(app: FastMCP) -> None:
    global _APP
    _APP = app

    app.tool(name="file_read")(file_read)
    app.tool(name="file_write")(file_write)
    app.tool(name="file_list")(file_list)
    app.tool(name="file_search")(file_search)
