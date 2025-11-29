import pytest
from unittest.mock import MagicMock

import mlops_mcp.router as router_module
from mlops_mcp.router import (
    list_modules,
    activate_module,
    deactivate_module,
    register_tier_one_tools,
    _ACTIVE_MODULES,
    _ACTIVE_MODULE_TOOLS,
    _RUNTIME_REGISTERED_TOOLS,
)


@pytest.fixture(autouse=True)
def reset_router_state():
    _ACTIVE_MODULES.clear()
    _ACTIVE_MODULE_TOOLS.clear()
    _RUNTIME_REGISTERED_TOOLS.clear()
    router_module._APP = None
    yield
    _ACTIVE_MODULES.clear()
    _ACTIVE_MODULE_TOOLS.clear()
    _RUNTIME_REGISTERED_TOOLS.clear()
    router_module._APP = None


@pytest.fixture
def mock_app():
    app = MagicMock()
    app.tool = MagicMock(return_value=lambda fn: fn)
    register_tier_one_tools(app)
    return app


class TestListModules:
    def test_returns_known_modules(self):
        r = list_modules()
        assert "fileops" in r
        assert "analysis" in r

    def test_each_module_has_description_and_count(self):
        for name, meta in list_modules().items():
            assert "description" in meta
            assert "tool_count" in meta
            assert meta["tool_count"] > 0


class TestActivateModule:
    def test_unknown_module(self, mock_app):
        r = activate_module("nonexistent_module")
        assert r["success"] is False
        assert "unknown module" in r["error"]

    def test_known_module_activates(self, mock_app):
        r = activate_module("analysis")
        assert r["success"] is True
        assert "analysis" in r["active_modules"]

    def test_double_activation_does_not_duplicate_tools(self, mock_app):
        activate_module("analysis")
        r = activate_module("analysis")
        # second call should register zero new tools
        assert r["activated_tools"] == []

    def test_response_contains_tool_list(self, mock_app):
        r = activate_module("fileops")
        assert isinstance(r["activated_tools"], list)
        assert len(r["activated_tools"]) > 0


class TestDeactivateModule:
    def test_deactivate_active_module(self, mock_app):
        activate_module("analysis")
        r = deactivate_module("analysis")
        assert r["success"] is True
        assert r["deactivated"] is True
        assert "analysis" not in r["active_modules"]

    def test_deactivate_inactive_module(self, mock_app):
        # deactivating something that was never activated
        r = deactivate_module("analysis")
        assert r["success"] is True
        assert r["deactivated"] is False

    def test_deactivate_unknown_module(self, mock_app):
        r = deactivate_module("imaginary_module")
        assert r["success"] is False


class TestTierOneTools:
    def test_tier_one_registered_on_startup(self, mock_app):
        # tier-one tools are registered when register_tier_one_tools is called
        # mock_app fixture already called it — verify the calls happened
        assert mock_app.tool.called

    def test_tier_one_tools_not_in_runtime_registered(self, mock_app):
        # tier-one tools bypass _try_register_tool, so they
        # should NOT appear in _RUNTIME_REGISTERED_TOOLS
        from mlops_mcp.router import _TIER_ONE_TOOLS
        for tool_name in _TIER_ONE_TOOLS:
            assert tool_name not in _RUNTIME_REGISTERED_TOOLS