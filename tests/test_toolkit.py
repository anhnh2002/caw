"""Unit tests for ToolKit base class and @tool decorator.

These tests do NOT require a live provider — they verify metadata,
discovery, wrapper signatures, and server wiring only.
"""

from __future__ import annotations

import inspect

from mcp.server.fastmcp import Context

from caw import ToolKit, tool
from caw.mcp import MCPServerHandle


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class DummyKit(ToolKit, server_name="dummy", display_name="Dummy Kit"):
    def __init__(self):
        self.data = []

    @tool(description="Do alpha")
    async def alpha(self) -> str:
        return "alpha"

    @tool(name="custom_beta", description="Do beta", title="Beta")
    async def beta(self, x: int) -> str:
        return f"beta-{x}"

    def _private_helper(self):
        """Not a tool."""
        return 42

    def public_but_not_tool(self):
        return "nope"


class CtxKit(ToolKit, server_name="ctxkit"):
    @tool(description="Needs context")
    async def with_ctx(self, value: str, ctx: Context) -> str:
        return value


class SyncKit(ToolKit, server_name="synckit"):
    @tool(description="Sync tool")
    def sync_method(self, n: int) -> int:
        return n * 2


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_tool_decorator_attaches_metadata():
    info = DummyKit.alpha._toolkit_tool_info
    assert info["description"] == "Do alpha"
    assert info["name"] is None  # defaults at registration time


def test_tool_decorator_custom_name():
    info = DummyKit.beta._toolkit_tool_info
    assert info["name"] == "custom_beta"
    assert info["title"] == "Beta"


def test_toolkit_discovers_tools():
    kit = DummyKit()
    handle = kit.as_server()
    assert isinstance(handle, MCPServerHandle)

    # FastMCP stores tools internally — list them
    tool_names = set(handle.server._tool_manager._tools.keys())
    assert "alpha" in tool_names
    assert "custom_beta" in tool_names


def test_toolkit_ignores_non_tool_methods():
    kit = DummyKit()
    handle = kit.as_server()
    tool_names = set(handle.server._tool_manager._tools.keys())
    assert "_private_helper" not in tool_names
    assert "public_but_not_tool" not in tool_names


def test_toolkit_server_has_correct_tool_count():
    kit = DummyKit()
    handle = kit.as_server()
    tool_names = set(handle.server._tool_manager._tools.keys())
    assert len(tool_names) == 2


def test_toolkit_wrapper_signature():
    """Generated wrapper should have no 'self' and should end with ctx: Context."""
    kit = DummyKit()
    handle = kit.as_server()
    beta_tool = handle.server._tool_manager._tools["custom_beta"]
    sig = inspect.signature(beta_tool.fn)
    param_names = list(sig.parameters.keys())
    assert "self" not in param_names
    assert "x" in param_names
    assert "ctx" in param_names
    assert param_names[-1] == "ctx"


def test_toolkit_as_server_default_id():
    kit = DummyKit()
    handle = kit.as_server()
    assert handle.server_id.startswith("dummy_")


def test_toolkit_as_server_custom_id():
    kit = DummyKit()
    handle = kit.as_server(server_id="my_custom_id")
    assert handle.server_id == "my_custom_id"


def test_toolkit_ctx_passthrough_signature():
    """If a method declares ctx: Context, wrapper should still have ctx at the end."""
    kit = CtxKit()
    handle = kit.as_server()
    wc_tool = handle.server._tool_manager._tools["with_ctx"]
    sig = inspect.signature(wc_tool.fn)
    param_names = list(sig.parameters.keys())
    assert "self" not in param_names
    assert "value" in param_names
    assert "ctx" in param_names


def test_toolkit_sync_method_registered():
    """Sync methods should also be registered as tools."""
    kit = SyncKit()
    handle = kit.as_server()
    tool_names = set(handle.server._tool_manager._tools.keys())
    assert "sync_method" in tool_names


def test_toolkit_class_vars_inherited():
    """Subclass should inherit _server_name and _display_name."""
    assert DummyKit._server_name == "dummy"
    assert DummyKit._display_name == "Dummy Kit"


def test_toolkit_class_vars_default():
    """ToolKit without explicit server_name uses empty string."""

    class Bare(ToolKit):
        @tool(description="bare tool")
        async def bare(self) -> str:
            return "bare"

    assert Bare._server_name == ""
    kit = Bare()
    handle = kit.as_server()
    # Falls back to class name
    assert handle.server_id.startswith("Bare_")
