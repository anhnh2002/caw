"""caw - Coding Agent Wrapper."""

__version__ = "0.1.0"

from caw.agent import Agent, Session, register_provider
from caw.display import Display, DisplayMode, get_global_display, set_global_display
from caw.storage import JsonlWriter, SessionStore
from caw.models import (
    AgentSpec,
    ContentBlock,
    MCPServer,
    MCPTool,
    TextBlock,
    ThinkingBlock,
    ToolGroup,
    ToolUse,
    Trajectory,
    Turn,
    UsageStats,
)
from caw.provider import Provider, ProviderSession
from caw.providers.claude_code import ClaudeCodeProvider
from caw.providers.codex import CodexProvider
from caw.mcp import (
    MCPServerHandle,
    create_mcp_http_server_bundle,
    create_subagent_tool_server,
    get_state_from_context,
    mcp_tool,
    register_tool,
)
from caw.toolkit import ToolKit, tool

# Auto-register built-in providers
register_provider("claude_code", ClaudeCodeProvider)
register_provider("codex", CodexProvider)

__all__ = [
    "Agent",
    "AgentSpec",
    "ClaudeCodeProvider",
    "CodexProvider",
    "JsonlWriter",
    "ContentBlock",
    "Display",
    "DisplayMode",
    "get_global_display",
    "set_global_display",
    "MCPServer",
    "MCPServerHandle",
    "MCPTool",
    "Provider",
    "ProviderSession",
    "Session",
    "SessionStore",
    "TextBlock",
    "ThinkingBlock",
    "ToolGroup",
    "ToolUse",
    "Trajectory",
    "Turn",
    "UsageStats",
    "create_mcp_http_server_bundle",
    "create_subagent_tool_server",
    "get_state_from_context",
    "mcp_tool",
    "register_provider",
    "register_tool",
    "tool",
    "ToolKit",
]
