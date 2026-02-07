"""caw - Coding Agent Wrapper."""

__version__ = "0.1.0"

from caw.agent import Agent, Session, register_provider
from caw.display import Display, DisplayMode
from caw.storage import SessionStore
from caw.models import (
    ContentBlock,
    MCPServer,
    MCPTool,
    TextBlock,
    ThinkingBlock,
    ToolUse,
    Trajectory,
    Turn,
    UsageStats,
)
from caw.provider import Provider, ProviderSession
from caw.providers.claude_code import ClaudeCodeProvider

# Auto-register built-in providers
register_provider("claude_code", ClaudeCodeProvider)

__all__ = [
    "Agent",
    "ClaudeCodeProvider",
    "ContentBlock",
    "Display",
    "DisplayMode",
    "MCPServer",
    "MCPTool",
    "Provider",
    "ProviderSession",
    "Session",
    "SessionStore",
    "TextBlock",
    "ThinkingBlock",
    "ToolUse",
    "Trajectory",
    "Turn",
    "UsageStats",
    "register_provider",
]
