"""Core data models for the coding agent wrapper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union


@dataclass
class MCPServer:
    """Configuration for an MCP server."""

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class MCPTool:
    """Descriptor for a tool provided by an MCP server."""

    name: str
    description: str = ""
    server: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)


@dataclass
class TextBlock:
    """A block of text output from the agent."""

    text: str


@dataclass
class ThinkingBlock:
    """A block of thinking/reasoning output from the agent."""

    text: str


@dataclass
class ToolUse:
    """A tool invocation paired with its result."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    output: str = ""
    is_error: bool = False


ContentBlock = Union[TextBlock, ThinkingBlock, ToolUse]


@dataclass
class UsageStats:
    """Token usage and cost statistics."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed (input + output)."""
        return self.input_tokens + self.output_tokens

    def __add__(self, other: UsageStats) -> UsageStats:
        return UsageStats(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
        )


@dataclass
class Turn:
    """A single turn: user sends a message, agent responds."""

    input: str
    output: list[ContentBlock] = field(default_factory=list)
    usage: UsageStats = field(default_factory=UsageStats)
    duration_ms: int = 0

    @property
    def result(self) -> str:
        """Last text block's content."""
        for block in reversed(self.output):
            if isinstance(block, TextBlock) and block.text:
                return block.text
        return ""

    @property
    def tool_calls(self) -> list[ToolUse]:
        """All tool calls made during this turn."""
        return [b for b in self.output if isinstance(b, ToolUse)]


@dataclass
class Trajectory:
    """Complete record of a session."""

    agent: str
    model: str = ""
    turns: list[Turn] = field(default_factory=list)
    usage: UsageStats = field(default_factory=UsageStats)
    duration_ms: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_turns(self) -> int:
        return len(self.turns)

    @property
    def result(self) -> str:
        """The final result from the last turn."""
        if self.turns:
            return self.turns[-1].result
        return ""

    @property
    def total_tool_calls(self) -> int:
        return sum(len(t.tool_calls) for t in self.turns)
