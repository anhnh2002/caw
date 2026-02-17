"""Abstract base classes for provider implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from caw.models import MCPServer, ModelTier, ToolGroup, Trajectory, Turn


class ProviderSession(ABC):
    """ABC that each provider implements to manage a live session."""

    @abstractmethod
    def send(self, message: str) -> Turn:
        """Send a message and return the agent's response turn."""
        ...

    @abstractmethod
    def end(self) -> Trajectory:
        """Finalize the session and return the complete trajectory."""
        ...

    @property
    @abstractmethod
    def trajectory(self) -> Trajectory:
        """The accumulated trajectory so far."""
        ...

    def detect_usage_limit(self, turn: Turn) -> int | None:
        """Check whether *turn* indicates the provider's usage limit was hit.

        Returns the number of minutes to wait before retrying, or ``None``
        if no limit was detected.  Override in provider subclasses to
        implement provider-specific detection logic.
        """
        return None

    @property
    def session_id(self) -> str | None:
        """Provider-assigned session ID (if any)."""
        return None

    @property
    def last_raw_output(self) -> str | None:
        """Raw CLI stdout from the most recent send() call (if available)."""
        return None

    def set_step_callback(self, callback: Callable[[list], None] | None) -> None:
        """Set callback invoked after each step within send()."""
        pass  # default no-op; concrete providers override


class Provider(ABC):
    """ABC that each coding agent backend implements."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier (e.g. 'claude_code', 'codex')."""
        ...

    def resolve_model(self, tier: ModelTier) -> str:
        """Translate a :class:`ModelTier` into a concrete model identifier.

        Each provider must override this to map abstract tiers (e.g.
        ``ModelTier.STRONGEST``) to the actual model string it supports.
        """
        raise NotImplementedError(
            f"{self.name} provider does not implement resolve_model(); "
            f"pass an explicit model string instead of ModelTier.{tier.name}"
        )

    def resolve_tool_restrictions(self, tools: ToolGroup) -> dict[str, Any]:
        """Translate ToolGroup into provider-specific session kwargs.

        Receives a concrete ToolGroup value (never None — the Agent layer
        applies the default before calling this).
        """
        return {}

    @abstractmethod
    def start_session(self, mcp_servers: list[MCPServer], **kwargs: Any) -> ProviderSession:
        """Create and return a new provider session."""
        ...
