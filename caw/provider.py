"""Abstract base classes for provider implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from caw.models import MCPServer, Trajectory, Turn


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

    @property
    def session_id(self) -> str | None:
        """Provider-assigned session ID (if any)."""
        return None

    @property
    def last_raw_output(self) -> str | None:
        """Raw CLI stdout from the most recent send() call (if available)."""
        return None


class Provider(ABC):
    """ABC that each coding agent backend implements."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier (e.g. 'claude_code', 'codex')."""
        ...

    @abstractmethod
    def start_session(self, mcp_servers: list[MCPServer], **kwargs: Any) -> ProviderSession:
        """Create and return a new provider session."""
        ...
