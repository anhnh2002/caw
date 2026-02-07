"""Concrete Agent and Session — the main user-facing API."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from caw.display import LOG_ENV_VAR, Display, DisplayMode
from caw.models import MCPServer, Trajectory, Turn
from caw.provider import Provider, ProviderSession
from caw.storage import SessionStore

DEFAULT_PROVIDER = "claude_code"
PROVIDER_ENV_VAR = "CAW_PROVIDER"

_PROVIDER_REGISTRY: dict[str, type[Provider]] = {}


def register_provider(name: str, cls: type[Provider]) -> None:
    """Register a provider class under the given name."""
    _PROVIDER_REGISTRY[name] = cls


def _resolve_provider(name: str | None) -> Provider:
    """Resolve provider: explicit name > env var > default."""
    provider_name = name or os.environ.get(PROVIDER_ENV_VAR) or DEFAULT_PROVIDER
    if provider_name not in _PROVIDER_REGISTRY:
        available = list(_PROVIDER_REGISTRY.keys())
        raise ValueError(f"Unknown provider {provider_name!r}. Available: {available}")
    return _PROVIDER_REGISTRY[provider_name]()


class Session:
    """A live interaction session with a coding agent."""

    def __init__(
        self,
        provider_session: ProviderSession,
        store: SessionStore | None = None,
    ) -> None:
        self._session = provider_session
        self._store = store

    def send(self, message: str) -> Turn:
        """Send a message and get the agent's response turn."""
        turn = self._session.send(message)

        if self._store is not None:
            # On the first turn, backfill model if the provider discovered it
            if self._session.trajectory.num_turns == 1:
                model = self._session.trajectory.model
                if model:
                    self._store.update_config(model=model)

            self._store.save_turn(message, self._session.last_raw_output)

        return turn

    def end(self) -> Trajectory:
        """End the session and return the complete trajectory."""
        traj = self._session.end()
        if self._store is not None:
            self._store.finalize(traj)
        return traj

    @property
    def trajectory(self) -> Trajectory:
        """Accumulated trajectory (available during and after the session)."""
        return self._session.trajectory

    @property
    def session_dir(self) -> Path | None:
        """Path to the session's data directory, or None if persistence is disabled."""
        if self._store is not None:
            return self._store.session_dir
        return None

    def __enter__(self) -> Session:
        return self

    def __exit__(self, *args: Any) -> None:
        self.end()


class Agent:
    """Coding agent wrapper — unified interface across providers.

    Provider resolution order:
    1. Explicit ``provider`` argument
    2. ``CAW_PROVIDER`` environment variable
    3. Default provider
    """

    def __init__(
        self,
        provider: str | None = None,
        data_dir: str | None = "caw_data",
        display: DisplayMode | Display | str | None = None,
        **kwargs: Any,
    ) -> None:
        self._provider_name = provider
        self._provider: Provider | None = None
        self._mcp_servers: list[MCPServer] = []
        self._data_dir = data_dir
        self._display = self._resolve_display(display)
        self._kwargs = kwargs

    @staticmethod
    def _resolve_display(display: DisplayMode | Display | str | None) -> Display | None:
        """Resolve a display argument into a Display instance or None.

        Resolution order: explicit argument > ``CAW_LOG`` env var > None.
        """
        if display is None:
            env_mode = os.environ.get(LOG_ENV_VAR)
            if env_mode is None:
                return None
            return Display(mode=env_mode)
        if isinstance(display, Display):
            return display
        return Display(mode=display)

    def set_provider(self, provider: str) -> None:
        """Set or change the provider before starting a session."""
        self._provider_name = provider
        self._provider = None

    @property
    def provider(self) -> Provider:
        """The resolved provider instance (lazily created)."""
        if self._provider is None:
            self._provider = _resolve_provider(self._provider_name)
        return self._provider

    @property
    def mcp_servers(self) -> list[MCPServer]:
        """Currently configured MCP servers."""
        return list(self._mcp_servers)

    def add_mcp_server(self, server: MCPServer) -> None:
        """Register an MCP server for tool access."""
        self._mcp_servers.append(server)

    def set_model(self, model: str) -> None:
        """Set the model to use for sessions."""
        self._kwargs["model"] = model

    def set_reasoning(self, reasoning: str) -> None:
        """Set the reasoning budget token (e.g. ``'medium'``)."""
        self._kwargs["reasoning"] = reasoning

    def start_session(self, **kwargs: Any) -> Session:
        """Start a new interactive session with the agent."""
        merged = {**self._kwargs, **kwargs}
        if self._display is not None and "display" not in merged:
            merged["display"] = self._display
        provider_session = self.provider.start_session(mcp_servers=list(self._mcp_servers), **merged)

        store: SessionStore | None = None
        if self._data_dir and provider_session.session_id:
            store = SessionStore(self._data_dir, provider_session.session_id)
            store.write_config(
                agent=self.provider.name,
                model=merged.get("model", ""),
                mcp_servers=list(self._mcp_servers),
            )

        return Session(provider_session, store=store)
