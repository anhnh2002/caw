"""Concrete Agent and Session — the main user-facing API."""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid as uuid_mod
from pathlib import Path
from typing import Any

from caw.models import AgentSpec, MCPServer, ToolUse, Trajectory, Turn
from caw.provider import Provider, ProviderSession
from caw.storage import SessionStore

# ---------------------------------------------------------------------------
# Environment variable overrides
#
# These let you configure caw globally without changing code.  Each one is
# used as a fallback when the corresponding value is not set explicitly via
# the Agent() constructor or method calls.
#
#   CAW_PROVIDER  — Provider backend ("claude_code", "codex", …)
#   CAW_MODEL     — Model name passed to the provider (e.g. "gpt-5.2-codex")
#   CAW_EFFORT    — Reasoning effort level (e.g. "high", "medium", "low")
# ---------------------------------------------------------------------------
DEFAULT_PROVIDER = "claude_code"
CAW_PROVIDER = "CAW_PROVIDER"
CAW_MODEL = "CAW_MODEL"
CAW_EFFORT = "CAW_EFFORT"

_PROVIDER_REGISTRY: dict[str, type[Provider]] = {}

# Must match caw.mcp._TRAJ_MARKER_PREFIX / _SUFFIX
_TRAJ_MARKER_RE = re.compile(r"\n<!-- caw_traj:([\w-]+) -->$")


def register_provider(name: str, cls: type[Provider]) -> None:
    """Register a provider class under the given name."""
    _PROVIDER_REGISTRY[name] = cls


def _resolve_provider(name: str | None) -> Provider:
    """Resolve provider: explicit name > env var > default."""
    provider_name = name or os.environ.get(CAW_PROVIDER) or DEFAULT_PROVIDER
    if provider_name not in _PROVIDER_REGISTRY:
        available = list(_PROVIDER_REGISTRY.keys())
        raise ValueError(f"Unknown provider {provider_name!r}. Available: {available}")
    return _PROVIDER_REGISTRY[provider_name]()


def _attach_subagent_trajectories(turn: Turn, traj_dir: str | None) -> None:
    """Scan a turn's tool outputs for trajectory markers and attach them.

    For each ToolUse whose output ends with ``<!-- caw_traj:<uuid> -->``:
    1. Load the trajectory JSON from ``traj_dir/<uuid>.json``
    2. Attach it as ``tool_use.subagent_trajectory``
    3. Strip the marker from ``tool_use.output``
    """
    if not traj_dir:
        return
    for block in turn.output:
        if not isinstance(block, ToolUse):
            continue
        m = _TRAJ_MARKER_RE.search(block.output)
        if not m:
            continue
        traj_id = m.group(1)
        traj_path = os.path.join(traj_dir, f"{traj_id}.json")
        try:
            with open(traj_path) as f:
                traj_dict = json.load(f)
            block.subagent_trajectory = Trajectory.from_dict(traj_dict)
        except (OSError, json.JSONDecodeError, KeyError):
            pass  # best-effort
        # Strip marker regardless
        block.output = block.output[: m.start()]


class Session:
    """A live interaction session with a coding agent."""

    def __init__(
        self,
        provider_session: ProviderSession,
        store: SessionStore | None = None,
        subagent_traj_dir: str | None = None,
        tool_handles: list[Any] | None = None,
    ) -> None:
        self._session = provider_session
        self._store = store
        self._subagent_traj_dir = subagent_traj_dir
        self._tool_handles = tool_handles or []

    def send(self, message: str) -> Turn:
        """Send a message and get the agent's response turn."""
        turn = self._session.send(message)

        # Attach subagent trajectories from marker files
        _attach_subagent_trajectories(turn, self._subagent_traj_dir)

        if self._store is not None:
            self._store.append_turn(turn, self._session.trajectory, raw_output=self._session.last_raw_output)

        return turn

    def end(self) -> Trajectory:
        """End the session and return the complete trajectory."""
        traj = self._session.end()
        if self._store is not None:
            self._store.finalize(traj)
        # Stop all tool server handles
        for handle in self._tool_handles:
            try:
                handle.stop_sync()
            except Exception:
                pass
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
        system_prompt: str | None = None,
        model: str | None = None,
        reasoning: str | None = None,
        name: str = "",
        description: str = "",
        **kwargs: Any,
    ) -> None:
        self._provider_name = provider
        self._provider: Provider | None = None
        self._mcp_servers: list[MCPServer] = []
        self._subagents: list[AgentSpec] = []
        self._tool_servers: list[Any] = []  # list[MCPServerHandle], lazy import
        self._data_dir = data_dir
        self._name = name
        self._description = description
        self._metadata: dict[str, Any] = {}
        if system_prompt is not None:
            kwargs["system_prompt"] = system_prompt
        if model is not None:
            kwargs["model"] = model
        elif os.environ.get(CAW_MODEL):
            kwargs["model"] = os.environ[CAW_MODEL]
        if reasoning is not None:
            kwargs["reasoning"] = reasoning
        elif os.environ.get(CAW_EFFORT):
            kwargs["reasoning"] = os.environ[CAW_EFFORT]
        self._kwargs = kwargs

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

    def add_tool_server(self, handle: Any) -> None:
        """Register a custom HTTP tool server (MCPServerHandle).

        The handle's lifecycle (start/stop) is managed by the session.
        """
        self._tool_servers.append(handle)

    def set_model(self, model: str) -> None:
        """Set the model to use for sessions."""
        self._kwargs["model"] = model

    def set_reasoning(self, reasoning: str) -> None:
        """Set the reasoning budget token (e.g. ``'medium'``)."""
        self._kwargs["reasoning"] = reasoning

    def set_system_prompt(self, system_prompt: str) -> None:
        """Set a system prompt that guides the agent's behavior for the session."""
        self._kwargs["system_prompt"] = system_prompt

    def add_subagent(self, spec: AgentSpec) -> None:
        """Register a subagent that will be exposed as a tool."""
        self._subagents.append(spec)

    def get_spec(self) -> AgentSpec:
        """Return an AgentSpec snapshot of this agent's current configuration."""
        return AgentSpec(
            name=self._name,
            description=self._description,
            system_prompt=self._kwargs.get("system_prompt", ""),
            model=self._kwargs.get("model", ""),
            reasoning=self._kwargs.get("reasoning", ""),
            mcp_servers=list(self._mcp_servers),
            metadata=dict(self._metadata),
        )

    def _subagent_tool_servers(self, traj_dir: str, jsonl_path: str | None = None) -> list[Any]:
        """Convert registered subagents into HTTP tool server handles."""
        from caw.mcp import create_subagent_tool_server

        handles = []
        for spec in self._subagents:
            handle = create_subagent_tool_server(spec, traj_dir, jsonl_path)
            handles.append(handle)
        return handles

    def completion(self, message: str, **kwargs: Any) -> Trajectory:
        """Send a single message and return the complete trajectory.

        Convenience wrapper for simple use cases where you don't need
        to maintain a multi-turn session::

            traj = agent.completion("Explain this code")
            print(traj.result)
        """
        session = self.start_session(**kwargs)
        session.send(message)
        return session.end()

    def start_session(self, **kwargs: Any) -> Session:
        """Start a new interactive session with the agent."""
        merged = {**self._kwargs, **kwargs}

        # Generate session_id early so the JSONL path is known before MCP configs
        session_id: str | None = None
        store: SessionStore | None = None
        if self._data_dir:
            session_id = str(uuid_mod.uuid4())
            store = SessionStore(self._data_dir, session_id)

        # Create temp dir for subagent trajectory files (if subagents exist)
        subagent_traj_dir: str | None = None
        if self._subagents:
            subagent_traj_dir = tempfile.mkdtemp(prefix="caw_subagent_traj_")

        # Collect all tool server handles (user-registered + subagent)
        all_handles: list[Any] = list(self._tool_servers)
        if self._subagents:
            all_handles += self._subagent_tool_servers(
                subagent_traj_dir,  # type: ignore[arg-type]
                jsonl_path=str(store.jsonl_path) if store else None,
            )

        # Start all HTTP tool servers and collect their MCPServer configs
        for handle in all_handles:
            handle.start_sync()

        all_mcp = list(self._mcp_servers)
        for handle in all_handles:
            all_mcp.append(MCPServer(name=handle.server_id, url=handle.url))

        # Pass our session_id so the provider uses it (instead of generating its own)
        if session_id:
            merged["session_id"] = session_id

        provider_session = self.provider.start_session(mcp_servers=all_mcp, **merged)

        if store:
            store.write_metadata(provider_session.trajectory)

        return Session(
            provider_session,
            store=store,
            subagent_traj_dir=subagent_traj_dir,
            tool_handles=all_handles,
        )
