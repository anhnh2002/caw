"""Concrete Agent and Session — the main user-facing API."""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid as uuid_mod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from caw.display import LOG_ENV_VAR, Display, DisplayMode
from caw.models import AgentSpec, MCPServer, ToolUse, Trajectory, Turn
from caw.provider import Provider, ProviderSession
from caw.storage import SessionStore

DEFAULT_PROVIDER = "claude_code"
PROVIDER_ENV_VAR = "CAW_PROVIDER"

_PROVIDER_REGISTRY: dict[str, type[Provider]] = {}

# Must match caw.mcp.subagent_server._TRAJ_MARKER_PREFIX / _SUFFIX
_TRAJ_MARKER_RE = re.compile(r"\n<!-- caw_traj:([\w-]+) -->$")


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
    ) -> None:
        self._session = provider_session
        self._store = store
        self._subagent_traj_dir = subagent_traj_dir

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
        self._data_dir = data_dir
        self._display = self._resolve_display(display)
        self._name = name
        self._description = description
        self._metadata: dict[str, Any] = {}
        if system_prompt is not None:
            kwargs["system_prompt"] = system_prompt
        if model is not None:
            kwargs["model"] = model
        if reasoning is not None:
            kwargs["reasoning"] = reasoning
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

    def _subagent_mcp_servers(self, traj_dir: str, jsonl_path: str | None = None) -> list[MCPServer]:
        """Convert registered subagents into MCP server configs."""
        import sys

        servers: list[MCPServer] = []
        for spec in self._subagents:
            servers.append(
                MCPServer(
                    name=f"subagent_{spec.name}",
                    command=sys.executable,
                    args=["-m", "caw.subagent_server"],
                    env={
                        "CAW_SUBAGENT_NAME": spec.name,
                        "CAW_SUBAGENT_DESCRIPTION": spec.description,
                        "CAW_SUBAGENT_SYSTEM_PROMPT": spec.system_prompt,
                        "CAW_SUBAGENT_MODEL": spec.model or "",
                        "CAW_SUBAGENT_TRAJ_DIR": traj_dir,
                        "CAW_SUBAGENT_JSONL_PATH": jsonl_path or "",
                        "CAW_SUBAGENT_DEBUG": os.environ.get("CAW_SUBAGENT_DEBUG", ""),
                    },
                )
            )
        return servers

    def start_session(self, **kwargs: Any) -> Session:
        """Start a new interactive session with the agent."""
        merged = {**self._kwargs, **kwargs}
        if self._display is not None and "display" not in merged:
            merged["display"] = self._display

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

        all_mcp = list(self._mcp_servers)
        if self._subagents:
            all_mcp += self._subagent_mcp_servers(
                subagent_traj_dir,  # type: ignore[arg-type]
                jsonl_path=str(store.jsonl_path) if store else None,
            )

        # Pass our session_id so the provider uses it (instead of generating its own)
        if session_id:
            merged["session_id"] = session_id

        provider_session = self.provider.start_session(mcp_servers=all_mcp, **merged)

        if store:
            provider_session.trajectory.session_id = session_id  # type: ignore[assignment]
            provider_session.trajectory.created_at = datetime.now(timezone.utc).isoformat()
            store.write_metadata(provider_session.trajectory)

        return Session(provider_session, store=store, subagent_traj_dir=subagent_traj_dir)
