"""Claude Code provider — wraps the ``claude`` CLI in stream-json mode."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import uuid
from typing import Any

from caw.display import Display
from caw.models import ContentBlock, MCPServer, TextBlock, ThinkingBlock, ToolUse, Trajectory, Turn, UsageStats
from caw.provider import Provider, ProviderSession


class ClaudeCodeSession(ProviderSession):
    """Live session backed by the ``claude`` CLI."""

    def __init__(
        self,
        mcp_servers: list[MCPServer],
        model: str | None = None,
        display: Display | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self._session_id = str(uuid.uuid4())
        self._model = model
        self._mcp_servers = mcp_servers
        self._display = display
        self._system_prompt = system_prompt
        self._has_sent = False
        self._turns: list[Turn] = []
        self._total_usage = UsageStats()
        self._total_duration_ms = 0
        self._mcp_config_path: str | None = None
        self._last_raw_output: str = ""

    # ------------------------------------------------------------------
    # MCP config helpers
    # ------------------------------------------------------------------

    def _ensure_mcp_config(self) -> str | None:
        """Write MCP server config to a temp file on first call, return path."""
        if not self._mcp_servers:
            return None
        if self._mcp_config_path is not None:
            return self._mcp_config_path

        config: dict[str, Any] = {"mcpServers": {}}
        for srv in self._mcp_servers:
            entry: dict[str, Any] = {"command": srv.command, "args": srv.args}
            if srv.env:
                entry["env"] = srv.env
            config["mcpServers"][srv.name] = entry

        fd, path = tempfile.mkstemp(suffix=".json", prefix="caw_mcp_")
        with os.fdopen(fd, "w") as f:
            json.dump(config, f)
        self._mcp_config_path = path
        return path

    # ------------------------------------------------------------------
    # Core send
    # ------------------------------------------------------------------

    def send(self, message: str) -> Turn:
        if self._display:
            if not self._has_sent:
                self._display.on_metadata(
                    agent="claude_code",
                    model=self._model or "",
                    session=self._session_id,
                )
            self._display.on_user_message(message)

        cmd = [
            "claude",
            "-p",
            "--verbose",
            "--output-format",
            "stream-json",
        ]

        if self._model:
            cmd += ["--model", self._model]

        if not self._has_sent:
            cmd += ["--session-id", self._session_id]
            if self._system_prompt:
                cmd += ["--system-prompt", self._system_prompt]
        else:
            cmd += ["--resume", self._session_id]

        mcp_path = self._ensure_mcp_config()
        if mcp_path:
            cmd += ["--mcp-config", mcp_path]

        try:
            proc = subprocess.run(
                cmd,
                input=message,
                capture_output=True,
                text=True,
                timeout=600,
            )
        except FileNotFoundError:
            raise RuntimeError("claude CLI not found. Install it with: npm install -g @anthropic-ai/claude-code")

        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        self._last_raw_output = stdout

        if proc.returncode != 0 and not stdout.strip():
            raise RuntimeError(f"claude CLI exited with code {proc.returncode}: {stderr}")

        self._has_sent = True

        # Parse JSONL output
        turn = self._parse_output(message, stdout)
        self._turns.append(turn)
        self._total_usage = self._total_usage + turn.usage
        self._total_duration_ms += turn.duration_ms
        return turn

    # ------------------------------------------------------------------
    # JSONL parsing
    # ------------------------------------------------------------------

    def _parse_output(self, user_message: str, stdout: str) -> Turn:
        blocks: list[ContentBlock] = []
        tool_blocks: dict[str, ToolUse] = {}  # tool_use_id -> ToolUse block
        usage = UsageStats()
        duration_ms = 0

        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            event_type = event.get("type")

            if event_type == "system" and event.get("subtype") == "init":
                if not self._model:
                    self._model = event.get("model", "")
                    if self._display and self._model:
                        self._display.on_metadata(model=self._model)

            elif event_type == "assistant":
                new_blocks = self._parse_assistant_blocks(event)
                for block in new_blocks:
                    blocks.append(block)
                    if self._display:
                        if isinstance(block, TextBlock):
                            self._display.on_text(block)
                        elif isinstance(block, ThinkingBlock):
                            self._display.on_thinking(block)
                        elif isinstance(block, ToolUse):
                            self._display.on_tool_call(block)
                    if isinstance(block, ToolUse):
                        tool_blocks[block.id] = block

            elif event_type == "user":
                # User events carry tool results — pair eagerly
                msg_data = event.get("message", {})
                for content in msg_data.get("content", []):
                    if content.get("type") == "tool_result":
                        tid = content.get("tool_use_id", "")
                        if tid:
                            text_parts: list[str] = []
                            raw_content = content.get("content", "")
                            if isinstance(raw_content, str):
                                text_parts.append(raw_content)
                            elif isinstance(raw_content, list):
                                for part in raw_content:
                                    if isinstance(part, dict) and part.get("type") == "text":
                                        text_parts.append(part.get("text", ""))
                            output = "\n".join(text_parts)
                            is_error = content.get("is_error", False)

                            if tid in tool_blocks:
                                tool_blocks[tid].output = output
                                tool_blocks[tid].is_error = is_error
                                if self._display:
                                    self._display.on_tool_result(tool_blocks[tid])

            elif event_type == "result":
                usage = self._parse_usage(event)
                duration_ms = event.get("duration_ms", 0)

        turn = Turn(input=user_message, output=blocks, usage=usage, duration_ms=duration_ms)

        if self._display:
            self._display.on_turn_end(turn.result, usage, duration_ms)

        return turn

    @staticmethod
    def _parse_assistant_blocks(event: dict[str, Any]) -> list[ContentBlock]:
        """Parse an assistant event into content blocks."""
        msg_data = event.get("message", {})
        content_blocks = msg_data.get("content", [])
        if not content_blocks:
            return []

        result: list[ContentBlock] = []

        for block in content_blocks:
            block_type = block.get("type")
            if block_type == "text":
                text = block.get("text", "")
                if text:
                    result.append(TextBlock(text=text))
            elif block_type == "thinking":
                text = block.get("thinking", "")
                if text:
                    result.append(ThinkingBlock(text=text))
            elif block_type == "tool_use":
                result.append(
                    ToolUse(
                        id=block.get("id", ""),
                        name=block.get("name", ""),
                        arguments=block.get("input", {}),
                    )
                )

        return result

    @staticmethod
    def _parse_usage(event: dict[str, Any]) -> UsageStats:
        u = event.get("usage", {})
        return UsageStats(
            input_tokens=u.get("input_tokens", 0),
            output_tokens=u.get("output_tokens", 0),
            cache_read_tokens=u.get("cache_read_input_tokens", 0),
            cache_write_tokens=u.get("cache_creation_input_tokens", 0),
            cost_usd=event.get("total_cost_usd", 0.0),
        )

    # ------------------------------------------------------------------
    # Trajectory / lifecycle
    # ------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def last_raw_output(self) -> str:
        return self._last_raw_output

    @property
    def trajectory(self) -> Trajectory:
        return Trajectory(
            agent="claude_code",
            model=self._model or "",
            turns=list(self._turns),
            usage=self._total_usage,
            duration_ms=self._total_duration_ms,
        )

    def end(self) -> Trajectory:
        traj = self.trajectory
        if self._mcp_config_path and os.path.exists(self._mcp_config_path):
            os.unlink(self._mcp_config_path)
            self._mcp_config_path = None
        return traj


class ClaudeCodeProvider(Provider):
    """Provider that delegates to the ``claude`` CLI."""

    @property
    def name(self) -> str:
        return "claude_code"

    def start_session(self, mcp_servers: list[MCPServer], **kwargs: Any) -> ClaudeCodeSession:
        model = kwargs.get("model")
        display = kwargs.get("display")
        system_prompt = kwargs.get("system_prompt")
        return ClaudeCodeSession(mcp_servers=mcp_servers, model=model, display=display, system_prompt=system_prompt)
