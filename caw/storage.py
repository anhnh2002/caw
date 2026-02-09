"""Session data persistence — writes raw session data to disk."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from caw.models import MCPServer, TextBlock, ThinkingBlock, ToolUse, Trajectory


class SessionStore:
    """Persists raw session data to a directory on disk.

    Layout::

        <data_dir>/sessions/<session_id>/
            config.json
            turns/
                000_input.txt
                000_raw_output.jsonl
                001_input.txt
                001_raw_output.jsonl
    """

    def __init__(self, data_dir: str | Path, session_id: str) -> None:
        self._session_id = session_id
        self._session_dir = Path(data_dir) / "sessions" / session_id
        self._turns_dir = self._session_dir / "turns"
        self._turns_dir.mkdir(parents=True, exist_ok=True)
        self._turn_counter = 0

    @property
    def session_dir(self) -> Path:
        """Path to this session's directory."""
        return self._session_dir

    # ------------------------------------------------------------------
    # config.json
    # ------------------------------------------------------------------

    def write_config(
        self,
        agent: str,
        model: str,
        mcp_servers: list[MCPServer],
        metadata: dict[str, Any] | None = None,
        system_prompt: str | None = None,
    ) -> None:
        """Write the initial config.json for this session."""
        config: dict[str, Any] = {
            "session_id": self._session_id,
            "agent": agent,
            "model": model,
            "mcp_servers": [asdict(s) for s in mcp_servers],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }
        if system_prompt:
            config["system_prompt"] = system_prompt
        self._write_json(self._session_dir / "config.json", config)

    def update_config(self, **updates: Any) -> None:
        """Read-modify-write merge into config.json."""
        path = self._session_dir / "config.json"
        config = self._read_json(path)
        config.update(updates)
        self._write_json(path, config)

    def finalize(self, trajectory: Trajectory) -> None:
        """Update config.json with final stats and write trajectory.json."""
        self.update_config(
            model=trajectory.model,
            num_turns=trajectory.num_turns,
            total_duration_ms=trajectory.duration_ms,
            total_usage=asdict(trajectory.usage),
        )
        self._write_json(
            self._session_dir / "trajectory.json",
            self._serialize_trajectory(trajectory),
        )

    # ------------------------------------------------------------------
    # Turn files
    # ------------------------------------------------------------------

    def save_turn(self, user_input: str, raw_output: str | None = None) -> None:
        """Write turn input and (optionally) raw CLI output to disk."""
        prefix = f"{self._turn_counter:03d}"

        input_path = self._turns_dir / f"{prefix}_input.txt"
        input_path.write_text(user_input, encoding="utf-8")

        if raw_output is not None:
            output_path = self._turns_dir / f"{prefix}_raw_output.jsonl"
            output_path.write_text(raw_output, encoding="utf-8")

        self._turn_counter += 1

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_block(block: TextBlock | ThinkingBlock | ToolUse) -> dict[str, Any]:
        if isinstance(block, TextBlock):
            return {"type": "text", "text": block.text}
        elif isinstance(block, ThinkingBlock):
            return {"type": "thinking", "text": block.text}
        else:
            d: dict[str, Any] = {
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "arguments": block.arguments,
                "output": block.output,
            }
            if block.is_error:
                d["is_error"] = True
            return d

    @staticmethod
    def _serialize_trajectory(trajectory: Trajectory) -> dict[str, Any]:
        return {
            "agent": trajectory.agent,
            "model": trajectory.model,
            "duration_ms": trajectory.duration_ms,
            "usage": asdict(trajectory.usage),
            "turns": [
                {
                    "input": turn.input,
                    "output": [SessionStore._serialize_block(b) for b in turn.output],
                    "usage": asdict(turn.usage),
                    "duration_ms": turn.duration_ms,
                }
                for turn in trajectory.turns
            ],
            "metadata": trajectory.metadata,
        }

    @staticmethod
    def _write_json(path: Path, data: dict[str, Any]) -> None:
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
