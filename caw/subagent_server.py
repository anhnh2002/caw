"""Minimal MCP server that exposes a subagent as a single callable tool.

Reads configuration from environment variables:
  CAW_SUBAGENT_NAME          - tool name
  CAW_SUBAGENT_DESCRIPTION   - tool description
  CAW_SUBAGENT_SYSTEM_PROMPT - system prompt for the claude CLI call
  CAW_SUBAGENT_MODEL         - model to use (optional)
  CAW_SUBAGENT_TRAJ_DIR      - directory to write trajectory files (optional)
  CAW_SUBAGENT_DEBUG         - set to "1" to enable debug logging to file

Communicates over stdio using the MCP protocol (JSON-RPC 2.0, newline-delimited).
"""

from __future__ import annotations

import json
import os
import sys
import uuid as uuid_mod

# Marker embedded in MCP responses so the parent can match trajectory files.
_TRAJ_MARKER_PREFIX = "\n<!-- caw_traj:"
_TRAJ_MARKER_SUFFIX = " -->"

_debug_file = None


def _debug(msg: str) -> None:
    """Write a debug message to the log file (if enabled)."""
    if _debug_file is not None:
        _debug_file.write(msg + "\n")
        _debug_file.flush()


def _read_message() -> dict | None:
    """Read one newline-delimited JSON-RPC message from stdin."""
    while True:
        line = sys.stdin.readline()
        if not line:  # EOF
            _debug("stdin EOF")
            return None
        line = line.strip()
        if not line:
            continue  # skip empty lines between messages
        _debug(f"<-- {line[:200]}")
        try:
            return json.loads(line)
        except json.JSONDecodeError as e:
            _debug(f"JSON parse error: {e}")
            continue


def _write_message(msg: dict) -> None:
    """Write one newline-delimited JSON-RPC message to stdout."""
    data = json.dumps(msg)
    _debug(f"--> {data[:200]}")
    sys.stdout.write(data + "\n")
    sys.stdout.flush()


def _response(id: object, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": id, "result": result}


def _error(id: object, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}


# ---------------------------------------------------------------------------
# Run subagent via caw Agent
# ---------------------------------------------------------------------------


def _run_subagent(prompt: str, system_prompt: str, model: str) -> tuple[str, dict | None, list | None]:
    """Run a single-turn subagent via caw Agent.

    Returns ``(result_text, trajectory_dict | None, turns | None)``.
    """
    from caw import Agent

    _debug(f"Creating Agent: model={model}")

    agent = Agent(
        system_prompt=system_prompt,
        model=model or None,
        data_dir=None,  # no persistence for subagent
    )

    try:
        with agent.start_session() as session:
            turn = session.send(prompt)
            traj = session.trajectory
    except Exception as e:
        _debug(f"Agent error: {e}")
        return f"Error: {e}", None, None

    result_text = turn.result
    _debug(f"Agent done: {len(result_text)} chars, {traj.usage.total_tokens} tokens")

    return result_text, traj.to_dict(), list(traj.turns)


# ---------------------------------------------------------------------------
# MCP server main loop
# ---------------------------------------------------------------------------


def main() -> None:
    global _debug_file

    name = os.environ.get("CAW_SUBAGENT_NAME", "subagent")
    description = os.environ.get("CAW_SUBAGENT_DESCRIPTION", "A subagent")
    system_prompt = os.environ.get("CAW_SUBAGENT_SYSTEM_PROMPT", "")
    model = os.environ.get("CAW_SUBAGENT_MODEL", "")
    traj_dir = os.environ.get("CAW_SUBAGENT_TRAJ_DIR", "")
    jsonl_path = os.environ.get("CAW_SUBAGENT_JSONL_PATH", "")

    # Debug logging to file
    debug = os.environ.get("CAW_SUBAGENT_DEBUG", "")
    if debug == "1":
        log_path = f"/tmp/caw_subagent_{name.replace(' ', '_')}_{os.getpid()}.log"
        _debug_file = open(log_path, "w")
        _debug(f"Starting subagent MCP server: name={name} pid={os.getpid()}")
        _debug(f"  model={model}")
        _debug(f"  traj_dir={traj_dir}")
        _debug(f"  log={log_path}")

    tool_def = {
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The task for the subagent",
                }
            },
            "required": ["prompt"],
        },
    }

    while True:
        msg = _read_message()
        if msg is None:
            _debug("No more messages, exiting")
            break

        method = msg.get("method")
        msg_id = msg.get("id")
        _debug(f"method={method} id={msg_id}")

        if method == "initialize":
            _write_message(
                _response(
                    msg_id,
                    {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": f"caw-subagent-{name}", "version": "0.1.0"},
                    },
                )
            )

        elif method == "notifications/initialized":
            pass

        elif method == "tools/list":
            _write_message(_response(msg_id, {"tools": [tool_def]}))

        elif method == "tools/call":
            params = msg.get("params", {})
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            _debug(f"tools/call: tool={tool_name} prompt={str(arguments.get('prompt', ''))[:80]}")

            if tool_name != name:
                _write_message(
                    _response(
                        msg_id,
                        {
                            "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}],
                            "isError": True,
                        },
                    )
                )
                continue

            prompt = arguments.get("prompt", "")
            if not prompt:
                _write_message(
                    _response(
                        msg_id,
                        {
                            "content": [{"type": "text", "text": "Error: prompt is required"}],
                            "isError": True,
                        },
                    )
                )
                continue

            result_text, traj_dict, turns = _run_subagent(prompt, system_prompt, model)
            _debug(f"subagent result: {len(result_text)} chars")

            # Write subagent events to parent's JSONL (file-locked)
            if turns and jsonl_path:
                try:
                    from caw.storage import JsonlWriter

                    writer = JsonlWriter(jsonl_path, subagent=name)
                    for i, t in enumerate(turns):
                        writer.write_turn_events(t, i)
                    _debug(f"Wrote {len(turns)} turn(s) to parent JSONL")
                except Exception as e:
                    _debug(f"Failed to write to parent JSONL: {e}")

            # Write trajectory to file and embed marker in response
            traj_marker = ""
            if traj_dict and traj_dir:
                traj_id = str(uuid_mod.uuid4())
                traj_path = os.path.join(traj_dir, f"{traj_id}.json")
                try:
                    os.makedirs(traj_dir, exist_ok=True)
                    with open(traj_path, "w") as f:
                        json.dump(traj_dict, f, indent=2)
                    traj_marker = f"{_TRAJ_MARKER_PREFIX}{traj_id}{_TRAJ_MARKER_SUFFIX}"
                    _debug(f"Wrote traj to {traj_path}")
                except OSError as e:
                    _debug(f"Failed to write traj: {e}")

            _write_message(
                _response(
                    msg_id,
                    {"content": [{"type": "text", "text": result_text + traj_marker}]},
                )
            )

        elif method == "ping":
            _write_message(_response(msg_id, {}))

        elif msg_id is not None:
            _write_message(_error(msg_id, -32601, f"Method not found: {method}"))

    if _debug_file:
        _debug_file.close()


if __name__ == "__main__":
    main()
