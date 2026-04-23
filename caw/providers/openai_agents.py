"""OpenAI Agents SDK provider."""

from __future__ import annotations

import asyncio
import glob as glob_mod
import json
import os
import re
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from caw.display import get_global_display
from caw.models import (
    AgentStreamEvent,
    ContentBlock,
    MCPServer,
    ModelTier,
    TextBlock,
    ThinkingBlock,
    ToolGroup,
    ToolUse,
    Trajectory,
    Turn,
    UsageStats,
)
from caw.pricing import compute_cost
from caw.provider import Provider, ProviderSession

CAW_OPENAI_BASE_URL = "CAW_OPENAI_BASE_URL"
CAW_OPENAI_API_KEY = "CAW_OPENAI_API_KEY"

_MODEL_TIER_MAP: dict[ModelTier, str] = {
    ModelTier.STRONGEST: os.environ.get("OPENAI_AGENTS_STRONGEST_MODEL", "gpt-5.4"),
    ModelTier.FAST: os.environ.get("OPENAI_AGENTS_FAST_MODEL", "gpt-5.4-mini"),
}


def _run_coro_sync(coro):
    """Run *coro* from sync code, including when another event loop is active."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, Any] = {}

    def _runner() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:
            result["error"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result.get("value")


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, indent=2, default=str)
    except TypeError:
        return str(value)


def _get_field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _parse_tool_arguments(raw: Any) -> dict[str, Any]:
    args = _get_field(raw, "arguments")
    if args is None:
        args = _get_field(raw, "input")
    if isinstance(args, dict):
        return args
    if isinstance(args, str) and args:
        try:
            parsed = json.loads(args)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        return {"input": args}
    return {}


def _tool_call_id(raw: Any) -> str:
    return str(
        _get_field(raw, "call_id")
        or _get_field(raw, "id")
        or _get_field(raw, "tool_call_id")
        or uuid.uuid4()
    )


def _tool_call_name(item: Any) -> str:
    raw = getattr(item, "raw_item", None)
    return str(
        _get_field(raw, "name")
        or _get_field(raw, "tool_name")
        or getattr(item, "title", None)
        or getattr(item, "description", None)
        or _get_field(raw, "type")
        or "tool"
    )


def _resolve_workspace_path(root: Path, path: str) -> Path:
    target = (root / path).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        raise ValueError(f"Path escapes workspace: {path}")
    return target


def _safe_glob(root: Path, pattern: str) -> list[str]:
    if os.path.isabs(pattern):
        raise ValueError("Absolute glob patterns are not allowed.")
    if ".." in Path(pattern).parts:
        raise ValueError("Glob patterns must stay inside the workspace.")
    matches = []
    for match in glob_mod.glob(str(root / pattern), recursive=True):
        path = Path(match).resolve()
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        matches.append(str(rel))
    return sorted(matches)


def _make_local_tools(workspace: Path, tools: ToolGroup):
    from agents import function_tool

    enabled = []

    if tools & ToolGroup.READER:

        @function_tool(strict_mode=False)
        def read_file(path: str, offset: int = 0, limit: int = 20000) -> str:
            """Read a UTF-8 text file from the workspace."""
            target = _resolve_workspace_path(workspace, path)
            text = target.read_text(errors="replace")
            if offset < 0:
                offset = 0
            if limit <= 0:
                return ""
            return text[offset : offset + limit]

        @function_tool(strict_mode=False)
        def list_dir(path: str = ".") -> str:
            """List files and directories under a workspace directory."""
            target = _resolve_workspace_path(workspace, path)
            entries = []
            for child in sorted(target.iterdir(), key=lambda p: p.name):
                suffix = "/" if child.is_dir() else ""
                entries.append(f"{child.name}{suffix}")
            return "\n".join(entries)

        @function_tool(strict_mode=False)
        def glob(pattern: str) -> str:
            """Find workspace paths matching a glob pattern."""
            return "\n".join(_safe_glob(workspace, pattern))

        @function_tool(strict_mode=False)
        def grep(pattern: str, glob: str = "**/*", ignore_case: bool = False, max_matches: int = 100) -> str:
            """Search text files in the workspace using a regular expression."""
            flags = re.IGNORECASE if ignore_case else 0
            regex = re.compile(pattern, flags)
            results: list[str] = []
            for rel in _safe_glob(workspace, glob):
                target = _resolve_workspace_path(workspace, rel)
                if not target.is_file():
                    continue
                try:
                    text = target.read_text(errors="replace")
                except OSError:
                    continue
                for line_no, line in enumerate(text.splitlines(), start=1):
                    if regex.search(line):
                        results.append(f"{rel}:{line_no}:{line}")
                        if len(results) >= max_matches:
                            return "\n".join(results)
            return "\n".join(results)

        enabled.extend([read_file, list_dir, glob, grep])

    if tools & ToolGroup.WRITER:

        @function_tool(strict_mode=False)
        def write_file(path: str, content: str, create_dirs: bool = True) -> str:
            """Write a UTF-8 text file inside the workspace."""
            target = _resolve_workspace_path(workspace, path)
            if create_dirs:
                target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
            return f"Wrote {path}"

        @function_tool(strict_mode=False)
        def edit_file(path: str, old: str, new: str, replace_all: bool = False) -> str:
            """Edit a text file by replacing an exact string."""
            target = _resolve_workspace_path(workspace, path)
            text = target.read_text(errors="replace")
            count = text.count(old)
            if count == 0:
                raise ValueError("Old text was not found.")
            updated = text.replace(old, new) if replace_all else text.replace(old, new, 1)
            target.write_text(updated)
            changed = count if replace_all else 1
            return f"Edited {path}: {changed} replacement(s)"

        enabled.extend([write_file, edit_file])

    if tools & ToolGroup.EXEC:

        @function_tool(strict_mode=False)
        async def run_shell(command: str, timeout: float = 60.0) -> str:
            """Run a shell command from the workspace and return output."""
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=str(workspace),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return f"Command timed out after {timeout}s"
            out = stdout.decode(errors="replace")
            err = stderr.decode(errors="replace")
            result = out
            if err:
                result += ("\n" if result else "") + err
            if proc.returncode:
                result += f"\n(exit code {proc.returncode})"
            return result

        enabled.append(run_shell)

    if tools & ToolGroup.WEB:

        @function_tool(strict_mode=False)
        async def web_fetch(url: str, timeout: float = 20.0, max_bytes: int = 200000) -> str:
            """Fetch a URL over HTTP(S) and return response text."""

            def _fetch() -> str:
                req = Request(url, headers={"User-Agent": "caw-openai-agents/0.1"})
                with urlopen(req, timeout=timeout) as resp:
                    data = resp.read(max_bytes)
                    charset = resp.headers.get_content_charset() or "utf-8"
                    return data.decode(charset, errors="replace")

            return await asyncio.to_thread(_fetch)

        enabled.append(web_fetch)

    return enabled


class OpenAIAgentsSession(ProviderSession):
    """Live session backed by the OpenAI Agents SDK."""

    def __init__(
        self,
        mcp_servers: list[MCPServer],
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        system_prompt: str | None = None,
        session_id: str | None = None,
        tools: ToolGroup = ToolGroup.ALL - ToolGroup.INTERACTION,
        reasoning: str | None = None,
        workspace: str | Path | None = None,
        max_turns: int = 100,
        tracing_disabled: bool = True,
        model_settings: Any = None,
        run_config: Any = None,
    ) -> None:
        self._session_id = session_id or str(uuid.uuid4())
        self._mcp_servers = mcp_servers
        self._model = model
        self._base_url = base_url or os.environ.get(CAW_OPENAI_BASE_URL)
        self._api_key = api_key or os.environ.get(CAW_OPENAI_API_KEY) or os.environ.get("OPENAI_API_KEY")
        self._system_prompt = system_prompt
        self._tools = tools
        self._reasoning = reasoning
        self._workspace = Path(workspace or os.getcwd()).resolve()
        self._max_turns = max_turns
        self._tracing_disabled = tracing_disabled
        self._model_settings = model_settings
        self._run_config = run_config
        self._created_at = datetime.now(timezone.utc).isoformat()
        self._turns: list[Turn] = []
        self._total_usage = UsageStats()
        self._total_duration_ms = 0
        self._input_items: list[Any] = []
        self._last_raw_output = ""
        self._step_callback: Callable[[list], None] | None = None

        if not self._model:
            raise ValueError("OpenAI Agents provider requires a model.")
        if not self._api_key:
            raise ValueError(
                "OpenAI Agents provider requires api_key, CAW_OPENAI_API_KEY, or OPENAI_API_KEY."
            )

    def set_step_callback(self, callback):
        self._step_callback = callback

    def send(self, message: str) -> Turn:
        async def _collect() -> Turn:
            final_turn: Turn | None = None
            async for event in self.stream_async(message):
                if event.is_final and event.turn is not None:
                    final_turn = event.turn
            if final_turn is None:
                raise RuntimeError("OpenAI Agents SDK run ended without a final turn.")
            return final_turn

        return _run_coro_sync(_collect())

    async def stream_async(self, message: str, **kwargs: Any) -> AsyncIterator[AgentStreamEvent]:
        from agents import (
            Agent as SDKAgent,
            AsyncOpenAI,
            ItemHelpers,
            ModelSettings,
            OpenAIChatCompletionsModel,
            RunConfig,
            Runner,
        )

        display = get_global_display()
        if display:
            if not self._turns:
                display.on_metadata(agent="openai_agents", model=self._model, session=self._session_id)
            display.on_user_message(message)

        raw_lines: list[str] = []
        blocks: list[ContentBlock] = []
        tool_blocks: dict[str, ToolUse] = {}

        client = AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)
        sdk_model = OpenAIChatCompletionsModel(model=self._model, openai_client=client)
        sdk_tools = _make_local_tools(self._workspace, self._tools)
        sdk_mcp_servers = self._sdk_mcp_servers()

        sdk_agent = SDKAgent(
            name="caw",
            instructions=self._system_prompt,
            model=sdk_model,
            tools=sdk_tools,
            mcp_servers=sdk_mcp_servers,
        )

        model_settings = kwargs.get("model_settings", self._model_settings)
        if model_settings is None and self._reasoning:
            model_settings = ModelSettings(reasoning={"effort": self._reasoning})

        run_config = kwargs.get("run_config", self._run_config)
        if run_config is None:
            run_config = RunConfig(tracing_disabled=self._tracing_disabled)
        elif getattr(run_config, "tracing_disabled", None) is False and self._tracing_disabled:
            run_config.tracing_disabled = True

        model_settings = model_settings or getattr(run_config, "model_settings", None)
        default_model_settings = ModelSettings(prompt_cache_retention="24h")
        model_settings = (
            default_model_settings
            if model_settings is None
            else default_model_settings.resolve(model_settings)
        )
        run_config.model_settings = model_settings.resolve(ModelSettings(include_usage=True))

        max_turns = int(kwargs.get("max_turns", self._max_turns))
        hooks = kwargs.get("hooks")
        context = kwargs.get("context")
        raw_events = bool(kwargs.get("raw_events", False))

        input_items: str | list[Any]
        if self._input_items:
            input_items = list(self._input_items) + [{"role": "user", "content": message}]
        else:
            input_items = message

        started = time.perf_counter()
        result = Runner.run_streamed(
            sdk_agent,
            input=input_items,
            context=context,
            max_turns=max_turns,
            hooks=hooks,
            run_config=run_config,
        )

        async for raw_event in result.stream_events():
            raw_lines.append(repr(raw_event))

            if raw_events:
                yield AgentStreamEvent(type="raw", raw=raw_event)

            if getattr(raw_event, "type", "") != "run_item_stream_event":
                if getattr(raw_event, "type", "") == "agent_updated_stream_event":
                    yield AgentStreamEvent(type="agent_updated", raw=raw_event)
                continue

            item = getattr(raw_event, "item", None)
            item_type = getattr(item, "type", "")

            if item_type == "tool_call_item":
                block = ToolUse(
                    id=_tool_call_id(getattr(item, "raw_item", None)),
                    name=_tool_call_name(item),
                    arguments=_parse_tool_arguments(getattr(item, "raw_item", None)),
                )
                blocks.append(block)
                tool_blocks[block.id] = block
                if display:
                    display.on_tool_call(block)
                self._emit_step(blocks)
                yield AgentStreamEvent(type="tool_call", block=block, raw=raw_event)

            elif item_type == "tool_call_output_item":
                raw_item = getattr(item, "raw_item", None)
                tool_id = str(_get_field(raw_item, "call_id") or _get_field(raw_item, "tool_call_id") or "")
                block = tool_blocks.get(tool_id)
                if block is None:
                    block = ToolUse(id=tool_id or str(uuid.uuid4()), name=_tool_call_name(item), arguments={})
                    blocks.append(block)
                    tool_blocks[block.id] = block
                block.output = _stringify(getattr(item, "output", None))
                status = str(_get_field(raw_item, "status", "")).lower()
                block.is_error = status in {"failed", "error"}
                if display:
                    display.on_tool_result(block)
                self._emit_step(blocks)
                yield AgentStreamEvent(type="tool_result", block=block, raw=raw_event)

            elif item_type == "message_output_item":
                text = ItemHelpers.text_message_output(item)
                if text:
                    block = TextBlock(text=text)
                    blocks.append(block)
                    if display:
                        display.on_text(block)
                    self._emit_step(blocks)
                    yield AgentStreamEvent(type="text", block=block, raw=raw_event)

            elif item_type == "reasoning_item":
                text = self._reasoning_text(getattr(item, "raw_item", None))
                if text:
                    block = ThinkingBlock(text=text)
                    blocks.append(block)
                    if display:
                        display.on_thinking(block)
                    self._emit_step(blocks)
                    yield AgentStreamEvent(type="thinking", block=block, raw=raw_event)

        self._last_raw_output = "\n".join(raw_lines)
        self._input_items = result.to_input_list()

        usage = self._usage_from_result(result)
        duration_ms = int((time.perf_counter() - started) * 1000)
        turn = Turn(input=message, output=blocks, usage=usage, duration_ms=duration_ms)
        self._turns.append(turn)
        self._total_usage = self._total_usage + usage
        self._total_duration_ms += duration_ms

        if display:
            display.on_turn_end(turn.result, usage, duration_ms)

        yield AgentStreamEvent(type="final", turn=turn, raw=result, is_final=True)

    def _emit_step(self, blocks: list[ContentBlock]) -> None:
        if self._step_callback:
            self._step_callback(list(blocks))

    def _sdk_mcp_servers(self) -> list[Any]:
        from agents.mcp import MCPServerStdio, MCPServerStreamableHttp

        servers = []
        for srv in self._mcp_servers:
            if srv.url:
                servers.append(MCPServerStreamableHttp({"url": srv.url}, name=srv.name))
            else:
                params: dict[str, Any] = {"command": srv.command}
                if srv.args:
                    params["args"] = srv.args
                if srv.env:
                    params["env"] = srv.env
                servers.append(MCPServerStdio(params, name=srv.name))
        return servers

    @staticmethod
    def _reasoning_text(raw_item: Any) -> str:
        summary = _get_field(raw_item, "summary")
        if isinstance(summary, list):
            parts = []
            for item in summary:
                text = _get_field(item, "text")
                if text:
                    parts.append(str(text))
            return "\n".join(parts)
        if summary:
            return _stringify(summary)
        content = _get_field(raw_item, "content")
        return _stringify(content)

    def _usage_from_result(self, result: Any) -> UsageStats:
        usage = UsageStats()
        for response in getattr(result, "raw_responses", []) or []:
            raw_usage = getattr(response, "usage", None)
            if raw_usage is None:
                continue
            cached = getattr(getattr(raw_usage, "input_tokens_details", None), "cached_tokens", 0)
            input_tokens = getattr(raw_usage, "input_tokens", 0) or 0
            usage.input_tokens += max(0, input_tokens - (cached or 0))
            usage.cache_read_tokens += cached or 0
            usage.output_tokens += getattr(raw_usage, "output_tokens", 0) or 0
        usage.cost_usd = compute_cost("openai_agents", self._model, usage)
        return usage

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def last_raw_output(self) -> str:
        return self._last_raw_output

    @property
    def trajectory(self) -> Trajectory:
        return Trajectory(
            agent="openai_agents",
            model=self._model,
            session_id=self._session_id,
            created_at=self._created_at,
            system_prompt=self._system_prompt or "",
            reasoning=self._reasoning or "",
            mcp_servers=list(self._mcp_servers),
            turns=list(self._turns),
            usage=self._total_usage,
            duration_ms=self._total_duration_ms,
            metadata={"base_url": self._base_url or ""},
        )

    def end(self) -> Trajectory:
        return self.trajectory


class OpenAIAgentsProvider(Provider):
    """Provider that runs agents through the OpenAI Agents SDK."""

    @property
    def name(self) -> str:
        return "openai_agents"

    def resolve_model(self, tier: ModelTier) -> str:
        return _MODEL_TIER_MAP[tier]

    def resolve_tool_restrictions(self, tools: ToolGroup) -> dict[str, Any]:
        if not tools:
            raise ValueError("ToolGroup must not be empty — at least one group is required.")
        return {"tools": tools}

    def _limit_probe_kwargs(self) -> dict[str, Any]:
        return {"tools": ToolGroup.READER, "max_turns": 1}

    def start_session(self, mcp_servers: list[MCPServer], **kwargs: Any) -> OpenAIAgentsSession:
        return OpenAIAgentsSession(
            mcp_servers=mcp_servers,
            model=kwargs.get("model") or "",
            base_url=kwargs.get("base_url"),
            api_key=kwargs.get("api_key"),
            system_prompt=kwargs.get("system_prompt"),
            session_id=kwargs.get("session_id"),
            tools=kwargs.get("tools", ToolGroup.ALL - ToolGroup.INTERACTION),
            reasoning=kwargs.get("reasoning"),
            workspace=kwargs.get("workspace"),
            max_turns=kwargs.get("max_turns", 10),
            tracing_disabled=kwargs.get("tracing_disabled", True),
            model_settings=kwargs.get("model_settings"),
            run_config=kwargs.get("run_config"),
        )
