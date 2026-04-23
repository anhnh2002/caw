from __future__ import annotations

from types import SimpleNamespace

import pytest

from caw import Agent, AgentStreamEvent, ToolGroup
from caw.models import TextBlock, ToolUse
from caw.providers.openai_agents import OpenAIAgentsProvider, OpenAIAgentsSession, _make_local_tools


def test_provider_aliases_registered():
    assert Agent(provider="openai_agents").provider.name == "openai_agents"
    assert Agent(provider="openai_agent").provider.name == "openai_agents"
    assert Agent(provider="oa").provider.name == "openai_agents"


def test_requires_api_key(monkeypatch):
    monkeypatch.delenv("CAW_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="requires api_key"):
        OpenAIAgentsSession(mcp_servers=[], model="test-model")


def test_uses_env_config(monkeypatch):
    monkeypatch.setenv("CAW_OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("CAW_OPENAI_BASE_URL", "https://example.test/v1")
    session = OpenAIAgentsSession(mcp_servers=[], model="test-model")
    assert session._api_key == "env-key"
    assert session._base_url == "https://example.test/v1"


def test_tool_group_maps_to_session_kwargs():
    provider = OpenAIAgentsProvider()
    assert provider.resolve_tool_restrictions(ToolGroup.READER) == {"tools": ToolGroup.READER}
    with pytest.raises(ValueError):
        provider.resolve_tool_restrictions(ToolGroup.ALL - ToolGroup.ALL)


def test_provider_start_session_uses_session_default_max_turns():
    session = OpenAIAgentsProvider().start_session(
        mcp_servers=[],
        model="test-model",
        api_key="key",
    )

    assert session._max_turns == 100


def test_provider_start_session_preserves_explicit_max_turns():
    session = OpenAIAgentsProvider().start_session(
        mcp_servers=[],
        model="test-model",
        api_key="key",
        max_turns=12,
    )

    assert session._max_turns == 12


def test_local_tools_are_gated_by_group(tmp_path):
    reader = _make_local_tools(tmp_path, ToolGroup.READER)
    reader_names = {tool.name for tool in reader}
    assert {"read_file", "list_dir", "glob", "grep"} <= reader_names
    assert "write_file" not in reader_names

    full = _make_local_tools(tmp_path, ToolGroup.READER | ToolGroup.WRITER | ToolGroup.EXEC)
    full_names = {tool.name for tool in full}
    assert {"write_file", "edit_file", "run_shell"} <= full_names


@pytest.mark.asyncio
async def test_stream_async_converts_sdk_events(monkeypatch):
    monkeypatch.setattr("agents.ItemHelpers.text_message_output", lambda item: item.text)

    class FakeResult:
        raw_responses = []

        async def stream_events(self):
            yield SimpleNamespace(
                type="run_item_stream_event",
                item=SimpleNamespace(
                    type="tool_call_item",
                    raw_item={"call_id": "call_1", "name": "read_file", "arguments": '{"path":"README.md"}'},
                ),
            )
            yield SimpleNamespace(
                type="run_item_stream_event",
                item=SimpleNamespace(
                    type="tool_call_output_item",
                    raw_item={"call_id": "call_1"},
                    output="file contents",
                ),
            )
            yield SimpleNamespace(
                type="run_item_stream_event",
                item=SimpleNamespace(type="message_output_item", raw_item=None, text="done"),
            )

        def to_input_list(self):
            return [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "done"}]

    monkeypatch.setattr("agents.Runner.run_streamed", lambda *args, **kwargs: FakeResult())

    session = OpenAIAgentsSession(mcp_servers=[], model="test-model", api_key="key")
    events: list[AgentStreamEvent] = []
    async for event in session.stream_async("hi", max_turns=3):
        events.append(event)

    assert [event.type for event in events] == ["tool_call", "tool_result", "text", "final"]
    assert isinstance(events[0].block, ToolUse)
    assert events[0].block.name == "read_file"
    assert events[0].block.arguments == {"path": "README.md"}
    assert events[1].block.output == "file contents"
    assert isinstance(events[2].block, TextBlock)
    assert events[-1].turn is not None
    assert events[-1].turn.result == "done"
    assert session.trajectory.num_turns == 1


@pytest.mark.asyncio
async def test_stream_async_sets_default_model_settings(monkeypatch):
    from agents import ModelSettings

    monkeypatch.setattr("agents.ItemHelpers.text_message_output", lambda item: item.text)
    captured = {}

    class FakeResult:
        raw_responses = []

        async def stream_events(self):
            yield SimpleNamespace(
                type="run_item_stream_event",
                item=SimpleNamespace(type="message_output_item", raw_item=None, text="done"),
            )

        def to_input_list(self):
            return [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "done"}]

    def fake_run_streamed(*args, **kwargs):
        captured["run_config"] = kwargs["run_config"]
        return FakeResult()

    monkeypatch.setattr("agents.Runner.run_streamed", fake_run_streamed)

    model_settings = ModelSettings(temperature=0.2)
    session = OpenAIAgentsSession(
        mcp_servers=[],
        model="test-model",
        api_key="key",
        model_settings=model_settings,
    )
    events = [event async for event in session.stream_async("hi")]

    assert events[-1].is_final
    assert captured["run_config"].model_settings.temperature == 0.2
    assert captured["run_config"].model_settings.include_usage is True
    assert captured["run_config"].model_settings.prompt_cache_retention == "24h"


@pytest.mark.asyncio
async def test_stream_async_preserves_explicit_prompt_cache_retention(monkeypatch):
    from agents import ModelSettings

    monkeypatch.setattr("agents.ItemHelpers.text_message_output", lambda item: item.text)
    captured = {}

    class FakeResult:
        raw_responses = []

        async def stream_events(self):
            yield SimpleNamespace(
                type="run_item_stream_event",
                item=SimpleNamespace(type="message_output_item", raw_item=None, text="done"),
            )

        def to_input_list(self):
            return [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "done"}]

    def fake_run_streamed(*args, **kwargs):
        captured["run_config"] = kwargs["run_config"]
        return FakeResult()

    monkeypatch.setattr("agents.Runner.run_streamed", fake_run_streamed)

    session = OpenAIAgentsSession(
        mcp_servers=[],
        model="test-model",
        api_key="key",
        model_settings=ModelSettings(prompt_cache_retention="in_memory", include_usage=False),
    )
    events = [event async for event in session.stream_async("hi")]

    assert events[-1].is_final
    assert captured["run_config"].model_settings.prompt_cache_retention == "in_memory"
    assert captured["run_config"].model_settings.include_usage is True


@pytest.mark.asyncio
async def test_stream_async_connects_and_cleans_up_mcp_servers(monkeypatch):
    monkeypatch.setattr("agents.ItemHelpers.text_message_output", lambda item: item.text)
    captured = {}

    class FakeMCPServer:
        def __init__(self):
            self.connected = False
            self.cleaned_up = False

        async def __aenter__(self):
            self.connected = True
            return self

        async def __aexit__(self, exc_type, exc_value, traceback):
            self.cleaned_up = True
            self.connected = False

    fake_mcp_server = FakeMCPServer()

    class FakeResult:
        raw_responses = []

        async def stream_events(self):
            yield SimpleNamespace(
                type="run_item_stream_event",
                item=SimpleNamespace(type="message_output_item", raw_item=None, text="done"),
            )

        def to_input_list(self):
            return [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "done"}]

    def fake_run_streamed(agent, *args, **kwargs):
        captured["server_connected_during_run"] = agent.mcp_servers[0].connected
        return FakeResult()

    monkeypatch.setattr("agents.Runner.run_streamed", fake_run_streamed)

    session = OpenAIAgentsSession(mcp_servers=[], model="test-model", api_key="key")
    monkeypatch.setattr(session, "_sdk_mcp_servers", lambda: [fake_mcp_server])
    events = [event async for event in session.stream_async("hi")]

    assert events[-1].is_final
    assert captured["server_connected_during_run"] is True
    assert fake_mcp_server.cleaned_up is True
    assert fake_mcp_server.connected is False


def test_usage_from_result_uses_openai_agents_pricing():
    session = OpenAIAgentsSession(
        mcp_servers=[],
        model="anthropic/claude-sonnet-4-6",
        api_key="key",
    )
    result = SimpleNamespace(
        raw_responses=[
            SimpleNamespace(
                usage=SimpleNamespace(
                    input_tokens=110,
                    output_tokens=20,
                    input_tokens_details=SimpleNamespace(cached_tokens=10),
                )
            )
        ]
    )

    usage = session._usage_from_result(result)

    assert usage.input_tokens == 100
    assert usage.cache_read_tokens == 10
    assert usage.output_tokens == 20
    assert usage.cost_usd == pytest.approx(0.000603)


def test_send_collects_stream(monkeypatch):
    monkeypatch.setattr("agents.ItemHelpers.text_message_output", lambda item: item.text)

    class FakeResult:
        raw_responses = []

        async def stream_events(self):
            yield SimpleNamespace(
                type="run_item_stream_event",
                item=SimpleNamespace(type="message_output_item", raw_item=None, text="sync done"),
            )

        def to_input_list(self):
            return [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "sync done"}]

    monkeypatch.setattr("agents.Runner.run_streamed", lambda *args, **kwargs: FakeResult())

    session = OpenAIAgentsSession(mcp_servers=[], model="test-model", api_key="key")
    turn = session.send("hi")
    assert turn.result == "sync done"
    assert session.trajectory.num_turns == 1


@pytest.mark.asyncio
async def test_local_read_tool_invocation(tmp_path):
    from agents.tool_context import ToolContext

    (tmp_path / "a.txt").write_text("alpha")
    tools = _make_local_tools(tmp_path, ToolGroup.READER)
    read_tool = next(tool for tool in tools if tool.name == "read_file")
    ctx = ToolContext(context=None, tool_name="read_file", tool_call_id="call_1", tool_arguments='{"path":"a.txt"}')
    output = await read_tool.on_invoke_tool(ctx, '{"path":"a.txt"}')
    assert output == "alpha"
