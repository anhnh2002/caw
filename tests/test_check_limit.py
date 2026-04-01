"""Unit tests for the check_limit API."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from caw.models import ModelTier, TextBlock, Turn
from caw.provider import ProviderSession


# ---------------------------------------------------------------------------
# Provider.check_limit
# ---------------------------------------------------------------------------


class TestProviderCheckLimit:
    """Tests for Provider.check_limit via concrete providers."""

    def test_no_limit_returns_none(self):
        """When the probe turn has no limit, check_limit returns None."""
        from caw.providers.claude_code import ClaudeCodeProvider

        ok_turn = Turn(input="hi", output=[TextBlock(text="ok")])

        with patch.object(ClaudeCodeProvider, "start_session") as mock_start:
            mock_session = MagicMock(spec=ProviderSession)
            mock_session.send.return_value = ok_turn
            mock_session.detect_usage_limit.return_value = None
            mock_start.return_value = mock_session

            provider = ClaudeCodeProvider()
            result = provider.check_limit(model="claude-sonnet-4-6")

            assert result is None
            mock_start.assert_called_once()
            mock_session.send.assert_called_once_with("hi")
            mock_session.detect_usage_limit.assert_called_once_with(ok_turn)
            mock_session.end.assert_called_once()

    def test_limit_returns_minutes(self):
        """When the probe turn hits a limit, check_limit returns wait minutes."""
        from caw.providers.claude_code import ClaudeCodeProvider

        limit_turn = Turn(
            input="hi",
            output=[TextBlock(text="You've hit your usage limit \u00b7 resets 3am (UTC)")],
        )

        with patch.object(ClaudeCodeProvider, "start_session") as mock_start:
            mock_session = MagicMock(spec=ProviderSession)
            mock_session.send.return_value = limit_turn
            mock_session.detect_usage_limit.return_value = 42
            mock_start.return_value = mock_session

            provider = ClaudeCodeProvider()
            result = provider.check_limit(model="claude-opus-4-6")

            assert result == 42
            mock_session.end.assert_called_once()

    def test_session_ended_on_error(self):
        """The probe session is always cleaned up, even on send() error."""
        from caw.providers.claude_code import ClaudeCodeProvider

        with patch.object(ClaudeCodeProvider, "start_session") as mock_start:
            mock_session = MagicMock(spec=ProviderSession)
            mock_session.send.side_effect = RuntimeError("cli exploded")
            mock_start.return_value = mock_session

            provider = ClaudeCodeProvider()
            with pytest.raises(RuntimeError, match="cli exploded"):
                provider.check_limit()

            mock_session.end.assert_called_once()

    def test_display_suppressed_during_probe(self):
        """The global display is suppressed during the probe and restored after."""
        from caw.display import get_global_display, set_global_display
        from caw.providers.claude_code import ClaudeCodeProvider

        sentinel_display = MagicMock()
        set_global_display(sentinel_display)

        ok_turn = Turn(input="hi", output=[TextBlock(text="ok")])
        captured_displays = []

        def fake_send(msg):
            captured_displays.append(get_global_display())
            return ok_turn

        try:
            with patch.object(ClaudeCodeProvider, "start_session") as mock_start:
                mock_session = MagicMock(spec=ProviderSession)
                mock_session.send.side_effect = fake_send
                mock_session.detect_usage_limit.return_value = None
                mock_start.return_value = mock_session

                provider = ClaudeCodeProvider()
                provider.check_limit()

            # During the probe, display should have been None
            assert captured_displays == [None]
            # After the probe, display should be restored
            assert get_global_display() is sentinel_display
        finally:
            set_global_display(None)

    def test_claude_probe_disallows_all_tools(self):
        """ClaudeCodeProvider passes disallowed_tools covering all groups."""
        from caw.providers.claude_code import ClaudeCodeProvider, _TOOL_GROUP_MAP

        ok_turn = Turn(input="hi", output=[TextBlock(text="ok")])

        with patch.object(ClaudeCodeProvider, "start_session") as mock_start:
            mock_session = MagicMock(spec=ProviderSession)
            mock_session.send.return_value = ok_turn
            mock_session.detect_usage_limit.return_value = None
            mock_start.return_value = mock_session

            provider = ClaudeCodeProvider()
            provider.check_limit()

            call_kwargs = mock_start.call_args
            assert call_kwargs.kwargs.get("disallowed_tools") or "disallowed_tools" in (
                call_kwargs[1] if len(call_kwargs) > 1 else {}
            )

            # Verify all tool names are covered
            all_expected: list[str] = []
            for names in _TOOL_GROUP_MAP.values():
                all_expected.extend(names)

            passed = call_kwargs.kwargs.get("disallowed_tools", [])
            assert set(passed) == set(all_expected)

    def test_codex_probe_uses_readonly_sandbox(self):
        """CodexProvider passes sandbox='read-only' for the probe."""
        from caw.providers.codex import CodexProvider

        ok_turn = Turn(input="hi", output=[TextBlock(text="ok")])

        with patch.object(CodexProvider, "start_session") as mock_start:
            mock_session = MagicMock(spec=ProviderSession)
            mock_session.send.return_value = ok_turn
            mock_session.detect_usage_limit.return_value = None
            mock_start.return_value = mock_session

            provider = CodexProvider()
            provider.check_limit()

            call_kwargs = mock_start.call_args
            assert call_kwargs.kwargs.get("sandbox") == "read-only"


# ---------------------------------------------------------------------------
# Agent.check_limit
# ---------------------------------------------------------------------------


class TestAgentCheckLimit:
    """Tests for Agent.check_limit."""

    def test_delegates_to_provider(self):
        from caw import Agent

        with patch("caw.agent._resolve_provider") as mock_resolve:
            mock_provider = MagicMock()
            mock_provider.check_limit.return_value = None
            mock_resolve.return_value = mock_provider

            agent = Agent(model="some-model")
            result = agent.check_limit()

            assert result is None
            mock_provider.check_limit.assert_called_once_with(model="some-model")

    def test_resolves_model_tier(self):
        from caw import Agent

        with patch("caw.agent._resolve_provider") as mock_resolve:
            mock_provider = MagicMock()
            mock_provider.resolve_model.return_value = "concrete-model"
            mock_provider.check_limit.return_value = None
            mock_resolve.return_value = mock_provider

            agent = Agent(model=ModelTier.STRONGEST)
            result = agent.check_limit()

            assert result is None
            mock_provider.resolve_model.assert_called_once_with(ModelTier.STRONGEST)
            mock_provider.check_limit.assert_called_once_with(model="concrete-model")

    def test_returns_wait_minutes(self):
        from caw import Agent

        with patch("caw.agent._resolve_provider") as mock_resolve:
            mock_provider = MagicMock()
            mock_provider.check_limit.return_value = 30
            mock_resolve.return_value = mock_provider

            agent = Agent()
            result = agent.check_limit()

            assert result == 30
