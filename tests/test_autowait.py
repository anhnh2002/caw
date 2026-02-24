"""Unit tests for the auto-wait-on-usage-limit feature."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from caw.providers.claude_code import (
    _DEFAULT_WAIT_MINUTES,
    _parse_reset_minutes,
    detect_usage_limit,
)
from caw.providers.codex import (
    _DEFAULT_WAIT_MINUTES as _CODEX_DEFAULT_WAIT_MINUTES,
    _parse_codex_reset_minutes,
    detect_codex_usage_limit,
)


# ---------------------------------------------------------------------------
# detect_usage_limit (free function in claude_code provider)
# ---------------------------------------------------------------------------


class TestDetectUsageLimit:
    """Tests for the Claude Code detection function."""

    def test_no_limit(self):
        assert detect_usage_limit("Here is your answer.") is None

    def test_missing_resets(self):
        assert detect_usage_limit("You've hit your limit.") is None

    def test_missing_limit(self):
        assert detect_usage_limit("The timer resets at 3am.") is None

    def test_simple_limit_message(self):
        text = "You've hit your usage limit · resets 3am (UTC)"
        result = detect_usage_limit(text)
        assert result is not None
        assert isinstance(result, int)
        assert result >= 1

    def test_limit_with_unparseable_time_returns_default(self):
        # Contains both keywords but no parseable reset time
        text = "You've hit your limit and it resets sometime soon"
        result = detect_usage_limit(text)
        assert result == _DEFAULT_WAIT_MINUTES


# ---------------------------------------------------------------------------
# _parse_reset_minutes
# ---------------------------------------------------------------------------


class TestParseResetMinutes:
    """Tests for the reset-time parser."""

    def test_no_match(self):
        assert _parse_reset_minutes("No reset info here") is None

    def test_utc_am(self):
        text = "resets 3am (UTC)"
        result = _parse_reset_minutes(text)
        assert result is not None
        assert result >= 1

    def test_utc_pm(self):
        text = "resets 3pm (UTC)"
        result = _parse_reset_minutes(text)
        assert result is not None
        assert result >= 1

    def test_with_minutes(self):
        text = "resets 3:30am (UTC)"
        result = _parse_reset_minutes(text)
        assert result is not None
        assert result >= 1

    def test_12am(self):
        text = "resets 12am (UTC)"
        result = _parse_reset_minutes(text)
        assert result is not None
        assert result >= 1

    def test_12pm(self):
        text = "resets 12pm (UTC)"
        result = _parse_reset_minutes(text)
        assert result is not None
        assert result >= 1

    def test_unknown_timezone_returns_none(self):
        text = "resets 3am (Mars/Olympus)"
        result = _parse_reset_minutes(text)
        assert result is None

    def test_includes_buffer(self):
        """The returned minutes should include a 5-minute safety buffer."""
        from datetime import datetime, timedelta, timezone

        text = "resets 3am (UTC)"
        result = _parse_reset_minutes(text)
        assert result is not None

        now = datetime.now(timezone.utc)
        reset = now.replace(hour=3, minute=0, second=0, microsecond=0)
        if reset <= now:
            reset += timedelta(days=1)
        raw_minutes = int((reset - now).total_seconds() / 60)

        assert result == raw_minutes + 5


# ---------------------------------------------------------------------------
# CAW_AUTOWAIT env var handling
# ---------------------------------------------------------------------------


class TestAutoWaitEnvVar:
    """Tests for CAW_AUTOWAIT environment variable resolution."""

    def test_default_is_auto_wait_on(self):
        from caw import Agent

        agent = Agent()
        # auto_wait should not be explicitly set to False in kwargs
        assert agent._kwargs.get("auto_wait") is not False

    @pytest.mark.parametrize("env_val", ["0", "false", "False", "no", "off"])
    def test_env_disables_auto_wait(self, monkeypatch, env_val):
        from caw import Agent

        monkeypatch.setenv("CAW_AUTOWAIT", env_val)
        agent = Agent()
        assert agent._kwargs["auto_wait"] is False

    @pytest.mark.parametrize("env_val", ["1", "true", "yes", ""])
    def test_env_does_not_disable_auto_wait(self, monkeypatch, env_val):
        from caw import Agent

        monkeypatch.setenv("CAW_AUTOWAIT", env_val)
        agent = Agent()
        assert agent._kwargs.get("auto_wait") is not False


# ---------------------------------------------------------------------------
# ProviderSession.detect_usage_limit default
# ---------------------------------------------------------------------------


class TestProviderSessionDefault:
    """The base ProviderSession.detect_usage_limit returns None."""

    def test_default_returns_none(self):
        from caw.models import TextBlock, Turn
        from caw.providers.claude_code import ClaudeCodeSession

        # A non-limit turn should return None
        session = ClaudeCodeSession(mcp_servers=[])
        turn = Turn(input="hi", output=[TextBlock(text="hello")])
        assert session.detect_usage_limit(turn) is None

    def test_detects_limit(self):
        from caw.models import TextBlock, Turn
        from caw.providers.claude_code import ClaudeCodeSession

        session = ClaudeCodeSession(mcp_servers=[])
        turn = Turn(
            input="hi",
            output=[TextBlock(text="You've hit your usage limit · resets 3am (UTC)")],
        )
        result = session.detect_usage_limit(turn)
        assert result is not None
        assert result >= 1


# ---------------------------------------------------------------------------
# Core Session auto-wait loop
# ---------------------------------------------------------------------------


class TestCoreSessionAutoWait:
    """Tests that the core Session.send() drives the wait loop."""

    @patch("caw.agent.time.sleep")
    def test_codex_auto_wait_sleeps_and_retries(self, mock_sleep):
        """Codex usage-limit turns trigger the same auto-wait loop."""
        from caw.agent import Session, _AUTO_WAIT_RESUME_MESSAGE
        from caw.models import TextBlock, Turn

        limit_turn = Turn(
            input="hi",
            output=[
                TextBlock(
                    text="You've hit your usage limit. Visit https://chatgpt.com/codex/settings/usage to purchase more credits or try again at 3:47 PM."
                )
            ],
        )
        ok_turn = Turn(input=_AUTO_WAIT_RESUME_MESSAGE, output=[TextBlock(text="Done!")])

        mock_ps = MagicMock()
        mock_ps.send.side_effect = [limit_turn, ok_turn]
        # Simulate CodexSession.detect_usage_limit behaviour
        mock_ps.detect_usage_limit.side_effect = lambda t: detect_codex_usage_limit(t.result)
        mock_ps.last_raw_output = ""
        mock_ps.trajectory = MagicMock()

        session = Session(mock_ps, auto_wait=True)
        result = session.send("hi")

        assert result is ok_turn
        mock_sleep.assert_called_once()
        assert mock_ps.send.call_count == 2

    def test_session_auto_wait_true_by_default(self):
        from caw.agent import Session

        mock_ps = MagicMock()
        session = Session(mock_ps)
        assert session._auto_wait is True

    def test_session_auto_wait_false(self):
        from caw.agent import Session

        mock_ps = MagicMock()
        session = Session(mock_ps, auto_wait=False)
        assert session._auto_wait is False

    def test_auto_wait_off_skips_detection(self):
        """When auto_wait=False, detect_usage_limit is never called."""
        from caw.agent import Session
        from caw.models import TextBlock, Turn

        limit_turn = Turn(
            input="hi",
            output=[TextBlock(text="You've hit your limit · resets 3am (UTC)")],
        )

        mock_ps = MagicMock()
        mock_ps.send.return_value = limit_turn
        mock_ps.detect_usage_limit.return_value = 60
        mock_ps.last_raw_output = ""
        mock_ps.trajectory = MagicMock()

        session = Session(mock_ps, auto_wait=False)
        result = session.send("hi")

        assert result is limit_turn
        mock_ps.detect_usage_limit.assert_not_called()

    @patch("caw.agent.time.sleep")
    def test_auto_wait_sleeps_and_retries(self, mock_sleep):
        """When auto_wait=True and provider detects a limit, Session sleeps and retries."""
        from caw.agent import Session, _AUTO_WAIT_RESUME_MESSAGE
        from caw.models import TextBlock, Turn

        limit_turn = Turn(
            input="hi",
            output=[TextBlock(text="You've hit your limit · resets 3am (UTC)")],
        )
        ok_turn = Turn(input=_AUTO_WAIT_RESUME_MESSAGE, output=[TextBlock(text="Done!")])

        mock_ps = MagicMock()
        mock_ps.send.side_effect = [limit_turn, ok_turn]
        mock_ps.detect_usage_limit.side_effect = [5, None]
        mock_ps.last_raw_output = ""
        mock_ps.trajectory = MagicMock()

        session = Session(mock_ps, auto_wait=True)
        result = session.send("hi")

        assert result is ok_turn
        mock_sleep.assert_called_once_with(5 * 60)
        assert mock_ps.send.call_count == 2
        assert mock_ps.send.call_args_list[1][0][0] == _AUTO_WAIT_RESUME_MESSAGE


# ---------------------------------------------------------------------------
# Codex usage-limit detection
# ---------------------------------------------------------------------------


class TestDetectCodexUsageLimit:
    """Tests for the Codex detection function."""

    def test_no_limit(self):
        assert detect_codex_usage_limit("Here is your answer.") is None

    def test_missing_usage_limit_phrase(self):
        assert detect_codex_usage_limit("try again at 3:47 PM") is None

    def test_simple_limit_message(self):
        text = (
            "You've hit your usage limit. Visit https://chatgpt.com/codex/settings/usage "
            "to purchase more credits or try again at 3:47 PM."
        )
        result = detect_codex_usage_limit(text)
        assert result is not None
        assert isinstance(result, int)
        assert result >= 1

    def test_limit_without_time_returns_default(self):
        text = "You've hit your usage limit. Please wait."
        result = detect_codex_usage_limit(text)
        assert result == _CODEX_DEFAULT_WAIT_MINUTES


class TestParseCodexResetMinutes:
    """Tests for the Codex reset-time parser."""

    def test_no_match(self):
        assert _parse_codex_reset_minutes("No reset info here") is None

    def test_pm_time(self):
        text = "try again at 3:47 PM"
        result = _parse_codex_reset_minutes(text)
        assert result is not None
        assert result >= 1

    def test_am_time(self):
        text = "try again at 6:00 AM"
        result = _parse_codex_reset_minutes(text)
        assert result is not None
        assert result >= 1

    def test_12pm(self):
        text = "try again at 12:00 PM"
        result = _parse_codex_reset_minutes(text)
        assert result is not None
        assert result >= 1

    def test_12am(self):
        text = "try again at 12:00 AM"
        result = _parse_codex_reset_minutes(text)
        assert result is not None
        assert result >= 1

    def test_includes_buffer(self):
        """The returned minutes should include a 5-minute safety buffer."""
        from datetime import datetime, timedelta

        text = "try again at 3:47 PM"
        result = _parse_codex_reset_minutes(text)
        assert result is not None

        now = datetime.now()
        reset = now.replace(hour=15, minute=47, second=0, microsecond=0)
        if reset <= now:
            reset += timedelta(days=1)
        raw_minutes = int((reset - now).total_seconds() / 60)

        assert result == raw_minutes + 5


class TestCodexSessionDetectUsageLimit:
    """Tests for CodexSession.detect_usage_limit override."""

    def test_returns_none_for_normal_turn(self):
        from caw.models import TextBlock, Turn
        from caw.providers.codex import CodexSession

        session = CodexSession(mcp_servers=[])
        turn = Turn(input="hi", output=[TextBlock(text="hello")])
        assert session.detect_usage_limit(turn) is None

    def test_detects_limit(self):
        from caw.models import TextBlock, Turn
        from caw.providers.codex import CodexSession

        session = CodexSession(mcp_servers=[])
        turn = Turn(
            input="hi",
            output=[
                TextBlock(
                    text="You've hit your usage limit. Visit https://chatgpt.com/codex/settings/usage to purchase more credits or try again at 3:47 PM."
                )
            ],
        )
        result = session.detect_usage_limit(turn)
        assert result is not None
        assert result >= 1
