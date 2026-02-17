"""Tests for ToolGroup abstraction and provider resolution."""

import pytest

from caw.models import TextBlock, ToolGroup, Trajectory, Turn
from caw.providers.claude_code import ClaudeCodeProvider
from caw.providers.codex import CodexProvider


# -- ToolGroup arithmetic -----------------------------------------------------


class TestToolGroupArithmetic:
    def test_all_minus_writer(self):
        result = ToolGroup.ALL - ToolGroup.WRITER
        assert result == (
            ToolGroup.READER | ToolGroup.EXEC | ToolGroup.WEB | ToolGroup.PARALLEL | ToolGroup.INTERACTION
        )

    def test_all_minus_all_is_falsy(self):
        result = ToolGroup.ALL - ToolGroup.ALL
        assert not result

    def test_all_minus_interaction_is_default(self):
        default = ToolGroup.ALL - ToolGroup.INTERACTION
        assert default == (ToolGroup.READER | ToolGroup.WRITER | ToolGroup.EXEC | ToolGroup.WEB | ToolGroup.PARALLEL)

    def test_union(self):
        result = ToolGroup.READER | ToolGroup.EXEC
        assert result & ToolGroup.READER
        assert result & ToolGroup.EXEC
        assert not (result & ToolGroup.WRITER)

    def test_subtract_returns_not_implemented_for_non_toolgroup(self):
        result = ToolGroup.ALL.__sub__(42)
        assert result is NotImplemented

    def test_all_contains_every_group(self):
        for group in (
            ToolGroup.READER,
            ToolGroup.WRITER,
            ToolGroup.EXEC,
            ToolGroup.WEB,
            ToolGroup.PARALLEL,
            ToolGroup.INTERACTION,
        ):
            assert ToolGroup.ALL & group


# -- Claude Code provider resolution ------------------------------------------


class TestClaudeCodeResolution:
    def setup_method(self):
        self.provider = ClaudeCodeProvider()

    def test_reader_only(self):
        result = self.provider.resolve_tool_restrictions(ToolGroup.READER)
        disallowed = result["disallowed_tools"]
        # Should disallow all non-reader tools
        assert "Write" in disallowed
        assert "Edit" in disallowed
        assert "NotebookEdit" in disallowed
        assert "Bash" in disallowed
        assert "WebFetch" in disallowed
        assert "WebSearch" in disallowed
        assert "Task" in disallowed
        assert "TaskOutput" in disallowed
        assert "TaskStop" in disallowed
        assert "AskUserQuestion" in disallowed
        # Reader tools should NOT be disallowed
        assert "Read" not in disallowed
        assert "Glob" not in disallowed
        assert "Grep" not in disallowed

    def test_default_disallows_interaction(self):
        default = ToolGroup.ALL - ToolGroup.INTERACTION
        result = self.provider.resolve_tool_restrictions(default)
        assert result == {"disallowed_tools": ["AskUserQuestion"]}

    def test_all_returns_empty(self):
        result = self.provider.resolve_tool_restrictions(ToolGroup.ALL)
        assert result == {}

    def test_empty_raises(self):
        empty = ToolGroup.ALL - ToolGroup.ALL
        with pytest.raises(ValueError):
            self.provider.resolve_tool_restrictions(empty)

    def test_reader_exec(self):
        tools = ToolGroup.READER | ToolGroup.EXEC
        result = self.provider.resolve_tool_restrictions(tools)
        disallowed = result["disallowed_tools"]
        assert "Read" not in disallowed
        assert "Bash" not in disallowed
        assert "Write" in disallowed
        assert "WebFetch" in disallowed


# -- Codex provider resolution ------------------------------------------------


class TestCodexResolution:
    def setup_method(self):
        self.provider = CodexProvider()

    def test_reader_only(self):
        result = self.provider.resolve_tool_restrictions(ToolGroup.READER)
        assert result == {"sandbox": "read-only"}

    def test_writer_no_exec(self):
        result = self.provider.resolve_tool_restrictions(ToolGroup.READER | ToolGroup.WRITER)
        assert result == {"sandbox": "workspace-write"}

    def test_exec_implies_full_access(self):
        result = self.provider.resolve_tool_restrictions(ToolGroup.READER | ToolGroup.EXEC)
        assert result == {"sandbox": "danger-full-access"}

    def test_all_returns_empty(self):
        result = self.provider.resolve_tool_restrictions(ToolGroup.ALL)
        assert result == {}

    def test_empty_raises(self):
        empty = ToolGroup.ALL - ToolGroup.ALL
        with pytest.raises(ValueError):
            self.provider.resolve_tool_restrictions(empty)


# -- Trajectory validation properties -----------------------------------------


class TestTrajectoryValidation:
    def test_empty_trajectory_not_usage_limited(self):
        traj = Trajectory(agent="test")
        assert not traj.is_usage_limited

    def test_empty_trajectory_not_complete(self):
        traj = Trajectory(agent="test")
        assert not traj.is_complete

    def test_normal_trajectory_is_complete(self):
        turn = Turn(input="hello", output=[TextBlock(text="world")])
        traj = Trajectory(agent="test", turns=[turn], completed_at="2025-06-15T00:00:00+00:00")
        assert traj.is_complete
        assert not traj.is_usage_limited

    def test_usage_limited_trajectory(self):
        turn = Turn(
            input="hello",
            output=[TextBlock(text="You've hit your usage limit. It resets at 3am (UTC).")],
        )
        traj = Trajectory(agent="test", turns=[turn], completed_at="2025-06-15T00:00:00+00:00", usage_limited=True)
        assert traj.is_usage_limited
        assert not traj.is_complete

    def test_metadata_preserved(self):
        traj = Trajectory(agent="test", metadata={"task_id": 42, "iteration": 3})
        assert traj.metadata == {"task_id": 42, "iteration": 3}
        d = traj.to_dict()
        restored = Trajectory.from_dict(d)
        assert restored.metadata == {"task_id": 42, "iteration": 3}


# -- Session metadata via providers -------------------------------------------


class TestSessionMetadata:
    """Metadata is owned by the core Session, not the provider."""

    def test_session_metadata_merges_onto_trajectory(self):
        from unittest.mock import MagicMock
        from caw.agent import Session

        mock_ps = MagicMock()
        mock_ps.trajectory = Trajectory(agent="test", metadata={"provider_key": "p"})
        session = Session(mock_ps, metadata={"user_key": "u"})
        traj = session.trajectory
        assert traj.metadata == {"provider_key": "p", "user_key": "u"}

    def test_session_metadata_overrides_provider(self):
        from unittest.mock import MagicMock
        from caw.agent import Session

        mock_ps = MagicMock()
        mock_ps.trajectory = Trajectory(agent="test", metadata={"key": "provider"})
        session = Session(mock_ps, metadata={"key": "session"})
        assert session.trajectory.metadata["key"] == "session"

    def test_no_metadata(self):
        from unittest.mock import MagicMock
        from caw.agent import Session

        mock_ps = MagicMock()
        mock_ps.trajectory = Trajectory(agent="test")
        session = Session(mock_ps)
        assert session.trajectory.metadata == {}

    def test_provider_session_no_metadata(self):
        from caw.providers.claude_code import ClaudeCodeSession

        session = ClaudeCodeSession(mcp_servers=[])
        assert session.trajectory.metadata == {}

    def test_codex_provider_session_no_metadata(self):
        from caw.providers.codex import CodexSession

        session = CodexSession(mcp_servers=[])
        assert session.trajectory.metadata == {}


# -- Claude Code disallowed tools in CLI args ----------------------------------


class TestClaudeCodeDisallowedTools:
    def test_session_stores_disallowed_tools(self):
        from caw.providers.claude_code import ClaudeCodeSession

        session = ClaudeCodeSession(
            mcp_servers=[],
            disallowed_tools=["Write", "Edit"],
        )
        assert session._disallowed_tools == ["Write", "Edit"]

    def test_session_no_disallowed_tools(self):
        from caw.providers.claude_code import ClaudeCodeSession

        session = ClaudeCodeSession(mcp_servers=[])
        assert session._disallowed_tools is None
