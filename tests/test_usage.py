"""End-to-end usage examples for the caw library.

Each test reads like a real user scenario.
These tests are meant to be run against a live provider backend.
"""

from __future__ import annotations

import pytest

from caw import Agent, MCPServer


# ===========================================================================
# Basic usage
# ===========================================================================


def test_basic_usage():
    """Simplest possible usage — one agent, one turn."""
    agent = Agent()

    with agent.start_session() as session:
        turn = session.send("What is 1 + 1?")
        assert turn.result  # agent responded with something

    traj = session.trajectory
    assert traj.num_turns == 1


def test_explicit_provider():
    """Pick a specific provider."""
    agent = Agent(provider="claude_code")

    with agent.start_session() as session:
        turn = session.send("Say hello")
        assert turn.result


def test_set_provider():
    """set_provider() switches the backend before starting a session."""
    agent = Agent()
    agent.set_provider("claude_code")

    with agent.start_session() as session:
        turn = session.send("Say hello")
        assert turn.result


def test_provider_from_env_var(monkeypatch):
    """CAW_PROVIDER env var selects the provider when none is specified."""
    monkeypatch.setenv("CAW_PROVIDER", "claude_code")

    agent = Agent()  # no explicit provider — picks up env var
    with agent.start_session() as session:
        turn = session.send("Say hello")
        assert turn.result


def test_unknown_provider_raises():
    """Helpful error when a provider name is not recognized."""
    agent = Agent(provider="does_not_exist")
    with pytest.raises(ValueError, match="Unknown provider"):
        agent.start_session()


# ===========================================================================
# Model / reasoning configuration
# ===========================================================================


def test_set_model():
    """set_model() configures the model before starting a session."""
    agent = Agent()
    agent.set_model("claude-sonnet-4-5-20250929")

    with agent.start_session() as session:
        turn = session.send("What is 2 + 2?")
        assert turn.result


def test_set_reasoning():
    """set_reasoning() configures the reasoning budget."""
    agent = Agent()
    agent.set_model("claude-sonnet-4-5-20250929")
    agent.set_reasoning("medium")

    with agent.start_session() as session:
        turn = session.send("What is 2 + 2?")
        assert turn.result


# ===========================================================================
# Multi-turn conversation
# ===========================================================================


def test_multi_turn_conversation(tmp_path):
    """Multi-turn session where the user and agent go back and forth."""
    agent = Agent()

    with agent.start_session() as session:
        session.send(f"Create a file called {tmp_path}/hello.py with print('hello')")
        session.send(f"Now rename it to {tmp_path}/greet.py")
        session.send(f"Read {tmp_path}/greet.py and tell me what's in it")

    traj = session.trajectory
    assert traj.num_turns == 3
    assert traj.result  # last turn has a response


# ===========================================================================
# Tool use
# ===========================================================================


def test_turn_with_tool_use():
    """Agent uses tools during a turn — inspect what it did."""
    agent = Agent()

    with agent.start_session() as session:
        turn = session.send("List the files in the current directory")

    # Coding agents typically use tools (bash, read_file, etc.)
    assert len(turn.tool_calls) > 0
    assert turn.result  # final text response after tool use


# ===========================================================================
# MCP server configuration
# ===========================================================================


def test_mcp_server_configuration():
    """Configure MCP servers before starting a session."""
    agent = Agent()
    agent.add_mcp_server(
        MCPServer(
            name="filesystem",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        )
    )
    agent.add_mcp_server(
        MCPServer(
            name="github",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"],
            env={"GITHUB_TOKEN": "ghp_xxx"},
        )
    )

    assert len(agent.mcp_servers) == 2

    with agent.start_session() as session:
        turn = session.send("List files in /tmp using the filesystem tool")
        assert turn.result


# ===========================================================================
# Trajectory inspection
# ===========================================================================


def test_trajectory_inspection(tmp_path):
    """After a session, inspect the full trajectory for logging / analysis."""
    agent = Agent()

    with agent.start_session() as session:
        session.send(f"Create a file called {tmp_path}/test.txt with content 'hello'")
        session.send(f"Read {tmp_path}/test.txt")

    traj = session.trajectory
    assert traj.num_turns == 2
    assert traj.total_tool_calls > 0
    assert traj.usage.input_tokens > 0
    assert traj.usage.output_tokens > 0
    assert traj.duration_ms > 0


def test_trajectory_accessible_mid_session():
    """trajectory property is live — available before end() is called."""
    agent = Agent()
    session = agent.start_session()

    session.send("Say hello")
    assert session.trajectory.num_turns == 1

    session.send("Say goodbye")
    assert session.trajectory.num_turns == 2

    traj = session.end()
    assert traj.num_turns == 2


# ===========================================================================
# Independent sessions
# ===========================================================================


def test_independent_sessions():
    """Each session is independent — separate trajectories."""
    agent = Agent()

    with agent.start_session() as s1:
        s1.send("Say 'alpha'")

    with agent.start_session() as s2:
        s2.send("Say 'beta'")

    assert s1.trajectory.num_turns == 1
    assert s2.trajectory.num_turns == 1
    assert s1.trajectory.turns[0].input == "Say 'alpha'"
    assert s2.trajectory.turns[0].input == "Say 'beta'"
