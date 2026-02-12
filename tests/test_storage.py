"""End-to-end tests for session data persistence.

These tests run against a live provider backend and verify
that the correct files are written to disk.
"""

from __future__ import annotations

import json

from caw import Agent


# ===========================================================================
# Single-turn persistence
# ===========================================================================


def test_single_turn_persists(tmp_path):
    """A one-turn session writes config.json and turn files to disk."""
    agent = Agent(data_dir=str(tmp_path))

    with agent.start_session() as session:
        session.send("What is 1 + 1?")
        session_dir = session.session_dir

    assert session_dir is not None
    assert session_dir.exists()

    # trajectory.json
    traj = json.loads((session_dir / "trajectory.json").read_text())
    assert traj["agent"] == "claude_code"
    assert traj["model"]  # backfilled from provider
    assert len(traj["turns"]) == 1
    assert traj["duration_ms"] > 0
    assert traj["total_usage"]["input_tokens"] > 0
    assert traj["total_usage"]["output_tokens"] > 0

    # Turn files
    turns_dir = session_dir / "turns"
    assert (turns_dir / "000_input.txt").read_text() == "What is 1 + 1?"
    assert (turns_dir / "000_raw_output.jsonl").exists()
    raw = (turns_dir / "000_raw_output.jsonl").read_text()
    assert len(raw) > 0  # non-empty raw CLI output


# ===========================================================================
# Multi-turn persistence
# ===========================================================================


def test_multi_turn_persists(tmp_path):
    """A multi-turn session creates numbered turn files for each turn."""
    agent = Agent(data_dir=str(tmp_path))

    with agent.start_session() as session:
        session.send("Say hello")
        session.send("Say goodbye")
        session_dir = session.session_dir

    assert session_dir is not None

    traj = json.loads((session_dir / "trajectory.json").read_text())
    assert len(traj["turns"]) == 2

    turns_dir = session_dir / "turns"
    assert (turns_dir / "000_input.txt").read_text() == "Say hello"
    assert (turns_dir / "000_raw_output.jsonl").exists()
    assert (turns_dir / "001_input.txt").read_text() == "Say goodbye"
    assert (turns_dir / "001_raw_output.jsonl").exists()


# ===========================================================================
# Persistence disabled
# ===========================================================================


def test_data_dir_none_disables_persistence(tmp_path):
    """data_dir=None disables persistence — no files written."""
    agent = Agent(data_dir=None)

    with agent.start_session() as session:
        session.send("Say hello")
        assert session.session_dir is None

    # tmp_path should remain empty (nothing was written anywhere)
    assert not any(tmp_path.iterdir())
