"""Tests for Session.save_trajectory / load_trajectory."""

from __future__ import annotations

import json

import pytest

from caw.agent import Session
from caw.models import TextBlock, Trajectory, Turn, UsageStats


def _make_trajectory() -> Trajectory:
    """Build a minimal in-memory trajectory for testing."""
    turn = Turn(
        input="What is 2+2?",
        output=[TextBlock(text="4")],
        usage=UsageStats(input_tokens=10, output_tokens=5),
        duration_ms=100,
    )
    return Trajectory(
        agent="test",
        model="test-model",
        session_id="test-session",
        created_at="2025-01-01T00:00:00Z",
        turns=[turn],
        usage=UsageStats(input_tokens=10, output_tokens=5),
        duration_ms=100,
    )


class TestSaveTrajectory:
    def test_writes_valid_json(self, tmp_path):
        traj = _make_trajectory()
        session = Session._from_trajectory(traj)

        out = tmp_path / "traj.json"
        session.save_trajectory(out)

        data = json.loads(out.read_text())
        assert data["agent"] == "test"
        assert data["model"] == "test-model"
        assert len(data["turns"]) == 1

    def test_creates_parent_dirs(self, tmp_path):
        traj = _make_trajectory()
        session = Session._from_trajectory(traj)

        out = tmp_path / "a" / "b" / "traj.json"
        session.save_trajectory(out)
        assert out.exists()


class TestLoadTrajectory:
    def test_round_trip(self, tmp_path):
        traj = _make_trajectory()
        session = Session._from_trajectory(traj)

        out = tmp_path / "traj.json"
        session.save_trajectory(out)

        loaded = Session.load_trajectory(out)
        assert loaded.trajectory.agent == "test"
        assert loaded.trajectory.model == "test-model"
        assert loaded.trajectory.session_id == "test-session"
        assert len(loaded.trajectory.turns) == 1
        assert loaded.trajectory.turns[0].input == "What is 2+2?"

    def test_loaded_session_is_readonly(self, tmp_path):
        traj = _make_trajectory()
        session = Session._from_trajectory(traj)

        out = tmp_path / "traj.json"
        session.save_trajectory(out)

        loaded = Session.load_trajectory(out)
        assert loaded._readonly is True

    def test_send_raises_on_loaded_session(self, tmp_path):
        traj = _make_trajectory()
        session = Session._from_trajectory(traj)

        out = tmp_path / "traj.json"
        session.save_trajectory(out)

        loaded = Session.load_trajectory(out)
        with pytest.raises(RuntimeError, match="Cannot send messages on a loaded session"):
            loaded.send("hello")

    def test_end_raises_on_loaded_session(self, tmp_path):
        traj = _make_trajectory()
        session = Session._from_trajectory(traj)

        out = tmp_path / "traj.json"
        session.save_trajectory(out)

        loaded = Session.load_trajectory(out)
        with pytest.raises(RuntimeError, match="Cannot send messages on a loaded session"):
            loaded.end()
