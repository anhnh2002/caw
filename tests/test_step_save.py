"""Tests for per-step trajectory saving (traj_path)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from caw.agent import Session
from caw.models import TextBlock, ToolUse, Trajectory, Turn, UsageStats


def _make_provider_session(
    blocks_sequence: list[list],
    turn: Turn,
    existing_turns: list[Turn] | None = None,
):
    """Build a mock ProviderSession that fires step callbacks during send().

    *blocks_sequence* is a list of block-lists; each entry triggers one
    step-callback invocation before ``send()`` returns *turn*.
    """
    captured_cb = {}
    base_turns = list(existing_turns or [])

    def _set_cb(cb):
        captured_cb["cb"] = cb

    def _send(message):
        cb = captured_cb.get("cb")
        if cb:
            for blocks in blocks_sequence:
                cb(blocks)
        return turn

    def _make_traj():
        return Trajectory(
            agent="test",
            model="m",
            session_id="s",
            created_at="2025-01-01T00:00:00Z",
            turns=list(base_turns),
            usage=UsageStats(),
            duration_ms=0,
        )

    mock_ps = MagicMock()
    mock_ps.set_step_callback.side_effect = _set_cb
    mock_ps.send.side_effect = _send
    mock_ps.detect_usage_limit.return_value = None
    mock_ps.last_raw_output = ""
    # Return a fresh Trajectory each time (matching real provider behavior)
    type(mock_ps).trajectory = property(lambda self: _make_traj())
    return mock_ps


class TestStepSaveTrajPath:
    """traj_path file is updated after each step, not just at session end."""

    def test_file_written_during_send(self, tmp_path):
        """The traj_path file should exist with partial data before send() returns."""
        text_block = TextBlock(text="partial")
        tool_block = ToolUse(id="t1", name="Read", arguments={})
        final_turn = Turn(
            input="hi",
            output=[text_block, tool_block],
            usage=UsageStats(input_tokens=5, output_tokens=3),
        )

        # Two step-callback invocations: first with one block, then two
        blocks_seq = [
            [text_block],
            [text_block, tool_block],
        ]

        mock_ps = _make_provider_session(blocks_seq, final_turn)
        out = tmp_path / "traj.json"
        session = Session(mock_ps, auto_wait=False)
        session._traj_path = str(out)

        # Collect snapshots written to disk during send()
        snapshots: list[dict] = []
        orig_set_cb = mock_ps.set_step_callback.side_effect

        def _intercept_set_cb(cb):
            if cb is None:
                return orig_set_cb(cb)

            def _wrapper(blocks):
                cb(blocks)
                # Read the file right after the callback fires
                if out.exists():
                    snapshots.append(json.loads(out.read_text()))

            return orig_set_cb(_wrapper)

        mock_ps.set_step_callback.side_effect = _intercept_set_cb

        session.send("hi")

        # Two step callbacks → two snapshots
        assert len(snapshots) == 2

        # First snapshot: partial turn with 1 block
        assert len(snapshots[0]["turns"]) == 1
        assert len(snapshots[0]["turns"][0]["output"]) == 1

        # Second snapshot: partial turn with 2 blocks
        assert len(snapshots[1]["turns"]) == 1
        assert len(snapshots[1]["turns"][0]["output"]) == 2

    def test_creates_parent_dirs(self, tmp_path):
        turn = Turn(input="hi", output=[TextBlock(text="ok")])
        mock_ps = _make_provider_session([[TextBlock(text="ok")]], turn)

        out = tmp_path / "a" / "b" / "traj.json"
        session = Session(mock_ps, auto_wait=False)
        session._traj_path = str(out)
        session.send("hi")

        assert out.exists()

    def test_callback_cleared_after_send(self, tmp_path):
        turn = Turn(input="hi", output=[TextBlock(text="done")])
        mock_ps = _make_provider_session([], turn)

        session = Session(mock_ps, auto_wait=False)
        session._traj_path = str(tmp_path / "traj.json")
        session.send("hi")

        # Last set_step_callback call should be None (clearing)
        calls = mock_ps.set_step_callback.call_args_list
        assert calls[-1][0][0] is None

    def test_no_traj_path_skips_file_write(self, tmp_path):
        """Without traj_path, no file should be created."""
        turn = Turn(input="hi", output=[TextBlock(text="ok")])
        mock_ps = _make_provider_session([[TextBlock(text="ok")]], turn)

        session = Session(mock_ps, auto_wait=False)
        # traj_path is None by default
        session.send("hi")

        # Nothing written to tmp_path
        assert list(tmp_path.iterdir()) == []

    def test_store_updated_per_step(self, tmp_path):
        """When a store is present, _save_trajectory is called per step."""
        text_block = TextBlock(text="hi")
        turn = Turn(input="hi", output=[text_block])
        mock_ps = _make_provider_session([[text_block], [text_block]], turn)

        mock_store = MagicMock()

        session = Session(mock_ps, store=mock_store, auto_wait=False)
        session.send("hi")

        # _save_trajectory called once per step callback
        assert mock_store._save_trajectory.call_count == 2

    def test_partial_turn_appended_to_completed_turns(self, tmp_path):
        """The step snapshot should contain completed turns plus a partial in-progress turn."""
        existing_turn = Turn(input="first", output=[TextBlock(text="done")])

        text_block = TextBlock(text="partial")
        final_turn = Turn(input="second", output=[text_block])
        mock_ps = _make_provider_session([[text_block]], final_turn, existing_turns=[existing_turn])

        out = tmp_path / "traj.json"
        session = Session(mock_ps, auto_wait=False)
        session._traj_path = str(out)
        session.send("second")

        data = json.loads(out.read_text())
        # Should have 2 turns: the completed one + the partial one
        assert len(data["turns"]) == 2
        assert data["turns"][0]["input"] == "first"
        assert data["turns"][1]["input"] == "second"
