"""Tests for caw.faststats.FastStats."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from caw.faststats import FastStats
from caw.models import TextBlock, ToolUse, Trajectory, Turn, UsageStats


def _trajectory(
    *,
    cost_usd: float = 1.25,
    input_tokens: int = 100,
    output_tokens: int = 50,
    cache_read_tokens: int = 200,
    cache_write_tokens: int = 30,
    long_system_prompt: bool = False,
    long_turns: int = 1,
    subagent_cost: float = 0.0,
) -> Trajectory:
    """Build an in-memory trajectory with optional padding to test sizes."""
    usage = UsageStats(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        cost_usd=cost_usd,
    )
    output_blocks = [TextBlock(text="x" * 4000)] if long_turns else [TextBlock(text="ok")]

    turns = []
    for i in range(long_turns):
        turn_output: list = list(output_blocks)
        if subagent_cost and i == 0:
            sub = Trajectory(
                agent="sub",
                model="claude-sub",
                session_id="sub-1",
                usage=UsageStats(cost_usd=subagent_cost, input_tokens=10, output_tokens=5),
                duration_ms=42,
            )
            turn_output.append(
                ToolUse(
                    id="t1",
                    name="run_subagent",
                    arguments={},
                    output="",
                    is_error=False,
                    subagent_trajectory=sub,
                )
            )
        turns.append(
            Turn(
                input=f"q{i}",
                output=turn_output,
                usage=UsageStats(input_tokens=1, output_tokens=1, cost_usd=0.001),
                duration_ms=100,
            )
        )

    return Trajectory(
        agent="claude_code",
        model="us.anthropic.claude-sonnet-4-6",
        session_id="abc-123",
        created_at="2026-03-07T23:21:38.632214+00:00",
        completed_at="2026-03-07T23:21:44.636179+00:00",
        usage_limited=False,
        system_prompt="prompt: " + ("p" * 5000) if long_system_prompt else "small",
        turns=turns,
        usage=usage,
        duration_ms=3270,
    )


def _write(traj: Trajectory, path: Path) -> Path:
    path.write_text(json.dumps(traj.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


class TestFromTrajectory:
    def test_basic_fields(self):
        traj = _trajectory()
        stats = FastStats.from_trajectory(traj)
        assert stats.agent == "claude_code"
        assert stats.model == "us.anthropic.claude-sonnet-4-6"
        assert stats.session_id == "abc-123"
        assert stats.created_at == "2026-03-07T23:21:38.632214+00:00"
        assert stats.completed_at == "2026-03-07T23:21:44.636179+00:00"
        assert stats.duration_ms == 3270
        assert stats.cost_usd == pytest.approx(1.25)
        assert stats.input_tokens == 100
        assert stats.total_tokens == 150
        assert stats.usage_limited is False
        assert stats.path is None

    def test_uses_total_usage_with_subagent(self):
        traj = _trajectory(cost_usd=1.0, subagent_cost=0.5)
        stats = FastStats.from_trajectory(traj)
        # total_usage = own usage + subagent.total_usage
        assert stats.cost_usd == pytest.approx(1.5)


class TestFromPath:
    def test_round_trip_small_file(self, tmp_path):
        traj = _trajectory(cost_usd=2.5, input_tokens=42, output_tokens=84)
        path = _write(traj, tmp_path / "trajectory.json")

        stats = FastStats.from_path(path)
        assert stats is not None
        assert stats.path == path
        assert stats.model == "us.anthropic.claude-sonnet-4-6"
        assert stats.cost_usd == pytest.approx(2.5)
        assert stats.input_tokens == 42
        assert stats.output_tokens == 84
        assert stats.duration_ms == 3270

    def test_round_trip_large_file(self, tmp_path):
        # Force the file well beyond head + tail buffers so the fast path
        # really has to seek.
        traj = _trajectory(cost_usd=99.99, long_turns=20, long_system_prompt=False)
        path = _write(traj, tmp_path / "trajectory.json")
        assert path.stat().st_size > 4096 * 2  # > head + tail

        stats = FastStats.from_path(path)
        assert stats is not None
        assert stats.cost_usd == pytest.approx(99.99)
        assert stats.model == "us.anthropic.claude-sonnet-4-6"
        assert stats.session_id == "abc-123"

    def test_long_system_prompt_pushes_header_fields(self, tmp_path):
        # The header fields all live before ``system_prompt`` in the
        # ``to_dict`` ordering, so even a 5KB system prompt should not
        # break extraction.
        traj = _trajectory(long_system_prompt=True)
        path = _write(traj, tmp_path / "trajectory.json")

        stats = FastStats.from_path(path)
        assert stats is not None
        assert stats.created_at == "2026-03-07T23:21:38.632214+00:00"
        assert stats.completed_at == "2026-03-07T23:21:44.636179+00:00"

    def test_returns_none_for_missing_file(self, tmp_path):
        assert FastStats.from_path(tmp_path / "nope.json") is None

    def test_returns_none_for_empty_file(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text("")
        assert FastStats.from_path(path) is None

    def test_returns_none_for_non_caw_json(self, tmp_path):
        path = tmp_path / "other.json"
        path.write_text(json.dumps({"hello": "world"}))
        # Fast path fails (no model field), full-parse fallback also fails.
        assert FastStats.from_path(path) is None

    def test_fallback_to_full_parse_for_compact_json(self, tmp_path):
        # Fast path expects indent=2; a compact dump should still work via
        # the json.loads fallback.
        traj = _trajectory(cost_usd=7.5)
        path = tmp_path / "compact.json"
        path.write_text(json.dumps(traj.to_dict()))

        stats = FastStats.from_path(path)
        assert stats is not None
        assert stats.cost_usd == pytest.approx(7.5)
        assert stats.model == "us.anthropic.claude-sonnet-4-6"


class TestIterDirectory:
    def test_finds_trajectories_via_default_patterns(self, tmp_path):
        # Mix the canonical layout and the .traj.json convention.
        sess = tmp_path / "sessions" / "s1"
        sess.mkdir(parents=True)
        _write(_trajectory(cost_usd=1.0), sess / "trajectory.json")
        (tmp_path / "logs").mkdir()
        _write(_trajectory(cost_usd=2.0), tmp_path / "logs" / "x.traj.json")

        results = list(FastStats.iter_directory(tmp_path))
        costs = sorted(r.cost_usd for r in results)
        assert costs == pytest.approx([1.0, 2.0])

    def test_skip_parts_excludes_subdirs(self, tmp_path):
        keep_dir = tmp_path / "keep"
        skip_dir = tmp_path / "drafts"
        keep_dir.mkdir()
        skip_dir.mkdir()
        _write(_trajectory(cost_usd=1.0), keep_dir / "a.traj.json")
        _write(_trajectory(cost_usd=2.0), skip_dir / "b.traj.json")

        results = list(FastStats.iter_directory(tmp_path, skip_parts={"drafts"}))
        assert [r.cost_usd for r in results] == pytest.approx([1.0])

    def test_directory_total_cost(self, tmp_path):
        _write(_trajectory(cost_usd=1.5), tmp_path / "a.traj.json")
        _write(_trajectory(cost_usd=2.25), tmp_path / "b.traj.json")
        assert FastStats.directory_total_cost(tmp_path) == pytest.approx(3.75)

    def test_directory_total_cost_missing_dir(self, tmp_path):
        assert FastStats.directory_total_cost(tmp_path / "missing") == 0.0

    def test_iter_directory_skips_unreadable(self, tmp_path):
        good = _write(_trajectory(cost_usd=1.0), tmp_path / "good.traj.json")
        bad = tmp_path / "bad.traj.json"
        bad.write_text("not json at all")
        results = list(FastStats.iter_directory(tmp_path))
        assert len(results) == 1
        assert results[0].path == good


class TestToDict:
    def test_serializable(self, tmp_path):
        traj = _trajectory()
        path = _write(traj, tmp_path / "trajectory.json")
        stats = FastStats.from_path(path)
        d = stats.to_dict()
        # Path is stringified for JSON serialization.
        assert isinstance(d["path"], str)
        assert d["model"] == "us.anthropic.claude-sonnet-4-6"
        # Round-trips through json.
        json.dumps(d)
