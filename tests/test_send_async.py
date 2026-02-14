"""Unit tests for Session.send_async — async send with FIFO ordering."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from caw.agent import Session
from caw.models import TextBlock, Turn


def _make_session(side_effect):
    """Create a Session with a mocked provider that returns turns from side_effect."""
    mock_ps = MagicMock()
    mock_ps.send.side_effect = side_effect
    mock_ps.detect_usage_limit.return_value = None
    mock_ps.last_raw_output = ""
    mock_ps.trajectory = MagicMock()
    return Session(mock_ps, auto_wait=False)


def _turn(msg: str) -> Turn:
    return Turn(input=msg, output=[TextBlock(text=f"reply to {msg}")])


class TestSendAsync:
    """Tests for Session.send_async()."""

    @pytest.mark.asyncio
    async def test_basic_send_async(self):
        """send_async returns the same Turn as send would."""
        session = _make_session([_turn("hello")])
        turn = await session.send_async("hello")
        assert turn.result == "reply to hello"

    @pytest.mark.asyncio
    async def test_fifo_ordering(self):
        """Multiple concurrent send_async calls execute in submission order."""
        call_order: list[str] = []

        def mock_send(msg):
            call_order.append(msg)
            # Small sleep so tasks actually overlap
            time.sleep(0.05)
            return _turn(msg)

        session = _make_session(mock_send)

        messages = [f"msg_{i}" for i in range(10)]
        tasks = [asyncio.create_task(session.send_async(m)) for m in messages]
        turns = await asyncio.gather(*tasks)

        assert call_order == messages
        assert [t.result for t in turns] == [f"reply to {m}" for m in messages]

    @pytest.mark.asyncio
    async def test_serialized_not_concurrent(self):
        """Only one send runs at a time — no overlapping execution."""
        active = 0
        max_active = 0

        def mock_send(msg):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            time.sleep(0.05)
            active -= 1
            return _turn(msg)

        session = _make_session(mock_send)

        tasks = [asyncio.create_task(session.send_async(f"m{i}")) for i in range(5)]
        await asyncio.gather(*tasks)

        assert max_active == 1

    @pytest.mark.asyncio
    async def test_can_do_work_while_sending(self):
        """Caller can do async work while send_async is in progress."""
        session = _make_session(lambda msg: (time.sleep(0.1), _turn(msg))[1])

        ticks = 0
        task = asyncio.create_task(session.send_async("slow"))
        while not task.done():
            await asyncio.sleep(0.02)
            ticks += 1
        turn = await task

        assert turn.result == "reply to slow"
        assert ticks >= 2  # Did meaningful async work while waiting

    @pytest.mark.asyncio
    async def test_readonly_session_raises(self):
        """send_async on a read-only (loaded) session raises RuntimeError."""
        from caw.models import Trajectory

        traj = Trajectory(agent="test", model="test", session_id="x", turns=[])
        session = Session._from_trajectory(traj)

        with pytest.raises(RuntimeError, match="Cannot send messages"):
            await session.send_async("hello")
