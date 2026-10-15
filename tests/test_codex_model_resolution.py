"""Tests for how CodexSession resolves the model it prices and reports.

With no ``-m``, codex runs the default from ``$CODEX_HOME/config.toml``; caw
must price that model rather than ``""`` (which has no rates, so cost was 0).
"""

from __future__ import annotations

import pytest

from caw.providers.codex import CodexSession, _codex_config_model


@pytest.fixture
def codex_home(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    return tmp_path


def _turn_completed(inp: int, out: int, cached: int = 0) -> dict:
    return {
        "type": "turn.completed",
        "usage": {"input_tokens": inp, "cached_input_tokens": cached, "output_tokens": out},
    }


def test_reads_top_level_model(codex_home):
    (codex_home / "config.toml").write_text(
        'model = "gpt-5.5"\nmodel_reasoning_effort = "high"\n\n[profiles.fast]\nmodel = "gpt-5.4-mini"\n'
    )
    assert _codex_config_model() == "gpt-5.5"


def test_ignores_model_inside_tables(codex_home):
    (codex_home / "config.toml").write_text('[profiles.fast]\nmodel = "gpt-5.4-mini"\n')
    assert _codex_config_model() is None


def test_missing_config(codex_home):
    assert _codex_config_model() is None


def test_explicit_model_wins(codex_home):
    (codex_home / "config.toml").write_text('model = "gpt-5.5"\n')
    session = CodexSession(mcp_servers=[], model="gpt-5.3-codex")
    assert session._effective_model() == "gpt-5.3-codex"


def test_extra_config_model_wins_over_config_toml(codex_home):
    (codex_home / "config.toml").write_text('model = "gpt-5.5"\n')
    session = CodexSession(mcp_servers=[], extra_config={"model": "gpt-5.4"})
    assert session._effective_model() == "gpt-5.4"


def test_default_model_is_priced_and_reported(codex_home):
    (codex_home / "config.toml").write_text('model = "gpt-5.5"\n')
    session = CodexSession(mcp_servers=[])
    usage = session._parse_usage(_turn_completed(1_000_000, 1_000_000, cached=0))
    # gpt-5.5 via the opencode openai/ fallback: 5 + 30
    assert usage.cost_usd == pytest.approx(35.0)
    assert session.trajectory.model == "gpt-5.5"
