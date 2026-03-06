from __future__ import annotations

from pathlib import Path

from tropical_mcp.runtime import resolve_runtime_settings
from tropical_mcp.server import runtime_info


def test_runtime_resolves_explicit_codex_path() -> None:
    settings = resolve_runtime_settings(
        {
            "TROPICAL_MCP_CLIENT": "codex",
            "CODEX_HOME": "/tmp/codex-home",
            "TROPICAL_MCP_ENABLE_TELEMETRY": "1",
        }
    )

    assert settings.client == "codex"
    assert settings.client_source == "explicit"
    assert settings.telemetry_enabled is True
    assert settings.telemetry_path == "/tmp/codex-home/state/tropical-mcp/telemetry.jsonl"


def test_runtime_resolves_explicit_claude_path() -> None:
    settings = resolve_runtime_settings(
        {
            "TROPICAL_MCP_CLIENT": "claude",
            "HOME": "/tmp/claude-home",
            "TROPICAL_MCP_ENABLE_TELEMETRY": "1",
        }
    )

    assert settings.client == "claude"
    assert settings.client_source == "explicit"
    assert settings.telemetry_path == "/tmp/claude-home/.claude/compactor-telemetry.jsonl"


def test_runtime_auto_detects_codex() -> None:
    settings = resolve_runtime_settings(
        {
            "CODEX_HOME": "/tmp/codex-home",
            "TROPICAL_MCP_ENABLE_TELEMETRY": "1",
        }
    )

    assert settings.client == "codex"
    assert settings.client_source == "auto"
    assert settings.telemetry_path == "/tmp/codex-home/state/tropical-mcp/telemetry.jsonl"


def test_runtime_generic_fallback_uses_xdg_state_home() -> None:
    settings = resolve_runtime_settings(
        {
            "XDG_STATE_HOME": "/tmp/state-home",
            "TROPICAL_MCP_ENABLE_TELEMETRY": "1",
        }
    )

    assert settings.client == "generic"
    assert settings.client_source == "auto"
    assert settings.telemetry_path == "/tmp/state-home/tropical-mcp/telemetry.jsonl"


def test_runtime_explicit_override_beats_default_path() -> None:
    settings = resolve_runtime_settings(
        {
            "TROPICAL_MCP_CLIENT": "codex",
            "CODEX_HOME": "/tmp/codex-home",
            "TROPICAL_MCP_TELEMETRY_PATH": "/tmp/custom/telemetry.jsonl",
            "TROPICAL_MCP_ENABLE_TELEMETRY": "1",
        }
    )

    assert settings.telemetry_path == "/tmp/custom/telemetry.jsonl"


def test_runtime_can_disable_telemetry() -> None:
    settings = resolve_runtime_settings(
        {
            "TROPICAL_MCP_CLIENT": "generic",
            "TROPICAL_MCP_ENABLE_TELEMETRY": "0",
        }
    )

    assert settings.telemetry_enabled is False
    assert settings.telemetry_path is None


def test_runtime_info_reports_shape(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TROPICAL_MCP_CLIENT", "codex")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    monkeypatch.setenv("TROPICAL_MCP_ENABLE_TELEMETRY", "1")

    info = runtime_info()

    assert info["client"] == "codex"
    assert info["client_source"] == "explicit"
    assert info["telemetry_enabled"] is True
    assert str(info["telemetry_path"]).endswith("state/tropical-mcp/telemetry.jsonl")
    assert "runtime_info" in info["supported_tools"]
    assert "l2_guarded" in info["supported_policies"]

