from __future__ import annotations

import json

from tropical_mcp.server import context_anchor, runtime_info
from tropical_mcp.telemetry_cli import main as telemetry_cli_main

MESSAGES = [
    {"id": "goal", "text": "Build the tropical MCP server.", "role_hint": "pivot"},
    {"id": "constraint", "text": "Constraint: preserve the public API.", "role_hint": "predecessor"},
]


def test_telemetry_cli_prints_summary(monkeypatch, tmp_path, capsys) -> None:
    telemetry_path = tmp_path / "telemetry.jsonl"
    monkeypatch.setenv("TROPICAL_MCP_CLIENT", "codex")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    monkeypatch.setenv("TROPICAL_MCP_ENABLE_TELEMETRY", "1")
    monkeypatch.setenv("TROPICAL_MCP_TELEMETRY_PATH", str(telemetry_path))

    runtime_info()
    context_anchor(MESSAGES, k=1)

    telemetry_cli_main(["--limit", "10"])
    out = json.loads(capsys.readouterr().out)

    assert out["scope"] == "latest_run"
    assert out["tool_counts"]["context_anchor"] == 1
    assert out["tool_counts"]["runtime_info"] == 1
    assert out["run_id"]
