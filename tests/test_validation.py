from __future__ import annotations

import json

from tropical_mcp import validation
from tropical_mcp.server import _log_telemetry
from tropical_mcp.validation import _certificate_fixture_gate, _policy_invariance_gate


def test_validation_fixture_refs_are_repo_relative() -> None:
    cert_gate = _certificate_fixture_gate()
    policy_gate = _policy_invariance_gate()

    assert cert_gate["fixture"] == "package:fixtures/dreams_memory_safety_certificate.json"
    assert policy_gate["fixture"] == "package:fixtures/policy_invariance.json"
    assert "/Users/" not in str(cert_gate["fixture"])
    assert "/Users/" not in str(policy_gate["fixture"])


def test_telemetry_write_failure_emits_warning(monkeypatch, caplog, tmp_path) -> None:
    monkeypatch.setenv("TROPICAL_MCP_CLIENT", "codex")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    monkeypatch.setenv("TROPICAL_MCP_ENABLE_TELEMETRY", "1")

    def raise_oserror(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("builtins.open", raise_oserror)
    caplog.set_level("WARNING", logger="tropical-mcp")

    _log_telemetry("runtime_info", {"client": "codex", "audit": {}}, context={})

    assert "Telemetry write skipped" in caplog.text


def test_validation_main_emits_public_report(capsys) -> None:
    validation.main()

    report = json.loads(capsys.readouterr().out)
    assert report["certificate_fixture"]["fixture"] == "package:fixtures/dreams_memory_safety_certificate.json"
    assert report["policy_invariance"]["fixture"] == "package:fixtures/policy_invariance.json"
    assert report["stdio_smoke"]["alive_for_1s"] is True
