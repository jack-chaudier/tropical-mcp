from __future__ import annotations

from tropical_mcp import compat


def test_tropical_compactor_alias_warns_and_calls_server(monkeypatch, capsys) -> None:
    called: list[str] = []

    def fake_main() -> None:
        called.append("server")

    monkeypatch.setattr("tropical_mcp.server.main", fake_main)

    compat.tropical_compactor()

    captured = capsys.readouterr()
    assert "Deprecated alias 'tropical-compactor'" in captured.err
    assert called == ["server"]


def test_tropical_compactor_replay_alias_warns_and_calls_replay(monkeypatch, capsys) -> None:
    called: list[str] = []

    def fake_main() -> None:
        called.append("replay")

    monkeypatch.setattr("tropical_mcp.benchmark_harness.main", fake_main)

    compat.tropical_compactor_replay()

    captured = capsys.readouterr()
    assert "Deprecated alias 'tropical-compactor-replay'" in captured.err
    assert called == ["replay"]


def test_tropical_compactor_full_validate_alias_warns_and_calls_validation(monkeypatch, capsys) -> None:
    called: list[str] = []

    def fake_main() -> None:
        called.append("validate")

    monkeypatch.setattr("tropical_mcp.validation.main", fake_main)

    compat.tropical_compactor_full_validate()

    captured = capsys.readouterr()
    assert "Deprecated alias 'tropical-compactor-full-validate'" in captured.err
    assert called == ["validate"]
