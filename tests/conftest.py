from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _disable_telemetry_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TROPICAL_MCP_ENABLE_TELEMETRY", "0")

