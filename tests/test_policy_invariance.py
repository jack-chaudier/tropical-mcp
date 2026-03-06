from __future__ import annotations

import json
from pathlib import Path

from tropical_mcp.golden import capture_policy_invariance_snapshot

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "policy_invariance.json"


def test_policy_invariance_matches_golden_fixture() -> None:
    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert capture_policy_invariance_snapshot() == expected

