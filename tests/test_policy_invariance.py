from __future__ import annotations

from tropical_mcp.golden import capture_policy_invariance_snapshot
from tropical_mcp.resources import fixture_json


def test_policy_invariance_matches_golden_fixture() -> None:
    expected = fixture_json("policy_invariance.json")
    assert capture_policy_invariance_snapshot() == expected
