from __future__ import annotations

import json
from pathlib import Path

from tropical_mcp.certificate_cli import main as certificate_cli_main
from tropical_mcp.server import certificate

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
EXPECTED_CERTIFICATE = FIXTURES / "dreams_memory_safety_certificate.json"
TRANSCRIPT = FIXTURES / "dreams_memory_safety_transcript.json"


def test_certificate_matches_public_dreams_fixture_shape() -> None:
    expected = json.loads(EXPECTED_CERTIFICATE.read_text(encoding="utf-8"))
    transcript = json.loads(TRANSCRIPT.read_text(encoding="utf-8"))

    actual = certificate(
        messages=transcript["messages"],
        token_budget=40,
        k=3,
        name="memory_safety_certificate_example",
    )

    assert _project_shape(actual, expected) == expected
    assert "metadata" in actual
    assert actual["metadata"]["full_context"]["k_max_feasible"] == 3
    assert actual["metadata"]["full_context"]["predecessor_ids"] == ["hc1", "hc2", "hc3"]


def test_certificate_cli_writes_expected_shape(tmp_path: Path) -> None:
    output_path = tmp_path / "certificate.json"
    certificate_cli_main(
        [
            "--input",
            str(TRANSCRIPT),
            "--output",
            str(output_path),
            "--token-budget",
            "40",
            "--k",
            "3",
            "--name",
            "memory_safety_certificate_example",
        ]
    )

    expected = json.loads(EXPECTED_CERTIFICATE.read_text(encoding="utf-8"))
    actual = json.loads(output_path.read_text(encoding="utf-8"))
    assert _project_shape(actual, expected) == expected
    assert "metadata" in actual


def _project_shape(actual: object, expected: object) -> object:
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        return {key: _project_shape(actual[key], value) for key, value in expected.items()}
    if isinstance(expected, list):
        assert isinstance(actual, list)
        assert len(actual) == len(expected)
        return [
            _project_shape(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected, strict=True)
        ]
    return actual
