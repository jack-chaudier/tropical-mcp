from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "codex"


def test_codex_example_bundle_exists() -> None:
    expected_files = [
        EXAMPLES / "config.toml",
        EXAMPLES / "compact_prompt.md",
        EXAMPLES / "Prompt.md",
        EXAMPLES / "Plan.md",
        EXAMPLES / "Implement.md",
        EXAMPLES / "Documentation.md",
    ]

    for path in expected_files:
        assert path.exists(), f"missing example file: {path}"


def test_codex_example_config_is_consistent() -> None:
    config = tomllib.loads((EXAMPLES / "config.toml").read_text(encoding="utf-8"))

    assert config["approval_policy"] == "on-request"
    assert config["sandbox_mode"] == "workspace-write"
    assert config["model_auto_compact_token_limit"] == 220000
    assert config["profiles"]["stress"]["model_auto_compact_token_limit"] < config["model_auto_compact_token_limit"]

    server = config["mcp_servers"]["tropical-mcp"]
    assert server["command"] == "uv"
    assert server["env"]["TROPICAL_MCP_CLIENT"] == "codex"
    assert server["startup_timeout_sec"] == 10
    assert server["tool_timeout_sec"] == 60

    prompt_path = ROOT / config["experimental_compact_prompt_file"]
    assert prompt_path.exists()


def test_codex_docs_reference_current_registration_flow() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    config_doc = (ROOT / "docs" / "configuration.md").read_text(encoding="utf-8")

    assert "[mcp_servers.tropical-mcp]" in readme
    assert "codex mcp add tropical-mcp" in readme
    assert "codex mcp list" in readme
    assert "runtime_info()" in config_doc
    assert "diagnose(...)" in config_doc
    assert "does not replace Codex's internal compactor automatically" in config_doc
