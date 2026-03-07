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

    assert "approval_policy" not in config
    assert "sandbox_mode" not in config
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
    doc_log = (EXAMPLES / "Documentation.md").read_text(encoding="utf-8")

    assert "[mcp_servers.tropical-mcp]" in readme
    assert "codex mcp add tropical-mcp" in readme
    assert "codex mcp list" in readme
    assert ".codex/config.toml" in readme
    assert "~/.codex/config.toml" not in readme
    assert ".codex/config.toml" in config_doc
    assert "runtime_info()" in config_doc
    assert "diagnose(...)" in config_doc
    assert "context_anchor(...)" in config_doc
    assert "telemetry_summary(...)" in config_doc
    assert "does not replace Codex's internal compactor automatically" in config_doc
    assert "v0.3.0" in config_doc
    assert "./scripts/validate_installed_wheel.sh" in doc_log
