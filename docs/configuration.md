# Client Configuration

## Claude Code

Claude Code can register the MCP and call the tools explicitly, but this package does not replace Claude Code's internal compaction path automatically.

```bash
claude mcp add tropical-mcp --scope user -- \
  uv --directory /absolute/path/to/tropical-mcp run tropical-mcp
```

## Codex

Codex supports named `mcp_servers` tables, project-scoped config, `experimental_compact_prompt_file`, `model_auto_compact_token_limit`, and per-server timeouts. The supported integration pattern is project config + explicit tool calls; this package does not replace Codex's internal compactor automatically.

Project-scoped `.codex/config.toml`

```toml
[mcp_servers.tropical-mcp]
command = "uv"
args = ["--directory", "/absolute/path/to/tropical-mcp", "run", "tropical-mcp"]
env = { TROPICAL_MCP_CLIENT = "codex" }
startup_timeout_sec = 10
tool_timeout_sec = 60

[profiles.stress]
model_auto_compact_token_limit = 60000
experimental_compact_prompt_file = "examples/codex/compact_prompt.md"
```

CLI registration

```bash
codex mcp add tropical-mcp --env TROPICAL_MCP_CLIENT=codex -- \
  uv --directory /absolute/path/to/tropical-mcp run tropical-mcp
codex mcp list
```

Manual smoke checklist

1. Call `runtime_info()`.
2. Call `diagnose(...)`.
3. Confirm telemetry lands in `${CODEX_HOME:-~/.codex}/state/tropical-mcp/telemetry.jsonl`.
4. Remove any existing `tropical-compactor` registration only after the new server passes smoke.

Migration checklist

1. Run `codex mcp list` and confirm whether `tropical-compactor` is still registered.
2. Add `tropical-mcp` with `TROPICAL_MCP_CLIENT=codex`.
3. Call `runtime_info()` and `diagnose(...)`.
4. Only after the smoke succeeds, run `codex mcp remove tropical-compactor`.
5. Keep the deprecated `tropical-compactor*` command aliases only as a temporary bridge for one release cycle.
