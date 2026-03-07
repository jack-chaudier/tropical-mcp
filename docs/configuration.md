# Client Configuration

`tropical-mcp` is used through **explicit MCP tool calls**. Register the server in the client, keep any compact-prompt or durable-memory files nearby, and call the tools deliberately. For Codex specifically, this package does not replace Codex's internal compactor automatically.

## Codex

Codex supports named `mcp_servers` tables, a project-scoped `.codex/config.toml`, `experimental_compact_prompt_file`, `model_auto_compact_token_limit`, and per-server timeouts.

Project-scoped `.codex/config.toml`

The example below intentionally omits `approval_policy` and `sandbox_mode` so it does not overwrite a user's local security posture by copy-paste. Add those locally only if you want a project-specific override.

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

Recommended smoke flow

1. Call `runtime_info()`.
2. Call `diagnose(...)`.
3. Call `context_anchor(...)`.
4. Call `compact_auto(...)`.
5. Call `certificate(...)`.
6. Call `telemetry_summary(...)`.
7. Confirm telemetry lands in `${CODEX_HOME:-~/.codex}/state/tropical-mcp/telemetry.jsonl`.
8. Remove any existing `tropical-compactor` registration only after the new server passes smoke.

Migration checklist

1. Run `codex mcp list` and confirm whether `tropical-compactor` is still registered.
2. Add `tropical-mcp` with `TROPICAL_MCP_CLIENT=codex`.
3. Call `runtime_info()`, `diagnose(...)`, `context_anchor(...)`, `compact_auto(...)`, `certificate(...)`, and `telemetry_summary(...)`.
4. Only after the smoke succeeds, run `codex mcp remove tropical-compactor`.
5. Keep the deprecated `tropical-compactor*` command aliases only as a temporary bridge through `v0.2.x`; remove them before `v0.3.0`.

## Claude Code

Claude Code can register the MCP and call the tools explicitly, but the same boundary applies: this package does not replace Claude Code's internal compaction path automatically.

```bash
claude mcp add tropical-mcp --scope user -- \
  uv --directory /absolute/path/to/tropical-mcp run tropical-mcp
```

## Example Bundle

See [`examples/codex/`](../examples/codex/) for the matching compact prompt and durable-memory templates.
