# tropical-mcp

Source-available MCP server for **guarded context compaction** in Codex, Claude Code, and similar tool-calling clients.

`tropical-mcp` is the evaluation implementation for the MirageKit research program. The public research showcase, working papers, replay artifacts, and live demo live in [`dreams`](https://github.com/jack-chaudier/dreams).

This package addresses a long-context reliability failure mode: an agent can remain answer-valid while silently switching the governing task intent under naive memory compression. `tropical-mcp` exposes explicit MCP tools that preserve pivot-critical structure when feasible and emit auditable artifacts when they cannot.

## Who This Is For

- Researchers evaluating long-context compaction or eviction behavior.
- Agent teams using Codex, Claude Code, or similar clients that can register an MCP server and call tools deliberately.
- Anyone who wants checkable artifacts such as `runtime_info()`, telemetry records, and `certificate(...)` outputs instead of opaque compression behavior.

## What It Ships

- `compact(messages, token_budget, policy, k)`
  - `l2_guarded` for protected pivot + predecessor retention
  - `l2_iterative_guarded` for iterative safe-removal checks
  - `recency` as the baseline comparison policy
- `diagnose(...)` for a one-call tagged horizon view
- `context_anchor(...)` for paste-ready objective + constraint anchors
- `inspect(...)` and `inspect_horizon(...)` for feasibility and witness inspection
- `compact_auto(...)` for adaptive `k` selection
- `certificate(...)` for portable memory-safety artifacts
- `telemetry_summary(...)` for run-scoped telemetry rollups
- `runtime_info()` for client/runtime/telemetry introspection
- `retention_floor(...)` and `tag(...)` for operational analysis and role diagnostics

## Integration Boundary

This server does **not** replace a host client's internal compactor automatically.

The supported pattern is:

1. Register `tropical-mcp` as an MCP server in Codex, Claude Code, or a similar client.
2. Keep compact-prompt guidance and durable memory files near the project when the host supports them.
3. Call `runtime_info()`, `diagnose(...)`, `context_anchor(...)`, `compact_auto(...)`, `certificate(...)`, `telemetry_summary(...)`, and related tools explicitly during long runs.

Not supported:

- automatic interception of Codex or Claude Code host compaction events
- magical drop-in replacement of client-owned compression internals

## Install For Evaluation

Current public evaluation path: clone this repository and install from source for academic research, peer review, or internal evaluation.

```bash
git clone https://github.com/jack-chaudier/tropical-mcp.git ~/tropical-mcp
cd ~/tropical-mcp
uv venv
source .venv/bin/activate
uv pip install -e '.[dev]'
```

## License Boundary

This repository is currently **source-available for evaluation**.

You may clone, install, run, reproduce, and create private local modifications to this repository for academic research, peer review, internal evaluation, and preparation of upstream patches. Public redistribution, hosted-service use, and commercial production use still require prior written consent.

See [`LICENSE`](./LICENSE) for the full terms. For broader rights, contact `jackgaff@umich.edu`.

## Codex Quick-Start

Project-scoped config file:

`.codex/config.toml`

```toml
[mcp_servers.tropical-mcp]
command = "uv"
args = ["--directory", "/absolute/path/to/tropical-mcp", "run", "tropical-mcp"]
env = { TROPICAL_MCP_CLIENT = "codex" }
startup_timeout_sec = 10
tool_timeout_sec = 60
```

Or register from the CLI:

```bash
codex mcp add tropical-mcp --env TROPICAL_MCP_CLIENT=codex -- \
  uv --directory /absolute/path/to/tropical-mcp run tropical-mcp
codex mcp list
```

After registration, the recommended first sequence is:

1. `runtime_info()`
2. `diagnose(...)`
3. `context_anchor(...)`
4. `compact_auto(...)`
5. `certificate(...)`
6. `telemetry_summary(...)`

Use the full example bundle in [`examples/codex/`](./examples/codex/) for:

- `config.toml`
- `compact_prompt.md`
- durable memory templates (`Prompt.md`, `Plan.md`, `Implement.md`, `Documentation.md`)

## Claude Code Quick-Start

```bash
claude mcp add tropical-mcp --scope user -- \
  uv --directory /absolute/path/to/tropical-mcp run tropical-mcp
```

The same boundary applies in Claude Code: register the server, keep any durable context files current, and call the MCP tools explicitly.

## Minimal Verification Snippet

This direct local smoke test exercises the same workflow you should use from the client.

```bash
uv run python - <<'PY'
from tropical_mcp.server import certificate, compact_auto, context_anchor, runtime_info, telemetry_summary

messages = [
    {"id": "goal", "role": "user", "content": "Build a long-running coding agent workflow for Codex.", "role_hint": "pivot"},
    {"id": "constraint_stdio", "role": "user", "content": "Use stdio transport and never emit JSON-RPC data to stdout logs.", "role_hint": "predecessor"},
    {"id": "constraint_clients", "role": "user", "content": "Support Codex and Claude-style clients through explicit MCP tool calls.", "role_hint": "predecessor"},
    {"id": "status", "role": "assistant", "content": "I am wiring the verification flow and docs.", "role_hint": "noise"},
]

info = runtime_info()
anchor = context_anchor(messages, k=2)
auto = compact_auto(messages, token_budget=45, k_target=2, mode="adaptive")
cert = certificate(messages, token_budget=45, k=2)
summary = telemetry_summary(limit=20)

print(info["client"], info["telemetry_path"], info["run_id"])
print(anchor["k_selected"], anchor["anchor_text"].splitlines()[0])
print(auto["audit"]["policy_selected"], auto["audit"]["k_selected"], auto["audit"]["guard_effective"])
print(cert["policies"]["recency"]["audit"]["dropped_ids"])
print(cert["policies"]["l2_guarded"]["audit"]["contract_satisfied"])
print(summary["tool_counts"])
PY
```

What to expect:

- `runtime_info()` resolves the client, package version, supported tools, telemetry path, and run ID.
- `context_anchor(...)` emits a paste-ready objective/constraint restatement before compaction.
- `compact_auto(...)` selects `l2_guarded` on the sample and reports the chosen `k`.
- `certificate(...)` emits a portable recency-vs-guarded artifact with kept/dropped IDs and audit flags.
- `telemetry_summary(...)` rolls up the current run so you can inspect what actually happened.

## Artifacts And Telemetry

- `runtime_info()` reports the resolved client, telemetry path, and active run ID before you rely on any tool output.
- Every tool call appends telemetry to a client-aware JSONL path.
- `telemetry_summary(...)` summarizes the active run by default, so the JSONL log is operational instead of opaque.
- `certificate(...)` produces a shareable artifact that can be compared against public fixtures in `dreams/results/`.

For Codex, telemetry defaults to `${CODEX_HOME:-~/.codex}/state/tropical-mcp/telemetry.jsonl`. For Claude-style clients it defaults to `~/.claude/compactor-telemetry.jsonl`. See [`docs/GUIDE.md`](./docs/GUIDE.md) and [`docs/configuration.md`](./docs/configuration.md) for the full workflow.

Quick CLI summary:

```bash
uv run tropical-mcp-telemetry --limit 25
```

## Validation And Release Signals

```bash
uv run --extra dev ruff check .
uv run --extra dev mypy src/tropical_mcp
uv run --extra dev pytest -q
uv build
uv run tropical-mcp-full-validate
```

Or run the bundled script:

```bash
./scripts/full_validation.sh
```

## Replay Benchmark

```bash
uv run tropical-mcp-replay \
  --fractions 1.0,0.8,0.65,0.5,0.4 \
  --policies recency,l2_guarded \
  --k 3 \
  --line-count 200 \
  --output-dir artifacts/cyberops_mcp_replay
```

## Migration Note

Temporary compatibility aliases remain for one release cycle:

- `tropical-compactor`
- `tropical-compactor-replay`
- `tropical-compactor-full-validate`

## Project Signals

- CI on push and pull request: lint, type-check, tests, build, and functional validation
- Client configuration guide: [`docs/configuration.md`](./docs/configuration.md)
- Full usage guide: [`docs/GUIDE.md`](./docs/GUIDE.md)
- Contribution guide: [`CONTRIBUTING.md`](./CONTRIBUTING.md)
- Security policy: [`SECURITY.md`](./SECURITY.md)
- Version history: [`CHANGELOG.md`](./CHANGELOG.md)
