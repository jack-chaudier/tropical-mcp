# tropical-mcp

Source-available [MCP](https://modelcontextprotocol.io/) server that gives AI coding agents explicit tools to preserve and inspect requirements during long sessions.

## The Problem

When you work with an AI coding agent (Codex, Claude Code, etc.) on a complex task, you give constraints across many messages: "use async I/O", "target Python 3.10", "don't break the public API". As the conversation grows, the client silently compresses older messages to stay within its context window — and your constraints can disappear with them.

The agent keeps working confidently, but it has lost 3 of your 7 requirements. You catch the drift, re-explain, and the agent refactors. This is the **validity mirage**: the context looks fine, the agent feels fine, but critical information is gone.

## What tropical-mcp Does

`tropical-mcp` gives you explicit MCP tools to **detect and prevent** this failure mode:

- **Before compaction**: inspect which constraints are protected and how many can be safely retained.
- **During compaction**: use policies that protect your core task and its constraints instead of blindly keeping only the newest messages.
- **After compaction**: get auditable artifacts (certificates, telemetry) that prove what was kept and what was dropped.

Under the hood, the server uses [tropical semiring algebra](./docs/GUIDE.md#how-it-works) to identify which messages contain your core task ("the pivot") and which contain constraints it depends on ("predecessors"), then protects them during compression.

This is the evaluation implementation for the [MirageKit](https://github.com/jack-chaudier/dreams) research program.

## Who This Is For

- **Researchers** evaluating long-context compaction or eviction behavior in LLM agents.
- **Agent teams** using Codex, Claude Code, or similar MCP-capable clients who want verifiable context management.
- **Anyone** who wants checkable artifacts instead of opaque compression — certificates, telemetry records, and retention audits you can inspect.

## Tools Overview

**Core compaction**
| Tool | Purpose |
|---|---|
| `compact_auto(...)` | Primary entry point — auto-selects the best protection level and compresses |
| `compact(...)` | Compress with a specific policy (`l2_guarded`, `l2_iterative_guarded`, or `recency`) |
| `certificate(...)` | Emit a portable artifact proving what was kept/dropped across policies |

**Inspection and diagnosis**
| Tool | Purpose |
|---|---|
| `runtime_info()` | Show resolved client, run ID, telemetry path, and package version |
| `diagnose(...)` | One-call tagged horizon view — shows feasible protection levels |
| `inspect(...)` / `inspect_horizon(...)` | Check feasibility at a specific or maximum protection level |
| `context_anchor(...)` | Build a paste-ready restatement of your objective and constraints |

**Telemetry and analysis**
| Tool | Purpose |
|---|---|
| `telemetry_summary(...)` | Roll up the current run's telemetry into a single report |
| `retention_floor(...)` | Estimate safe retention over multiple compaction epochs |
| `tag(...)` | Classify messages as pivot / predecessor / noise |

## How It Fits In

`tropical-mcp` is registered as an MCP server alongside your client. It does **not** replace the client's internal compactor automatically — you call the tools explicitly.

1. Register `tropical-mcp` as an MCP server in Codex, Claude Code, or a similar client.
2. Optionally keep compact-prompt guidance and durable memory files near the project.
3. Use `runtime_info()`, `compact_auto(...)`, and `certificate(...)` as the minimum smoke test.
4. For deeper analysis, extend with `diagnose(...)`, `context_anchor(...)`, and `telemetry_summary(...)`.

## Install

Clone and install from source. Python 3.10+ and [uv](https://docs.astral.sh/uv/) are required.

```bash
git clone https://github.com/jack-chaudier/tropical-mcp.git ~/tropical-mcp
cd ~/tropical-mcp
uv venv
source .venv/bin/activate
uv pip install -e '.[dev]'
```

> **License**: This repository is **source-available for evaluation**. You may clone, install, run, and create private modifications for research, peer review, and internal evaluation. Public redistribution and commercial production use require prior written consent. See [`LICENSE`](./LICENSE) for the full terms; contact `jackgaff@umich.edu` for broader rights.

## Quick-Start: Codex

**Option A** — project-scoped config (`.codex/config.toml`):

```toml
[mcp_servers.tropical-mcp]
command = "uv"
args = ["--directory", "/absolute/path/to/tropical-mcp", "run", "tropical-mcp"]
env = { TROPICAL_MCP_CLIENT = "codex" }
startup_timeout_sec = 10
tool_timeout_sec = 60
```

**Option B** — CLI registration:

```bash
codex mcp add tropical-mcp --env TROPICAL_MCP_CLIENT=codex -- \
  uv --directory /absolute/path/to/tropical-mcp run tropical-mcp
codex mcp list
```

After registration, the minimum smoke sequence is: `runtime_info()` → `compact_auto(...)` → `certificate(...)`.

For a fuller research review, extend to: `runtime_info()` → `diagnose(...)` → `context_anchor(...)` → `compact_auto(...)` → `certificate(...)` → `telemetry_summary(...)`.

See [`examples/codex/`](./examples/codex/) for a complete bundle including config, compact prompt, and durable memory templates.

## Quick-Start: Claude Code

```bash
claude mcp add tropical-mcp --scope user -- \
  uv --directory /absolute/path/to/tropical-mcp run tropical-mcp
```

After registration, verify the same way: `runtime_info()` → `compact_auto(...)` → `certificate(...)`.

The same integration pattern applies — register the server and call the MCP tools explicitly. See [`docs/configuration.md`](./docs/configuration.md) for details.

## Verification Snippets

### Minimal smoke test

Run this locally to verify the install works. The four sample messages simulate a real conversation: one task ("the pivot"), two constraints ("predecessors"), and one routine status update ("noise").

```bash
uv run python - <<'PY'
from tropical_mcp.server import certificate, compact_auto, runtime_info

messages = [
    {"id": "goal", "role": "user", "content": "Build a long-running coding agent workflow for Codex.", "role_hint": "pivot"},
    {"id": "constraint_stdio", "role": "user", "content": "Use stdio transport and never emit JSON-RPC data to stdout logs.", "role_hint": "predecessor"},
    {"id": "constraint_clients", "role": "user", "content": "Support Codex and Claude-style clients through explicit MCP tool calls.", "role_hint": "predecessor"},
    {"id": "status", "role": "assistant", "content": "I am wiring the verification flow and docs.", "role_hint": "noise"},
]

info = runtime_info()
auto = compact_auto(messages, token_budget=45, k_target=2, mode="adaptive")
cert = certificate(messages, token_budget=45, k=2)

print(info["client"], info["telemetry_path"], info["run_id"])
print(auto["audit"]["policy_selected"], auto["audit"]["k_selected"], auto["audit"]["guard_effective"])
print(cert["policies"]["recency"]["audit"]["dropped_ids"])
print(cert["policies"]["l2_guarded"]["audit"]["contract_satisfied"])
PY
```

What to expect:

- `runtime_info()` resolves the client, package version, supported tools, telemetry path, and run ID.
- `compact_auto(...)` selects `l2_guarded` on the sample and reports the chosen protection level (`k`).
- `certificate(...)` emits a portable artifact comparing recency vs. guarded policies, with kept/dropped IDs and audit flags.

### Full research sequence

This extends the smoke test with diagnosis, anchoring, and telemetry — useful for evaluating the full workflow in one pass.

```bash
uv run python - <<'PY'
from tropical_mcp.server import (
    certificate,
    compact_auto,
    context_anchor,
    diagnose,
    runtime_info,
    telemetry_summary,
)

messages = [
    {"id": "goal", "role": "user", "content": "Build a long-running coding agent workflow for Codex.", "role_hint": "pivot"},
    {"id": "constraint_stdio", "role": "user", "content": "Use stdio transport and never emit JSON-RPC data to stdout logs.", "role_hint": "predecessor"},
    {"id": "constraint_clients", "role": "user", "content": "Support Codex and Claude-style clients through explicit MCP tool calls.", "role_hint": "predecessor"},
    {"id": "status", "role": "assistant", "content": "I am wiring the verification flow and docs.", "role_hint": "noise"},
]

info = runtime_info()
diagnosis = diagnose(messages, k_max=2)
anchor = context_anchor(messages, k=2)
auto = compact_auto(messages, token_budget=45, k_target=2, mode="adaptive")
cert = certificate(messages, token_budget=45, k=2)
summary = telemetry_summary(limit=20)

print(info["client"], info["telemetry_path"], info["run_id"])
print(diagnosis["feasible_slots"])
print(anchor["k_selected"], anchor["anchor_text"].splitlines()[0])
print(auto["audit"]["policy_selected"], auto["audit"]["k_selected"], auto["audit"]["guard_effective"])
print(cert["policies"]["recency"]["audit"]["dropped_ids"])
print(cert["policies"]["l2_guarded"]["audit"]["contract_satisfied"])
print(summary["tool_counts"])
PY
```

What to expect:

- `diagnose(...)` shows the feasible protection levels before you compact.
- `context_anchor(...)` emits a paste-ready objective/constraint restatement.
- `telemetry_summary(...)` rolls up the current run into a single operational report.

## Artifacts And Telemetry

- `runtime_info()` reports the resolved client, telemetry path, and active run ID before you rely on any tool output.
- Every tool call appends telemetry metadata to a client-aware JSONL path; raw conversation text is not written to telemetry.
- Tool outputs can still expose anchor text, message previews, local telemetry paths, and run IDs. Treat screenshots, pasted outputs, and shared transcripts as review artifacts, not automatically safe-to-publish logs.
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
uv run --extra dev pytest
uv build
./scripts/validate_installed_wheel.sh
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

Temporary compatibility aliases remain only through `v0.2.x` and will be removed before `v0.3.0`:

- `tropical-compactor`
- `tropical-compactor-replay`
- `tropical-compactor-full-validate`

## Further Reading

**Using the tools**: [`docs/GUIDE.md`](./docs/GUIDE.md) · [`docs/configuration.md`](./docs/configuration.md) · [`docs/ARTIFACT_INDEX.md`](./docs/ARTIFACT_INDEX.md)

**Understanding the design**: [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md) · [`docs/METHODOLOGY.md`](./docs/METHODOLOGY.md) · [`docs/SOURCE_OF_TRUTH.md`](./docs/SOURCE_OF_TRUTH.md)

**Research and citation**: [`dreams`](https://github.com/jack-chaudier/dreams) · [`CITATION.cff`](./CITATION.cff) · <https://x.com/J_C_Gaffney>

**Contributing and maintaining**: [`CONTRIBUTING.md`](./CONTRIBUTING.md) · [`docs/RELEASE.md`](./docs/RELEASE.md) · [`docs/MAINTAINER_MAP.md`](./docs/MAINTAINER_MAP.md) · [`CHANGELOG.md`](./CHANGELOG.md)

**Policies**: [`CODE_OF_CONDUCT.md`](./CODE_OF_CONDUCT.md) · [`SECURITY.md`](./SECURITY.md)
