# Architecture

This document describes the structure and boundaries of the `tropical-mcp` codebase. For usage instructions see [`GUIDE.md`](./GUIDE.md); for client setup see [`configuration.md`](./configuration.md).

## System boundary

`tropical-mcp` is an MCP server that exposes explicit tool calls for compaction analysis and guarded context retention. It is registered in clients such as Codex or Claude Code and called deliberately — it does not intercept or replace a host client's internal compactor.

## Core modules

- `src/tropical_mcp/server.py` wires the MCP server surface and exported tools.
- `src/tropical_mcp/compactor.py` implements policy application, guarded retention behavior, and audit outputs.
- `src/tropical_mcp/runtime.py` resolves client identity, telemetry location, and runtime metadata.
- `src/tropical_mcp/validation.py` drives the functional validation path.
- `src/tropical_mcp/benchmark_harness.py` generates the public replay witness.
- `src/tropical_mcp/resources.py` and `src/tropical_mcp/telemetry_cli.py` expose packaged fixtures and telemetry inspection.

## Operational flow

- The minimum smoke path is `runtime_info()`, `compact_auto(...)`, and `certificate(...)`.
- The fuller research workflow adds `diagnose(...)`, `context_anchor(...)`, and `telemetry_summary(...)`.
- Telemetry is written to a client-aware JSONL path so review remains auditable after the tool call returns.

## Public artifacts

- `examples/codex/` contains the example configuration and durable-memory bundle for Codex-style clients.
- `src/tropical_mcp/fixtures/` contains the packaged public fixtures used by installed-wheel and validation checks.
- `tests/` verifies both algorithmic behavior and public documentation expectations.

## Companion repo boundary

- The papers, mirrored validation logs, replay summaries, and live evidence site live in `dreams`.
- When behavior documentation conflicts, prefer this implementation repo first and then update `dreams` to match.
