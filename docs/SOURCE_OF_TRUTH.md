# Source of Truth

This file defines what wins when `tropical-mcp` docs, examples, and release artifacts disagree.

## Resolution order

1. Packaged code and machine-readable fixtures.
2. Validation and release scripts that exercise those artifacts.
3. Narrative documentation and README copy.

## Canonical sources by question

- Runtime behavior and exported tools:
  `src/tropical_mcp/server.py`
- Compaction semantics and audit shape:
  `src/tropical_mcp/compactor.py`
- Runtime and telemetry resolution:
  `src/tropical_mcp/runtime.py`
- Packaged validation fixtures:
  `src/tropical_mcp/fixtures/`
- Example client configuration:
  `examples/codex/config.toml`
- Release integrity:
  `examples/codex/SHA256SUMS.txt`, `src/tropical_mcp/fixtures/SHA256SUMS.txt`, and `scripts/update_release_checksums.sh`

## Workflow boundary

- The minimum supported smoke path is `runtime_info()`, `compact_auto(...)`, and `certificate(...)`.
- The fuller research workflow adds `diagnose(...)`, `context_anchor(...)`, and `telemetry_summary(...)`.
- If older docs still imply that the fuller workflow is the only supported first check, update them to preserve this distinction.

## Companion repo boundary

- `dreams` mirrors public validation outputs and research-facing narrative material.
- When `dreams` and `tropical-mcp` disagree about implementation behavior, prefer `tropical-mcp` and then sync `dreams`.
