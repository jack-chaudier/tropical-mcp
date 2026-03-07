# Changelog

All notable changes to this project are documented here.

## [Unreleased]

### Changed

- Clarified the evaluation license boundary so private local modifications and upstream patch submission are explicitly allowed without implying redistribution rights
- Unified public contact details on the README, package metadata, and security surface
- Hardened telemetry handling to warn on write failures instead of failing silently
- Sanitized validation fixture references so public artifacts no longer leak maintainer-local absolute paths
- Expanded tests and CI smoke coverage for replay and compatibility CLI entrypoints
- Moved golden validation fixtures into the installed package so wheel-based validation matches source-checkout validation
- Added an installed-wheel validation script and CI step to prevent package/repo drift
- Added citation metadata, a release checklist, and safer Codex example guidance for public evaluator workflows

## [0.2.0] - 2026-03-06

### Added

- Codex-aware runtime resolution with client-specific telemetry paths and `runtime_info()`
- Policy-invariance golden fixture gate for compact/compact_auto/replay behavior
- `certificate(...)` MCP tool and `tropical-mcp-certificate` CLI
- Public `dreams` certificate compatibility fixture and validation gate
- Temporary `tropical-compactor*` compatibility aliases for one release cycle
- `examples/codex/` bundle with config, compact prompt, and durable memory templates

### Changed

- Codex docs now use named `[mcp_servers.tropical-mcp]` tables and the `codex mcp add` / `codex mcp list` workflow
- Build/test validation now covers runtime, certificate shape compatibility, and docs/example consistency
- README, configuration docs, and Codex example notes now make the public repo split explicit: `dreams` for research/evidence, `tropical-mcp` for installation
- Verification guidance now centers the supported `runtime_info()`, `compact_auto(...)`, and `certificate(...)` flow while keeping `diagnose(...)` available for horizon inspection

## [0.1.0] - 2026-02-25

### Added

- Initial public release of MCP server tools:
  - `compact`
  - `inspect`
  - `inspect_horizon`
  - `compact_auto`
  - `retention_floor`
  - `tag`
- Replay benchmark harness and validation scripts
- Unit, property, and functional validation suites
