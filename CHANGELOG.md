# Changelog

All notable changes to this project are documented here.

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
