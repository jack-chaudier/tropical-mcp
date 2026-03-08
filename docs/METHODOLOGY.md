# Methodology

This note explains what the public `tropical-mcp` release is designed to demonstrate and what it is not designed to prove by itself.

## Evaluation goal

- Show that compaction behavior can be inspected through explicit MCP tool calls instead of inferred from opaque host behavior.
- Preserve pivot-critical structure when feasible and emit auditable artifacts when it is not.
- Make the runtime, chosen policy, certificate output, and telemetry visible enough for independent review.

## Public verification paths

- Minimum smoke flow:
  `runtime_info()`, `compact_auto(...)`, `certificate(...)`
- Fuller research workflow:
  `runtime_info()`, `diagnose(...)`, `context_anchor(...)`, `compact_auto(...)`, `certificate(...)`, `telemetry_summary(...)`

## Validation profile

- Static quality gate: Ruff, mypy, and pytest.
- Packaging gate: `uv build` plus installed-wheel validation.
- Functional gate: `tropical-mcp-full-validate`.
- Replay witness: `tropical-mcp-replay`, mirrored into the public `dreams` repository.

## Interpretation boundary

- The package demonstrates a guarded compaction workflow and auditable release surface.
- The small packaged fixtures and replay witness are inspection-friendly by design; they are not meant to stand in for every broader paper-level result by themselves.
- Public theory and narrative claims should stay aligned with the implementation-facing contract exposed here.
