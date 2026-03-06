# Documentation / Audit Log

## Current Status
- Active milestone:
- Completed:
- In progress:
- Next:

## Decisions
- Record decisions that must survive compaction.

## Validation Log
- Record each command run and whether it passed.

## Open Risks
- Where the agent could still drift silently:

## Manual Smoke Checklist
1. Register the MCP in Codex.
2. Call `runtime_info()`.
3. Call `compact_auto(...)`.
4. Call `certificate(...)`.
5. Call `diagnose(...)` if you want the tagged horizon view.
6. Confirm telemetry lands in the Codex path.
7. Remove any old `tropical-compactor` registration only after the new server passes smoke.
