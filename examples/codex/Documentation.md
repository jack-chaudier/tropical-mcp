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
3. Call `diagnose(...)`.
4. Confirm telemetry lands in the Codex path.
5. Remove any old `tropical-compactor` registration only after the new server passes smoke.

