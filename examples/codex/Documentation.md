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
2. Call `runtime_info()` and confirm the reported client and telemetry path.
3. Call `compact_auto(...)` and confirm the guarded policy is selected on the witness payload.
4. Call `certificate(...)` and confirm the recency-vs-guarded kept/dropped IDs are populated.
5. Call `diagnose(...)` if you want the tagged horizon view.
6. Confirm telemetry lands in the Codex path.
7. If you are validating from a source checkout, run `./scripts/validate_installed_wheel.sh` and expect `Installed wheel validation passed ...`.
8. Remove any old `tropical-compactor` registration only after the new server passes smoke.
