# Implementation Rules

1. Read `Prompt.md`, `Plan.md`, and `Documentation.md` before coding.
2. Keep changes scoped to the active milestone.
3. Preserve task identity over opportunistic cleanup.
4. Do not weaken acceptance criteria without logging the tradeoff.
5. After each milestone:
   - run the required checks,
   - fix failures before moving on,
   - update `Documentation.md`.
6. When compaction pressure rises, refresh the durable files before continuing.
7. Use `runtime_info()`, `diagnose(...)`, `compact_auto(...)`, and `certificate(...)` explicitly; do not assume the host client will route internal compaction through this MCP automatically.

