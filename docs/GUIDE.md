# tropical-mcp — Full Guide

`tropical-mcp` is the source-available evaluation implementation of the MirageKit research program. This guide explains the supported workflow for Codex, Claude Code, and similar tool-calling clients: register the MCP, keep any compact prompt and durable memory files nearby, and call the tools explicitly. The package does not intercept a host client's internal compactor automatically.

## The Problem: Validity Mirage

When you work on a complex task, the user gives constraints across multiple messages. After enough turns, a host client may auto-compact and silently evict older messages using recency-like heuristics — keeping recent messages, dropping old ones.

The constraints were in those old messages. After compaction:
- You confidently keep building, but you've lost 3 of 7 constraints
- You implement synchronous I/O because you forgot the async requirement
- The user catches it, has to re-explain, you have to refactor

This is the **validity mirage** — context looks fine, you feel fine, but critical information is gone. You cannot detect this reliably by introspection alone. The supported way to verify preservation is to check the witness algebraically.

## How It Works

The compactor uses **tropical semiring algebra** (max-plus: T_max = (R ∪ {-∞}, max, +)) to identify which messages contain the user's task (the **pivot**) and which contain constraints the task depends on (**predecessors**). When compaction happens, it protects these messages first.

### L2 Composition Rule

```
W_new[j] = max(W_prev[j], W_incoming[max(0, j - d_prev)])
```

A pivot at slot 0 gets "lifted" to slot j by consuming j predecessors before it. The algebra propagates this automatically — no search needed.

### k_max_feasible

The highest j where W[j] is finite. This tells you how many predecessor constraints can be algebraically protected. Higher = better.

### Semantic Reordering

Real conversations often have the pivot first ("Build me X") then constraints later ("use SQLite"). The `_semantic_reorder()` function moves post-pivot predecessors before the pivot so the algebra can count them.

## Tools Reference

| Tool | What it does | When to use |
|---|---|---|
| `runtime_info` | Show resolved client, run ID, telemetry path, and package version | First smoke check after client registration |
| `diagnose` | Tag + inspect_horizon in one call | Session start (replaces separate tag + inspect) |
| `context_anchor` | Build a paste-ready objective + constraint anchor | Before compaction or after 15+ tool calls |
| `tag` | Classify messages as pivot/predecessor/noise | When new constraints arrive |
| `inspect_horizon` | Show max feasible k across all slots | Every 5 tool calls; after compaction |
| `inspect` | Check feasibility at a specific k | Before compaction; reviewing context health |
| `compact_auto` | Auto-select best k and compress | Primary compaction entry point |
| `compact` | Compress with a specific policy | When you need precise control |
| `certificate` | Emit a portable memory-safety artifact | Before publishing or comparing runs |
| `telemetry_summary` | Roll up the latest run's telemetry in one view | After a compaction or certificate sequence |
| `retention_floor` | Estimate safe retention over H epochs | Planning compression aggressiveness |

## Tagging Best Practices

### Role Hints
- `"pivot"` — the user's core task/goal
- `"predecessor"` — constraints, errors, decisions the task depends on
- `"noise"` — chatter, acknowledgments, routine output

### Critical Rule: Tag Every Message Separately

Do NOT summarize multiple messages into one synthetic blob. Each user message that contains a constraint is its own chunk with its own `id`. The algebra can only protect what it sees — if you collapse 5 constraint messages into 1 synthetic summary, k_max_feasible will be 1 instead of 5.

**Internal case-study note**: in one internal trace, a lazy 2-message tagging scheme yielded k=1, while proper 6-message tagging yielded k=3.

## Context Anchoring

After 15+ tool calls, call `context_anchor(...)` to build a **context anchor** — a short message restating the pivot and ALL constraints as a numbered checklist. Paste the returned `anchor_text` into the conversation as a fresh message, then call `compact_auto` with `mode="adaptive"`.

This "launders" critical context into recent messages so it survives recency-based auto-compaction. Don't wait until you "feel" pressure — by then it's too late.

## Compaction Policies

- `l2_guarded`: Protects pivot + k predecessors identified by L2 scan. Default for `compact_auto`.
- `l2_iterative_guarded`: Removes chunks one at a time, re-scanning after each removal. Blocks any eviction that would reduce W[k]. Safest but slowest.
- `recency`: Keep newest chunks only. Fallback when no feasible pivot exists.

**Rule**: `l2_guarded` beats recency whenever a feasible pivot exists. Never use plain recency if a pivot exists.

## Key Technical Notes

- `compact_auto` with `mode="adaptive"` picks the best k automatically
- `context_anchor(...)` falls back to the best feasible witness when the requested `k` is too high
- If inspect shows `feasible: false`, lower k or add `role_hint` overrides
- Never hardcode k — always check via `inspect_horizon`
- All tools return `k_max_feasible` for convenient monitoring
- Telemetry path is client-aware; use `runtime_info()` to inspect the resolved location and active run ID

## Client Boundaries

- Codex: supported pattern is a project-scoped `.codex/config.toml` + explicit MCP tool calls + compact prompt override + durable memory files.
- Claude Code: supported pattern is explicit MCP tool calls plus compact guidance in the client.
- Neither client is intercepted automatically by this package; it does not replace the host's internal compactor.

## Telemetry

Every tool call automatically appends telemetry metadata to the resolved telemetry path. Raw conversation text is not written to this log. For Codex, the default is `${CODEX_HOME:-~/.codex}/state/tropical-mcp/telemetry.jsonl`; for Claude-style clients it remains `~/.claude/compactor-telemetry.jsonl`; otherwise it falls back to `${XDG_STATE_HOME:-~/.local/state}/tropical-mcp/telemetry.jsonl`.

Current telemetry includes client/runtime details and artifact-grade fields such as run ID, policy selection, token counts, feasibility, guard reason, and pivot identity. Use `telemetry_summary(...)` after a long run to turn the JSONL stream into a run-scoped operational report.

CLI:
```bash
uv run tropical-mcp-telemetry --limit 25
```

Example:
```json
{"schema_version": 1, "timestamp": "ISO8601Z", "client": "codex", "tool": "compact_auto", "policy_selected": "l2_guarded", "tokens_after": 1200, "pivot_id": "msg_42"}
```

## Stress Test Results

These numbers are internal case-study notes that motivated the public release. The public, inspectable witness and archived evidence live in `dreams/results/`, `dreams/site/evidence.html`, and the flagship working paper.

### PennyTree (Python novel writer, ~931 tool calls)
- k stuck at 0-1 (98%) due to lazy tagging
- Fixed: proper per-message tagging → k=3 instantly
- Acid test: 13/13 constraints recalled (no compaction between constraints and test)

### Pixl (Rust pixel art editor, ~452 tool calls)
- k climbed to 4 with proper tagging before first compaction
- k collapsed to 1 after auto-compaction (continuation summary blob)
- Acid test after 2 compactions: only 3/7 prompts recalled (lost foundational constraints)
- This suggests the compactor is most valuable during compaction events, which is also when it is hardest to invoke
