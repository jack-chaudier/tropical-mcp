# tropical-mcp — Full Guide

This guide explains how to use `tropical-mcp` to protect your task context during long AI coding sessions. It covers the problem being solved, the underlying approach, all available tools, and best practices.

**Workflow summary**: Register the MCP server in your client (Codex, Claude Code, etc.), optionally keep compact-prompt and durable-memory files nearby, and call the tools explicitly. The package does not intercept the client's internal compactor automatically.

## The Problem: Validity Mirage

When you work with an AI agent on a complex task, you give constraints across many messages. After enough turns, the client silently compresses older messages to stay within its context window — typically keeping the newest messages and dropping the oldest.

Your constraints were in those older messages. After compaction:
- The agent keeps working confidently, but it has lost 3 of 7 constraints
- It implements synchronous I/O because the async requirement was evicted
- You catch the drift, re-explain, and the agent has to refactor

This is the **validity mirage** — the context looks fine, the agent feels fine, but critical information is gone. The agent cannot detect this by introspection alone. The only reliable way to verify preservation is to check mathematically which messages survived.

## How It Works

The server classifies each message in the conversation by its role:
- **Pivot**: The user's core task or goal (e.g., "Build me a REST API for the inventory system")
- **Predecessor**: A constraint the task depends on (e.g., "use async I/O", "target Python 3.10")
- **Noise**: Routine chatter, acknowledgments, or status updates that can be safely dropped

It then uses **tropical semiring algebra** — a mathematical framework where "addition" is `max` and "multiplication" is `+` — to compute exactly how many predecessor constraints can be protected during compaction. This is computed algebraically rather than by heuristic search, which makes it deterministic and auditable.

### The composition rule

Each time a new message arrives, the protection frontier is updated:

```
W_new[j] = max(W_prev[j], W_incoming[max(0, j - d_prev)])
```

In plain terms: a pivot at position 0 "absorbs" predecessors as they appear. The value at position `j` represents the feasibility of protecting the pivot plus `j` predecessors. The algebra propagates this automatically — no search required.

### k_max_feasible

The highest `j` where the frontier `W[j]` is finite. This is the maximum number of predecessor constraints that can be algebraically protected during compaction. Higher is better — it means more of your constraints will survive.

### Semantic reordering

Real conversations often have the pivot first ("Build me X") followed by constraints later ("use SQLite", "keep it under 100ms"). The `_semantic_reorder()` function moves post-pivot predecessors before the pivot so the algebra can count them correctly.

## Tools Reference

All tools are called explicitly through the MCP interface. The table below summarizes each tool and when to use it.

| Tool | What it does | When to use |
|---|---|---|
| `runtime_info` | Show resolved client, run ID, telemetry path, and package version | First check after client registration |
| `diagnose` | Classify messages and show the protection frontier in one call | Session start; replaces separate `tag` + `inspect` |
| `context_anchor` | Build a paste-ready restatement of your objective and constraints | Before compaction, or after 15+ tool calls as a safety refresh |
| `tag` | Classify messages as pivot / predecessor / noise | When new constraints arrive mid-session |
| `inspect_horizon` | Show the maximum feasible protection level across all slots | Periodic health check (every ~5 tool calls) |
| `inspect` | Check whether a specific protection level `k` is feasible | Before compaction; reviewing whether constraints are safe |
| `compact_auto` | Auto-select the best protection level and compress | **Primary compaction entry point** |
| `compact` | Compress with a specific policy and protection level | When you need precise manual control |
| `certificate` | Emit a portable artifact comparing policies with kept/dropped IDs | Before publishing results or comparing runs |
| `telemetry_summary` | Roll up the current run's telemetry into a single report | After a compaction or certificate sequence |
| `retention_floor` | Estimate safe retention over multiple compaction epochs | Planning how aggressive compression can be |

## Tagging Best Practices

### Role hints

Each message gets a `role_hint` that tells the compactor how important it is:

- `"pivot"` — the user's core task or goal (e.g., "Build me a REST API")
- `"predecessor"` — a constraint, error, or decision the task depends on (e.g., "use async I/O")
- `"noise"` — routine chatter, acknowledgments, or status updates (safe to drop)

### Critical rule: tag every message separately

Do NOT summarize multiple messages into one synthetic blob. Each user message that contains a constraint must be its own chunk with its own `id`. The algebra can only protect what it can see — if you collapse 5 constraint messages into 1 synthetic summary, the maximum protection level drops from 5 to 1.

Keep the original message boundaries whenever possible.

## Context Anchoring

After 15+ tool calls (or whenever a session is getting long), call `context_anchor(...)` to build a **context anchor** — a short message restating the pivot and all constraints as a numbered checklist. Paste the returned `anchor_text` into the conversation as a fresh message, then call `compact_auto` with `mode="adaptive"`.

This moves critical context into a recent message so it survives recency-based auto-compaction even if the client compresses before you act. Don't wait until you "feel" context pressure — by then constraints may already be gone.

## Compaction Policies

Three policies are available, in order of protection strength:

1. **`l2_iterative_guarded`**: Removes messages one at a time, re-scanning after each removal. Blocks any eviction that would reduce the protection frontier. Safest but slowest.
2. **`l2_guarded`**: Protects the pivot plus `k` predecessors identified by the algebra. Default for `compact_auto`. Good balance of safety and speed.
3. **`recency`**: Keeps only the newest messages. Used as a fallback when no feasible pivot exists.

**Rule of thumb**: Always prefer `l2_guarded` (or `l2_iterative_guarded`) over `recency` when a pivot exists. Plain recency will drop constraints without checking.

## Key Technical Notes

- `compact_auto` with `mode="adaptive"` picks the best protection level (`k`) automatically — this is the recommended default.
- `context_anchor(...)` falls back to the best feasible level when the requested `k` is too high.
- If `inspect` shows `feasible: false`, lower `k` or add `role_hint` overrides to your messages.
- Never hardcode `k` — always check the current feasible range via `inspect_horizon`.
- All tools return `k_max_feasible` so you can monitor protection capacity as the conversation grows.
- The telemetry path is client-aware; use `runtime_info()` to see where logs are written and the active run ID.

## Client Boundaries

- Codex: supported pattern is a project-scoped `.codex/config.toml` + explicit MCP tool calls + compact prompt override + durable memory files.
- Claude Code: supported pattern is explicit MCP tool calls plus compact guidance in the client.
- Neither client is intercepted automatically by this package; it does not replace the host's internal compactor.

## Telemetry

Every tool call automatically appends telemetry metadata to the resolved telemetry path. Raw conversation text is not written to this log. For Codex, the default is `${CODEX_HOME:-~/.codex}/state/tropical-mcp/telemetry.jsonl`; for Claude-style clients it remains `~/.claude/compactor-telemetry.jsonl`; otherwise it falls back to `${XDG_STATE_HOME:-~/.local/state}/tropical-mcp/telemetry.jsonl`.

Current telemetry includes client/runtime details and artifact-grade fields such as run ID, policy selection, token counts, feasibility, guard reason, and pivot identity. Use `telemetry_summary(...)` after a long run to turn the JSONL stream into a run-scoped operational report.

Tool outputs still deserve review before you share them. `runtime_info()` can reveal local telemetry paths and run IDs, while `context_anchor(...)`, `tag(...)`, and `diagnose(...)` can echo anchor text or message previews. Treat screenshots, pasted outputs, and transcripts as sensitive until you have checked them.

CLI:
```bash
uv run tropical-mcp-telemetry --limit 25
```

Example:
```json
{"schema_version": 1, "timestamp": "ISO8601Z", "client": "codex", "tool": "compact_auto", "policy_selected": "l2_guarded", "tokens_after": 1200, "pivot_id": "msg_42"}
```

## Public Evidence Boundary

The public, inspectable evidence for this release lives in the MirageKit showcase and committed validation artifacts, not in private case-study traces.

Use these reviewer-visible surfaces when you want evidence you can cite directly:

- `dreams/results/replay/replay_summary.json` for the deterministic replay witness
- `dreams/site/evidence.html` for the curated evidence dossier
- this repository's validation commands and CI logs for software quality and packaging checks
