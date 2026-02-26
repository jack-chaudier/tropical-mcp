# Tropical Compactor — Full Guide

## The Problem: Validity Mirage

When you work on a complex task, the user gives constraints across multiple messages. After 30-40 tool calls, Claude Code's built-in auto-compaction activates and silently evicts older messages using recency — keeping recent messages, dropping old ones.

The constraints were in those old messages. After compaction:
- You confidently keep building, but you've lost 3 of 7 constraints
- You implement synchronous I/O because you forgot the async requirement
- The user catches it, has to re-explain, you have to refactor

This is the **validity mirage** — context looks fine, you feel fine, but critical information is gone. You cannot detect this by introspection. The only way to know is to check algebraically.

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
| `diagnose` | Tag + inspect_horizon in one call | Session start (replaces separate tag + inspect) |
| `tag` | Classify messages as pivot/predecessor/noise | When new constraints arrive |
| `inspect_horizon` | Show max feasible k across all slots | Every 5 tool calls; after compaction |
| `inspect` | Check feasibility at a specific k | Before compaction; reviewing context health |
| `compact_auto` | Auto-select best k and compress | Primary compaction entry point |
| `compact` | Compress with a specific policy | When you need precise control |
| `retention_floor` | Estimate safe retention over H epochs | Planning compression aggressiveness |

## Tagging Best Practices

### Role Hints
- `"pivot"` — the user's core task/goal
- `"predecessor"` — constraints, errors, decisions the task depends on
- `"noise"` — chatter, acknowledgments, routine output

### Critical Rule: Tag Every Message Separately

Do NOT summarize multiple messages into one synthetic blob. Each user message that contains a constraint is its own chunk with its own `id`. The algebra can only protect what it sees — if you collapse 5 constraint messages into 1 synthetic summary, k_max_feasible will be 1 instead of 5.

**Measured**: lazy 2-message tagging = k=1; proper 6-message tagging = k=3.

## Context Anchoring

After 15+ tool calls, write a **context anchor** — a short message restating the pivot and ALL constraints as a numbered checklist. Then call `compact_auto` with `mode="adaptive"`.

This "launders" critical context into recent messages so it survives recency-based auto-compaction. Don't wait until you "feel" pressure — by then it's too late.

## Compaction Policies

- `l2_guarded`: Protects pivot + k predecessors identified by L2 scan. Default for `compact_auto`.
- `l2_iterative_guarded`: Removes chunks one at a time, re-scanning after each removal. Blocks any eviction that would reduce W[k]. Safest but slowest.
- `recency`: Keep newest chunks only. Fallback when no feasible pivot exists.

**Rule**: `l2_guarded` beats recency whenever a feasible pivot exists. Never use plain recency if a pivot exists.

## Key Technical Notes

- `compact_auto` with `mode="adaptive"` picks the best k automatically
- If inspect shows `feasible: false`, lower k or add `role_hint` overrides
- Never hardcode k — always check via `inspect_horizon`
- All tools return `k_max_feasible` for convenient monitoring
- Telemetry is auto-logged to `~/.claude/compactor-telemetry.jsonl` by the server

## Telemetry

Every tool call automatically appends to `~/.claude/compactor-telemetry.jsonl`:
```json
{"timestamp": "ISO8601Z", "tool": "tool_name", "k_max_feasible": int|null, "pivot_id": "id", "predecessor_count": int, "auto": true}
```

Analyze with: `python scripts/analyze_telemetry.py`

## Stress Test Results

### PennyTree (Python novel writer, ~931 tool calls)
- k stuck at 0-1 (98%) due to lazy tagging
- Fixed: proper per-message tagging → k=3 instantly
- Acid test: 13/13 constraints recalled (no compaction between constraints and test)

### Pixl (Rust pixel art editor, ~452 tool calls)
- k climbed to 4 with proper tagging before first compaction
- k collapsed to 1 after auto-compaction (continuation summary blob)
- Acid test after 2 compactions: only 3/7 prompts recalled (lost foundational constraints)
- Proves: compactor is needed most DURING compaction events, which is when it's hardest to invoke
