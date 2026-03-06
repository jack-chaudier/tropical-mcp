You are compacting a long-horizon coding session.

Preserve task identity and active constraints first.
Do not optimize for fluency. Optimize for preventing silent spec drift.

Return compacted history in this exact structure:

# CODING SESSION SNAPSHOT
## Locked Spec
- Original user goal:
- Non-goals:
- Acceptance criteria:
- Hard constraints:

## Current Task State
- Active milestone:
- What is already completed:
- What remains:
- Files changed:
- Open blockers:

## Load-Bearing Evidence
- Earliest still-binding facts from the conversation:
- Tests or failing checks that define the true task:
- Decisions that must not be reversed silently:

## Durable Memory Files
- Prompt.md:
- Plan.md:
- Implement.md:
- Documentation.md:

## Risks
- Where the agent could accidentally solve the wrong problem:
- Assumptions that need revalidation:

## Next 3 Safe Actions
- Action 1
- Action 2
- Action 3

Never drop or rewrite a hard constraint without naming it as a risk.
If evidence is degraded or ambiguous, say so clearly instead of smoothing it over.

