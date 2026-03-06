"""Golden snapshots that guard compaction policy invariance."""

from __future__ import annotations

from typing import Any, cast

from .benchmark_harness import run_replay, summarize_rows
from .server import compact, compact_auto


def _normalize_for_golden(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 12)
    if isinstance(value, list):
        return [_normalize_for_golden(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_for_golden(item) for key, item in value.items()}
    return value


def fixture_k3() -> list[dict[str, str]]:
    return [
        {"id": "n1", "role": "user", "text": "n01 casual note", "role_hint": "noise"},
        {"id": "n2", "role": "assistant", "text": "n02 casual response", "role_hint": "noise"},
        {"id": "c1", "role": "user", "text": "Must use SQLite only.", "role_hint": "predecessor"},
        {"id": "c2", "role": "user", "text": "Must keep /api/v1 routes.", "role_hint": "predecessor"},
        {"id": "c3", "role": "user", "text": "Must include audit_log table.", "role_hint": "predecessor"},
        {"id": "c4", "role": "user", "text": "Must not store plaintext passwords.", "role_hint": "predecessor"},
        {
            "id": "c5",
            "role": "user",
            "text": "Must include regression tests for login and refresh.",
            "role_hint": "predecessor",
        },
        {
            "id": "pivot",
            "role": "assistant",
            "text": "Understood all constraints. I will now build the app step by step.",
            "role_hint": "pivot",
        },
        {"id": "f1", "role": "user", "text": "n03 filler", "role_hint": "noise"},
        {"id": "f2", "role": "user", "text": "n04 filler", "role_hint": "noise"},
        {"id": "f3", "role": "user", "text": "n05 filler", "role_hint": "noise"},
        {"id": "qa", "role": "user", "text": "Run final QA now.", "role_hint": "noise"},
        {
            "id": "summary",
            "role": "assistant",
            "text": "Final QA complete. All constraints verified.",
            "role_hint": "noise",
        },
    ]


def capture_policy_invariance_snapshot() -> dict[str, Any]:
    messages = fixture_k3()
    snapshot: dict[str, Any] = {"compact": [], "compact_auto": []}

    for policy in ("recency", "l2_guarded", "l2_iterative_guarded"):
        for budget in (79, 60, 35):
            out = compact(messages, token_budget=budget, policy=policy, k=3)
            audit = out["audit"]
            snapshot["compact"].append(
                {
                    "policy": policy,
                    "budget": budget,
                    "kept_ids": [str(msg["id"]) for msg in out["messages"]],
                    "dropped_ids": list(audit.get("dropped_ids", [])),
                    "protected_ids": list(audit.get("protected_ids", [])),
                    "breach_ids": list(audit.get("breach_ids", [])),
                    "tokens_after": audit.get("tokens_after"),
                    "feasible": audit.get("feasible"),
                    "contract_satisfied": audit.get("contract_satisfied"),
                    "guard_effective": audit.get("guard_effective"),
                }
            )

    for kwargs, label in (
        ({"token_budget": 120, "k_target": 5, "mode": "adaptive"}, "adaptive_120_k5"),
        ({"token_budget": 60, "k_target": 5, "mode": "strict"}, "strict_60_k5"),
    ):
        out = compact_auto(messages, **kwargs)
        audit = out["audit"]
        snapshot["compact_auto"].append(
            {
                "label": label,
                "kept_ids": [str(msg["id"]) for msg in out["messages"]],
                "dropped_ids": list(audit.get("dropped_ids", [])),
                "tokens_after": audit.get("tokens_after"),
                "feasible": audit.get("feasible"),
                "contract_satisfied": audit.get("contract_satisfied"),
                "guard_effective": audit.get("guard_effective"),
                "k_selected": audit.get("k_selected"),
                "policy_selected": audit.get("policy_selected"),
            }
        )

    rows = run_replay(
        fractions=[1.0, 0.8, 0.65, 0.5, 0.4],
        policies=["recency", "l2_guarded"],
        k=3,
        line_count=200,
    )
    snapshot["replay_summary"] = summarize_rows(rows)
    return cast(dict[str, Any], _normalize_for_golden(snapshot))
