#!/usr/bin/env python3
"""Comprehensive functional validation for tropical-mcp."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import cast

from .golden import capture_policy_invariance_snapshot, fixture_k3
from .server import certificate, compact, compact_auto, inspect, inspect_horizon, retention_floor

ROOT = Path(__file__).resolve().parents[2]


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _run_policy_comparison() -> dict[str, object]:
    messages = fixture_k3()
    horizon = inspect_horizon(messages, k_max=5)
    _assert("error" not in horizon, f"inspect_horizon failed: {horizon}")
    _assert(horizon["k_max_feasible"] is not None, "k horizon should be feasible for fixture")

    inspect_k3 = inspect(messages, k=3)
    _assert(inspect_k3["feasible"] is True, "fixture should be feasible at k=3")
    _assert(inspect_k3["pivot_id"] == "pivot", "pivot id mismatch")

    budgets = [79, 60, 35]
    rows: list[dict[str, object]] = []
    for budget in budgets:
        rec = compact(messages, token_budget=budget, policy="recency", k=3)
        l2 = compact(messages, token_budget=budget, policy="l2_guarded", k=3)
        l2_iter = compact(messages, token_budget=budget, policy="l2_iterative_guarded", k=3)

        _assert("error" not in rec, f"recency failed at {budget}: {rec}")
        _assert("error" not in l2, f"l2_guarded failed at {budget}: {l2}")
        _assert("error" not in l2_iter, f"l2_iterative_guarded failed at {budget}: {l2_iter}")

        rec_ids = {m["id"] for m in rec["messages"]}
        l2_ids = {m["id"] for m in l2["messages"]}
        l2_iter_ids = {m["id"] for m in l2_iter["messages"]}

        _assert(l2["audit"]["tokens_after"] <= budget, "l2 budget overflow")
        _assert(l2_iter["audit"]["tokens_after"] <= budget, "l2_iter budget overflow")
        _assert(l2_iter["audit"]["iterative_checked"] >= 0, "iterative audit missing")
        _assert(l2_iter["audit"]["policy"] == "l2_iterative_guarded", "iterative policy mismatch")

        if budget <= 60:
            _assert("pivot" in l2_ids, "l2_guarded should keep pivot in tight budget")
            _assert("pivot" in l2_iter_ids, "l2_iterative_guarded should keep pivot in tight budget")
        if budget <= 35:
            _assert("pivot" not in rec_ids, "recency should drop pivot in tight budget")

        rows.append(
            {
                "budget": budget,
                "recency_kept": sorted(rec_ids),
                "l2_guarded_kept": sorted(l2_ids),
                "l2_iterative_kept": sorted(l2_iter_ids),
                "l2_guard_effective": l2["audit"]["guard_effective"],
                "l2_iter_guard_effective": l2_iter["audit"]["guard_effective"],
                "l2_iter_checked": l2_iter["audit"]["iterative_checked"],
                "l2_iter_blocked": l2_iter["audit"]["iterative_blocked"],
            }
        )

    return {"horizon": horizon, "inspect_k3": inspect_k3, "rows": rows}


def _run_auto_and_floor_checks() -> dict[str, object]:
    messages = fixture_k3()
    auto = compact_auto(messages, token_budget=120, k_target=5, mode="adaptive")
    _assert("error" not in auto, f"compact_auto failed: {auto}")
    _assert(auto["audit"]["policy_selected"] == "l2_guarded", "auto should select l2_guarded")
    _assert(auto["audit"]["k_selected"] is not None, "auto should select a k")

    strict = compact_auto(messages, token_budget=60, k_target=5, mode="strict")
    _assert("error" not in strict, f"compact_auto strict failed: {strict}")
    _assert(strict["audit"]["policy_selected"] in {"none", "l2_guarded"}, "unexpected strict policy")

    floor = retention_floor(messages, k=3, horizon=100, failure_prob=0.01)
    _assert("error" not in floor, f"retention_floor failed: {floor}")
    _assert(floor["required_predecessor_count"] >= 0, "invalid predecessor floor")
    _assert("model_note" in floor, "missing floor note")

    return {"auto": auto["audit"], "strict": strict["audit"], "floor": floor}


def _stdio_smoke() -> dict[str, object]:
    proc = subprocess.Popen(
        [sys.executable, "-m", "tropical_mcp.server"],
        cwd=str(ROOT),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(1.0)
    alive = proc.poll() is None
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    _assert(alive, "stdio server did not stay alive during smoke window")
    return {"alive_for_1s": alive, "returncode": proc.returncode}


def _policy_invariance_gate() -> dict[str, object]:
    expected_path = ROOT / "fixtures" / "policy_invariance.json"
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    actual = capture_policy_invariance_snapshot()
    _assert(actual == expected, "policy invariance golden fixture drifted")
    return {"fixture": str(expected_path), "matched": True}


def _certificate_fixture_gate() -> dict[str, object]:
    expected_path = ROOT / "fixtures" / "dreams_memory_safety_certificate.json"
    transcript_path = ROOT / "fixtures" / "dreams_memory_safety_transcript.json"
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    actual = certificate(
        messages=transcript["messages"],
        token_budget=40,
        k=3,
        name="memory_safety_certificate_example",
    )
    projected = _project_shape(actual, expected)
    _assert(projected == expected, "certificate fixture drifted from public dreams shape")
    return {"fixture": str(expected_path), "matched": True}


def _project_shape(actual: object, expected: object) -> object:
    if isinstance(expected, dict):
        _assert(isinstance(actual, dict), "certificate projection expected a dict")
        actual_dict = cast(dict[str, object], actual)
        return {key: _project_shape(actual_dict[key], value) for key, value in expected.items()}
    if isinstance(expected, list):
        _assert(isinstance(actual, list), "certificate projection expected a list")
        actual_list = cast(list[object], actual)
        _assert(len(actual_list) == len(expected), "certificate projection length mismatch")
        return [
            _project_shape(a_item, e_item)
            for a_item, e_item in zip(actual_list, expected, strict=True)
        ]
    return actual


def main() -> None:
    report = {
        "policy_comparison": _run_policy_comparison(),
        "auto_and_floor": _run_auto_and_floor_checks(),
        "certificate_fixture": _certificate_fixture_gate(),
        "policy_invariance": _policy_invariance_gate(),
        "stdio_smoke": _stdio_smoke(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
