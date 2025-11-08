from __future__ import annotations

from typing import Dict


def compute_reward(metrics: Dict[str, float], latency_s: float) -> float:
    passed_tests = metrics.get("tests_passed", 0.0)
    lint_score = metrics.get("lint_improvement", 0.0)
    parallel_bonus = metrics.get("parallel_groups", 0.0)
    regression = metrics.get("regressions", 0.0)

    reward = (
        2.0 * passed_tests
        + 0.5 * lint_score
        + 0.2 * parallel_bonus
        - 1.0 * regression
        - 0.01 * latency_s
    )
    return reward


__all__ = ["compute_reward"]
