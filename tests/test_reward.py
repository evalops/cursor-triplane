from trainer.reward import compute_reward


def test_compute_reward_balances_signals():
    metrics = {
        "tests_passed": 1,
        "lint_improvement": 0.5,
        "parallel_groups": 3,
        "regressions": 0,
    }
    reward = compute_reward(metrics, latency_s=10.0)
    assert reward > 0


def test_compute_reward_penalizes_regressions():
    metrics = {
        "tests_passed": 0,
        "lint_improvement": 0,
        "parallel_groups": 0,
        "regressions": 2,
    }
    reward = compute_reward(metrics, latency_s=5.0)
    assert reward < 0
