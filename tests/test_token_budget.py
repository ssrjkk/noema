"""Tests for TokenBudget — token budget with graceful degradation."""

from noema.budget.token_budget import BudgetAction, TokenBudget


def test_default_budget_not_exceeded():
    tb = TokenBudget()
    assert tb.used == 0
    assert tb.remaining == tb.max_tokens
    assert tb.fraction_used == 0.0
    assert tb.check() == BudgetAction.ALLOW
    assert not tb.should_degrade()


def test_token_tracking_increments():
    tb = TokenBudget(max_tokens=1000)
    tb.record(100)
    assert tb.used == 100
    assert tb.remaining == 900
    assert tb.fraction_used == 0.1

    tb.record(200)
    assert tb.used == 300
    assert tb.remaining == 700


def test_warn_at_triggers_degrade():
    tb = TokenBudget(max_tokens=1000, warn_at=0.5, hard_cap_at=1.0)
    tb.record(600)
    action = tb.check(estimated_cost=100)
    assert action == BudgetAction.DEGRADE
    assert tb.should_degrade()


def test_should_degrade_returns_true_when_degraded():
    tb = TokenBudget(max_tokens=1000, warn_at=0.5, hard_cap_at=1.0)
    tb.record(600)
    assert tb.should_degrade()


def test_hard_cap_at_blocks_tokens():
    tb = TokenBudget(max_tokens=1000, hard_cap_at=0.8)
    tb.record(900)
    action = tb.check(estimated_cost=100)
    assert action == BudgetAction.SKIP


def test_hard_cap_skipped_step_is_recorded():
    tb = TokenBudget(max_tokens=1000, hard_cap_at=0.8)
    tb.record(900)
    tb.check(estimated_cost=100, step_name="expensive-call")
    assert "expensive-call" in tb.skipped_steps()


def test_skip_rejected_steps_tracked():
    tb = TokenBudget(max_tokens=1000, hard_cap_at=0.8)
    tb.record(900)
    tb.check(estimated_cost=100, step_name="step1")
    tb.check(estimated_cost=100, step_name="step2")
    assert len(tb.skipped_steps()) == 2


def test_mutiple_tasks_reset_budget():
    tb = TokenBudget(max_tokens=10000, warn_at=0.5, hard_cap_at=0.95)
    tb.record(8000)
    assert tb.used == 8000
    assert tb.should_degrade()

    tb.reset()
    assert tb.used == 0
    assert tb.remaining == 10000
    assert not tb.should_degrade()
    assert tb.skipped_steps() == []


def test_edge_case_zero_max_tokens():
    tb = TokenBudget(max_tokens=0)
    assert tb.remaining == 0
    assert tb.fraction_used == 0.0
    tb.record(0)
    assert tb.used == 0
    assert tb.check() in (BudgetAction.SKIP, BudgetAction.DEGRADE)


def test_edge_case_negative_tokens():
    tb = TokenBudget(max_tokens=1000)
    tb.record(-100)
    assert tb.used == -100
    assert tb.remaining == 1100


def test_fraction_used_with_zero_max():
    tb = TokenBudget(max_tokens=0)
    assert tb.fraction_used == 0.0
    tb.record(50)
    # max(0, 1) = 1, so fraction = 50/1 = 50
    assert tb.fraction_used == 50.0


def test_stats_output():
    tb = TokenBudget(max_tokens=10000, warn_at=0.7, hard_cap_at=0.8)
    tb.record(8500)
    tb.check(estimated_cost=1000, step_name="gen")
    stats = tb.stats()
    assert stats["max_tokens"] == 10000
    assert stats["used"] == 8500
    assert stats["remaining"] == 1500
    # Hard cap hit first → SKIP, _degraded never set
    assert stats["degraded"] is False
    assert stats["skipped_steps"] == 1


def test_budget_prompt_vs_completion():
    """Verify budget limits per token type can work with separate instances."""
    prompt_budget = TokenBudget(max_tokens=2000, warn_at=0.8, hard_cap_at=1.0)
    completion_budget = TokenBudget(max_tokens=4000, warn_at=0.8, hard_cap_at=1.0)

    prompt_budget.record(1800)
    completion_budget.record(1000)

    assert prompt_budget.should_degrade()
    assert not completion_budget.should_degrade()

    prompt_budget.reset()
    assert not prompt_budget.should_degrade()
