"""Unit and property tests for RewardSystem."""
from __future__ import annotations

import copy
import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ondc_env.reward_system import RewardSystem
from ondc_env.types import (
    ActionType,
    BecknPhase,
    EpisodeState,
    N_ACTIONS,
    OrderStatus,
    RewardWeights,
    SellerOffer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_state(
    phase: BecknPhase = BecknPhase.CONFIRM,
    order_status: OrderStatus | None = None,
    total_spent: float = 0.0,
    budget: float = 1000.0,
    urgency: float = 0.5,
    delivery_eta: float = 0.0,
    invalid_action_count: int = 0,
    selected_offer: SellerOffer | None = None,
) -> EpisodeState:
    return EpisodeState(
        task_id="test",
        target_item="laptop",
        budget=budget,
        initial_budget=budget,
        urgency=urgency,
        steps_remaining=50,
        max_steps=50,
        current_phase=phase,
        order_status=order_status,
        total_spent=total_spent,
        delivery_eta=delivery_eta,
        invalid_action_count=invalid_action_count,
        selected_offer=selected_offer,
    )


def make_offer(price: float = 500.0, rating: float = 4.5, eta: int = 3) -> SellerOffer:
    return SellerOffer(
        seller_id="seller_0",
        item_id="item_0",
        name="laptop",
        price=price,
        rating=rating,
        delivery_eta=eta,
        stock=10,
        is_available=True,
        fulfillment_type="standard",
    )


rs = RewardSystem(RewardWeights())


# ---------------------------------------------------------------------------
# Positive reward components
# ---------------------------------------------------------------------------

class TestPositiveRewards:
    def test_task_completion_on_confirm(self):
        offer = make_offer(price=800.0, rating=4.5, eta=3)
        prev = make_state(budget=1000.0, selected_offer=offer)
        nxt = make_state(order_status=OrderStatus.CONFIRMED, budget=1000.0, selected_offer=offer)
        result = rs.compute(ActionType.CONFIRM_ORDER, prev, nxt)
        assert "task_completion" in result.breakdown
        assert result.breakdown["task_completion"] == RewardWeights().task_completion

    def test_good_price_fires_when_price_below_90pct_budget(self):
        offer = make_offer(price=800.0)   # 800 <= 1000 * 0.9 = 900
        prev = make_state(budget=1000.0, selected_offer=offer)
        nxt = make_state(order_status=OrderStatus.CONFIRMED, budget=1000.0, selected_offer=offer)
        result = rs.compute(ActionType.CONFIRM_ORDER, prev, nxt)
        assert "good_price" in result.breakdown

    def test_good_price_does_not_fire_when_price_above_90pct_budget(self):
        offer = make_offer(price=950.0)   # 950 > 1000 * 0.9 = 900
        prev = make_state(budget=1000.0, selected_offer=offer)
        nxt = make_state(order_status=OrderStatus.CONFIRMED, budget=1000.0, selected_offer=offer)
        result = rs.compute(ActionType.CONFIRM_ORDER, prev, nxt)
        assert "good_price" not in result.breakdown

    def test_fast_delivery_fires_within_urgency_threshold(self):
        # urgency=0.8 → threshold = 10*(1-0.8) = 2.0; eta=1 qualifies
        offer = make_offer(eta=1)
        prev = make_state(urgency=0.8, budget=1000.0, selected_offer=offer)
        nxt = make_state(order_status=OrderStatus.CONFIRMED, urgency=0.8, budget=1000.0, selected_offer=offer)
        result = rs.compute(ActionType.CONFIRM_ORDER, prev, nxt)
        assert "fast_delivery" in result.breakdown

    def test_high_seller_rating_fires_when_rating_gte_4(self):
        offer = make_offer(rating=4.0)
        prev = make_state(budget=1000.0, selected_offer=offer)
        nxt = make_state(order_status=OrderStatus.CONFIRMED, budget=1000.0, selected_offer=offer)
        result = rs.compute(ActionType.CONFIRM_ORDER, prev, nxt)
        assert "high_seller_rating" in result.breakdown

    def test_high_seller_rating_does_not_fire_below_4(self):
        offer = make_offer(rating=3.5)
        prev = make_state(budget=1000.0, selected_offer=offer)
        nxt = make_state(order_status=OrderStatus.CONFIRMED, budget=1000.0, selected_offer=offer)
        result = rs.compute(ActionType.CONFIRM_ORDER, prev, nxt)
        assert "high_seller_rating" not in result.breakdown


# ---------------------------------------------------------------------------
# Penalty components
# ---------------------------------------------------------------------------

class TestPenalties:
    def test_invalid_action_penalty_fires(self):
        prev = make_state(invalid_action_count=0)
        nxt = make_state(invalid_action_count=1)
        result = rs.compute(ActionType.SEARCH_PRODUCTS, prev, nxt)
        assert "invalid_action" in result.breakdown
        assert result.breakdown["invalid_action"] == RewardWeights().invalid_action

    def test_budget_exceeded_penalty_fires(self):
        prev = make_state(budget=1000.0, total_spent=0.0)
        nxt = make_state(budget=1000.0, total_spent=1100.0)
        result = rs.compute(ActionType.WAIT, prev, nxt)
        assert "budget_exceeded" in result.breakdown

    def test_budget_exceeded_does_not_fire_within_budget(self):
        prev = make_state(budget=1000.0, total_spent=0.0)
        nxt = make_state(budget=1000.0, total_spent=500.0)
        result = rs.compute(ActionType.WAIT, prev, nxt)
        assert "budget_exceeded" not in result.breakdown

    def test_order_failed_penalty_fires(self):
        prev = make_state()
        nxt = make_state(order_status=OrderStatus.FAILED)
        result = rs.compute(ActionType.INIT_ORDER, prev, nxt)
        assert "order_failed" in result.breakdown

    def test_unnecessary_wait_penalty_fires(self):
        prev = make_state()
        nxt = make_state()
        result = rs.compute(ActionType.WAIT, prev, nxt)
        assert "unnecessary_wait" in result.breakdown

    def test_late_delivery_penalty_fires_when_overdue(self):
        # urgency=0.9 → threshold = 10*(1-0.9) = 1.0; eta=5 is overdue
        prev = make_state(urgency=0.9)
        nxt = make_state(urgency=0.9, delivery_eta=5.0, order_status=OrderStatus.SHIPPED)
        result = rs.compute(ActionType.TRACK_ORDER, prev, nxt)
        assert "late_delivery" in result.breakdown


# ---------------------------------------------------------------------------
# Total consistency
# ---------------------------------------------------------------------------

class TestTotalConsistency:
    def test_total_equals_breakdown_sum_on_confirm(self):
        offer = make_offer(price=800.0, rating=4.5, eta=1)
        prev = make_state(budget=1000.0, urgency=0.8, selected_offer=offer)
        nxt = make_state(order_status=OrderStatus.CONFIRMED, budget=1000.0, urgency=0.8, selected_offer=offer)
        result = rs.compute(ActionType.CONFIRM_ORDER, prev, nxt)
        assert abs(result.total - sum(result.breakdown.values())) < 1e-6

    def test_total_is_finite(self):
        prev = make_state()
        nxt = make_state()
        result = rs.compute(ActionType.WAIT, prev, nxt)
        assert math.isfinite(result.total)

    def test_empty_breakdown_total_is_zero(self):
        prev = make_state(phase=BecknPhase.SEARCH)
        nxt = make_state(phase=BecknPhase.SELECT)
        result = rs.compute(ActionType.SEARCH_PRODUCTS, prev, nxt)
        assert result.total == sum(result.breakdown.values())


# ---------------------------------------------------------------------------
# Property: reward total == sum of breakdown (Property 2)
# ---------------------------------------------------------------------------

@given(
    action=st.integers(min_value=0, max_value=N_ACTIONS - 1),
    total_spent=st.floats(min_value=0.0, max_value=2000.0, allow_nan=False, allow_infinity=False),
    budget=st.floats(min_value=100.0, max_value=2000.0, allow_nan=False, allow_infinity=False),
    invalid_count=st.integers(min_value=0, max_value=10),
)
@settings(max_examples=300)
def test_reward_total_equals_breakdown_sum(action, total_spent, budget, invalid_count):
    """Property 2: Reward total equals sum of breakdown components.

    **Validates: Requirements 6.1, 6.12**
    """
    prev = make_state(invalid_action_count=invalid_count)
    nxt = make_state(total_spent=total_spent, budget=budget, invalid_action_count=invalid_count + 1)
    result = rs.compute(action, prev, nxt)
    assert abs(result.total - sum(result.breakdown.values())) < 1e-6
    assert math.isfinite(result.total)


# ---------------------------------------------------------------------------
# Property: RewardSystem does not mutate input states (Property 13)
# ---------------------------------------------------------------------------

@given(
    action=st.integers(min_value=0, max_value=N_ACTIONS - 1),
    total_spent=st.floats(min_value=0.0, max_value=2000.0, allow_nan=False, allow_infinity=False),
    budget=st.floats(min_value=100.0, max_value=2000.0, allow_nan=False, allow_infinity=False),
    invalid_count=st.integers(min_value=0, max_value=10),
    order_status=st.sampled_from([None] + list(OrderStatus)),
)
@settings(max_examples=300)
def test_reward_system_does_not_mutate_states(action, total_spent, budget, invalid_count, order_status):
    """Property 13: RewardSystem does not mutate input states.

    For any action and pair of EpisodeState instances, calling
    RewardSystem.compute() must not modify prev_state or next_state.

    **Validates: Requirements 6.11**
    """
    import dataclasses

    prev = make_state(
        total_spent=total_spent,
        budget=budget,
        invalid_action_count=invalid_count,
        order_status=order_status,
    )
    nxt = make_state(
        total_spent=total_spent,
        budget=budget,
        invalid_action_count=invalid_count + 1,
        order_status=order_status,
    )
    prev_snapshot = dataclasses.asdict(prev)
    nxt_snapshot = dataclasses.asdict(nxt)

    rs.compute(action, prev, nxt)

    assert dataclasses.asdict(prev) == prev_snapshot, "prev_state was mutated by RewardSystem.compute()"
    assert dataclasses.asdict(nxt) == nxt_snapshot, "next_state was mutated by RewardSystem.compute()"


# ---------------------------------------------------------------------------
# Property: reward components fire on correct conditions (Property 16)
# ---------------------------------------------------------------------------

@given(
    budget=st.floats(min_value=100.0, max_value=2000.0, allow_nan=False, allow_infinity=False),
    overspend=st.floats(min_value=0.01, max_value=5000.0, allow_nan=False, allow_infinity=False),
    action=st.integers(min_value=0, max_value=N_ACTIONS - 1),
)
@settings(max_examples=300)
def test_budget_exceeded_fires_when_total_spent_exceeds_budget(budget, overspend, action):
    """Property 16: budget_exceeded fires whenever total_spent > budget.

    For any episode state where total_spent > budget, the budget_exceeded
    penalty must appear in the reward breakdown regardless of the action taken.

    **Validates: Requirements 6.7**
    """
    total_spent = budget + overspend  # guaranteed > budget
    prev = make_state(budget=budget)
    nxt = make_state(budget=budget, total_spent=total_spent)
    result = rs.compute(action, prev, nxt)
    assert "budget_exceeded" in result.breakdown


@given(
    phase=st.sampled_from(list(BecknPhase)),
    order_status=st.sampled_from([None] + list(OrderStatus)),
    total_spent=st.floats(min_value=0.0, max_value=500.0, allow_nan=False, allow_infinity=False),
    budget=st.floats(min_value=600.0, max_value=2000.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=300)
def test_unnecessary_wait_fires_for_any_wait_action(phase, order_status, total_spent, budget):
    """Property 16: unnecessary_wait fires for any WAIT action regardless of state.

    For any episode state, taking a WAIT action must always include the
    unnecessary_wait penalty in the reward breakdown.

    **Validates: Requirements 6.9**
    """
    prev = make_state(phase=phase, order_status=order_status, budget=budget)
    nxt = make_state(phase=phase, order_status=order_status, total_spent=total_spent, budget=budget)
    result = rs.compute(ActionType.WAIT, prev, nxt)
    assert "unnecessary_wait" in result.breakdown


@given(
    budget=st.floats(min_value=100.0, max_value=2000.0, allow_nan=False, allow_infinity=False),
    price_fraction=st.floats(min_value=0.0, max_value=0.9, allow_nan=False, allow_infinity=False),
    rating=st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    eta=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=300)
def test_good_price_fires_when_confirmed_price_at_or_below_90pct_budget(
    budget, price_fraction, rating, eta
):
    """Property 16: good_price fires when confirmed order price <= 0.9 * budget.

    For any confirmed order where selected_offer.price <= 0.9 * budget,
    the good_price component must appear in the reward breakdown.

    **Validates: Requirements 6.3**
    """
    price = price_fraction * budget  # guaranteed <= 0.9 * budget
    offer = make_offer(price=price, rating=rating, eta=eta)
    prev = make_state(budget=budget, selected_offer=offer)
    nxt = make_state(order_status=OrderStatus.CONFIRMED, budget=budget, selected_offer=offer)
    result = rs.compute(ActionType.CONFIRM_ORDER, prev, nxt)
    assert "good_price" in result.breakdown
