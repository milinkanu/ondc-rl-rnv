"""
RewardSystem: modular, composable reward function with per-component breakdown.
"""
from __future__ import annotations

import math

from ondc_env.types import (
    ActionType,
    EpisodeState,
    OrderStatus,
    RewardResult,
    RewardWeights,
)


def _urgency_eta_threshold(urgency: float, max_eta: int = 10) -> float:
    """Map urgency [0,1] to an ETA threshold in steps (higher urgency → lower threshold)."""
    return max_eta * (1.0 - urgency)


class RewardSystem:
    """Computes scalar reward from weighted sub-reward components."""

    def __init__(self, weights: RewardWeights) -> None:
        self.weights = weights

    def compute(
        self,
        action: int,
        prev_state: EpisodeState,
        next_state: EpisodeState,
    ) -> RewardResult:
        """
        Compute reward for a transition. Does NOT mutate prev_state or next_state.
        Returns RewardResult where total == sum(breakdown.values()).
        """
        w = self.weights
        breakdown: dict[str, float] = {}

        try:
            action_type = ActionType(action)
        except ValueError:
            action_type = None

        # ----------------------------------------------------------------
        # Positive rewards
        # ----------------------------------------------------------------

        if action_type == ActionType.CONFIRM_ORDER and next_state.order_status == OrderStatus.CONFIRMED:
            breakdown["task_completion"] = w.task_completion

            if next_state.selected_offer is not None:
                # Good price: confirmed price <= 90% of budget
                if next_state.selected_offer.price <= prev_state.budget * 0.9:
                    breakdown["good_price"] = w.good_price

                # Fast delivery: ETA within urgency threshold
                eta_threshold = _urgency_eta_threshold(next_state.urgency)
                if next_state.selected_offer.delivery_eta <= eta_threshold:
                    breakdown["fast_delivery"] = w.fast_delivery

                # High seller rating
                if next_state.selected_offer.rating >= 4.0:
                    breakdown["high_seller_rating"] = w.high_seller_rating

        if action_type == ActionType.ACCEPT_DELIVERY and next_state.order_status == OrderStatus.DELIVERED:
            # Partial task completion on delivery acceptance
            breakdown["task_completion"] = w.task_completion * 0.5

        if action_type in (ActionType.RETURN_ITEM, ActionType.FILE_GRIEVANCE):
            if next_state.order_status == OrderStatus.CANCELLED:
                breakdown["successful_return"] = w.successful_return

        # ----------------------------------------------------------------
        # Negative rewards (penalties)
        # ----------------------------------------------------------------

        # Invalid action penalty — caller signals this via action being out of range
        # or the env sets a flag; we detect it by checking if action_type is None
        # OR the env passes the invalid flag via next_state having incremented count
        if next_state.invalid_action_count > prev_state.invalid_action_count:
            breakdown["invalid_action"] = w.invalid_action

        # Budget exceeded
        if next_state.total_spent > next_state.budget:
            breakdown["budget_exceeded"] = w.budget_exceeded

        # Order failed
        if next_state.order_status == OrderStatus.FAILED:
            breakdown["order_failed"] = w.order_failed

        # Unnecessary wait
        if action_type == ActionType.WAIT:
            breakdown["unnecessary_wait"] = w.unnecessary_wait

        # Late delivery: delivery_eta exceeded urgency deadline
        if next_state.delivery_eta > 0:
            eta_threshold = _urgency_eta_threshold(next_state.urgency)
            if next_state.delivery_eta > eta_threshold and next_state.order_status in (
                OrderStatus.SHIPPED, OrderStatus.PENDING
            ):
                breakdown["late_delivery"] = w.late_delivery

        total = sum(breakdown.values())

        # Guard against NaN/Inf
        if not math.isfinite(total):
            total = 0.0
            breakdown = {}

        return RewardResult(total=total, breakdown=breakdown)
