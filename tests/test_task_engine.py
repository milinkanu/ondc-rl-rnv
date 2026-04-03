"""Unit and property tests for TaskEngine."""
from __future__ import annotations

import copy

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ondc_env.task_engine import TaskEngine, VALID_ACTIONS, _TRANSITIONS
from ondc_env.types import (
    ActionType,
    BecknPhase,
    EpisodeState,
    OrderStatus,
    N_ACTIONS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_state(
    phase: BecknPhase = BecknPhase.SEARCH,
    order_status: OrderStatus | None = None,
    total_spent: float = 0.0,
    budget: float = 1000.0,
) -> EpisodeState:
    return EpisodeState(
        task_id="test",
        target_item="laptop",
        budget=budget,
        initial_budget=budget,
        urgency=0.5,
        steps_remaining=50,
        max_steps=50,
        current_phase=phase,
        order_status=order_status,
        total_spent=total_spent,
    )


engine = TaskEngine()


# ---------------------------------------------------------------------------
# validate_action — invalid cases
# ---------------------------------------------------------------------------

class TestValidateActionInvalid:
    def test_confirm_in_search_phase(self):
        state = make_state(BecknPhase.SEARCH)
        result = engine.validate_action(ActionType.CONFIRM_ORDER, state)
        assert not result.is_valid
        assert result.reason != ""

    def test_init_in_search_phase(self):
        state = make_state(BecknPhase.SEARCH)
        result = engine.validate_action(ActionType.INIT_ORDER, state)
        assert not result.is_valid
        assert result.reason != ""

    def test_search_in_confirm_phase(self):
        state = make_state(BecknPhase.CONFIRM)
        result = engine.validate_action(ActionType.SEARCH_PRODUCTS, state)
        assert not result.is_valid
        assert result.reason != ""

    def test_unknown_action_returns_invalid(self):
        state = make_state(BecknPhase.SEARCH)
        result = engine.validate_action(999, state)
        assert not result.is_valid
        assert result.reason != ""

    @pytest.mark.parametrize("phase", list(BecknPhase))
    def test_every_phase_has_some_invalid_action(self, phase):
        """Each phase must reject at least one action."""
        state = make_state(phase)
        allowed = VALID_ACTIONS[phase]
        invalid_actions = [a for a in ActionType if a not in allowed]
        assert invalid_actions, f"Phase {phase} has no invalid actions — check VALID_ACTIONS"
        result = engine.validate_action(invalid_actions[0], state)
        assert not result.is_valid


# ---------------------------------------------------------------------------
# validate_action — valid cases
# ---------------------------------------------------------------------------

class TestValidateActionValid:
    @pytest.mark.parametrize("phase,action", [
        (BecknPhase.SEARCH, ActionType.SEARCH_PRODUCTS),
        (BecknPhase.SELECT, ActionType.SELECT_SELLER_0),
        (BecknPhase.SELECT, ActionType.SELECT_SELLER_2),
        (BecknPhase.INIT, ActionType.INIT_ORDER),
        (BecknPhase.CONFIRM, ActionType.CONFIRM_ORDER),
        (BecknPhase.CONFIRM, ActionType.CANCEL_BEFORE_CONFIRM),
        (BecknPhase.TRACK, ActionType.TRACK_ORDER),
        (BecknPhase.POST_ORDER, ActionType.ACCEPT_DELIVERY),
        (BecknPhase.POST_ORDER, ActionType.RETURN_ITEM),
    ])
    def test_valid_action_returns_true(self, phase, action):
        state = make_state(phase)
        result = engine.validate_action(action, state)
        assert result.is_valid

    def test_wait_valid_in_all_phases(self):
        for phase in BecknPhase:
            state = make_state(phase)
            result = engine.validate_action(ActionType.WAIT, state)
            assert result.is_valid, f"WAIT should be valid in {phase.name}"


# ---------------------------------------------------------------------------
# Phase transitions
# ---------------------------------------------------------------------------

class TestTransitions:
    @pytest.mark.parametrize("phase,action,expected", [
        (BecknPhase.SEARCH, ActionType.SEARCH_PRODUCTS, BecknPhase.SELECT),
        (BecknPhase.SELECT, ActionType.SELECT_SELLER_0, BecknPhase.INIT),
        (BecknPhase.SELECT, ActionType.SELECT_SELLER_4, BecknPhase.INIT),
        (BecknPhase.INIT, ActionType.INIT_ORDER, BecknPhase.CONFIRM),
        (BecknPhase.CONFIRM, ActionType.CONFIRM_ORDER, BecknPhase.TRACK),
        (BecknPhase.TRACK, ActionType.TRACK_ORDER, BecknPhase.POST_ORDER),
    ])
    def test_forward_transitions(self, phase, action, expected):
        assert engine.transition(phase, action) == expected

    def test_cancel_from_confirm_goes_to_search(self):
        assert engine.transition(BecknPhase.CONFIRM, ActionType.CANCEL_BEFORE_CONFIRM) == BecknPhase.SEARCH

    def test_cancel_from_track_goes_to_search(self):
        assert engine.transition(BecknPhase.TRACK, ActionType.CANCEL_ORDER) == BecknPhase.SEARCH

    def test_wait_keeps_phase(self):
        for phase in BecknPhase:
            assert engine.transition(phase, ActionType.WAIT) == phase

    def test_all_valid_pairs_have_transition(self):
        for phase, actions in VALID_ACTIONS.items():
            for action in actions:
                result = engine.transition(phase, action)
                assert isinstance(result, BecknPhase)


# ---------------------------------------------------------------------------
# is_terminal
# ---------------------------------------------------------------------------

class TestIsTerminal:
    def test_confirmed_order_is_terminal(self):
        state = make_state(order_status=OrderStatus.CONFIRMED)
        assert engine.is_terminal(state)

    def test_cancelled_order_is_terminal(self):
        state = make_state(order_status=OrderStatus.CANCELLED)
        assert engine.is_terminal(state)

    def test_delivered_order_is_terminal(self):
        state = make_state(order_status=OrderStatus.DELIVERED)
        assert engine.is_terminal(state)

    def test_pending_order_not_terminal(self):
        state = make_state(order_status=OrderStatus.PENDING)
        assert not engine.is_terminal(state)

    def test_no_order_not_terminal(self):
        state = make_state()
        assert not engine.is_terminal(state)

    def test_budget_exhausted_is_terminal(self):
        state = make_state(total_spent=1000.0, budget=1000.0)
        assert engine.is_terminal(state)

    def test_budget_not_exhausted_not_terminal(self):
        state = make_state(total_spent=500.0, budget=1000.0)
        assert not engine.is_terminal(state)

    def test_cancel_from_confirm_is_valid(self):
        state = make_state(BecknPhase.CONFIRM)
        result = engine.validate_action(ActionType.CANCEL_BEFORE_CONFIRM, state)
        assert result.is_valid

    def test_cancel_from_track_is_valid(self):
        state = make_state(BecknPhase.TRACK)
        result = engine.validate_action(ActionType.CANCEL_ORDER, state)
        assert result.is_valid


# ---------------------------------------------------------------------------
# Property: validate_action does not mutate state
# ---------------------------------------------------------------------------

@given(
    action=st.integers(min_value=0, max_value=N_ACTIONS + 5),
    phase=st.sampled_from(list(BecknPhase)),
)
@settings(max_examples=300)
def test_validate_action_does_not_mutate_state(action, phase):
    """Property 14: TaskEngine does not mutate state during validation."""
    state = make_state(phase)
    before = copy.deepcopy(state)
    engine.validate_action(action, state)
    assert state.current_phase == before.current_phase
    assert state.budget == before.budget
    assert state.total_spent == before.total_spent
    assert state.invalid_action_count == before.invalid_action_count
    assert state.steps_remaining == before.steps_remaining
