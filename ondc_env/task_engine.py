"""
TaskEngine: enforces Beckn protocol phase transitions and validates agent actions.
"""
from __future__ import annotations

from ondc_env.types import (
    ActionType,
    BecknPhase,
    EpisodeState,
    OrderStatus,
    ValidationResult,
)

# ---------------------------------------------------------------------------
# Valid actions per phase
# ---------------------------------------------------------------------------

VALID_ACTIONS: dict[BecknPhase, set[ActionType]] = {
    BecknPhase.SEARCH: {
        ActionType.SEARCH_PRODUCTS,
        ActionType.WAIT,
    },
    BecknPhase.SELECT: {
        ActionType.SELECT_SELLER_0,
        ActionType.SELECT_SELLER_1,
        ActionType.SELECT_SELLER_2,
        ActionType.SELECT_SELLER_3,
        ActionType.SELECT_SELLER_4,
        ActionType.WAIT,
    },
    BecknPhase.INIT: {
        ActionType.INIT_ORDER,
        ActionType.WAIT,
    },
    BecknPhase.CONFIRM: {
        ActionType.CONFIRM_ORDER,
        ActionType.CANCEL_BEFORE_CONFIRM,
        ActionType.WAIT,
    },
    BecknPhase.TRACK: {
        ActionType.TRACK_ORDER,
        ActionType.CANCEL_ORDER,
        ActionType.WAIT,
    },
    BecknPhase.POST_ORDER: {
        ActionType.ACCEPT_DELIVERY,
        ActionType.CANCEL_ORDER,
        ActionType.RETURN_ITEM,
        ActionType.FILE_GRIEVANCE,
        ActionType.WAIT,
    },
}

# Phase transitions for valid (phase, action) pairs
_TRANSITIONS: dict[tuple[BecknPhase, ActionType], BecknPhase] = {
    # SEARCH → SELECT
    (BecknPhase.SEARCH, ActionType.SEARCH_PRODUCTS): BecknPhase.SELECT,
    (BecknPhase.SEARCH, ActionType.WAIT): BecknPhase.SEARCH,

    # SELECT → INIT
    (BecknPhase.SELECT, ActionType.SELECT_SELLER_0): BecknPhase.INIT,
    (BecknPhase.SELECT, ActionType.SELECT_SELLER_1): BecknPhase.INIT,
    (BecknPhase.SELECT, ActionType.SELECT_SELLER_2): BecknPhase.INIT,
    (BecknPhase.SELECT, ActionType.SELECT_SELLER_3): BecknPhase.INIT,
    (BecknPhase.SELECT, ActionType.SELECT_SELLER_4): BecknPhase.INIT,
    (BecknPhase.SELECT, ActionType.WAIT): BecknPhase.SELECT,

    # INIT → CONFIRM
    (BecknPhase.INIT, ActionType.INIT_ORDER): BecknPhase.CONFIRM,
    (BecknPhase.INIT, ActionType.WAIT): BecknPhase.INIT,

    # CONFIRM → TRACK or back to SEARCH on cancel
    (BecknPhase.CONFIRM, ActionType.CONFIRM_ORDER): BecknPhase.TRACK,
    (BecknPhase.CONFIRM, ActionType.CANCEL_BEFORE_CONFIRM): BecknPhase.SEARCH,
    (BecknPhase.CONFIRM, ActionType.WAIT): BecknPhase.CONFIRM,

    # TRACK → POST_ORDER or back to SEARCH on cancel
    (BecknPhase.TRACK, ActionType.TRACK_ORDER): BecknPhase.POST_ORDER,
    (BecknPhase.TRACK, ActionType.CANCEL_ORDER): BecknPhase.SEARCH,
    (BecknPhase.TRACK, ActionType.WAIT): BecknPhase.TRACK,

    # POST_ORDER → terminal / SEARCH
    (BecknPhase.POST_ORDER, ActionType.ACCEPT_DELIVERY): BecknPhase.POST_ORDER,
    (BecknPhase.POST_ORDER, ActionType.CANCEL_ORDER): BecknPhase.SEARCH,
    (BecknPhase.POST_ORDER, ActionType.RETURN_ITEM): BecknPhase.POST_ORDER,
    (BecknPhase.POST_ORDER, ActionType.FILE_GRIEVANCE): BecknPhase.POST_ORDER,
    (BecknPhase.POST_ORDER, ActionType.WAIT): BecknPhase.POST_ORDER,
}

_TERMINAL_STATUSES = {
    OrderStatus.CONFIRMED,
    OrderStatus.CANCELLED,
    OrderStatus.DELIVERED,
}


class TaskEngine:
    """Validates actions and drives Beckn phase transitions."""

    def validate_action(self, action: int, state: EpisodeState) -> ValidationResult:
        """
        Check whether `action` is valid in the current phase.
        Does NOT mutate state.
        """
        try:
            action_type = ActionType(action)
        except ValueError:
            return ValidationResult(
                is_valid=False,
                reason=f"Unknown action {action}; must be in [0, {len(ActionType) - 1}]",
            )

        allowed = VALID_ACTIONS.get(state.current_phase, set())
        if action_type not in allowed:
            return ValidationResult(
                is_valid=False,
                reason=(
                    f"{action_type.name} is not valid in phase "
                    f"{state.current_phase.name}; "
                    f"allowed: {[a.name for a in sorted(allowed)]}"
                ),
            )

        return ValidationResult(is_valid=True)

    def transition(self, current_phase: BecknPhase, action: int) -> BecknPhase:
        """Return the next phase for a valid (phase, action) pair."""
        action_type = ActionType(action)
        return _TRANSITIONS.get((current_phase, action_type), current_phase)

    def is_terminal(self, state: EpisodeState) -> bool:
        """
        True when the episode should end:
        - order confirmed, cancelled, or delivered
        - budget exhausted (total_spent >= budget)
        - order status is RETURNED (represented via CANCELLED after return)
        """
        if state.order_status in _TERMINAL_STATUSES:
            return True
        if state.total_spent >= state.budget and state.budget > 0:
            return True
        return False
