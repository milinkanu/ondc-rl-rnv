"""
ONDCAgentEnv: Gymnasium-compatible RL environment for the ONDC buyer lifecycle.
"""
from __future__ import annotations

import copy
import json
import uuid
from dataclasses import asdict
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ondc_env.types import (
    MAX_SELLERS,
    N_ACTIONS,
    N_OBS_DIM,
    ActionType,
    BecknPhase,
    EnvConfig,
    EpisodeState,
    SearchQuery,
    SellerConfig,
)
from ondc_env.task_engine import TaskEngine
from ondc_env.seller_simulator import SellerSimulator
from ondc_env.reward_system import RewardSystem


# ---------------------------------------------------------------------------
# Observation schema index constants (mirrors OBS_SCHEMA in design doc)
# ---------------------------------------------------------------------------

_IDX_BUDGET_NORM = 0
_IDX_URGENCY = 1
_IDX_STEPS_NORM = 2
_IDX_PHASE_ONEHOT = slice(3, 9)       # 6 phases
_IDX_SELLERS = slice(9, 49)           # 5 sellers × 8 features
_IDX_SEL_PRICE = 49
_IDX_SEL_RATING = 50
_IDX_SEL_ETA = 51
_IDX_HAS_SELECTION = 52
_IDX_ORDER_STATUS = slice(53, 58)     # 5 statuses one-hot
_IDX_TOTAL_SPENT_NORM = 58
_IDX_INVALID_RATIO = 59
_IDX_DELIVERY_ETA_NORM = 60

_N_PHASES = 6
_N_ORDER_STATUSES = 5
_SELLER_FEATURES = 8                  # per seller: price, rating, eta, stock, discount, avail, fulfil(3)
_FULFILLMENT_TYPES = ["standard", "express", "same_day"]


class ONDCAgentEnv(gym.Env):
    """
    Gymnasium environment simulating the ONDC buyer-seller lifecycle.

    Observation: flat float32 array of shape (N_OBS_DIM,) = (61,)
    Action:      Discrete(N_ACTIONS) = Discrete(15)
    """

    metadata = {"render_modes": ["human", "json"]}

    def __init__(self, config: EnvConfig | None = None) -> None:
        super().__init__()
        self.config = config or EnvConfig()

        # Spaces
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(
            low=0.0,
            high=10.0,
            shape=(N_OBS_DIM,),
            dtype=np.float32,
        )

        # Sub-components
        self.task_engine = TaskEngine()
        self.seller_sim = SellerSimulator(
            n_sellers=self.config.n_sellers,
            config=self.config.seller_config,
            random_event_prob=self.config.random_event_prob,
        )
        self.reward_system = RewardSystem(self.config.reward_weights)

        # Episode state (initialised on reset)
        self.state: EpisodeState | None = None
        self._episode_started = False

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Reset the environment to an initial state.

        Returns (obs, info) where obs.shape == (N_OBS_DIM,) and obs.dtype == float32.
        """
        super().reset(seed=seed)

        # Resolve seed: explicit arg > config seed > None
        effective_seed = seed if seed is not None else self.config.seed

        # Resolve budget from options or config
        budget = self.config.initial_budget
        if options is not None:
            budget = float(options.get("budget", budget))

        # Resolve other episode parameters from options
        urgency = float(
            (options or {}).get(
                "urgency",
                self.np_random.uniform(*self.config.urgency_range),
            )
        )
        target_item = str((options or {}).get("target_item", "item"))
        task_id = str((options or {}).get("task_id", uuid.uuid4()))

        # Initialise sellers
        sellers = self.seller_sim.reset(seed=effective_seed)

        self.state = EpisodeState(
            task_id=task_id,
            target_item=target_item,
            budget=budget,
            initial_budget=budget,
            urgency=urgency,
            steps_remaining=self.config.max_steps,
            max_steps=self.config.max_steps,
            current_phase=BecknPhase.SEARCH,
            sellers=sellers,
        )
        self._episode_started = True

        obs = self._build_obs(self.state)
        info: dict[str, Any] = {
            "phase": self.state.current_phase,
            "budget": self.state.budget,
            "urgency": self.state.urgency,
        }
        return obs, info

    def step(
        self,
        action: int | np.integer,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one environment step.

        Returns (obs, reward, terminated, truncated, info).
        - Invalid action: increments invalid_action_count, applies penalty, returns current obs.
        - Valid action: applies effects, advances phase, ticks sellers, decrements steps.
        """
        assert self._episode_started, "Call reset() before step()"
        action = int(action)

        prev_state = copy.deepcopy(self.state)

        # 1. Validate action
        validation = self.task_engine.validate_action(action, self.state)

        if not validation.is_valid:
            self.state.invalid_action_count += 1
            reward_result = self.reward_system.compute(action, prev_state, self.state)
            self.state.total_reward += reward_result.total
            obs = self._build_obs(self.state)
            info: dict[str, Any] = {
                "reward_breakdown": reward_result.breakdown,
                "phase": self.state.current_phase,
                "events": [],
                "invalid_action": True,
                "reason": validation.reason,
            }
            return obs, reward_result.total, False, False, info

        # 2. Apply action effects via SellerSimulator
        events = self._apply_action_effects(action)

        # 3. Advance Beckn phase
        self.state.current_phase = self.task_engine.transition(
            self.state.current_phase, action
        )

        # 4. Tick seller simulator
        tick_events = self.seller_sim.tick()
        self._apply_tick_events(tick_events)
        all_events = events + tick_events

        # 5. Decrement steps
        self.state.steps_remaining -= 1

        # 6. Compute reward
        reward_result = self.reward_system.compute(action, prev_state, self.state)
        self.state.total_reward += reward_result.total

        # 7. Check termination
        terminated = self.task_engine.is_terminal(self.state)
        truncated = self.state.steps_remaining <= 0

        # 8. Build observation
        obs = self._build_obs(self.state)

        info = {
            "reward_breakdown": reward_result.breakdown,
            "phase": self.state.current_phase,
            "events": all_events,
        }
        return obs, reward_result.total, terminated, truncated, info

    def render(self, render_mode: str | None = None) -> str | None:
        """
        Render the current environment state.

        render_mode="human": formatted string with phase, sellers, offer, reward breakdown.
        render_mode="json":  JSON-serializable string of current EpisodeState.

        The optional render_mode argument overrides self.render_mode for this call.
        """
        if self.state is None:
            return None

        mode = render_mode if render_mode is not None else self.render_mode

        if mode == "json":
            return self._render_json()

        if mode == "human":
            return self._render_human()

        return None

    def close(self) -> None:
        """Clean up resources."""
        self.state = None
        self._episode_started = False

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_obs(self, state: EpisodeState) -> np.ndarray:
        """
        Construct flat float32 observation of shape (N_OBS_DIM,) from EpisodeState.

        Layout (mirrors OBS_SCHEMA in design doc):
          [0]      budget_normalized
          [1]      urgency
          [2]      steps_remaining_norm
          [3:9]    phase one-hot (6 phases)
          [9:49]   per-seller features (5 × 8)
          [49]     selected_price_norm
          [50]     selected_rating_norm
          [51]     selected_eta_norm
          [52]     has_selection
          [53:58]  order_status one-hot (5 statuses)
          [58]     total_spent_norm
          [59]     invalid_action_ratio
          [60]     delivery_eta_norm
        """
        obs = np.zeros(N_OBS_DIM, dtype=np.float32)

        initial_budget = state.initial_budget if state.initial_budget > 0 else 1.0
        max_steps = state.max_steps if state.max_steps > 0 else 1

        # --- Task context [0:3] ---
        obs[_IDX_BUDGET_NORM] = state.budget / initial_budget
        obs[_IDX_URGENCY] = state.urgency
        obs[_IDX_STEPS_NORM] = state.steps_remaining / max_steps

        # --- Phase one-hot [3:9] ---
        phase_idx = int(state.current_phase)
        if 0 <= phase_idx < _N_PHASES:
            obs[3 + phase_idx] = 1.0

        # --- Per-seller features [9:49] (5 sellers × 8 features) ---
        cfg = self.config.seller_config
        price_range = cfg.price_range
        price_span = (price_range[1] - price_range[0]) or 1.0
        eta_max = float(cfg.eta_range[1]) or 1.0
        stock_max = float(cfg.stock_range[1]) or 1.0

        for i in range(MAX_SELLERS):
            base = 9 + i * _SELLER_FEATURES
            if i < len(state.sellers):
                s = state.sellers[i]
                obs[base + 0] = (s.price - price_range[0]) / price_span
                obs[base + 1] = s.rating / 5.0
                obs[base + 2] = s.delivery_eta / eta_max
                obs[base + 3] = s.stock / stock_max
                obs[base + 4] = s.discount_pct
                obs[base + 5] = 1.0 if s.is_available else 0.0
                # fulfillment encoded as normalized scalar: standard=0.0, express=0.5, same_day=1.0
                ft_idx = _FULFILLMENT_TYPES.index(s.fulfillment_type) if s.fulfillment_type in _FULFILLMENT_TYPES else 0
                obs[base + 6] = ft_idx / max(len(_FULFILLMENT_TYPES) - 1, 1)
                obs[base + 7] = 0.0  # reserved / padding
            # else: slot stays zero-padded

        # --- Selected offer [49:53] ---
        if state.selected_offer is not None:
            offer = state.selected_offer
            obs[_IDX_SEL_PRICE] = (offer.price - price_range[0]) / price_span
            obs[_IDX_SEL_RATING] = offer.rating / 5.0
            obs[_IDX_SEL_ETA] = offer.delivery_eta / eta_max
            obs[_IDX_HAS_SELECTION] = 1.0
        # else: indices 49–52 stay 0.0

        # --- Order status one-hot [53:58] ---
        if state.order_status is not None:
            status_idx = int(state.order_status)
            if 0 <= status_idx < _N_ORDER_STATUSES:
                obs[53 + status_idx] = 1.0

        # --- Episode metrics [58:61] ---
        obs[_IDX_TOTAL_SPENT_NORM] = state.total_spent / initial_budget
        total_steps_taken = max_steps - state.steps_remaining
        obs[_IDX_INVALID_RATIO] = (
            state.invalid_action_count / total_steps_taken
            if total_steps_taken > 0
            else 0.0
        )
        obs[_IDX_DELIVERY_ETA_NORM] = state.delivery_eta / eta_max

        return obs

    # ------------------------------------------------------------------
    # Action effect helpers
    # ------------------------------------------------------------------

    def _apply_action_effects(self, action: int) -> list:
        """Apply action-specific effects to the episode state. Returns any events."""
        from ondc_env.types import ActionType, OrderStatus

        try:
            action_type = ActionType(action)
        except ValueError:
            return []

        state = self.state

        # SELECT_SELLER_* — pick a seller and set selected_offer
        _select_map = {
            ActionType.SELECT_SELLER_0: 0,
            ActionType.SELECT_SELLER_1: 1,
            ActionType.SELECT_SELLER_2: 2,
            ActionType.SELECT_SELLER_3: 3,
            ActionType.SELECT_SELLER_4: 4,
        }
        if action_type in _select_map:
            idx = _select_map[action_type]
            query = SearchQuery(item_name=state.target_item)
            catalog = self.seller_sim.get_catalog(query)
            if idx < len(catalog):
                offer = catalog[idx]
                result = self.seller_sim.apply_selection(offer.seller_id, offer.item_id, 1)
                if result.success:
                    state.selected_offer = result.offer

        elif action_type == ActionType.CONFIRM_ORDER:
            if state.selected_offer is not None:
                state.order_id = str(uuid.uuid4())
                state.order_status = OrderStatus.CONFIRMED
                state.total_spent += state.selected_offer.price
                state.budget -= state.selected_offer.price
                state.delivery_eta = float(state.selected_offer.delivery_eta)

        elif action_type == ActionType.CANCEL_BEFORE_CONFIRM:
            state.selected_offer = None
            state.order_status = OrderStatus.CANCELLED

        elif action_type == ActionType.CANCEL_ORDER:
            state.order_status = OrderStatus.CANCELLED

        elif action_type == ActionType.TRACK_ORDER:
            # Simulate delivery progress
            if state.delivery_eta > 0:
                state.delivery_eta = max(0.0, state.delivery_eta - 1)
            if state.delivery_eta == 0 and state.order_status == OrderStatus.CONFIRMED:
                state.order_status = OrderStatus.SHIPPED

        elif action_type == ActionType.ACCEPT_DELIVERY:
            state.order_status = OrderStatus.DELIVERED

        elif action_type == ActionType.RETURN_ITEM:
            state.order_status = OrderStatus.CANCELLED

        elif action_type == ActionType.SEARCH_PRODUCTS:
            # Refresh seller catalog (no state change needed; sellers already in state)
            pass

        return []

    def _apply_tick_events(self, events: list) -> None:
        """Sync seller states in EpisodeState after a tick."""
        # Sellers are shared by reference via seller_sim; update state.sellers
        self.state.sellers = list(self.seller_sim.sellers)

        # If selected offer's seller went down, mark order as failed
        if self.state.selected_offer is not None:
            sel_id = self.state.selected_offer.seller_id
            for event in events:
                if event.seller_id == sel_id and event.event_type in ("STOCKOUT", "SELLER_DOWN"):
                    from ondc_env.types import OrderStatus
                    if self.state.order_status not in (
                        OrderStatus.CONFIRMED, OrderStatus.SHIPPED,
                        OrderStatus.DELIVERED, OrderStatus.CANCELLED,
                    ):
                        self.state.order_status = OrderStatus.FAILED

    # ------------------------------------------------------------------
    # Render helpers
    # ------------------------------------------------------------------

    def _render_human(self) -> str:
        state = self.state
        lines = [
            f"Phase:        {state.current_phase.name}",
            f"Budget:       {state.budget:.2f} / {state.initial_budget:.2f}",
            f"Steps left:   {state.steps_remaining}",
            f"Total spent:  {state.total_spent:.2f}",
            f"Order status: {state.order_status.name if state.order_status else 'None'}",
            "",
            "Sellers:",
        ]
        for s in state.sellers:
            avail = "✓" if s.is_available else "✗"
            lines.append(
                f"  [{avail}] {s.name:12s}  price={s.price:.2f}  "
                f"rating={s.rating:.1f}  eta={s.delivery_eta}  stock={s.stock}"
            )
        if state.selected_offer:
            o = state.selected_offer
            lines += [
                "",
                f"Selected:     {o.seller_id}  price={o.price:.2f}  "
                f"rating={o.rating:.1f}  eta={o.delivery_eta}",
            ]
        return "\n".join(lines)

    def _render_json(self) -> str:
        state = self.state

        def _default(obj):
            if hasattr(obj, "__int__"):
                return int(obj)
            if hasattr(obj, "__float__"):
                return float(obj)
            return str(obj)

        data = {
            "task_id": state.task_id,
            "target_item": state.target_item,
            "budget": state.budget,
            "initial_budget": state.initial_budget,
            "urgency": state.urgency,
            "steps_remaining": state.steps_remaining,
            "max_steps": state.max_steps,
            "current_phase": int(state.current_phase),
            "current_phase_name": state.current_phase.name,
            "sellers": [
                {
                    "seller_id": s.seller_id,
                    "name": s.name,
                    "price": s.price,
                    "original_price": s.original_price,
                    "stock": s.stock,
                    "rating": s.rating,
                    "delivery_eta": s.delivery_eta,
                    "is_available": s.is_available,
                    "discount_pct": s.discount_pct,
                    "fulfillment_type": s.fulfillment_type,
                }
                for s in state.sellers
            ],
            "selected_offer": (
                {
                    "seller_id": state.selected_offer.seller_id,
                    "item_id": state.selected_offer.item_id,
                    "name": state.selected_offer.name,
                    "price": state.selected_offer.price,
                    "rating": state.selected_offer.rating,
                    "delivery_eta": state.selected_offer.delivery_eta,
                    "stock": state.selected_offer.stock,
                    "is_available": state.selected_offer.is_available,
                    "fulfillment_type": state.selected_offer.fulfillment_type,
                }
                if state.selected_offer is not None
                else None
            ),
            "order_id": state.order_id,
            "order_status": int(state.order_status) if state.order_status is not None else None,
            "order_status_name": state.order_status.name if state.order_status is not None else None,
            "delivery_eta": state.delivery_eta,
            "tracking_updates": state.tracking_updates,
            "total_spent": state.total_spent,
            "total_reward": state.total_reward,
            "invalid_action_count": state.invalid_action_count,
        }
        return json.dumps(data, default=_default)
