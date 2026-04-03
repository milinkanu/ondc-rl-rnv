"""
Unit tests for ONDCAgentEnv.step() and render().
Requirements: 7.1–7.8, 11.1, 11.2
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from ondc_env import ActionType, BecknPhase, EnvConfig, N_OBS_DIM
from ondc_env.env import ONDCAgentEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_env(**kwargs) -> ONDCAgentEnv:
    return ONDCAgentEnv(EnvConfig(**kwargs))


def fresh_env(**kwargs) -> ONDCAgentEnv:
    env = make_env(**kwargs)
    env.reset(seed=0)
    return env


# ---------------------------------------------------------------------------
# Invalid action behaviour
# ---------------------------------------------------------------------------

class TestInvalidAction:
    def test_invalid_action_increments_count(self):
        env = fresh_env()
        # In SEARCH phase, CONFIRM_ORDER is invalid
        obs_before, _ = env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(ActionType.CONFIRM_ORDER)
        assert env.state.invalid_action_count == 1

    def test_invalid_action_returns_unchanged_obs(self):
        env = fresh_env()
        obs_before, _ = env.reset(seed=0)
        obs, *_ = env.step(ActionType.CONFIRM_ORDER)
        np.testing.assert_array_equal(obs, obs_before)

    def test_invalid_action_not_terminated(self):
        env = fresh_env()
        _, reward, terminated, truncated, _ = env.step(ActionType.CONFIRM_ORDER)
        assert not terminated
        assert not truncated

    def test_invalid_action_penalty_in_breakdown(self):
        env = fresh_env()
        _, _, _, _, info = env.step(ActionType.CONFIRM_ORDER)
        assert "invalid_action" in info["reward_breakdown"]

    def test_invalid_action_does_not_change_phase(self):
        env = fresh_env()
        env.step(ActionType.CONFIRM_ORDER)
        assert env.state.current_phase == BecknPhase.SEARCH

    def test_invalid_action_does_not_decrement_steps(self):
        env = fresh_env(max_steps=50)
        env.reset(seed=0)
        steps_before = env.state.steps_remaining
        env.step(ActionType.CONFIRM_ORDER)
        assert env.state.steps_remaining == steps_before


# ---------------------------------------------------------------------------
# Valid action behaviour
# ---------------------------------------------------------------------------

class TestValidAction:
    def test_valid_action_advances_phase(self):
        env = fresh_env()
        env.step(ActionType.SEARCH_PRODUCTS)
        assert env.state.current_phase == BecknPhase.SELECT

    def test_valid_action_decrements_steps(self):
        env = fresh_env(max_steps=50)
        env.reset(seed=0)
        steps_before = env.state.steps_remaining
        env.step(ActionType.SEARCH_PRODUCTS)
        assert env.state.steps_remaining == steps_before - 1

    def test_valid_action_returns_correct_shape(self):
        env = fresh_env()
        obs, *_ = env.step(ActionType.SEARCH_PRODUCTS)
        assert obs.shape == (N_OBS_DIM,)
        assert obs.dtype == np.float32

    def test_info_contains_required_keys(self):
        env = fresh_env()
        _, _, _, _, info = env.step(ActionType.SEARCH_PRODUCTS)
        assert "reward_breakdown" in info
        assert "phase" in info
        assert "events" in info

    def test_reward_is_finite(self):
        env = fresh_env()
        _, reward, _, _, _ = env.step(ActionType.SEARCH_PRODUCTS)
        assert np.isfinite(reward)


# ---------------------------------------------------------------------------
# Truncation and termination
# ---------------------------------------------------------------------------

class TestTermination:
    def test_truncated_when_steps_exhausted(self):
        env = make_env(max_steps=1)
        env.reset(seed=0)
        _, _, terminated, truncated, _ = env.step(ActionType.SEARCH_PRODUCTS)
        assert truncated or terminated  # steps_remaining hits 0

    def test_truncated_flag_true_at_zero_steps(self):
        env = make_env(max_steps=2)
        env.reset(seed=0)
        env.step(ActionType.SEARCH_PRODUCTS)
        _, _, _, truncated, _ = env.step(ActionType.WAIT)
        assert truncated

    def test_terminated_on_confirmed_order(self):
        """Drive env to CONFIRM phase and confirm order."""
        env = make_env(max_steps=50, seed=0)
        env.reset(seed=0)
        env.step(ActionType.SEARCH_PRODUCTS)   # → SELECT
        env.step(ActionType.SELECT_SELLER_0)   # → INIT
        env.step(ActionType.INIT_ORDER)        # → CONFIRM
        _, _, terminated, _, _ = env.step(ActionType.CONFIRM_ORDER)  # → TRACK
        assert terminated

    def test_terminated_on_cancel(self):
        env = make_env(max_steps=50, seed=0)
        env.reset(seed=0)
        env.step(ActionType.SEARCH_PRODUCTS)
        env.step(ActionType.SELECT_SELLER_0)
        env.step(ActionType.INIT_ORDER)
        _, _, terminated, _, _ = env.step(ActionType.CANCEL_BEFORE_CONFIRM)
        # CANCEL_BEFORE_CONFIRM sets order_status=CANCELLED → terminal
        assert terminated


# ---------------------------------------------------------------------------
# Monotonicity invariants
# ---------------------------------------------------------------------------

class TestMonotonicity:
    def test_steps_remaining_non_increasing(self):
        env = fresh_env(max_steps=20)
        env.reset(seed=0)
        prev_steps = env.state.steps_remaining
        for action in [ActionType.SEARCH_PRODUCTS, ActionType.WAIT, ActionType.WAIT]:
            env.step(action)
            assert env.state.steps_remaining <= prev_steps
            prev_steps = env.state.steps_remaining

    def test_invalid_action_count_non_decreasing(self):
        env = fresh_env()
        env.reset(seed=0)
        prev_count = env.state.invalid_action_count
        for _ in range(5):
            env.step(ActionType.CONFIRM_ORDER)  # always invalid in SEARCH
            assert env.state.invalid_action_count >= prev_count
            prev_count = env.state.invalid_action_count


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

class TestRender:
    def test_render_json_returns_valid_json(self):
        env = ONDCAgentEnv.__new__(ONDCAgentEnv)
        env.__init__(EnvConfig())
        env.render_mode = "json"
        env.reset(seed=0)
        result = env.render()
        assert result is not None
        data = json.loads(result)
        assert "current_phase" in data
        assert "budget" in data
        assert "sellers" in data

    def test_render_human_returns_string(self):
        env = ONDCAgentEnv.__new__(ONDCAgentEnv)
        env.__init__(EnvConfig())
        env.render_mode = "human"
        env.reset(seed=0)
        result = env.render()
        assert isinstance(result, str)
        assert "Phase" in result

    def test_render_none_before_reset(self):
        env = ONDCAgentEnv(EnvConfig())
        env.render_mode = "json"
        # state is None before reset
        env.state = None
        result = env.render()
        assert result is None

    def test_render_json_contains_all_episode_state_fields(self):
        env = ONDCAgentEnv.__new__(ONDCAgentEnv)
        env.__init__(EnvConfig())
        env.render_mode = "json"
        env.reset(seed=0)
        data = json.loads(env.render())
        required_fields = [
            "task_id", "target_item", "budget", "initial_budget", "urgency",
            "steps_remaining", "max_steps", "current_phase", "sellers",
            "selected_offer", "order_id", "order_status", "delivery_eta",
            "total_spent", "total_reward", "invalid_action_count",
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"
