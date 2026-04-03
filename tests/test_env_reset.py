"""
Unit tests for ONDCAgentEnv.reset() and _build_obs().
Requirements: 1.2, 2.1–2.4, 4.1–4.7
"""
from __future__ import annotations

import numpy as np
import pytest

from ondc_env import (
    BecknPhase,
    EnvConfig,
    N_OBS_DIM,
    OrderStatus,
    SellerConfig,
)
from ondc_env.env import ONDCAgentEnv, _IDX_HAS_SELECTION


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_env(**kwargs) -> ONDCAgentEnv:
    return ONDCAgentEnv(EnvConfig(**kwargs))


# ---------------------------------------------------------------------------
# reset() — shape and dtype
# ---------------------------------------------------------------------------

class TestResetShapeAndDtype:
    def test_obs_shape(self):
        env = make_env()
        obs, _ = env.reset()
        assert obs.shape == (N_OBS_DIM,)

    def test_obs_dtype(self):
        env = make_env()
        obs, _ = env.reset()
        assert obs.dtype == np.float32

    def test_obs_shape_with_seed(self):
        env = make_env(seed=42)
        obs, _ = env.reset(seed=42)
        assert obs.shape == (N_OBS_DIM,)

    def test_obs_shape_multiple_resets(self):
        env = make_env()
        for _ in range(3):
            obs, _ = env.reset()
            assert obs.shape == (N_OBS_DIM,)


# ---------------------------------------------------------------------------
# reset() — initial state
# ---------------------------------------------------------------------------

class TestResetInitialState:
    def test_phase_is_search(self):
        env = make_env()
        env.reset()
        assert env.state.current_phase == BecknPhase.SEARCH

    def test_steps_remaining_equals_max_steps(self):
        env = make_env(max_steps=30)
        env.reset()
        assert env.state.steps_remaining == 30

    def test_budget_from_options(self):
        env = make_env(initial_budget=1000.0)
        env.reset(options={"budget": 500.0})
        assert env.state.budget == 500.0

    def test_budget_from_config_when_no_option(self):
        env = make_env(initial_budget=750.0)
        env.reset()
        assert env.state.budget == 750.0

    def test_sellers_initialised(self):
        env = make_env(n_sellers=3)
        env.reset()
        assert len(env.state.sellers) == 3

    def test_no_selected_offer_after_reset(self):
        env = make_env()
        env.reset()
        assert env.state.selected_offer is None

    def test_invalid_action_count_zero(self):
        env = make_env()
        env.reset()
        assert env.state.invalid_action_count == 0

    def test_total_spent_zero(self):
        env = make_env()
        env.reset()
        assert env.state.total_spent == 0.0

    def test_episode_started_flag(self):
        env = make_env()
        env.reset()
        assert env._episode_started is True


# ---------------------------------------------------------------------------
# _build_obs() — index-level encoding
# ---------------------------------------------------------------------------

class TestBuildObs:
    def _make_state_and_obs(self, **env_kwargs):
        env = make_env(**env_kwargs)
        obs, _ = env.reset()
        return env, obs

    def test_budget_normalized_at_index_0(self):
        env = make_env(initial_budget=1000.0)
        env.reset(options={"budget": 500.0})
        obs = env._build_obs(env.state)
        # budget / initial_budget = 500 / 500 = 1.0 (initial_budget set from options)
        assert obs[0] == pytest.approx(1.0, abs=1e-5)

    def test_urgency_at_index_1(self):
        env = make_env()
        env.reset(options={"urgency": 0.7})
        obs = env._build_obs(env.state)
        assert obs[1] == pytest.approx(0.7, abs=1e-5)

    def test_steps_remaining_norm_at_index_2(self):
        env = make_env(max_steps=50)
        env.reset()
        obs = env._build_obs(env.state)
        # steps_remaining / max_steps = 50 / 50 = 1.0
        assert obs[2] == pytest.approx(1.0, abs=1e-5)

    def test_phase_onehot_search(self):
        env = make_env()
        env.reset()
        obs = env._build_obs(env.state)
        # SEARCH = 0 → index 3 should be 1, rest 0
        phase_slice = obs[3:9]
        assert phase_slice[0] == pytest.approx(1.0)
        assert phase_slice[1:].sum() == pytest.approx(0.0)

    def test_seller_features_non_zero_for_available_sellers(self):
        env = make_env(n_sellers=5)
        env.reset()
        obs = env._build_obs(env.state)
        # At least one seller slot should have non-zero features
        seller_block = obs[9:49]
        assert seller_block.sum() > 0.0

    def test_has_selection_zero_when_no_offer(self):
        env = make_env()
        env.reset()
        obs = env._build_obs(env.state)
        assert obs[_IDX_HAS_SELECTION] == pytest.approx(0.0)

    def test_selected_offer_indices_zero_when_no_offer(self):
        env = make_env()
        env.reset()
        obs = env._build_obs(env.state)
        assert obs[49] == pytest.approx(0.0)
        assert obs[50] == pytest.approx(0.0)
        assert obs[51] == pytest.approx(0.0)
        assert obs[52] == pytest.approx(0.0)

    def test_order_status_onehot_all_zero_when_none(self):
        env = make_env()
        env.reset()
        obs = env._build_obs(env.state)
        assert obs[53:58].sum() == pytest.approx(0.0)

    def test_total_spent_norm_zero_at_start(self):
        env = make_env()
        env.reset()
        obs = env._build_obs(env.state)
        assert obs[58] == pytest.approx(0.0)

    def test_invalid_action_ratio_zero_at_start(self):
        env = make_env()
        env.reset()
        obs = env._build_obs(env.state)
        assert obs[59] == pytest.approx(0.0)

    def test_obs_length_is_n_obs_dim(self):
        env = make_env()
        env.reset()
        obs = env._build_obs(env.state)
        assert len(obs) == N_OBS_DIM

    def test_obs_all_finite(self):
        env = make_env()
        env.reset()
        obs = env._build_obs(env.state)
        assert np.all(np.isfinite(obs))

    def test_has_selection_one_when_offer_set(self):
        """Manually set a selected_offer and verify has_selection flips to 1."""
        from ondc_env import SellerOffer
        env = make_env()
        env.reset()
        env.state.selected_offer = SellerOffer(
            seller_id="seller_0",
            item_id="item_0",
            name="test",
            price=200.0,
            rating=4.5,
            delivery_eta=3,
            stock=10,
            is_available=True,
            fulfillment_type="standard",
        )
        obs = env._build_obs(env.state)
        assert obs[_IDX_HAS_SELECTION] == pytest.approx(1.0)

    def test_order_status_onehot_confirmed(self):
        env = make_env()
        env.reset()
        env.state.order_status = OrderStatus.CONFIRMED
        obs = env._build_obs(env.state)
        # CONFIRMED = 1 → index 53+1 = 54
        assert obs[54] == pytest.approx(1.0)
        assert obs[53] == pytest.approx(0.0)
