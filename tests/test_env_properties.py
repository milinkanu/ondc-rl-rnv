"""
Property-based tests for ONDCAgentEnv.

Property 1:  Observation shape is always consistent (Requirements 1.2, 1.3, 4.1)
Property 10: reset() always returns to SEARCH phase with correct initial state
             (Requirements 2.1, 2.2, 2.3, 2.4)
"""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from ondc_env import BecknPhase, EnvConfig, N_OBS_DIM
from ondc_env.env import ONDCAgentEnv


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

valid_seeds = st.integers(min_value=0, max_value=2**31 - 1)
valid_budgets = st.floats(min_value=100.0, max_value=10_000.0, allow_nan=False, allow_infinity=False)
valid_urgency = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
valid_max_steps = st.integers(min_value=1, max_value=200)
valid_n_sellers = st.integers(min_value=1, max_value=5)


# ---------------------------------------------------------------------------
# Property 1: Observation shape is always consistent
# ---------------------------------------------------------------------------

@given(seed=valid_seeds)
@settings(max_examples=200)
def test_property1_obs_shape_after_reset(seed):
    """
    For any seed, reset() must return obs with shape (N_OBS_DIM,) and dtype float32.
    """
    env = ONDCAgentEnv(EnvConfig(seed=seed))
    obs, _ = env.reset(seed=seed)
    assert obs.shape == (N_OBS_DIM,), f"Expected shape ({N_OBS_DIM},), got {obs.shape}"
    assert obs.dtype == np.float32, f"Expected float32, got {obs.dtype}"


@given(
    seed=valid_seeds,
    max_steps=valid_max_steps,
    n_sellers=valid_n_sellers,
)
@settings(max_examples=200)
def test_property1_obs_shape_varied_config(seed, max_steps, n_sellers):
    """
    For any valid EnvConfig, reset() must return obs with shape (N_OBS_DIM,).
    """
    config = EnvConfig(max_steps=max_steps, n_sellers=n_sellers, seed=seed)
    env = ONDCAgentEnv(config)
    obs, _ = env.reset(seed=seed)
    assert obs.shape == (N_OBS_DIM,)
    assert obs.dtype == np.float32


@given(seed=valid_seeds)
@settings(max_examples=200)
def test_property1_obs_all_finite(seed):
    """
    All observation values must be finite (no NaN or Inf).
    """
    env = ONDCAgentEnv(EnvConfig(seed=seed))
    obs, _ = env.reset(seed=seed)
    assert np.all(np.isfinite(obs)), "Observation contains non-finite values"


# ---------------------------------------------------------------------------
# Property 10: reset() always returns to SEARCH phase with correct initial state
# ---------------------------------------------------------------------------

@given(seed=valid_seeds)
@settings(max_examples=200)
def test_property10_phase_is_search_after_reset(seed):
    """
    For any seed, reset() must set current_phase to BecknPhase.SEARCH.
    """
    env = ONDCAgentEnv(EnvConfig(seed=seed))
    env.reset(seed=seed)
    assert env.state.current_phase == BecknPhase.SEARCH


@given(seed=valid_seeds, max_steps=valid_max_steps)
@settings(max_examples=200)
def test_property10_steps_remaining_equals_max_steps(seed, max_steps):
    """
    After reset(), steps_remaining must equal config.max_steps.
    """
    config = EnvConfig(max_steps=max_steps, seed=seed)
    env = ONDCAgentEnv(config)
    env.reset(seed=seed)
    assert env.state.steps_remaining == max_steps


@given(seed=valid_seeds, budget=valid_budgets)
@settings(max_examples=200)
def test_property10_budget_from_options(seed, budget):
    """
    When options contains 'budget', reset() must use that value.
    """
    env = ONDCAgentEnv(EnvConfig(seed=seed))
    env.reset(seed=seed, options={"budget": budget})
    assert env.state.budget == pytest.approx(budget, abs=1e-5)


@given(seed=valid_seeds, budget=valid_budgets)
@settings(max_examples=200)
def test_property10_budget_from_config_when_no_option(seed, budget):
    """
    When no budget option is provided, reset() must use config.initial_budget.
    """
    config = EnvConfig(initial_budget=budget, seed=seed)
    env = ONDCAgentEnv(config)
    env.reset(seed=seed)
    assert env.state.budget == pytest.approx(budget, abs=1e-5)


@given(seed=valid_seeds)
@settings(max_examples=200)
def test_property10_reset_clears_selection_and_metrics(seed):
    """
    After reset(), selected_offer must be None and counters must be zero.
    """
    env = ONDCAgentEnv(EnvConfig(seed=seed))
    env.reset(seed=seed)
    assert env.state.selected_offer is None
    assert env.state.invalid_action_count == 0
    assert env.state.total_spent == 0.0
    assert env.state.total_reward == 0.0


@given(seed=valid_seeds)
@settings(max_examples=100)
def test_property10_multiple_resets_consistent(seed):
    """
    Calling reset() twice with the same seed must produce identical observations.
    """
    env = ONDCAgentEnv(EnvConfig())
    obs1, _ = env.reset(seed=seed)
    obs2, _ = env.reset(seed=seed)
    np.testing.assert_array_equal(obs1, obs2)


# ---------------------------------------------------------------------------
# Property 17: JSON render is a round-trip
# ---------------------------------------------------------------------------

import json
from dataclasses import fields as dataclass_fields
from ondc_env.types import EpisodeState

# All EpisodeState field names that must appear in the JSON output
_EPISODE_STATE_FIELDS = [f.name for f in dataclass_fields(EpisodeState)]


@given(seed=valid_seeds)
@settings(max_examples=200)
def test_property17_json_render_round_trip_after_reset(seed):
    """
    **Validates: Requirements 11.2**

    Property 17: JSON render is a round-trip.
    For any seed, after reset(), render(render_mode="json") must return valid JSON
    containing all EpisodeState fields.
    """
    env = ONDCAgentEnv(EnvConfig(seed=seed))
    env.reset(seed=seed)

    result = env.render(render_mode="json")

    # Must return a non-None string
    assert result is not None
    assert isinstance(result, str)

    # Must be valid JSON
    data = json.loads(result)

    # Must contain all EpisodeState fields
    for field_name in _EPISODE_STATE_FIELDS:
        assert field_name in data, f"Missing EpisodeState field in JSON: {field_name}"
