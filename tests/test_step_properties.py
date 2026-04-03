"""
Property-based tests for ONDCAgentEnv.step().

Property 3:  Phase transitions are monotonically forward (Requirements 3.4, 3.6)
Property 4:  Total spent is non-negative (Requirements 7.3)
Property 5:  Invalid action count is non-decreasing (Requirements 7.8)
Property 6:  Steps remaining is non-increasing (Requirements 7.7)
Property 8:  Episode terminates within max_steps (Requirements 7.4)
Property 11: Seeded episodes are reproducible (Requirements 2.5, 13.3)
Property 12: Invalid action step leaves observation unchanged and increments count
             (Requirements 7.2, 6.6)
"""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from ondc_env import ActionType, BecknPhase, EnvConfig, N_OBS_DIM
from ondc_env.env import ONDCAgentEnv

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

valid_seeds = st.integers(min_value=0, max_value=2**31 - 1)
valid_actions = st.integers(min_value=0, max_value=N_OBS_DIM - 1)  # wider than N_ACTIONS to test OOB too
valid_n_actions = st.integers(min_value=0, max_value=14)
valid_max_steps = st.integers(min_value=2, max_value=30)
action_sequences = st.lists(
    st.integers(min_value=0, max_value=14),
    min_size=1,
    max_size=20,
)

_CANCEL_ACTIONS = {
    int(ActionType.CANCEL_BEFORE_CONFIRM),
    int(ActionType.CANCEL_ORDER),
}


# ---------------------------------------------------------------------------
# Property 6: Steps remaining is non-increasing
# ---------------------------------------------------------------------------

@given(seed=valid_seeds, actions=action_sequences)
@settings(max_examples=200)
def test_property6_steps_remaining_non_increasing(seed, actions):
    """
    For any action sequence, steps_remaining must be non-increasing across steps.
    """
    env = ONDCAgentEnv(EnvConfig(max_steps=50, seed=seed))
    env.reset(seed=seed)

    prev_steps = env.state.steps_remaining
    for action in actions:
        _, _, terminated, truncated, _ = env.step(action)
        assert env.state.steps_remaining <= prev_steps, (
            f"steps_remaining increased: {prev_steps} → {env.state.steps_remaining}"
        )
        prev_steps = env.state.steps_remaining
        if terminated or truncated:
            break


# ---------------------------------------------------------------------------
# Property 5: Invalid action count is non-decreasing
# ---------------------------------------------------------------------------

@given(seed=valid_seeds, actions=action_sequences)
@settings(max_examples=200)
def test_property5_invalid_action_count_non_decreasing(seed, actions):
    """
    For any action sequence, invalid_action_count must be non-decreasing.
    """
    env = ONDCAgentEnv(EnvConfig(max_steps=50, seed=seed))
    env.reset(seed=seed)

    prev_count = env.state.invalid_action_count
    for action in actions:
        _, _, terminated, truncated, _ = env.step(action)
        assert env.state.invalid_action_count >= prev_count, (
            f"invalid_action_count decreased: {prev_count} → {env.state.invalid_action_count}"
        )
        prev_count = env.state.invalid_action_count
        if terminated or truncated:
            break


# ---------------------------------------------------------------------------
# Property 8: Episode terminates within max_steps
# ---------------------------------------------------------------------------

@given(seed=valid_seeds, max_steps=valid_max_steps)
@settings(max_examples=200)
def test_property8_episode_terminates_within_max_steps(seed, max_steps):
    """
    For any config and action sequence, total steps taken must not exceed max_steps.
    """
    env = ONDCAgentEnv(EnvConfig(max_steps=max_steps, seed=seed))
    env.reset(seed=seed)

    steps_taken = 0
    done = False
    while not done:
        # Use WAIT (always valid in any phase) to drive the episode to termination
        _, _, terminated, truncated, _ = env.step(ActionType.WAIT)
        steps_taken += 1
        done = terminated or truncated

    assert steps_taken <= max_steps, (
        f"Episode took {steps_taken} steps but max_steps={max_steps}"
    )


# ---------------------------------------------------------------------------
# Property 12: Invalid action step leaves observation unchanged and increments count
# ---------------------------------------------------------------------------

@given(seed=valid_seeds)
@settings(max_examples=200)
def test_property12_invalid_action_obs_unchanged_count_incremented(seed):
    """
    For any state, an invalid action must return the same obs and increment count by 1.
    CONFIRM_ORDER is always invalid in SEARCH phase (the initial phase).
    """
    env = ONDCAgentEnv(EnvConfig(seed=seed))
    obs_before, _ = env.reset(seed=seed)

    count_before = env.state.invalid_action_count
    # CONFIRM_ORDER is invalid in SEARCH phase
    obs_after, _, _, _, info = env.step(ActionType.CONFIRM_ORDER)

    np.testing.assert_array_equal(obs_before, obs_after)
    assert env.state.invalid_action_count == count_before + 1
    assert "invalid_action" in info["reward_breakdown"]


# ---------------------------------------------------------------------------
# Property 4: Total spent is non-negative
# ---------------------------------------------------------------------------

@given(seed=valid_seeds, actions=action_sequences)
@settings(max_examples=200)
def test_property4_total_spent_non_negative(seed, actions):
    """
    For any episode trajectory, total_spent must always be >= 0.
    """
    env = ONDCAgentEnv(EnvConfig(max_steps=50, seed=seed))
    env.reset(seed=seed)

    for action in actions:
        env.step(action)
        assert env.state.total_spent >= 0.0, (
            f"total_spent went negative: {env.state.total_spent}"
        )
        if env.task_engine.is_terminal(env.state) or env.state.steps_remaining <= 0:
            break


# ---------------------------------------------------------------------------
# Property 11: Seeded episodes are reproducible
# ---------------------------------------------------------------------------

@given(seed=valid_seeds, actions=action_sequences)
@settings(max_examples=100)
def test_property11_seeded_reproducibility(seed, actions):
    """
    Two envs with the same seed and same actions must produce identical obs and rewards.
    """
    config = EnvConfig(max_steps=50, seed=seed)

    env1 = ONDCAgentEnv(config)
    env2 = ONDCAgentEnv(config)

    obs1, _ = env1.reset(seed=seed)
    obs2, _ = env2.reset(seed=seed)
    np.testing.assert_array_equal(obs1, obs2)

    for action in actions:
        r1 = env1.step(action)
        r2 = env2.step(action)

        np.testing.assert_array_equal(r1[0], r2[0], err_msg="obs mismatch")
        assert r1[1] == pytest.approx(r2[1], abs=1e-6), "reward mismatch"
        assert r1[2] == r2[2], "terminated mismatch"
        assert r1[3] == r2[3], "truncated mismatch"

        if r1[2] or r1[3]:
            break


# ---------------------------------------------------------------------------
# Property 3: Phase transitions are monotonically forward
# ---------------------------------------------------------------------------

@given(seed=valid_seeds, actions=action_sequences)
@settings(max_examples=200)
def test_property3_phase_transitions_monotonically_forward(seed, actions):
    """
    Phase value must be non-decreasing, except when a CANCEL action is taken
    (which is explicitly allowed to backtrack to SEARCH).
    """
    env = ONDCAgentEnv(EnvConfig(max_steps=50, seed=seed))
    env.reset(seed=seed)

    prev_phase = int(env.state.current_phase)
    for action in actions:
        _, _, terminated, truncated, _ = env.step(action)
        curr_phase = int(env.state.current_phase)

        if action not in _CANCEL_ACTIONS:
            assert curr_phase >= prev_phase, (
                f"Phase went backward without cancel: {prev_phase} → {curr_phase} "
                f"(action={action})"
            )

        prev_phase = curr_phase
        if terminated or truncated:
            break
