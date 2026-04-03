# Requirements Document

## Introduction

ONDCAgentEnv is a reinforcement learning environment that simulates the ONDC (Open Network for Digital Commerce) buyer-seller lifecycle. An RL agent acts as a buyer navigating the full Beckn protocol flow — SEARCH → SELECT → INIT → CONFIRM → TRACK → CANCEL/RETURN — optimizing decisions across multi-seller markets with dynamic pricing, stock changes, and real-world constraints like budget, delivery urgency, and seller ratings. The system exposes a Gymnasium-compatible interface, a FastAPI HTTP layer for demo and inference, and a standalone inference script.

## Glossary

- **ONDCAgentEnv**: The Gymnasium-compatible RL environment wrapping the full ONDC lifecycle simulation.
- **TaskEngine**: The component that enforces Beckn protocol phase transitions and validates agent actions.
- **SellerSimulator**: The component that simulates N sellers with dynamic pricing, stock, ratings, and random events.
- **RewardSystem**: The component that computes scalar rewards from weighted sub-reward components.
- **API**: The FastAPI HTTP layer exposing session and training endpoints.
- **InferenceScript**: The standalone script that loads a trained model and runs a full demo episode.
- **EpisodeState**: The full internal state of a running episode, including phase, sellers, budget, and order status.
- **BecknPhase**: An enum representing the current protocol phase: SEARCH, SELECT, INIT, CONFIRM, TRACK, POST_ORDER.
- **ActionType**: An integer enum of all discrete actions available to the agent (N_ACTIONS = 15).
- **ObsDict**: The flattened float32 numpy array of shape (N_OBS_DIM,) returned as the observation.
- **RewardResult**: A dataclass containing the scalar total reward and a per-component breakdown dict.
- **SellerState**: Per-seller data including price, stock, rating, delivery ETA, and availability.
- **SellerOffer**: A seller's response to a search or selection query.
- **EnvConfig**: Configuration dataclass for the environment (max_steps, n_sellers, budget, etc.).
- **SellerConfig**: Configuration dataclass for the seller simulator (price range, volatility, etc.).
- **RewardWeights**: Configuration dataclass for reward component weights.
- **Session**: A server-side object managed by the API that holds an ONDCAgentEnv instance and its state.
- **N_OBS_DIM**: The fixed observation vector dimension (61).
- **N_ACTIONS**: The total number of discrete actions (15).
- **MAX_SELLERS**: The maximum number of seller slots in the observation (5).

---

## Requirements

### Requirement 1: Gymnasium-Compatible Environment Interface

**User Story:** As an RL researcher, I want a Gymnasium-compatible environment, so that I can use standard RL libraries like Stable-Baselines3 to train agents without custom wrappers.

#### Acceptance Criteria

1. THE ONDCAgentEnv SHALL implement the `gymnasium.Env` interface including `reset()`, `step()`, `render()`, and `close()` methods.
2. WHEN `reset()` is called, THE ONDCAgentEnv SHALL return a tuple of `(obs, info)` where `obs` has shape `(N_OBS_DIM,)` and dtype `float32`.
3. WHEN `step(action)` is called, THE ONDCAgentEnv SHALL return a tuple of `(obs, reward, terminated, truncated, info)` where `obs` has shape `(N_OBS_DIM,)` and dtype `float32`.
4. THE ONDCAgentEnv SHALL expose a `action_space` of type `gymnasium.spaces.Discrete(N_ACTIONS)`.
5. THE ONDCAgentEnv SHALL expose an `observation_space` consistent with the `(N_OBS_DIM,)` float32 observation vector.
6. WHEN `gymnasium.utils.env_checker.check_env(env)` is run, THE ONDCAgentEnv SHALL pass all checks without errors.

---

### Requirement 2: Episode Initialization

**User Story:** As an RL researcher, I want the environment to reset cleanly to a well-defined initial state, so that training episodes are reproducible and independent.

#### Acceptance Criteria

1. WHEN `reset()` is called, THE ONDCAgentEnv SHALL set `current_phase` to `BecknPhase.SEARCH`.
2. WHEN `reset()` is called, THE ONDCAgentEnv SHALL set `steps_remaining` to `config.max_steps`.
3. WHEN `reset()` is called with an `options` dict containing `budget`, THE ONDCAgentEnv SHALL set the episode budget to the provided value.
4. WHEN `reset()` is called without a `budget` option, THE ONDCAgentEnv SHALL set the episode budget to `config.initial_budget`.
5. WHEN `reset()` is called with a `seed`, THE ONDCAgentEnv SHALL initialize all random state deterministically from that seed.
6. WHEN `reset()` is called, THE ONDCAgentEnv SHALL initialize the SellerSimulator with a fresh set of seller states.

---

### Requirement 3: Beckn Protocol Phase Enforcement

**User Story:** As an RL researcher, I want the environment to enforce the Beckn protocol phase ordering, so that the agent learns realistic ONDC buyer behavior.

#### Acceptance Criteria

1. THE TaskEngine SHALL define a valid action set for each BecknPhase.
2. WHEN an agent submits an action that is not valid for the current BecknPhase, THE TaskEngine SHALL return a `ValidationResult` with `is_valid = False` and a non-empty `reason` string.
3. WHEN an agent submits a valid action, THE TaskEngine SHALL return a `ValidationResult` with `is_valid = True`.
4. WHEN a valid action is applied, THE TaskEngine SHALL transition `current_phase` to the next appropriate BecknPhase.
5. THE TaskEngine SHALL not mutate `EpisodeState` during `validate_action()`.
6. WHEN a CANCEL action is taken from CONFIRM or TRACK phase, THE TaskEngine SHALL allow the phase transition without treating it as an invalid action.
7. WHEN `is_terminal()` is called, THE TaskEngine SHALL return `True` if the order is confirmed, cancelled, returned, or the budget is exhausted.

---

### Requirement 4: Observation Construction

**User Story:** As an RL researcher, I want a consistent, well-defined observation vector, so that the agent receives a stable input representation across all steps.

#### Acceptance Criteria

1. THE ONDCAgentEnv SHALL build observations as a flat `np.ndarray` of shape `(N_OBS_DIM,)` and dtype `float32`.
2. THE ONDCAgentEnv SHALL encode task context (budget normalized, urgency, steps remaining normalized, phase one-hot) at indices 0–8 of the observation vector.
3. THE ONDCAgentEnv SHALL encode per-seller features for up to MAX_SELLERS sellers at indices 9–48, padding unavailable seller slots with zeros.
4. THE ONDCAgentEnv SHALL encode selected offer features at indices 49–52 of the observation vector.
5. THE ONDCAgentEnv SHALL encode order status as a one-hot vector at indices 53–57 of the observation vector.
6. THE ONDCAgentEnv SHALL encode episode metrics (total spent normalized, invalid action ratio, delivery ETA normalized) at indices 58–60 of the observation vector.
7. WHEN no seller is selected, THE ONDCAgentEnv SHALL set `has_selection` (index 52) to 0.

---

### Requirement 5: Seller Simulation

**User Story:** As an RL researcher, I want a realistic seller simulator with dynamic pricing and random events, so that the agent learns to handle real-world market variability.

#### Acceptance Criteria

1. WHEN `SellerSimulator.reset()` is called, THE SellerSimulator SHALL initialize `n_sellers` sellers with prices, ratings, ETAs, and stock values sampled within the configured ranges.
2. WHEN `get_catalog()` is called, THE SellerSimulator SHALL return only sellers where `is_available == True`.
3. WHEN `get_catalog()` is called, THE SellerSimulator SHALL return offers where each offer has `price > 0`, `rating` in `[0, 5]`, and `delivery_eta > 0`.
4. WHEN `tick()` is called, THE SellerSimulator SHALL apply Gaussian price drift to each seller's price.
5. WHEN `tick()` is called, THE SellerSimulator SHALL clamp all seller prices to remain within `config.price_range`.
6. WHEN `tick()` is called and a random event fires for a seller, THE SellerSimulator SHALL apply one of: STOCKOUT, PRICE_SPIKE, DELAY, or SELLER_DOWN effects.
7. WHEN `tick()` is called, THE SellerSimulator SHALL return a list of `SellerEvent` objects for all events that occurred.
8. THE SellerSimulator SHALL not mutate seller states during `get_catalog()`.

---

### Requirement 6: Reward Computation

**User Story:** As an RL researcher, I want a modular, configurable reward function with per-component breakdown, so that I can tune agent behavior and debug training.

#### Acceptance Criteria

1. WHEN `RewardSystem.compute()` is called, THE RewardSystem SHALL return a `RewardResult` where `result.total` equals the sum of all values in `result.breakdown`.
2. WHEN the agent confirms an order, THE RewardSystem SHALL include a `task_completion` reward component.
3. WHEN the confirmed order price is at or below 90% of the remaining budget, THE RewardSystem SHALL include a `good_price` reward component.
4. WHEN the confirmed order delivery ETA is within the urgency threshold, THE RewardSystem SHALL include a `fast_delivery` reward component.
5. WHEN the selected seller rating is at or above 4.0, THE RewardSystem SHALL include a `high_seller_rating` reward component.
6. WHEN an invalid action is taken, THE RewardSystem SHALL include an `invalid_action` penalty component.
7. WHEN `total_spent` exceeds `budget`, THE RewardSystem SHALL include a `budget_exceeded` penalty component.
8. WHEN the order status becomes FAILED, THE RewardSystem SHALL include an `order_failed` penalty component.
9. WHEN the agent takes a WAIT action, THE RewardSystem SHALL include an `unnecessary_wait` penalty component.
10. WHEN delivery is overdue relative to the urgency deadline, THE RewardSystem SHALL include a `late_delivery` penalty component.
11. THE RewardSystem SHALL not mutate `prev_state` or `next_state` during `compute()`.
12. THE RewardSystem SHALL return a `result.total` that is a finite float (no NaN or Inf).
13. WHERE reward weights are configured, THE RewardSystem SHALL use the provided `RewardWeights` values for all reward components.

---

### Requirement 7: Environment Step Execution

**User Story:** As an RL researcher, I want the environment step to correctly orchestrate all sub-components, so that each step produces a valid transition.

#### Acceptance Criteria

1. WHEN `step(action)` is called, THE ONDCAgentEnv SHALL validate the action via the TaskEngine before applying any effects.
2. WHEN an invalid action is submitted, THE ONDCAgentEnv SHALL increment `invalid_action_count`, apply the invalid action penalty, and return the current observation unchanged.
3. WHEN a valid action is submitted, THE ONDCAgentEnv SHALL apply action effects via the SellerSimulator, advance the BecknPhase, tick the SellerSimulator, decrement `steps_remaining` by 1, and compute the reward.
4. WHEN `steps_remaining` reaches 0, THE ONDCAgentEnv SHALL set `truncated = True` in the step return.
5. WHEN the TaskEngine reports a terminal state, THE ONDCAgentEnv SHALL set `terminated = True` in the step return.
6. THE ONDCAgentEnv SHALL include `reward_breakdown`, `phase`, and `events` in the `info` dict returned by `step()`.
7. THE ONDCAgentEnv SHALL ensure `steps_remaining` is non-increasing across all steps in an episode.
8. THE ONDCAgentEnv SHALL ensure `invalid_action_count` is non-decreasing across all steps in an episode.

---

### Requirement 8: FastAPI Session Management

**User Story:** As a demo operator, I want an HTTP API to manage RL environment sessions, so that I can drive inference from a web client or demo UI without embedding Python.

#### Acceptance Criteria

1. WHEN `POST /session/start` is called with a valid task config, THE API SHALL create a new session, call `env.reset()`, and return `{session_id, obs, info}` with HTTP 200.
2. WHEN `POST /session/{id}/step` is called with a valid action, THE API SHALL call `env.step(action)` and return `{obs, reward, done, info}` with HTTP 200.
3. WHEN `GET /session/{id}/state` is called for an existing session, THE API SHALL return the full current `EpisodeState` with HTTP 200.
4. WHEN `DELETE /session/{id}` is called for an existing session, THE API SHALL remove the session and return `{ok}` with HTTP 200.
5. WHEN `POST /session/{id}/step` is called with an unknown `session_id`, THE API SHALL return HTTP 404 with `{"detail": "Session not found"}`.
6. WHEN `GET /health` is called, THE API SHALL return HTTP 200 with `{ok}`.
7. THE API SHALL validate all request bodies using Pydantic models and return HTTP 422 for malformed inputs.
8. THE API SHALL assign each session a UUID to prevent cross-session state access.

---

### Requirement 9: Training Endpoint

**User Story:** As an RL researcher, I want to trigger and monitor training runs via the API, so that I can orchestrate training without direct Python access.

#### Acceptance Criteria

1. WHEN `POST /train` is called with a valid training config, THE API SHALL start a training run and return `{run_id}` with HTTP 200.
2. WHEN `GET /train/{run_id}/status` is called for an existing run, THE API SHALL return `{status, metrics}` with HTTP 200.

---

### Requirement 10: Inference Script

**User Story:** As a demo operator, I want a standalone inference script, so that I can run a full demo episode from the command line with a trained model.

#### Acceptance Criteria

1. WHEN `run_demo()` is called with a valid `model_path` and `task_config`, THE InferenceScript SHALL load the model, run a full episode, and return an `EpisodeResult`.
2. WHEN `render=True` is passed to `run_demo()`, THE InferenceScript SHALL call `env.render()` after each step to display the current state.
3. WHEN the episode exceeds `max_steps`, THE InferenceScript SHALL terminate the episode and return the result.

---

### Requirement 11: Environment Rendering

**User Story:** As a developer, I want a human-readable render output, so that I can visually inspect the agent's behavior during debugging and demos.

#### Acceptance Criteria

1. WHEN `render()` is called with `render_mode="human"`, THE ONDCAgentEnv SHALL output a formatted string showing the current phase, seller states, selected offer, and reward breakdown.
2. WHEN `render()` is called with `render_mode="json"`, THE ONDCAgentEnv SHALL return a JSON-serializable string of the current `EpisodeState`.

---

### Requirement 12: Error Handling and Resilience

**User Story:** As a developer, I want the system to handle error conditions gracefully, so that training runs and demos are not interrupted by recoverable failures.

#### Acceptance Criteria

1. WHEN a seller becomes unavailable due to a STOCKOUT or SELLER_DOWN event after selection, THE TaskEngine SHALL allow the agent to backtrack to the SELECT phase.
2. WHEN an order is confirmed with `price > remaining_budget`, THE RewardSystem SHALL apply the `budget_exceeded` penalty and THE ONDCAgentEnv SHALL continue the episode without crashing.
3. WHEN an unexpected exception occurs inside `env.step()`, THE API SHALL return HTTP 500 with `{"detail": "Environment error", "error": "<message>"}` and mark the session as failed.
4. IF a request body fails Pydantic validation, THEN THE API SHALL return HTTP 422 with a descriptive error message.

---

### Requirement 13: Configuration and Reproducibility

**User Story:** As an RL researcher, I want all environment parameters to be configurable and seeded, so that experiments are reproducible and comparable.

#### Acceptance Criteria

1. THE ONDCAgentEnv SHALL accept an `EnvConfig` at construction time that controls `max_steps`, `n_sellers`, `initial_budget`, `urgency_range`, `random_event_prob`, and `reward_weights`.
2. THE SellerSimulator SHALL accept a `SellerConfig` that controls `price_range`, `rating_range`, `eta_range`, `stock_range`, and `price_volatility`.
3. WHEN a `seed` is provided to `EnvConfig`, THE ONDCAgentEnv SHALL produce identical episode trajectories for identical action sequences.
4. THE RewardSystem SHALL accept a `RewardWeights` instance and use its values for all reward computations without requiring code changes.
