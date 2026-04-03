# Implementation Plan: ONDCAgentEnv

## Overview

Implement the ONDC reinforcement learning environment in Python, following the Beckn protocol lifecycle. The plan builds incrementally: data models → core components → full step loop → API layer → inference script.

## Tasks

- [x] 1. Scaffold project structure and core data models
  - Create directory layout: `ondc_env/`, `api/`, `scripts/`, `tests/`
  - Create `ondc_env/__init__.py`, `ondc_env/types.py`
  - Implement `BecknPhase`, `ActionType`, `OrderStatus` enums
  - Implement `SellerState`, `SellerOffer`, `SellerEvent`, `SearchQuery`, `SelectResult` dataclasses
  - Implement `EpisodeState`, `EnvConfig`, `SellerConfig`, `RewardWeights`, `RewardResult`, `ValidationResult` dataclasses
  - Define `N_OBS_DIM = 61`, `N_ACTIONS = 15`, `MAX_SELLERS = 5` constants
  - _Requirements: 1.4, 1.5, 4.1, 13.1, 13.2, 13.4_

- [x] 2. Implement TaskEngine
  - [x] 2.1 Implement `TaskEngine` class in `ondc_env/task_engine.py`
    - Define `VALID_ACTIONS` mapping from each `BecknPhase` to its allowed `ActionType` set
    - Implement `validate_action(action, state) -> ValidationResult` — returns `is_valid=False` with reason for protocol violations, no state mutation
    - Implement `transition(current_phase, action) -> BecknPhase` — returns next phase for valid actions
    - Implement `is_terminal(state) -> bool` — True when order confirmed, cancelled, returned, or budget exhausted
    - Handle CANCEL backtrack from CONFIRM/TRACK phases
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_

  - [x] 2.2 Write unit tests for TaskEngine
    - Test invalid action in wrong phase returns `is_valid=False` with non-empty reason
    - Test valid action returns `is_valid=True`
    - Test phase transitions for each valid `(phase, action)` pair
    - Test `is_terminal()` for all terminal conditions
    - Test CANCEL from CONFIRM and TRACK phases is valid
    - Test `validate_action()` does not mutate state
    - _Requirements: 3.1–3.7_

  - [x] 2.3 Write property test for TaskEngine immutability
    - **Property 14: TaskEngine does not mutate state during validation**
    - **Validates: Requirements 3.5**

- [x] 3. Implement SellerSimulator
  - [x] 3.1 Implement `SellerSimulator` class in `ondc_env/seller_simulator.py`
    - Implement `reset(seed) -> list[SellerState]` — initialize `n_sellers` sellers with values sampled within configured ranges
    - Implement `get_catalog(query) -> list[SellerOffer]` — return only available sellers, no state mutation
    - Implement `apply_selection(seller_id, item_id, quantity) -> SelectResult`
    - Implement `tick() -> list[SellerEvent]` — Gaussian price drift, clamp to `price_range`, fire random events (STOCKOUT, PRICE_SPIKE, DELAY, SELLER_DOWN)
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8_

  - [x] 3.2 Write unit tests for SellerSimulator
    - Test `reset()` produces sellers with values within configured ranges
    - Test `get_catalog()` excludes unavailable sellers
    - Test `get_catalog()` does not mutate seller states
    - Test `tick()` returns `SellerEvent` list for fired events
    - Test STOCKOUT event sets `stock=0` and `is_available=False`
    - _Requirements: 5.1–5.8_

  - [x] 3.3 Write property test for seller price bounds
    - **Property 7: Seller prices stay within configured bounds after tick**
    - **Validates: Requirements 5.5**

  - [x] 3.4 Write property test for catalog immutability
    - **Property 15: SellerSimulator does not mutate state during get_catalog()**
    - **Validates: Requirements 5.8**

  - [x] 3.5 Write property test for catalog correctness
    - **Property 9: Catalog only returns available sellers**
    - **Validates: Requirements 5.2, 5.3**

- [x] 4. Implement RewardSystem
  - [x] 4.1 Implement `RewardSystem` class in `ondc_env/reward_system.py`
    - Implement `compute(action, prev_state, next_state) -> RewardResult`
    - Include all reward components: `task_completion`, `good_price`, `fast_delivery`, `high_seller_rating`, `successful_return`
    - Include all penalty components: `invalid_action`, `budget_exceeded`, `order_failed`, `unnecessary_wait`, `late_delivery`
    - Ensure `result.total == sum(result.breakdown.values())`
    - Ensure no mutation of `prev_state` or `next_state`
    - Use `RewardWeights` values for all components
    - _Requirements: 6.1–6.13_

  - [x] 4.2 Write unit tests for RewardSystem
    - Test `task_completion` fires on CONFIRM_ORDER with CONFIRMED status
    - Test `good_price` fires when price <= 90% of budget
    - Test `fast_delivery` fires when ETA within urgency threshold
    - Test `high_seller_rating` fires when rating >= 4.0
    - Test `invalid_action` penalty fires for invalid actions
    - Test `budget_exceeded` penalty fires when `total_spent > budget`
    - Test `order_failed` penalty fires when status is FAILED
    - Test `unnecessary_wait` penalty fires for WAIT action
    - Test `late_delivery` penalty fires when delivery overdue
    - Test `result.total` is always a finite float
    - _Requirements: 6.1–6.13_

  - [x] 4.3 Write property test for reward total consistency
    - **Property 2: Reward total equals sum of breakdown components**
    - **Validates: Requirements 6.1, 6.12**

  - [x] 4.4 Write property test for RewardSystem immutability
    - **Property 13: RewardSystem does not mutate input states**
    - **Validates: Requirements 6.11**

  - [x] 4.5 Write property test for reward component conditions
    - **Property 16: Reward components fire on correct conditions**
    - **Validates: Requirements 6.3, 6.7, 6.9**

- [x] 5. Checkpoint — Ensure all component tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement ONDCAgentEnv core (reset and observation)
  - [x] 6.1 Implement `ONDCAgentEnv` class skeleton in `ondc_env/env.py`
    - Subclass `gymnasium.Env`, set `metadata`, define `action_space` and `observation_space`
    - Implement `__init__(config: EnvConfig)` — instantiate TaskEngine, SellerSimulator, RewardSystem
    - Implement `reset(seed, options) -> (obs, info)` — initialize EpisodeState, call `seller_sim.reset()`, set phase to SEARCH, set budget from options or config
    - Implement `_build_obs(state) -> np.ndarray` — construct flat float32 array of shape `(N_OBS_DIM,)` following OBS_SCHEMA layout
    - Implement `close()`
    - _Requirements: 1.1, 1.2, 1.4, 1.5, 2.1–2.6, 4.1–4.7, 13.1, 13.3_

  - [x] 6.2 Write unit tests for reset and observation
    - Test `reset()` returns obs with shape `(N_OBS_DIM,)` and dtype `float32`
    - Test `reset()` sets phase to SEARCH and steps_remaining to max_steps
    - Test `reset()` with budget option overrides config budget
    - Test `reset()` without budget option uses `config.initial_budget`
    - Test `_build_obs()` encodes each schema section at correct indices
    - Test `has_selection` (index 52) is 0 when no seller selected
    - _Requirements: 1.2, 2.1–2.4, 4.1–4.7_

  - [x] 6.3 Write property test for observation shape consistency
    - **Property 1: Observation shape is always consistent**
    - **Validates: Requirements 1.2, 1.3, 4.1**

  - [x] 6.4 Write property test for reset initial state
    - **Property 10: reset() always returns to SEARCH phase with correct initial state**
    - **Validates: Requirements 2.1, 2.2, 2.3, 2.4**

- [x] 7. Implement ONDCAgentEnv.step() and full episode loop
  - [x] 7.1 Implement `step(action) -> (obs, reward, terminated, truncated, info)` in `ondc_env/env.py`
    - Validate action via TaskEngine; on invalid: increment `invalid_action_count`, compute penalty reward, return current obs unchanged
    - On valid: apply action effects via SellerSimulator, advance phase via TaskEngine, tick SellerSimulator, decrement `steps_remaining`
    - Compute reward via RewardSystem
    - Set `terminated` from `task_engine.is_terminal()`, `truncated` when `steps_remaining <= 0`
    - Include `reward_breakdown`, `phase`, `events` in info dict
    - _Requirements: 7.1–7.8_

  - [x] 7.2 Implement `render(render_mode) -> str | None` in `ondc_env/env.py`
    - `"human"` mode: output formatted string with phase, seller states, selected offer, reward breakdown
    - `"json"` mode: return JSON-serializable string of current EpisodeState
    - _Requirements: 11.1, 11.2_

  - [x] 7.3 Write unit tests for step()
    - Test invalid action increments `invalid_action_count` and returns unchanged obs
    - Test valid action advances phase and decrements `steps_remaining`
    - Test `truncated=True` when `steps_remaining` reaches 0
    - Test `terminated=True` on terminal conditions
    - Test info dict contains `reward_breakdown`, `phase`, `events`
    - Test `steps_remaining` is non-increasing across steps
    - Test `invalid_action_count` is non-decreasing across steps
    - _Requirements: 7.1–7.8_

  - [x] 7.4 Write property test for steps remaining monotonicity
    - **Property 6: Steps remaining is non-increasing**
    - **Validates: Requirements 7.7**

  - [x] 7.5 Write property test for invalid action count monotonicity
    - **Property 5: Invalid action count is non-decreasing**
    - **Validates: Requirements 7.8**

  - [x] 7.6 Write property test for episode termination bound
    - **Property 8: Episode terminates within max_steps**
    - **Validates: Requirements 7.4**

  - [x] 7.7 Write property test for invalid action step behavior
    - **Property 12: Invalid action step leaves observation unchanged and increments count**
    - **Validates: Requirements 7.2, 6.6**

  - [x] 7.8 Write property test for total spent non-negativity
    - **Property 4: Total spent is non-negative**
    - **Validates: Requirements 7.3**

  - [x] 7.9 Write property test for seeded reproducibility
    - **Property 11: Seeded episodes are reproducible**
    - **Validates: Requirements 2.5, 13.3**

  - [x] 7.10 Write property test for phase transition monotonicity
    - **Property 3: Phase transitions are monotonically forward**
    - **Validates: Requirements 3.4, 3.6**

- [x] 8. Verify Gymnasium compatibility
  - Run `gymnasium.utils.env_checker.check_env(env)` and fix any issues
  - Ensure `action_space` is `Discrete(N_ACTIONS)` and `observation_space` matches `(N_OBS_DIM,)` float32
  - _Requirements: 1.1, 1.4, 1.5, 1.6_

- [x] 9. Checkpoint — Ensure all tests pass and env_checker passes
  - Ensure all tests pass, ask the user if questions arise.

- [x] 10. Implement FastAPI layer
  - [x] 10.1 Implement session management and endpoints in `api/main.py`
    - Define Pydantic request/response models for all endpoints
    - Implement in-memory session store with UUID session IDs
    - Implement `POST /session/start` — create session, call `env.reset()`, return `{session_id, obs, info}`
    - Implement `POST /session/{id}/step` — call `env.step(action)`, return `{obs, reward, done, info}`
    - Implement `GET /session/{id}/state` — return full `EpisodeState`
    - Implement `DELETE /session/{id}` — remove session, return `{ok}`
    - Implement `GET /health` — return `{ok}`
    - Return HTTP 404 for unknown session IDs, HTTP 422 for malformed inputs, HTTP 500 for env exceptions
    - _Requirements: 8.1–8.8, 12.3, 12.4_

  - [x] 10.2 Implement training endpoints in `api/main.py`
    - Implement `POST /train` — start training run, return `{run_id}`
    - Implement `GET /train/{run_id}/status` — return `{status, metrics}`
    - _Requirements: 9.1, 9.2_

  - [x] 10.3 Write unit tests for API endpoints
    - Test `POST /session/start` returns 200 with `session_id`, `obs`, `info`
    - Test `POST /session/{id}/step` returns 200 with `obs`, `reward`, `done`, `info`
    - Test `GET /session/{id}/state` returns 200 with full state
    - Test `DELETE /session/{id}` returns 200 with `{ok}`
    - Test unknown session ID returns 404 with `{"detail": "Session not found"}`
    - Test `GET /health` returns 200 with `{ok}`
    - Test malformed request body returns 422
    - Test env exception returns 500 with `{"detail": "Environment error", ...}`
    - _Requirements: 8.1–8.8, 12.3, 12.4_

  - [x] 10.4 Write property test for session ID uniqueness
    - **Property 18: API session IDs are unique**
    - **Validates: Requirements 8.8**

- [x] 11. Implement inference script
  - Implement `run_demo(model_path, task_config, render, max_steps) -> EpisodeResult` in `scripts/run_demo.py`
  - Load trained model, run full episode, call `env.render()` after each step when `render=True`
  - Terminate episode when `max_steps` exceeded and return `EpisodeResult`
  - Add CLI entry point using `argparse`
  - _Requirements: 10.1, 10.2, 10.3_

- [x] 12. Verify JSON render round-trip
  - Implement and test `render(render_mode="json")` returns valid JSON containing all EpisodeState fields
  - _Requirements: 11.2_

  - [x] 12.1 Write property test for JSON render round-trip
    - **Property 17: JSON render is a round-trip**
    - **Validates: Requirements 11.2**

- [x] 13. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests use `hypothesis` library; unit tests use `pytest`
- Install dependencies: `gymnasium`, `stable-baselines3`, `numpy`, `fastapi`, `uvicorn`, `pydantic`, `httpx`, `pytest`, `hypothesis`, `rich`
