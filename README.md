---
title: ONDC RL Simulator
emoji: 🛒
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
---

# ONDCAgentEnv

A reinforcement learning environment that simulates the [ONDC (Open Network for Digital Commerce)](https://ondc.org/) buyer-seller lifecycle. An RL agent acts as a buyer navigating the full Beckn protocol flow, making decisions across a dynamic multi-seller market with real-world constraints like budget, delivery urgency, and seller reliability.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Folder Structure](#3-folder-structure)
4. [API Endpoints](#4-api-endpoints)
5. [Agent Workflow](#5-agent-workflow)
6. [Environment & Simulation Logic](#6-environment--simulation-logic)
7. [Key Algorithms & Decision Logic](#7-key-algorithms--decision-logic)
8. [Setup & Run Instructions](#8-setup--run-instructions)

---

## 1. Project Overview

### What is this?

ONDCAgentEnv is a [Gymnasium](https://gymnasium.farama.org/)-compatible RL environment. It simulates a buyer on the ONDC network trying to find, select, and purchase a product from multiple competing sellers — all while managing a budget, delivery urgency, and unpredictable market events (price spikes, stockouts, seller failures).

A trained RL agent learns to navigate this process optimally: picking the right seller, confirming orders at the right time, and avoiding costly mistakes like overspending or acting out of protocol order.

### Why does it exist?

ONDC uses the **Beckn protocol** — a structured, phase-based flow for commerce transactions. This project models that flow as an RL problem so agents can be trained to make smart buyer decisions in a realistic, dynamic environment.

### Key capabilities

- Full Beckn protocol lifecycle: `SEARCH → SELECT → INIT → CONFIRM → TRACK → POST_ORDER`
- 5 simulated sellers with dynamic pricing, stock, ratings, and random disruptions
- Configurable reward shaping (price, speed, rating, penalties)
- FastAPI HTTP layer for external clients and demo UIs
- PPO training via Stable-Baselines3
- Property-based testing with Hypothesis

---

## 2. Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────┐
│                    External Clients                      │
│           (Demo UI / HTTP Client / CLI Script)           │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                   FastAPI Layer (api/)                   │
│   Session management, training endpoints, health check   │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              ONDCAgentEnv (ondc_env/env.py)              │
│         Gymnasium-compatible RL environment core         │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │  TaskEngine  │  │SellerSimulat.│  │ RewardSystem  │  │
│  │ Phase rules  │  │ Market sim   │  │ Reward shaping│  │
│  │ & validation │  │ & events     │  │ & breakdown   │  │
│  └──────────────┘  └──────────────┘  └───────────────┘  │
│                                                          │
│              EpisodeState (shared state)                 │
│              ObservationBuilder (61-dim vector)          │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│           Training / Inference (scripts/)                │
│     train.py (PPO via SB3)  |  run_demo.py (inference)  │
└─────────────────────────────────────────────────────────┘
```

### How components interact

Each call to `env.step(action)` flows through all three sub-components in sequence:

1. **TaskEngine** validates whether the action is legal in the current Beckn phase
2. **SellerSimulator** applies the action's market effects and advances time (tick)
3. **RewardSystem** computes a scalar reward with a per-component breakdown

All three read from and write to a shared `EpisodeState` dataclass.

---

## 3. Folder Structure

```
ondc-agent-env/
│
├── ondc_env/               # Core RL environment package
│   ├── __init__.py         # Public exports
│   ├── env.py              # ONDCAgentEnv — main Gymnasium environment
│   ├── types.py            # All dataclasses, enums, and constants
│   ├── task_engine.py      # Beckn phase rules and action validation
│   ├── seller_simulator.py # Dynamic seller market simulation
│   └── reward_system.py    # Modular reward computation
│
├── api/
│   └── main.py             # FastAPI app — session and training endpoints
│
├── scripts/
│   ├── train.py            # PPO training script (Stable-Baselines3)
│   └── run_demo.py         # Inference/demo script for a trained model
│
├── tests/                  # pytest + Hypothesis test suite
│   ├── test_env_reset.py
│   ├── test_env_step.py
│   ├── test_env_properties.py
│   ├── test_step_properties.py
│   ├── test_task_engine.py
│   ├── test_reward_system.py
│   ├── test_seller_simulator.py
│   ├── test_api.py
│   └── test_api_properties.py
│
├── models/
│   └── ppo_ondc.zip        # Pre-trained PPO model checkpoint
│
└── requirements.txt
```

### Why each folder exists

| Folder | Purpose |
|---|---|
| `ondc_env/` | The environment itself — everything the RL agent interacts with |
| `api/` | HTTP interface so non-Python clients can drive the environment |
| `scripts/` | Standalone CLI tools for training and running demos |
| `tests/` | Unit, integration, and property-based tests for correctness guarantees |
| `models/` | Saved model checkpoints from training runs |

---

## 4. API Endpoints

The FastAPI server exposes endpoints for session management, state inspection, and training orchestration. Start the server with:

```bash
uvicorn api.main:app --reload
```

### Session Endpoints

#### `POST /session/start`

Creates a new environment session and returns the initial observation.

**Why:** Allows external clients to start a fresh episode without embedding Python.

**Request body:**
```json
{
  "max_steps": 50,
  "n_sellers": 5,
  "initial_budget": 1000.0,
  "target_item": "laptop",
  "urgency": 0.7,
  "seed": 42
}
```

**Response:**
```json
{
  "session_id": "3f8a1c2d-...",
  "obs": [0.95, 0.7, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...],
  "info": {
    "phase": 0,
    "budget": 1000.0,
    "urgency": 0.7
  }
}
```

---

#### `POST /session/{session_id}/step`

Executes one action in the environment and returns the result.

**Why:** Drives the agent forward one step at a time from an external client.

**Request body:**
```json
{ "action": 0 }
```

**Response:**
```json
{
  "obs": [0.95, 0.7, 0.98, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, ...],
  "reward": 0.0,
  "done": false,
  "info": {
    "phase": 1,
    "reward_breakdown": {},
    "events": [],
    "invalid_action": false
  }
}
```

---

#### `GET /session/{session_id}/state`

Returns the full internal `EpisodeState` for inspection or debugging.

**Response:** Full JSON of the current episode state including sellers, order status, budget, phase, etc.

---

#### `DELETE /session/{session_id}`

Removes a session and frees its memory.

**Response:**
```json
{ "ok": true }
```

---

### Training Endpoints

#### `POST /train`

Registers a training run and returns a run ID. (Training itself runs via `scripts/train.py`.)

**Request body:**
```json
{
  "max_steps": 50,
  "n_sellers": 5,
  "initial_budget": 1000.0,
  "total_timesteps": 10000,
  "seed": 42
}
```

**Response:**
```json
{ "run_id": "a1b2c3d4-..." }
```

---

#### `GET /train/{run_id}/status`

Polls the status and metrics of a training run.

**Response:**
```json
{
  "status": "running",
  "metrics": {
    "total_timesteps": 10000,
    "steps_done": 4200
  }
}
```

---

#### `GET /health`

Health check endpoint.

**Response:**
```json
{ "ok": true }
```

---

### Error Responses

| Status | Condition |
|---|---|
| `404` | Session or run ID not found |
| `422` | Invalid request body (Pydantic validation failure) |
| `500` | Unexpected exception inside `env.step()` |

---

## 5. Agent Workflow

The agent follows the **Beckn protocol** — a strict phase-based flow. Each phase allows only specific actions. Taking an action outside the allowed set is penalized.

### Phase Flow

```
SEARCH ──► SELECT ──► INIT ──► CONFIRM ──► TRACK ──► POST_ORDER
  │                              │            │
  │◄─────────────────────────────┘            │
  │         (CANCEL_BEFORE_CONFIRM)           │
  │◄──────────────────────────────────────────┘
              (CANCEL_ORDER)
```

### Step-by-Step Agent Lifecycle

**Step 1 — SEARCH phase**
- Agent takes action `SEARCH_PRODUCTS` (action `0`)
- Environment returns a catalog of up to 5 sellers with prices, ratings, ETAs, and stock
- Phase advances to `SELECT`

**Step 2 — SELECT phase**
- Agent picks one of the sellers: `SELECT_SELLER_0` through `SELECT_SELLER_4` (actions `1–5`)
- The chosen seller's offer is stored as `selected_offer` in the episode state
- Phase advances to `INIT`

**Step 3 — INIT phase**
- Agent takes `INIT_ORDER` (action `6`) to initialize the order
- Phase advances to `CONFIRM`

**Step 4 — CONFIRM phase**
- Agent either confirms (`CONFIRM_ORDER`, action `7`) or cancels (`CANCEL_BEFORE_CONFIRM`, action `8`)
- On confirm: budget is deducted, order ID is assigned, phase moves to `TRACK`
- On cancel: phase resets to `SEARCH`

**Step 5 — TRACK phase**
- Agent takes `TRACK_ORDER` (action `9`) to simulate delivery progress
- Each track step decrements `delivery_eta` by 1
- When ETA reaches 0, order status becomes `SHIPPED`; phase moves to `POST_ORDER`

**Step 6 — POST_ORDER phase**
- Agent can `ACCEPT_DELIVERY` (action `10`), `CANCEL_ORDER` (action `11`), `RETURN_ITEM` (action `12`), or `FILE_GRIEVANCE` (action `13`)
- Accepting delivery marks the order as `DELIVERED` and ends the episode

### Action Table

| Action ID | Name | Valid Phase |
|---|---|---|
| 0 | `SEARCH_PRODUCTS` | SEARCH |
| 1–5 | `SELECT_SELLER_0` to `SELECT_SELLER_4` | SELECT |
| 6 | `INIT_ORDER` | INIT |
| 7 | `CONFIRM_ORDER` | CONFIRM |
| 8 | `CANCEL_BEFORE_CONFIRM` | CONFIRM |
| 9 | `TRACK_ORDER` | TRACK |
| 10 | `ACCEPT_DELIVERY` | POST_ORDER |
| 11 | `CANCEL_ORDER` | TRACK, POST_ORDER |
| 12 | `RETURN_ITEM` | POST_ORDER |
| 13 | `FILE_GRIEVANCE` | POST_ORDER |
| 14 | `WAIT` | Any phase |

---

## 6. Environment & Simulation Logic

### Observation Space

Every call to `reset()` or `step()` returns a flat `float32` numpy array of shape `(61,)`. Here's what each section encodes:

```
Index   Description
─────────────────────────────────────────────────────────
[0]     budget_normalized       = budget / initial_budget
[1]     urgency                 = 0.0 (low) to 1.0 (high)
[2]     steps_remaining_norm    = steps_remaining / max_steps
[3:9]   phase_onehot            = one-hot over 6 Beckn phases
[9:49]  per_seller_features     = 5 sellers × 8 features each
          per seller: price_norm, rating_norm, eta_norm,
                      stock_norm, discount_pct, is_available,
                      fulfillment_type_norm, (reserved)
[49]    selected_price_norm
[50]    selected_rating_norm
[51]    selected_eta_norm
[52]    has_selection           = 1.0 if a seller is selected
[53:58] order_status_onehot     = one-hot over 5 order statuses
[58]    total_spent_norm        = total_spent / initial_budget
[59]    invalid_action_ratio    = invalid_actions / steps_taken
[60]    delivery_eta_norm       = delivery_eta / max_eta
```

### Seller Simulation

The `SellerSimulator` maintains 5 sellers. On every valid `step()`, it calls `tick()` which:

1. Applies **Gaussian price drift** to each seller: `new_price = price + Normal(0, price × volatility)`, clamped to `[price_range[0], price_range[1]]`
2. With probability `random_event_prob` (default 10%), fires a **random event** for each seller:

| Event | Effect | Probability |
|---|---|---|
| `STOCKOUT` | Sets stock to 0, marks seller unavailable | 30% |
| `PRICE_SPIKE` | Multiplies price by 1.1–1.5× | 30% |
| `DELAY` | Adds 1–3 steps to delivery ETA | 30% |
| `SELLER_DOWN` | Marks seller unavailable | 10% |

If the agent's selected seller goes down after selection but before confirmation, the order status is set to `FAILED`.

### Episode Termination

An episode ends when any of these conditions are met:

- Order status becomes `CONFIRMED`, `CANCELLED`, or `DELIVERED` → `terminated = True`
- `total_spent >= budget` → `terminated = True`
- `steps_remaining` reaches 0 → `truncated = True`

### Reward System

Rewards are computed as a weighted sum of components. The breakdown is returned in `info["reward_breakdown"]` every step.

**Positive rewards (earned):**

| Component | Condition | Default Weight |
|---|---|---|
| `task_completion` | Order confirmed | +10.0 |
| `good_price` | Confirmed price ≤ 90% of budget | +2.0 |
| `fast_delivery` | ETA within urgency threshold | +1.5 |
| `high_seller_rating` | Selected seller rating ≥ 4.0 | +1.0 |
| `task_completion` (partial) | Delivery accepted | +5.0 |
| `successful_return` | Return/grievance resolved | +3.0 |

**Penalties (incurred):**

| Component | Condition | Default Weight |
|---|---|---|
| `invalid_action` | Action not valid in current phase | -1.0 |
| `budget_exceeded` | `total_spent > budget` | -5.0 |
| `order_failed` | Seller failure after selection | -3.0 |
| `unnecessary_wait` | `WAIT` action taken | -0.1 |
| `late_delivery` | ETA exceeds urgency deadline | -2.0 |

The urgency deadline is computed as: `max_eta × (1.0 - urgency)`. Higher urgency = tighter deadline.

---

## 7. Key Algorithms & Decision Logic

### Environment Step Algorithm

```
step(action):
  1. Validate action via TaskEngine
     → If invalid: increment invalid_action_count, apply penalty, return current obs
  2. Apply action effects (select seller, confirm order, track, etc.)
  3. Advance Beckn phase via TaskEngine.transition()
  4. Tick SellerSimulator (price drift + random events)
  5. Sync seller states into EpisodeState
  6. Decrement steps_remaining by 1
  7. Compute reward via RewardSystem
  8. Check termination (TaskEngine.is_terminal)
  9. Build and return 61-dim observation
```

### Phase Transition Table

```
(Current Phase, Action)          → Next Phase
────────────────────────────────────────────────
(SEARCH,  SEARCH_PRODUCTS)       → SELECT
(SELECT,  SELECT_SELLER_*)       → INIT
(INIT,    INIT_ORDER)            → CONFIRM
(CONFIRM, CONFIRM_ORDER)         → TRACK
(CONFIRM, CANCEL_BEFORE_CONFIRM) → SEARCH
(TRACK,   TRACK_ORDER)           → POST_ORDER
(TRACK,   CANCEL_ORDER)          → SEARCH
(POST_ORDER, ACCEPT_DELIVERY)    → POST_ORDER (terminal)
(*, WAIT)                        → same phase
```

### Seller Price Drift (Gaussian Random Walk)

Each tick, every seller's price evolves as:

```python
delta = Normal(mean=0, std=price × price_volatility)
new_price = clamp(price + delta, price_range[0], price_range[1])
```

With default `price_volatility = 0.05`, a seller priced at ₹500 has a standard deviation of ₹25 per step. This creates realistic price fluctuation that the agent must account for when deciding when to confirm.

### Urgency-ETA Threshold

The fast delivery reward and late delivery penalty both use this threshold:

```python
eta_threshold = max_eta × (1.0 - urgency)
```

Examples:
- `urgency=0.9` → threshold = `10 × 0.1 = 1 step` (very tight)
- `urgency=0.2` → threshold = `10 × 0.8 = 8 steps` (relaxed)

This means a high-urgency agent must pick sellers with very fast delivery to earn the `fast_delivery` bonus.

---

## 8. Setup & Run Instructions

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repo
git clone <repo-url>
cd ondc-agent-env

# Create and activate a virtual environment (recommended)
python -m venv ondc_env_venv
source ondc_env_venv/bin/activate   # Windows: ondc_env_venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Tests

```bash
pytest tests/
```

For property-based tests only:
```bash
pytest tests/test_env_properties.py tests/test_step_properties.py tests/test_api_properties.py -v
```

### Train a Model

```bash
python scripts/train.py --timesteps 20000 --output models/ppo_ondc.zip --seed 42
```

Options:
- `--timesteps` — total training timesteps (default: 20000)
- `--output` — path to save the model (default: `models/ppo_ondc.zip`)
- `--seed` — random seed for reproducibility

### Run a Demo Episode

```bash
python scripts/run_demo.py --model models/ppo_ondc.zip --budget 800 --urgency 0.7 --target-item laptop
```

Options:
- `--model` — path to a trained `.zip` model (required)
- `--budget` — episode budget (default: 1000.0)
- `--urgency` — urgency level 0.0–1.0 (default: 0.5)
- `--target-item` — item name to search for (default: "laptop")
- `--max-steps` — max steps before truncation (default: 50)
- `--no-render` — disable per-step console output
- `--seed` — random seed

### Start the API Server

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

API docs available at: `http://localhost:8000/docs`

### Use the Environment Directly (Python)

```python
from ondc_env import ONDCAgentEnv, EnvConfig

env = ONDCAgentEnv(EnvConfig(max_steps=50, initial_budget=1000.0))
obs, info = env.reset(seed=42, options={"target_item": "phone", "urgency": 0.6})

done = False
while not done:
    action = env.action_space.sample()          # replace with your agent
    obs, reward, terminated, truncated, info = env.step(action)
    print(env.render(render_mode="human"))
    done = terminated or truncated

env.close()
```

### Configuration Reference

```python
from ondc_env import EnvConfig, SellerConfig, RewardWeights

config = EnvConfig(
    max_steps=50,               # steps per episode
    n_sellers=5,                # number of simulated sellers
    initial_budget=1000.0,      # starting budget
    urgency_range=(0.2, 0.9),   # urgency sampled from this range on reset
    random_event_prob=0.1,      # probability of a seller event per tick
    seed=42,                    # set for reproducibility
    seller_config=SellerConfig(
        price_range=(100.0, 900.0),
        rating_range=(2.5, 5.0),
        eta_range=(1, 10),
        stock_range=(0, 50),
        price_volatility=0.05,
    ),
    reward_weights=RewardWeights(
        task_completion=10.0,
        good_price=2.0,
        fast_delivery=1.5,
        high_seller_rating=1.0,
        invalid_action=-1.0,
        budget_exceeded=-5.0,
    ),
)
```
