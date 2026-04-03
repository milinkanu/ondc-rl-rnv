"""
Core data models, enums, and constants for ONDCAgentEnv.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_OBS_DIM: int = 61
N_ACTIONS: int = 15
MAX_SELLERS: int = 5

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class BecknPhase(IntEnum):
    SEARCH = 0
    SELECT = 1
    INIT = 2
    CONFIRM = 3
    TRACK = 4
    POST_ORDER = 5


class ActionType(IntEnum):
    # Phase: SEARCH
    SEARCH_PRODUCTS = 0

    # Phase: SELECT (one action per seller slot, up to MAX_SELLERS)
    SELECT_SELLER_0 = 1
    SELECT_SELLER_1 = 2
    SELECT_SELLER_2 = 3
    SELECT_SELLER_3 = 4
    SELECT_SELLER_4 = 5

    # Phase: INIT
    INIT_ORDER = 6

    # Phase: CONFIRM
    CONFIRM_ORDER = 7
    CANCEL_BEFORE_CONFIRM = 8

    # Phase: TRACK
    TRACK_ORDER = 9

    # Phase: POST_ORDER
    ACCEPT_DELIVERY = 10
    CANCEL_ORDER = 11
    RETURN_ITEM = 12
    FILE_GRIEVANCE = 13

    # Universal
    WAIT = 14


class OrderStatus(IntEnum):
    PENDING = 0
    CONFIRMED = 1
    SHIPPED = 2
    DELIVERED = 3
    CANCELLED = 4
    FAILED = 5


# ---------------------------------------------------------------------------
# Seller models
# ---------------------------------------------------------------------------

@dataclass
class SellerState:
    seller_id: str
    name: str
    price: float
    original_price: float
    stock: int
    rating: float                   # 0.0–5.0
    delivery_eta: int               # steps
    is_available: bool
    discount_pct: float
    fulfillment_type: str           # "standard" | "express" | "same_day"


@dataclass
class SellerOffer:
    seller_id: str
    item_id: str
    name: str
    price: float
    rating: float
    delivery_eta: int
    stock: int
    is_available: bool
    fulfillment_type: str


@dataclass
class SellerEvent:
    seller_id: str
    event_type: str                 # "STOCKOUT" | "PRICE_SPIKE" | "DELAY" | "SELLER_DOWN"
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchQuery:
    item_name: str
    max_price: float | None = None
    min_rating: float | None = None


@dataclass
class SelectResult:
    success: bool
    offer: SellerOffer | None
    reason: str = ""


# ---------------------------------------------------------------------------
# Episode / environment state
# ---------------------------------------------------------------------------

@dataclass
class EpisodeState:
    # Task context
    task_id: str
    target_item: str
    budget: float
    initial_budget: float
    urgency: float                          # 0.0 (low) to 1.0 (high)
    steps_remaining: int
    max_steps: int

    # Beckn phase
    current_phase: BecknPhase

    # Search results (padded to MAX_SELLERS)
    sellers: list[SellerState] = field(default_factory=list)

    # Selected offer
    selected_offer: SellerOffer | None = None

    # Order state
    order_id: str | None = None
    order_status: OrderStatus | None = None
    delivery_eta: float = 0.0
    tracking_updates: list[str] = field(default_factory=list)

    # Episode metrics
    total_spent: float = 0.0
    total_reward: float = 0.0
    invalid_action_count: int = 0


# ---------------------------------------------------------------------------
# Config models
# ---------------------------------------------------------------------------

@dataclass
class RewardWeights:
    # Positive signals
    task_completion: float = 10.0
    good_price: float = 2.0
    fast_delivery: float = 1.5
    high_seller_rating: float = 1.0
    successful_return: float = 3.0

    # Negative signals
    invalid_action: float = -1.0
    budget_exceeded: float = -5.0
    order_failed: float = -3.0
    unnecessary_wait: float = -0.1
    late_delivery: float = -2.0


@dataclass
class SellerConfig:
    price_range: tuple[float, float] = (100.0, 900.0)
    rating_range: tuple[float, float] = (2.5, 5.0)
    eta_range: tuple[int, int] = (1, 10)
    stock_range: tuple[int, int] = (0, 50)
    price_volatility: float = 0.05


@dataclass
class EnvConfig:
    max_steps: int = 50
    n_sellers: int = MAX_SELLERS
    initial_budget: float = 1000.0
    urgency_range: tuple[float, float] = (0.2, 0.9)
    random_event_prob: float = 0.1
    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    seller_config: SellerConfig = field(default_factory=SellerConfig)
    seed: int | None = None


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

@dataclass
class RewardResult:
    total: float
    breakdown: dict[str, float] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    is_valid: bool
    reason: str = ""
