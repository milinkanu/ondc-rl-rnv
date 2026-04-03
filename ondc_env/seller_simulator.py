"""
SellerSimulator: simulates N sellers with dynamic pricing and random events.
"""
from __future__ import annotations

import random
import uuid
from typing import List

from ondc_env.types import (
    MAX_SELLERS,
    SearchQuery,
    SelectResult,
    SellerConfig,
    SellerEvent,
    SellerOffer,
    SellerState,
)

_EVENT_TYPES = ["STOCKOUT", "PRICE_SPIKE", "DELAY", "SELLER_DOWN"]
_EVENT_WEIGHTS = [0.3, 0.3, 0.3, 0.1]


class SellerSimulator:
    """Simulates a market of n_sellers with Gaussian price drift and random events."""

    def __init__(self, n_sellers: int, config: SellerConfig, random_event_prob: float = 0.1) -> None:
        self.n_sellers = n_sellers
        self.config = config
        self.random_event_prob = random_event_prob
        self.sellers: list[SellerState] = []
        self._rng = random.Random()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> list[SellerState]:
        """Initialise sellers with values sampled within configured ranges."""
        self._rng = random.Random(seed)
        cfg = self.config
        self.sellers = []

        for i in range(self.n_sellers):
            price = self._rng.uniform(*cfg.price_range)
            self.sellers.append(
                SellerState(
                    seller_id=f"seller_{i}",
                    name=f"Seller {i}",
                    price=round(price, 2),
                    original_price=round(price, 2),
                    stock=self._rng.randint(*cfg.stock_range),
                    rating=round(self._rng.uniform(*cfg.rating_range), 2),
                    delivery_eta=self._rng.randint(*cfg.eta_range),
                    is_available=True,
                    discount_pct=round(self._rng.uniform(0.0, 0.3), 2),
                    fulfillment_type=self._rng.choice(["standard", "express", "same_day"]),
                )
            )

        return self.sellers

    def get_catalog(self, query: SearchQuery) -> list[SellerOffer]:
        """Return offers from available sellers only. Does NOT mutate seller states."""
        offers: list[SellerOffer] = []
        for seller in self.sellers:
            if not seller.is_available:
                continue
            if seller.stock <= 0:
                continue
            offer = SellerOffer(
                seller_id=seller.seller_id,
                item_id=f"{seller.seller_id}_item",
                name=query.item_name,
                price=seller.price,
                rating=seller.rating,
                delivery_eta=seller.delivery_eta,
                stock=seller.stock,
                is_available=seller.is_available,
                fulfillment_type=seller.fulfillment_type,
            )
            offers.append(offer)
        return offers

    def apply_selection(self, seller_id: str, item_id: str, quantity: int) -> SelectResult:
        """Attempt to select a seller/item. Returns SelectResult."""
        seller = self._find_seller(seller_id)
        if seller is None:
            return SelectResult(success=False, offer=None, reason=f"Seller {seller_id} not found")
        if not seller.is_available:
            return SelectResult(success=False, offer=None, reason=f"Seller {seller_id} is unavailable")
        if seller.stock < quantity:
            return SelectResult(success=False, offer=None, reason=f"Insufficient stock ({seller.stock} < {quantity})")

        offer = SellerOffer(
            seller_id=seller.seller_id,
            item_id=item_id,
            name=item_id,
            price=seller.price,
            rating=seller.rating,
            delivery_eta=seller.delivery_eta,
            stock=seller.stock,
            is_available=seller.is_available,
            fulfillment_type=seller.fulfillment_type,
        )
        return SelectResult(success=True, offer=offer)

    def tick(self) -> list[SellerEvent]:
        """Advance time: apply Gaussian price drift and fire random events."""
        cfg = self.config
        events: list[SellerEvent] = []

        for seller in self.sellers:
            # Gaussian price drift
            delta = self._rng.gauss(0.0, seller.price * cfg.price_volatility)
            new_price = seller.price + delta
            # Round first, then clamp — avoids round-up exceeding bounds
            rounded_price = round(new_price, 2)
            seller.price = max(cfg.price_range[0], min(cfg.price_range[1], rounded_price))

            # Random event
            if self._rng.random() < self.random_event_prob:
                event_type = self._rng.choices(_EVENT_TYPES, weights=_EVENT_WEIGHTS, k=1)[0]
                self._apply_event(seller, event_type)
                events.append(SellerEvent(seller_id=seller.seller_id, event_type=event_type))

        return events

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_seller(self, seller_id: str) -> SellerState | None:
        for s in self.sellers:
            if s.seller_id == seller_id:
                return s
        return None

    def _apply_event(self, seller: SellerState, event_type: str) -> None:
        cfg = self.config
        if event_type == "STOCKOUT":
            seller.stock = 0
            seller.is_available = False
        elif event_type == "PRICE_SPIKE":
            multiplier = self._rng.uniform(1.1, 1.5)
            rounded_spike = round(seller.price * multiplier, 2)
            seller.price = max(cfg.price_range[0], min(cfg.price_range[1], rounded_spike))
        elif event_type == "DELAY":
            seller.delivery_eta += self._rng.randint(1, 3)
        elif event_type == "SELLER_DOWN":
            seller.is_available = False
