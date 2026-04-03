"""Unit and property tests for SellerSimulator."""
from __future__ import annotations

import copy

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ondc_env.seller_simulator import SellerSimulator
from ondc_env.types import SearchQuery, SellerConfig, SellerState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_sim(n: int = 5, seed: int = 42) -> SellerSimulator:
    sim = SellerSimulator(n, SellerConfig(), random_event_prob=0.1)
    sim.reset(seed=seed)
    return sim


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

class TestReset:
    def test_returns_correct_count(self):
        sim = make_sim(n=5)
        assert len(sim.sellers) == 5

    def test_prices_within_range(self):
        cfg = SellerConfig(price_range=(100.0, 900.0))
        sim = SellerSimulator(5, cfg)
        sim.reset()
        for s in sim.sellers:
            assert cfg.price_range[0] <= s.price <= cfg.price_range[1]

    def test_ratings_within_range(self):
        cfg = SellerConfig(rating_range=(2.5, 5.0))
        sim = SellerSimulator(5, cfg)
        sim.reset()
        for s in sim.sellers:
            assert cfg.rating_range[0] <= s.rating <= cfg.rating_range[1]

    def test_eta_within_range(self):
        cfg = SellerConfig(eta_range=(1, 10))
        sim = SellerSimulator(5, cfg)
        sim.reset()
        for s in sim.sellers:
            assert cfg.eta_range[0] <= s.delivery_eta <= cfg.eta_range[1]

    def test_stock_within_range(self):
        cfg = SellerConfig(stock_range=(0, 50))
        sim = SellerSimulator(5, cfg)
        sim.reset()
        for s in sim.sellers:
            assert cfg.stock_range[0] <= s.stock <= cfg.stock_range[1]

    def test_seeded_reset_is_reproducible(self):
        sim1 = SellerSimulator(5, SellerConfig())
        sim2 = SellerSimulator(5, SellerConfig())
        sellers1 = sim1.reset(seed=99)
        sellers2 = sim2.reset(seed=99)
        for s1, s2 in zip(sellers1, sellers2):
            assert s1.price == s2.price
            assert s1.rating == s2.rating


# ---------------------------------------------------------------------------
# get_catalog()
# ---------------------------------------------------------------------------

class TestGetCatalog:
    def test_excludes_unavailable_sellers(self):
        sim = make_sim()
        sim.sellers[0].is_available = False
        query = SearchQuery(item_name="laptop")
        offers = sim.get_catalog(query)
        ids = [o.seller_id for o in offers]
        assert "seller_0" not in ids

    def test_excludes_zero_stock_sellers(self):
        sim = make_sim()
        sim.sellers[1].stock = 0
        query = SearchQuery(item_name="laptop")
        offers = sim.get_catalog(query)
        ids = [o.seller_id for o in offers]
        assert "seller_1" not in ids

    def test_all_available_sellers_included(self):
        sim = make_sim()
        # ensure all are available with stock
        for s in sim.sellers:
            s.is_available = True
            s.stock = 10
        query = SearchQuery(item_name="laptop")
        offers = sim.get_catalog(query)
        assert len(offers) == 5

    def test_does_not_mutate_seller_states(self):
        sim = make_sim()
        before = copy.deepcopy(sim.sellers)
        sim.get_catalog(SearchQuery(item_name="laptop"))
        for orig, after in zip(before, sim.sellers):
            assert orig.price == after.price
            assert orig.stock == after.stock
            assert orig.is_available == after.is_available
            assert orig.rating == after.rating

    def test_empty_catalog_when_all_unavailable(self):
        sim = make_sim()
        for s in sim.sellers:
            s.is_available = False
        offers = sim.get_catalog(SearchQuery(item_name="laptop"))
        assert offers == []


# ---------------------------------------------------------------------------
# tick()
# ---------------------------------------------------------------------------

class TestTick:
    def test_returns_list(self):
        sim = make_sim()
        events = sim.tick()
        assert isinstance(events, list)

    def test_prices_stay_in_bounds_after_many_ticks(self):
        cfg = SellerConfig(price_range=(100.0, 900.0))
        sim = SellerSimulator(5, cfg, random_event_prob=0.0)
        sim.reset(seed=0)
        for _ in range(500):
            sim.tick()
        for s in sim.sellers:
            assert cfg.price_range[0] <= s.price <= cfg.price_range[1]

    def test_stockout_event_sets_unavailable(self):
        sim = make_sim()
        # Force a STOCKOUT on seller_0 directly
        sim._apply_event(sim.sellers[0], "STOCKOUT")
        assert sim.sellers[0].stock == 0
        assert not sim.sellers[0].is_available

    def test_seller_down_event_sets_unavailable(self):
        sim = make_sim()
        sim._apply_event(sim.sellers[0], "SELLER_DOWN")
        assert not sim.sellers[0].is_available

    def test_delay_event_increases_eta(self):
        sim = make_sim()
        original_eta = sim.sellers[0].delivery_eta
        sim._apply_event(sim.sellers[0], "DELAY")
        assert sim.sellers[0].delivery_eta > original_eta

    def test_price_spike_increases_price(self):
        sim = make_sim()
        original_price = sim.sellers[0].price
        sim._apply_event(sim.sellers[0], "PRICE_SPIKE")
        assert sim.sellers[0].price >= original_price

    def test_tick_events_reference_valid_seller_ids(self):
        sim = SellerSimulator(5, SellerConfig(), random_event_prob=1.0)
        sim.reset(seed=7)
        events = sim.tick()
        valid_ids = {s.seller_id for s in sim.sellers}
        for e in events:
            assert e.seller_id in valid_ids


# ---------------------------------------------------------------------------
# Property: prices stay within bounds after tick (Property 7)
# ---------------------------------------------------------------------------

@given(
    price_lo=st.floats(min_value=1.0, max_value=500.0, allow_nan=False, allow_infinity=False),
    price_hi_offset=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    price_volatility=st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False),
    n_sellers=st.integers(min_value=1, max_value=5),
    n_ticks=st.integers(min_value=1, max_value=200),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=100)
def test_seller_prices_stay_within_bounds(
    price_lo, price_hi_offset, price_volatility, n_sellers, n_ticks, seed
):
    """**Validates: Requirements 5.5**

    Property 7: Seller prices stay within configured bounds after tick.
    For any SellerConfig and any number of tick() calls, all seller prices
    must remain within [config.price_range[0], config.price_range[1]].
    """
    price_hi = price_lo + price_hi_offset
    cfg = SellerConfig(
        price_range=(price_lo, price_hi),
        price_volatility=price_volatility,
    )
    sim = SellerSimulator(n_sellers, cfg, random_event_prob=0.0)
    sim.reset(seed=seed)
    for _ in range(n_ticks):
        sim.tick()
    for s in sim.sellers:
        assert cfg.price_range[0] <= s.price <= cfg.price_range[1], (
            f"Price {s.price} out of bounds {cfg.price_range}"
        )


# ---------------------------------------------------------------------------
# Property: get_catalog does not mutate state (Property 15)
# ---------------------------------------------------------------------------

@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=100)
def test_get_catalog_does_not_mutate_state(seed):
    """Property 15: SellerSimulator does not mutate state during get_catalog()."""
    sim = SellerSimulator(5, SellerConfig())
    sim.reset(seed=seed)
    before = copy.deepcopy(sim.sellers)
    sim.get_catalog(SearchQuery(item_name="phone"))
    for orig, after in zip(before, sim.sellers):
        assert orig.price == after.price
        assert orig.stock == after.stock
        assert orig.is_available == after.is_available


# ---------------------------------------------------------------------------
# Property: catalog only returns available sellers (Property 9)
# ---------------------------------------------------------------------------

@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=100)
def test_catalog_only_returns_available_sellers(seed):
    """Property 9: Catalog only returns available sellers."""
    sim = SellerSimulator(5, SellerConfig())
    sim.reset(seed=seed)
    offers = sim.get_catalog(SearchQuery(item_name="item"))
    for offer in offers:
        assert offer.is_available
        assert offer.price > 0
        assert 0.0 <= offer.rating <= 5.0
        assert offer.delivery_eta > 0
