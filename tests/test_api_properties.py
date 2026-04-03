"""
Property-based tests for the FastAPI layer.

Property 18: API session IDs are unique
  For any number of POST /session/start calls, all returned session_id values
  must be distinct UUIDs.

Validates: Requirements 8.8
"""
from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient
from hypothesis import given, settings
from hypothesis import strategies as st

from api.main import app, _sessions, _runs


@pytest.fixture(autouse=True)
def clear_stores():
    _sessions.clear()
    _runs.clear()
    yield
    _sessions.clear()
    _runs.clear()


@given(n=st.integers(min_value=2, max_value=20))
@settings(max_examples=50)
def test_session_ids_are_unique(n):
    """
    Property 18: API session IDs are unique

    For any number of POST /session/start calls, all returned session_id values
    must be distinct UUIDs.

    Validates: Requirements 8.8
    """
    _sessions.clear()
    client = TestClient(app)

    session_ids = []
    for _ in range(n):
        r = client.post("/session/start", json={})
        assert r.status_code == 200
        sid = r.json()["session_id"]
        session_ids.append(sid)

    # All IDs must be valid UUIDs
    for sid in session_ids:
        uuid.UUID(sid)  # raises ValueError if not a valid UUID

    # All IDs must be distinct
    assert len(session_ids) == len(set(session_ids)), (
        f"Duplicate session IDs found among {len(session_ids)} sessions"
    )
