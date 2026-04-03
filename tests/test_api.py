"""
Unit tests for the FastAPI layer (api/main.py).
Requirements: 8.1–8.8, 12.3, 12.4
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app, _sessions, _runs

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_stores():
    """Reset in-memory stores before each test."""
    _sessions.clear()
    _runs.clear()
    yield
    _sessions.clear()
    _runs.clear()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def session_id(client):
    """Create a session and return its ID."""
    r = client.post("/session/start", json={})
    assert r.status_code == 200
    return r.json()["session_id"]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_health_returns_200_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


# ---------------------------------------------------------------------------
# POST /session/start
# ---------------------------------------------------------------------------


def test_session_start_returns_200_with_required_fields(client):
    r = client.post("/session/start", json={})
    assert r.status_code == 200
    data = r.json()
    assert "session_id" in data
    assert "obs" in data
    assert "info" in data


def test_session_start_obs_is_list_of_floats(client):
    r = client.post("/session/start", json={})
    obs = r.json()["obs"]
    assert isinstance(obs, list)
    assert len(obs) == 61
    assert all(isinstance(v, (int, float)) for v in obs)


def test_session_start_session_id_is_uuid_string(client):
    import uuid
    r = client.post("/session/start", json={})
    sid = r.json()["session_id"]
    # Should parse as a valid UUID
    uuid.UUID(sid)


def test_session_start_with_custom_config(client):
    r = client.post("/session/start", json={
        "max_steps": 20,
        "initial_budget": 500.0,
        "target_item": "laptop",
        "urgency": 0.7,
    })
    assert r.status_code == 200
    assert "session_id" in r.json()


# ---------------------------------------------------------------------------
# POST /session/{id}/step
# ---------------------------------------------------------------------------


def test_session_step_returns_200_with_required_fields(client, session_id):
    r = client.post(f"/session/{session_id}/step", json={"action": 0})
    assert r.status_code == 200
    data = r.json()
    assert "obs" in data
    assert "reward" in data
    assert "done" in data
    assert "info" in data


def test_session_step_obs_is_list_of_61_floats(client, session_id):
    r = client.post(f"/session/{session_id}/step", json={"action": 0})
    obs = r.json()["obs"]
    assert isinstance(obs, list)
    assert len(obs) == 61


def test_session_step_reward_is_float(client, session_id):
    r = client.post(f"/session/{session_id}/step", json={"action": 0})
    assert isinstance(r.json()["reward"], (int, float))


def test_session_step_done_is_bool(client, session_id):
    r = client.post(f"/session/{session_id}/step", json={"action": 0})
    assert isinstance(r.json()["done"], bool)


# ---------------------------------------------------------------------------
# GET /session/{id}/state
# ---------------------------------------------------------------------------


def test_session_state_returns_200_with_full_state(client, session_id):
    r = client.get(f"/session/{session_id}/state")
    assert r.status_code == 200
    data = r.json()
    # Check key EpisodeState fields are present
    assert "task_id" in data
    assert "budget" in data
    assert "current_phase" in data
    assert "steps_remaining" in data
    assert "sellers" in data


# ---------------------------------------------------------------------------
# DELETE /session/{id}
# ---------------------------------------------------------------------------


def test_session_delete_returns_200_ok(client, session_id):
    r = client.delete(f"/session/{session_id}")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_session_delete_removes_session(client, session_id):
    client.delete(f"/session/{session_id}")
    # Subsequent step should 404
    r = client.post(f"/session/{session_id}/step", json={"action": 0})
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# 404 for unknown session IDs
# ---------------------------------------------------------------------------


def test_step_unknown_session_returns_404(client):
    r = client.post("/session/nonexistent-id/step", json={"action": 0})
    assert r.status_code == 404
    assert r.json()["detail"] == "Session not found"


def test_state_unknown_session_returns_404(client):
    r = client.get("/session/nonexistent-id/state")
    assert r.status_code == 404
    assert r.json()["detail"] == "Session not found"


def test_delete_unknown_session_returns_404(client):
    r = client.delete("/session/nonexistent-id")
    assert r.status_code == 404
    assert r.json()["detail"] == "Session not found"


# ---------------------------------------------------------------------------
# 422 for malformed inputs
# ---------------------------------------------------------------------------


def test_step_missing_action_returns_422(client, session_id):
    r = client.post(f"/session/{session_id}/step", json={})
    assert r.status_code == 422


def test_step_invalid_action_type_returns_422(client, session_id):
    r = client.post(f"/session/{session_id}/step", json={"action": "not_an_int"})
    assert r.status_code == 422


def test_session_start_invalid_budget_returns_422(client):
    r = client.post("/session/start", json={"initial_budget": -100})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# 500 for env exceptions
# ---------------------------------------------------------------------------


def test_step_env_exception_returns_500(client, monkeypatch):
    """Patch env.step to raise an exception and verify 500 response."""
    r = client.post("/session/start", json={})
    sid = r.json()["session_id"]

    from ondc_env import ONDCAgentEnv

    def boom(*args, **kwargs):
        raise RuntimeError("simulated crash")

    monkeypatch.setattr(ONDCAgentEnv, "step", boom)

    r = client.post(f"/session/{sid}/step", json={"action": 0})
    assert r.status_code == 500
    body = r.json()
    assert body["detail"]["detail"] == "Environment error"
    assert "error" in body["detail"]
